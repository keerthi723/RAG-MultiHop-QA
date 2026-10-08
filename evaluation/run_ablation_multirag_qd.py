import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import json
import csv
import argparse
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.passage_utils import extract_structured_paragraphs
from retrieval.multi_query import decompose_question, run_multi_query_retrieval, retrieve_route
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank_candidates
from retrieval.diversity import select_diverse_evidence
from generation.generator import generate_answer_from_evidence
from evaluation.answer_metrics import exact_match, token_f1
from evaluation.retrieval_metrics import compute_retrieval_metrics

STAGES = [
    ("A", "Baseline RAG (Single query, no QD, no RRF, no CE, no Div)"),
    ("B", "QD only (Concatenated subqueries -> dense retrieval)"),
    ("C", "QD + Multi-Query (Independent subquery retrieval, max sim pool)"),
    ("D", "QD + MQ + RRF (Reciprocal Rank Fusion k=60)"),
    ("E", "QD + MQ + RRF + CE (Cross-Encoder reranking by original query)"),
    ("F", "Full MultiRAG-QD (+ Evidence Diversity Selection lambda=0.15)"),
]


def run_ablation(dataset_path, limit=30, output_dir=None):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:limit]
    n_samples = len(samples)

    if output_dir is None:
        out_dir = BASE_DIR / "results" / "evaluation" / "ablation_multirag_qd"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"RUNNING MULTIRAG-QD ABLATION STUDY (SAMPLE-BY-SAMPLE)")
    print(f"Dataset: {dataset_path} ({n_samples} samples)")
    print(f"Stages: A, B, C, D, E, F")
    print(f"Output: {out_dir}")
    print(f"=======================================================\n")

    # Load existing checkpoint if present
    checkpoint_file = out_dir / "ablation_checkpoint.json"
    per_sample_results = []
    done_indices = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                per_sample_results = json.load(f)
            done_indices = {r["index"] for r in per_sample_results}
            print(f"Found existing checkpoint: {len(done_indices)} samples already processed. Resuming.")
        except Exception:
            per_sample_results = []
            done_indices = set()

    for idx, sample in enumerate(samples):
        if idx in done_indices:
            continue

        print(f"\n[{idx+1}/{n_samples}] Processing sample index {idx}...")
        t_sample_start = time.perf_counter()

        question = sample["question"]
        structured_paras = extract_structured_paragraphs(sample)
        k = 5

        # Generation cache for identical evidence sets within this question
        gen_cache = {}

        def get_generation(evidence_items):
            # Key by sorted doc_ids
            doc_key = tuple(sorted(item["doc_id"] for item in evidence_items))
            if doc_key in gen_cache:
                return gen_cache[doc_key]
            t_g0 = time.perf_counter()
            res = generate_answer_from_evidence(question, evidence_items)
            t_g1 = time.perf_counter()
            res["gen_time"] = round(t_g1 - t_g0, 4)
            gen_cache[doc_key] = res
            return res

        # -----------------------------------------------------------------
        # Stage A: Baseline RAG
        # -----------------------------------------------------------------
        t_a0 = time.perf_counter()
        base_passages = retrieve_route(question, structured_paras, k=k, route_name="R0")
        base_gen = get_generation(base_passages)
        t_a1 = time.perf_counter()
        stage_A_time = round(t_a1 - t_a0, 4)

        # -----------------------------------------------------------------
        # Step 1: Decompose Question once for Stages B–F
        # -----------------------------------------------------------------
        t_dec0 = time.perf_counter()
        subquestions = decompose_question(question)
        t_dec1 = time.perf_counter()
        decomp_time = round(t_dec1 - t_dec0, 4)
        print(f"   Decomposition ({decomp_time}s): {len(subquestions)} subquestions")

        # -----------------------------------------------------------------
        # Stage B: QD only (Concatenated subqueries -> dense retrieval)
        # -----------------------------------------------------------------
        t_b0 = time.perf_counter()
        decomposed_query = " ".join(subquestions)
        stage_b_passages = retrieve_route(decomposed_query, structured_paras, k=k, route_name="DecompQuery")
        stage_b_gen = get_generation(stage_b_passages)
        t_b1 = time.perf_counter()
        stage_B_time = round(decomp_time + (t_b1 - t_b0), 4)

        # -----------------------------------------------------------------
        # Step 2: Multi-Query Retrieval (R0, R1, R2, R3)
        # -----------------------------------------------------------------
        t_mq0 = time.perf_counter()
        routes_results = run_multi_query_retrieval(question, subquestions, structured_paras, k=k)
        t_mq1 = time.perf_counter()
        mq_retrieval_time = round(t_mq1 - t_mq0, 4)

        # -----------------------------------------------------------------
        # Stage C: QD + Multi-Query (max similarity pool)
        # -----------------------------------------------------------------
        t_c0 = time.perf_counter()
        best_sim_per_doc = {}
        doc_obj = {}
        for r_name, r_items in routes_results.items():
            for item in r_items:
                doc_id = item["doc_id"]
                score = item["similarity_score"]
                if doc_id not in best_sim_per_doc or score > best_sim_per_doc[doc_id]:
                    best_sim_per_doc[doc_id] = score
                    doc_obj[doc_id] = item
        sorted_docs = sorted(best_sim_per_doc.items(), key=lambda x: x[1], reverse=True)[:k]
        stage_c_passages = [doc_obj[doc_id] for doc_id, _ in sorted_docs]
        stage_c_gen = get_generation(stage_c_passages)
        t_c1 = time.perf_counter()
        stage_C_time = round(decomp_time + mq_retrieval_time + (t_c1 - t_c0), 4)

        # -----------------------------------------------------------------
        # Stage D: QD + MQ + RRF (Reciprocal Rank Fusion k=60)
        # -----------------------------------------------------------------
        t_d0 = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(routes_results, k_rrf=60)
        stage_d_passages = fused_candidates[:k]
        stage_d_gen = get_generation(stage_d_passages)
        t_d1 = time.perf_counter()
        stage_D_time = round(decomp_time + mq_retrieval_time + (t_d1 - t_d0), 4)

        # -----------------------------------------------------------------
        # Stage E: QD + MQ + RRF + Cross-Encoder
        # -----------------------------------------------------------------
        t_e0 = time.perf_counter()
        top_candidates = fused_candidates[:10]
        reranked_candidates = rerank_candidates(question, top_candidates, top_n=10)
        stage_e_passages = reranked_candidates[:k]
        stage_e_gen = get_generation(stage_e_passages)
        t_e1 = time.perf_counter()
        stage_E_time = round(decomp_time + mq_retrieval_time + (t_e1 - t_e0), 4)

        # -----------------------------------------------------------------
        # Stage F: Full MultiRAG-QD (+ Evidence Diversity Selection lambda=0.15)
        # -----------------------------------------------------------------
        t_f0 = time.perf_counter()
        diverse_passages = select_diverse_evidence(reranked_candidates, top_k=k, lambda_diversity=0.15)
        stage_f_gen = get_generation(diverse_passages)
        t_f1 = time.perf_counter()
        stage_F_time = round(decomp_time + mq_retrieval_time + (t_f1 - t_f0), 4)

        # -----------------------------------------------------------------
        # Post-Inference Evaluation (Gold data accessed strictly here)
        # -----------------------------------------------------------------
        ground_truth = sample["answer"]

        def score_stage(stage_code, passages, gen_res, latency):
            ans = gen_res["answer"]
            failed = gen_res["failed"]
            em = exact_match(ans, ground_truth) if not failed else 0
            f1 = token_f1(ans, ground_truth) if not failed else 0.0
            ret_m = compute_retrieval_metrics(passages, sample)
            return {
                "stage": stage_code,
                "answer": ans,
                "failed": failed,
                "em": em,
                "f1": round(f1, 4),
                "recall@1": ret_m["recall@1"],
                "recall@3": ret_m["recall@3"],
                "recall@5": ret_m["recall@5"],
                "recall@10": ret_m["recall@10"],
                "sf_recall@5": ret_m["supporting_fact_recall@5"],
                "mrr": ret_m["mrr"],
                "latency_sec": latency,
                "evidence_titles": [p.get("title", "") for p in passages],
            }

        sample_entry = {
            "index": idx,
            "question_id": sample.get("_id", str(idx)),
            "question": question,
            "ground_truth": ground_truth,
            "subquestions": subquestions,
            "stages": {
                "A": score_stage("A", base_passages, base_gen, stage_A_time),
                "B": score_stage("B", stage_b_passages, stage_b_gen, stage_B_time),
                "C": score_stage("C", stage_c_passages, stage_c_gen, stage_C_time),
                "D": score_stage("D", stage_d_passages, stage_d_gen, stage_D_time),
                "E": score_stage("E", stage_e_passages, stage_e_gen, stage_E_time),
                "F": score_stage("F", diverse_passages, stage_f_gen, stage_F_time),
            },
        }

        per_sample_results.append(sample_entry)
        t_sample_end = time.perf_counter()

        sA = sample_entry["stages"]["A"]
        sE = sample_entry["stages"]["E"]
        sF = sample_entry["stages"]["F"]
        print(f"   Done in {t_sample_end - t_sample_start:.2f}s | Base F1={sA['f1']:.3f} | E(CE) F1={sE['f1']:.3f} | F(Full) F1={sF['f1']:.3f}")
        print(f"   Recall@5: Base={sA['recall@5']:.2f} -> E={sE['recall@5']:.2f} -> F={sF['recall@5']:.2f}")

        # Save checkpoint after each sample
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(per_sample_results, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------
    # Compute Final Aggregated Ablation Summary
    # -----------------------------------------------------------------
    num_completed = len(per_sample_results)
    ablation_summary = []

    for stage_code, stage_name in STAGES:
        stage_data = [r["stages"][stage_code] for r in per_sample_results]
        row = {
            "stage": stage_code,
            "description": stage_name,
            "em": round(sum(s["em"] for s in stage_data) / num_completed, 4),
            "f1": round(sum(s["f1"] for s in stage_data) / num_completed, 4),
            "recall@1": round(sum(s["recall@1"] for s in stage_data) / num_completed, 4),
            "recall@3": round(sum(s["recall@3"] for s in stage_data) / num_completed, 4),
            "recall@5": round(sum(s["recall@5"] for s in stage_data) / num_completed, 4),
            "recall@10": round(sum(s["recall@10"] for s in stage_data) / num_completed, 4),
            "sf_recall@5": round(sum(s["sf_recall@5"] for s in stage_data) / num_completed, 4),
            "mrr": round(sum(s["mrr"] for s in stage_data) / num_completed, 4),
            "avg_latency_sec": round(sum(s["latency_sec"] for s in stage_data) / num_completed, 4),
        }
        ablation_summary.append(row)

    # Compute Incremental Deltas: B-A, C-B, D-C, E-D, F-E
    stage_map = {row["stage"]: row for row in ablation_summary}
    step_pairs = [("B", "A"), ("C", "B"), ("D", "C"), ("E", "D"), ("F", "E")]
    incremental_table = []
    for cur, prev in step_pairs:
        c_row = stage_map[cur]
        p_row = stage_map[prev]
        inc = {
            "step": f"{cur} - {prev}",
            "delta_em": round(c_row["em"] - p_row["em"], 4),
            "delta_f1": round(c_row["f1"] - p_row["f1"], 4),
            "delta_recall@5": round(c_row["recall@5"] - p_row["recall@5"], 4),
            "delta_sf_recall@5": round(c_row["sf_recall@5"] - p_row["sf_recall@5"], 4),
            "delta_mrr": round(c_row["mrr"] - p_row["mrr"], 4),
            "delta_latency_sec": round(c_row["avg_latency_sec"] - p_row["avg_latency_sec"], 4),
        }
        incremental_table.append(inc)

    # Save JSON summary
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": ablation_summary,
            "incremental_deltas": incremental_table,
            "num_samples": num_completed,
        }, f, indent=2, ensure_ascii=False)

    # Save CSV comparison table
    fieldnames = [
        "stage", "description", "em", "f1", "recall@1", "recall@3", "recall@5",
        "recall@10", "sf_recall@5", "mrr", "avg_latency_sec"
    ]
    with open(out_dir / "table_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_summary)

    # Save CSV incremental table
    with open(out_dir / "table_incremental.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "delta_em", "delta_f1", "delta_recall@5", "delta_sf_recall@5", "delta_mrr", "delta_latency_sec"])
        writer.writeheader()
        writer.writerows(incremental_table)

    print("\n" + "=" * 75)
    print("VALIDATION ABLATION STUDY COMPLETE (30 Questions)")
    print("=" * 75)
    print(f"{'Stage':<6} {'EM':<8} {'F1':<8} {'R@5':<8} {'SF-R@5':<10} {'MRR':<8} {'Latency':<8} Description")
    print("-" * 75)
    for r in ablation_summary:
        print(f"{r['stage']:<6} {r['em']:<8.4f} {r['f1']:<8.4f} {r['recall@5']:<8.4f} {r['sf_recall@5']:<10.4f} {r['mrr']:<8.4f} {r['avg_latency_sec']:<8.2f}s {r['description']}")

    print("\nINCREMENTAL IMPROVEMENT TABLE:")
    print("-" * 75)
    print(f"{'Step':<10} {'Delta EM':<10} {'Delta F1':<10} {'Delta R@5':<12} {'Delta SF-R@5':<14} {'Delta MRR':<10}")
    for inc in incremental_table:
        print(f"{inc['step']:<10} {inc['delta_em']:<+10.4f} {inc['delta_f1']:<+10.4f} {inc['delta_recall@5']:<+12.4f} {inc['delta_sf_recall@5']:<+14.4f} {inc['delta_mrr']:<+10.4f}")

    return ablation_summary, incremental_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_val_30.json"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_ablation(dataset_path=args.dataset, limit=args.limit, output_dir=args.output_dir)
