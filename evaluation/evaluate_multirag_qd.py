import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import json
import argparse
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.passage_utils import load_question
from retrieval.multirag_qd import run_baseline_pipeline, run_multirag_qd_pipeline
from evaluation.answer_metrics import exact_match, token_f1
from evaluation.retrieval_metrics import compute_retrieval_metrics


def evaluate_sample(sample, k=5, ablation_stage="F"):
    """
    Evaluates a single sample with both Baseline and MultiRAG-QD.
    Gold information is ONLY extracted AFTER both pipelines complete.
    """
    # 1. Inference: Baseline
    base_res = run_baseline_pipeline(sample, k=k)

    # 2. Inference: MultiRAG-QD
    prop_res = run_multirag_qd_pipeline(sample, k=k, ablation_stage=ablation_stage)

    # 3. Post-Inference Evaluation (Gold data access restricted to this stage)
    ground_truth = sample["answer"]
    question = sample["question"]

    base_em = exact_match(base_res["answer"], ground_truth) if not base_res["failed"] else 0
    base_f1 = token_f1(base_res["answer"], ground_truth) if not base_res["failed"] else 0.0

    prop_em = exact_match(prop_res["answer"], ground_truth) if not prop_res["failed"] else 0
    prop_f1 = token_f1(prop_res["answer"], ground_truth) if not prop_res["failed"] else 0.0

    base_ret_metrics = compute_retrieval_metrics(base_res["evidence_passages"], sample)
    prop_ret_metrics = compute_retrieval_metrics(prop_res["evidence_passages"], sample)

    # Classification
    f1_delta = round(prop_f1 - base_f1, 4)
    em_delta = prop_em - base_em
    if prop_f1 > base_f1 + 0.001 or prop_em > base_em:
        status = "IMPROVED"
    elif prop_f1 < base_f1 - 0.001 or prop_em < base_em:
        status = "DEGRADED"
    else:
        status = "UNCHANGED"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "baseline_answer": base_res["answer"],
        "proposed_answer": prop_res["answer"],
        "subquestions": prop_res.get("subquestions", []),
        "status": status,
        "answer_metrics": {
            "baseline_em": base_em,
            "baseline_f1": round(base_f1, 4),
            "proposed_em": prop_em,
            "proposed_f1": round(prop_f1, 4),
            "em_delta": em_delta,
            "f1_delta": f1_delta,
        },
        "retrieval_metrics": {
            "baseline": base_ret_metrics,
            "proposed": prop_ret_metrics,
        },
        "evidence_summary": {
            "baseline_titles": [p.get("title", "") for p in base_res["evidence_passages"]],
            "proposed_titles": [p.get("title", "") for p in prop_res["evidence_passages"]],
            "gold_supporting_titles": base_ret_metrics["gold_titles"],
        },
        "timings": {
            "baseline": base_res["timings"],
            "proposed": prop_res["timings"],
        },
    }


def run_evaluation_suite(dataset_path, limit=100, start_idx=0, output_dir=None, checkpoint_every=5):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_dir is None:
        out_dir = BASE_DIR / "results" / "evaluation" / "multirag_qd"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_idx = min(start_idx + limit, len(data))
    samples_to_eval = data[start_idx:end_idx]
    n_total = len(samples_to_eval)

    print(f"\n=======================================================")
    print(f"Starting MultiRAG-QD Evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Range: [{start_idx} : {end_idx}] ({n_total} samples)")
    print(f"Output: {out_dir}")
    print(f"=======================================================\n")

    results = []
    t_suite_start = time.perf_counter()

    for i, sample in enumerate(samples_to_eval, start=1):
        actual_idx = start_idx + i - 1
        print(f"[{i}/{n_total}] Evaluating index {actual_idx}...")
        try:
            res = evaluate_sample(sample, k=5, ablation_stage="F")
            res["index"] = actual_idx
            results.append(res)

            am = res["answer_metrics"]
            bm = res["retrieval_metrics"]["baseline"]
            pm = res["retrieval_metrics"]["proposed"]
            print(f"   Status: {res['status']} | Base EM/F1: {am['baseline_em']}/{am['baseline_f1']:.3f} -> Prop EM/F1: {am['proposed_em']}/{am['proposed_f1']:.3f}")
            print(f"   Recall@5: Base {bm['recall@5']:.2f} -> Prop {pm['recall@5']:.2f} | Gold Titles: {bm['gold_titles']}")
        except Exception as e:
            print(f"   FAILED on index {actual_idx}: {e}")
            results.append({
                "index": actual_idx,
                "question": sample.get("question", ""),
                "status": "ERROR",
                "error": str(e),
            })

        if i % checkpoint_every == 0 or i == n_total:
            with open(out_dir / "detailed_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"   [Checkpoint saved: {i}/{n_total}]")

    t_suite_end = time.perf_counter()

    # Compute summary statistics
    valid_results = [r for r in results if r.get("status") in ["IMPROVED", "DEGRADED", "UNCHANGED"]]
    num_valid = len(valid_results)

    if num_valid > 0:
        base_em_avg = sum(r["answer_metrics"]["baseline_em"] for r in valid_results) / num_valid
        base_f1_avg = sum(r["answer_metrics"]["baseline_f1"] for r in valid_results) / num_valid
        prop_em_avg = sum(r["answer_metrics"]["proposed_em"] for r in valid_results) / num_valid
        prop_f1_avg = sum(r["answer_metrics"]["proposed_f1"] for r in valid_results) / num_valid

        improved_count = sum(1 for r in valid_results if r["status"] == "IMPROVED")
        degraded_count = sum(1 for r in valid_results if r["status"] == "DEGRADED")
        unchanged_count = sum(1 for r in valid_results if r["status"] == "UNCHANGED")

        # Retrieval metrics averages
        b_r1 = sum(r["retrieval_metrics"]["baseline"]["recall@1"] for r in valid_results) / num_valid
        p_r1 = sum(r["retrieval_metrics"]["proposed"]["recall@1"] for r in valid_results) / num_valid
        b_r3 = sum(r["retrieval_metrics"]["baseline"]["recall@3"] for r in valid_results) / num_valid
        p_r3 = sum(r["retrieval_metrics"]["proposed"]["recall@3"] for r in valid_results) / num_valid
        b_r5 = sum(r["retrieval_metrics"]["baseline"]["recall@5"] for r in valid_results) / num_valid
        p_r5 = sum(r["retrieval_metrics"]["proposed"]["recall@5"] for r in valid_results) / num_valid
        b_r10 = sum(r["retrieval_metrics"]["baseline"]["recall@10"] for r in valid_results) / num_valid
        p_r10 = sum(r["retrieval_metrics"]["proposed"]["recall@10"] for r in valid_results) / num_valid

        b_sf5 = sum(r["retrieval_metrics"]["baseline"]["supporting_fact_recall@5"] for r in valid_results) / num_valid
        p_sf5 = sum(r["retrieval_metrics"]["proposed"]["supporting_fact_recall@5"] for r in valid_results) / num_valid
        b_mrr = sum(r["retrieval_metrics"]["baseline"]["mrr"] for r in valid_results) / num_valid
        p_mrr = sum(r["retrieval_metrics"]["proposed"]["mrr"] for r in valid_results) / num_valid

        # Timings averages
        avg_decomp_sec = sum(r["timings"]["proposed"].get("decomp_sec", 0.0) for r in valid_results) / num_valid
        avg_ret_sec = sum(r["timings"]["proposed"].get("retrieval_sec", 0.0) for r in valid_results) / num_valid
        avg_rrf_sec = sum(r["timings"]["proposed"].get("rrf_sec", 0.0) for r in valid_results) / num_valid
        avg_rerank_sec = sum(r["timings"]["proposed"].get("rerank_sec", 0.0) for r in valid_results) / num_valid
        avg_div_sec = sum(r["timings"]["proposed"].get("diversity_sec", 0.0) for r in valid_results) / num_valid
        avg_gen_sec = sum(r["timings"]["proposed"].get("generation_sec", 0.0) for r in valid_results) / num_valid
        avg_total_prop_sec = sum(r["timings"]["proposed"].get("total_sec", 0.0) for r in valid_results) / num_valid
        avg_total_base_sec = sum(r["timings"]["baseline"].get("total_sec", 0.0) for r in valid_results) / num_valid

        summary = {
            "num_evaluated": num_valid,
            "exact_match": {
                "baseline": round(base_em_avg, 4),
                "proposed": round(prop_em_avg, 4),
                "delta": round(prop_em_avg - base_em_avg, 4),
            },
            "token_f1": {
                "baseline": round(base_f1_avg, 4),
                "proposed": round(prop_f1_avg, 4),
                "delta": round(prop_f1_avg - base_f1_avg, 4),
            },
            "sample_classification": {
                "improved": improved_count,
                "degraded": degraded_count,
                "unchanged": unchanged_count,
            },
            "retrieval_metrics": {
                "recall@1": {"baseline": round(b_r1, 4), "proposed": round(p_r1, 4), "delta": round(p_r1 - b_r1, 4)},
                "recall@3": {"baseline": round(b_r3, 4), "proposed": round(p_r3, 4), "delta": round(p_r3 - b_r3, 4)},
                "recall@5": {"baseline": round(b_r5, 4), "proposed": round(p_r5, 4), "delta": round(p_r5 - b_r5, 4)},
                "recall@10": {"baseline": round(b_r10, 4), "proposed": round(p_r10, 4), "delta": round(p_r10 - b_r10, 4)},
                "supporting_fact_recall@5": {"baseline": round(b_sf5, 4), "proposed": round(p_sf5, 4), "delta": round(p_sf5 - b_sf5, 4)},
                "mrr": {"baseline": round(b_mrr, 4), "proposed": round(p_mrr, 4), "delta": round(p_mrr - b_mrr, 4)},
            },
            "average_latency_sec": {
                "baseline_total": round(avg_total_base_sec, 4),
                "proposed_total": round(avg_total_prop_sec, 4),
                "decomposition": round(avg_decomp_sec, 4),
                "multi_query_retrieval": round(avg_ret_sec, 4),
                "rrf_fusion": round(avg_rrf_sec, 4),
                "cross_encoder_rerank": round(avg_rerank_sec, 4),
                "diversity_selection": round(avg_div_sec, 4),
                "llm_generation": round(avg_gen_sec, 4),
            },
            "total_suite_runtime_sec": round(t_suite_end - t_suite_start, 2),
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Build error analysis categorizing reasons for degraded / improved
        error_analysis = {
            "improved_samples": [
                {
                    "index": r["index"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "base_answer": r["baseline_answer"],
                    "prop_answer": r["proposed_answer"],
                    "base_f1": r["answer_metrics"]["baseline_f1"],
                    "prop_f1": r["answer_metrics"]["proposed_f1"],
                    "subquestions": r["subquestions"],
                    "base_retrieved_titles": r["evidence_summary"]["baseline_titles"],
                    "prop_retrieved_titles": r["evidence_summary"]["proposed_titles"],
                    "gold_supporting_titles": r["evidence_summary"]["gold_supporting_titles"],
                }
                for r in valid_results if r["status"] == "IMPROVED"
            ],
            "degraded_samples": [
                {
                    "index": r["index"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "base_answer": r["baseline_answer"],
                    "prop_answer": r["proposed_answer"],
                    "base_f1": r["answer_metrics"]["baseline_f1"],
                    "prop_f1": r["answer_metrics"]["proposed_f1"],
                    "subquestions": r["subquestions"],
                    "base_retrieved_titles": r["evidence_summary"]["baseline_titles"],
                    "prop_retrieved_titles": r["evidence_summary"]["proposed_titles"],
                    "gold_supporting_titles": r["evidence_summary"]["gold_supporting_titles"],
                }
                for r in valid_results if r["status"] == "DEGRADED"
            ],
        }

        with open(out_dir / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("EVALUATION SUITE COMPLETE")
        print("=" * 70)
        print(f"Evaluated: {num_valid} questions")
        print(f"Exact Match: Baseline {summary['exact_match']['baseline']} -> Proposed {summary['exact_match']['proposed']} (Delta: {summary['exact_match']['delta']:+.4f})")
        print(f"Token F1:    Baseline {summary['token_f1']['baseline']} -> Proposed {summary['token_f1']['proposed']} (Delta: {summary['token_f1']['delta']:+.4f})")
        print(f"Recall@5:    Baseline {summary['retrieval_metrics']['recall@5']['baseline']} -> Proposed {summary['retrieval_metrics']['recall@5']['proposed']} (Delta: {summary['retrieval_metrics']['recall@5']['delta']:+.4f})")
        print(f"SF-Recall@5: Baseline {summary['retrieval_metrics']['supporting_fact_recall@5']['baseline']} -> Proposed {summary['retrieval_metrics']['supporting_fact_recall@5']['proposed']}")
        print(f"MRR:         Baseline {summary['retrieval_metrics']['mrr']['baseline']} -> Proposed {summary['retrieval_metrics']['mrr']['proposed']}")
        print(f"Classification: Improved={improved_count}, Degraded={degraded_count}, Unchanged={unchanged_count}")
        print(f"Latency: Baseline {summary['average_latency_sec']['baseline_total']}s vs Proposed {summary['average_latency_sec']['proposed_total']}s")
        print(f"Saved to: {out_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_100.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_evaluation_suite(
        dataset_path=args.dataset,
        limit=args.limit,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
    )
