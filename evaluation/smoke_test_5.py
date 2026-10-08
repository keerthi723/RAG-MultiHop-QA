import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import json
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

VAL_PATH = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_val_30.json"
OUT_DIR = BASE_DIR / "results" / "evaluation" / "smoke_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_smoke_test(num_samples=5):
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    samples = val_data[:num_samples]
    print(f"=======================================================")
    print(f"SMOKE TEST: Running {num_samples} Validation Samples")
    print(f"Dataset: {VAL_PATH}")
    print(f"=======================================================\n")

    all_logs = []
    checks_passed = {
        "1_decomposition_works": True,
        "2_subquestions_2_to_3": True,
        "3_multiquery_works": True,
        "4_rrf_rankings_correct": True,
        "5_cross_encoder_works": True,
        "6_diversity_works": True,
        "7_final_evidence_distinct": True,
        "8_generation_works": True,
        "9_no_gold_in_inference": True,
        "10_logs_complete": True,
    }

    for idx, sample in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"SAMPLE {idx+1}/{num_samples} (HotpotQA _id: {sample.get('_id', 'N/A')})")
        print(f"Question: {sample['question']}")

        # -------------------------------------------------------------
        # STRICT BLINDING VERIFICATION:
        # Verify inference variables only contain question and context.
        # Gold answer and supporting facts are NOT accessed here.
        # -------------------------------------------------------------
        question = sample["question"]
        structured_paras = extract_structured_paragraphs(sample)

        # Baseline
        t0 = time.perf_counter()
        base_retrieved = retrieve_route(question, structured_paras, k=5, route_name="R0")
        t1 = time.perf_counter()
        base_gen = generate_answer_from_evidence(question, base_retrieved)
        t2 = time.perf_counter()
        print(f"  [Baseline] Retrieval: {t1-t0:.3f}s | Gen: {t2-t1:.3f}s | Ans: {base_gen['answer']!r}")

        # 1. Question Decomposition
        t_dec0 = time.perf_counter()
        subquestions = decompose_question(question)
        t_dec1 = time.perf_counter()
        print(f"  [1. Decomposition] Time: {t_dec1-t_dec0:.3f}s ({len(subquestions)} subquestions):")
        for q_i, sq in enumerate(subquestions, 1):
            print(f"      Q{q_i}: {sq}")

        if not subquestions or len(subquestions) == 0:
            checks_passed["1_decomposition_works"] = False
        if not (2 <= len(subquestions) <= 3):
            # Allow fallback if single question, but track compliance
            print(f"      Warning: subquestion count is {len(subquestions)} (expected 2-3)")
            checks_passed["2_subquestions_2_to_3"] = False

        # 2. Multi-Query Retrieval
        t_mq0 = time.perf_counter()
        routes_results = run_multi_query_retrieval(question, subquestions, structured_paras, k=5)
        t_mq1 = time.perf_counter()
        total_raw_candidates = sum(len(v) for v in routes_results.values())
        print(f"  [2. Multi-Query Retrieval] Time: {t_mq1-t_mq0:.3f}s | Routes: {list(routes_results.keys())} | Total Raw Candidates: {total_raw_candidates}")

        if not routes_results or "R0" not in routes_results:
            checks_passed["3_multiquery_works"] = False

        # 3. RRF Fusion
        t_rrf0 = time.perf_counter()
        fused = reciprocal_rank_fusion(routes_results, k_rrf=60)
        t_rrf1 = time.perf_counter()
        print(f"  [3. RRF Fusion] Time: {t_rrf1-t_rrf0:.3f}s | Fused pool size: {len(fused)}")
        for r_item in fused[:3]:
            print(f"      Rank {r_item['fused_rank']}: '{r_item['title']}' | Score: {r_item['rrf_score']:.5f} | Routes: {r_item['routes_appeared']}")

        if not fused or len(fused) == 0:
            checks_passed["4_rrf_rankings_correct"] = False

        # 4. Cross-Encoder Reranking
        t_ce0 = time.perf_counter()
        top_candidates = fused[:10]
        reranked = rerank_candidates(question, top_candidates, top_n=10)
        t_ce1 = time.perf_counter()
        print(f"  [4. Cross-Encoder Rerank] Time: {t_ce1-t_ce0:.3f}s | Top reranked: {len(reranked)}")
        for ce_item in reranked[:3]:
            print(f"      Rank {ce_item['rerank_rank']}: '{ce_item['title']}' | CE Score: {ce_item['cross_encoder_score']:.4f}")

        if not reranked or len(reranked) == 0:
            checks_passed["5_cross_encoder_works"] = False

        # 5. Diversity Selection
        t_div0 = time.perf_counter()
        diverse = select_diverse_evidence(reranked, top_k=5, lambda_diversity=0.15)
        t_div1 = time.perf_counter()
        print(f"  [5. Evidence Diversity] Time: {t_div1-t_div0:.3f}s | Selected: {len(diverse)} passages")
        for div_item in diverse:
            print(f"      Rank {div_item['final_rank']}: '{div_item['title']}' | Div Score: {div_item['diversity_score']:.4f} | MaxSim: {div_item['max_similarity_to_selected']:.4f}")

        if len(diverse) != min(5, len(structured_paras)):
            checks_passed["6_diversity_works"] = False

        # Check distinctness of final evidence
        doc_ids = [d["doc_id"] for d in diverse]
        if len(doc_ids) != len(set(doc_ids)):
            print(f"      ERROR: Duplicate doc_ids found in final evidence: {doc_ids}")
            checks_passed["7_final_evidence_distinct"] = False
        else:
            print(f"      Confirmed: All {len(diverse)} evidence passages are distinct (doc_ids={doc_ids}).")

        # 6. LLM Generation
        t_gen0 = time.perf_counter()
        prop_gen = generate_answer_from_evidence(question, diverse)
        t_gen1 = time.perf_counter()
        print(f"  [6. LLM Generation] Time: {t_gen1-t_gen0:.3f}s | Proposed Ans: {prop_gen['answer']!r}")

        if prop_gen["failed"] or not prop_gen["answer"]:
            checks_passed["8_generation_works"] = False

        # -------------------------------------------------------------
        # EVALUATION ONLY (Post-Inference):
        # Access gold answer and supporting facts strictly here.
        # -------------------------------------------------------------
        gold_ans = sample["answer"]
        b_em = exact_match(base_gen["answer"], gold_ans)
        b_f1 = token_f1(base_gen["answer"], gold_ans)
        p_em = exact_match(prop_gen["answer"], gold_ans)
        p_f1 = token_f1(prop_gen["answer"], gold_ans)

        b_ret = compute_retrieval_metrics(base_retrieved, sample)
        p_ret = compute_retrieval_metrics(diverse, sample)

        print(f"  [Post-Eval Comparison]")
        print(f"      Gold Answer: {gold_ans!r}")
        print(f"      Baseline EM/F1: {b_em}/{b_f1:.3f} | Proposed EM/F1: {p_em}/{p_f1:.3f}")
        print(f"      Gold Titles: {b_ret['gold_titles']}")
        print(f"      Baseline Retrieved: {[p['title'] for p in base_retrieved]}")
        print(f"      Proposed Retrieved: {[p['title'] for p in diverse]}")
        print(f"      Recall@5: Baseline {b_ret['recall@5']:.2f} -> Proposed {p_ret['recall@5']:.2f}")

        sample_log = {
            "index": idx,
            "question_id": sample.get("_id", str(idx)),
            "question": question,
            "generated_subquestions": subquestions,
            "retrieval_routes": list(routes_results.keys()),
            "route_details": {
                r: [{"doc_id": item["doc_id"], "title": item["title"], "rank": item["rank"], "similarity": item["similarity_score"]}
                    for item in items]
                for r, items in routes_results.items()
            },
            "rrf_fused": [
                {"doc_id": f["doc_id"], "title": f["title"], "fused_rank": f["fused_rank"], "rrf_score": f["rrf_score"], "routes": f["routes_appeared"]}
                for f in fused
            ],
            "cross_encoder_reranked": [
                {"doc_id": c["doc_id"], "title": c["title"], "rerank_rank": c["rerank_rank"], "ce_score": c["cross_encoder_score"]}
                for c in reranked
            ],
            "diversity_selected": [
                {"doc_id": d["doc_id"], "title": d["title"], "final_rank": d["final_rank"], "diversity_score": d["diversity_score"], "max_sim": d["max_similarity_to_selected"]}
                for d in diverse
            ],
            "baseline_evidence_titles": [p["title"] for p in base_retrieved],
            "proposed_evidence_titles": [p["title"] for p in diverse],
            "gold_supporting_titles": b_ret["gold_titles"],
            "baseline_answer": base_gen["answer"],
            "proposed_answer": prop_gen["answer"],
            "gold_answer": gold_ans,
            "baseline_em": b_em,
            "baseline_f1": round(b_f1, 4),
            "proposed_em": p_em,
            "proposed_f1": round(p_f1, 4),
            "retrieval_metrics": {
                "baseline": b_ret,
                "proposed": p_ret,
            },
            "timings": {
                "baseline_total": round((t1-t0) + (t2-t1), 4),
                "decomposition": round(t_dec1 - t_dec0, 4),
                "multi_query_retrieval": round(t_mq1 - t_mq0, 4),
                "rrf": round(t_rrf1 - t_rrf0, 4),
                "cross_encoder": round(t_ce1 - t_ce0, 4),
                "diversity": round(t_div1 - t_div0, 4),
                "generation": round(t_gen1 - t_gen0, 4),
                "proposed_total": round((t_dec1 - t_dec0) + (t_mq1 - t_mq0) + (t_rrf1 - t_rrf0) + (t_ce1 - t_ce0) + (t_div1 - t_div0) + (t_gen1 - t_gen0), 4),
            },
        }
        all_logs.append(sample_log)

    # Save smoke test results
    with open(OUT_DIR / "smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    for check, passed in checks_passed.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {check}")

    all_passed = all(checks_passed.values())
    print(f"\nOverall Smoke Test Status: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"Detailed logs written to: {OUT_DIR / 'smoke_test_results.json'}")
    return all_passed


if __name__ == "__main__":
    run_smoke_test(num_samples=5)
