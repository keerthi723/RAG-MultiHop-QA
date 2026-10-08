import time
from retrieval.passage_utils import extract_structured_paragraphs
from retrieval.multi_query import decompose_question, run_multi_query_retrieval, retrieve_route
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import rerank_candidates
from retrieval.diversity import select_diverse_evidence
from generation.generator import generate_answer_from_evidence


def run_baseline_pipeline(sample, k=5):
    """
    Config A (Clean Baseline):
    Original Question -> all-MiniLM-L6-v2 dense retrieval -> Top-5 -> Llama 3.2 3B -> Answer.
    """
    t0 = time.perf_counter()
    question = sample["question"]
    structured_paragraphs = extract_structured_paragraphs(sample)

    t_ret0 = time.perf_counter()
    baseline_passages = retrieve_route(question, structured_paragraphs, k=k, route_name="R0")
    t_ret1 = time.perf_counter()

    t_gen0 = time.perf_counter()
    gen_result = generate_answer_from_evidence(question, baseline_passages)
    t_gen1 = time.perf_counter()

    t1 = time.perf_counter()

    return {
        "config": "Config A (Baseline)",
        "question": question,
        "evidence_passages": baseline_passages,
        "answer": gen_result["answer"],
        "failed": gen_result["failed"],
        "timings": {
            "retrieval_sec": round(t_ret1 - t_ret0, 4),
            "generation_sec": round(t_gen1 - t_gen0, 4),
            "total_sec": round(t1 - t0, 4),
        },
    }


def run_multirag_qd_pipeline(sample, k=5, ablation_stage="F"):
    """
    MultiRAG-QD Pipeline with ablation support:
      Stage B: QD only (concatenated subqueries -> dense retrieval top-5)
      Stage C: QD + Multi-Query (multi-query union by similarity top-5)
      Stage D: QD + Multi-Query + RRF (fused by RRF top-5)
      Stage E: QD + Multi-Query + RRF + Cross-Encoder (reranked top-5)
      Stage F: Full Proposed Method (QD + Multi-Query + RRF + CE + Diversity top-5)
    """
    t_start = time.perf_counter()
    question = sample["question"]
    structured_paragraphs = extract_structured_paragraphs(sample)

    # 1. Question Decomposition
    t_dec0 = time.perf_counter()
    subquestions = decompose_question(question)
    t_dec1 = time.perf_counter()
    decomp_time = t_dec1 - t_dec0

    # If Stage B: Decomposed query string -> dense retrieval
    if ablation_stage == "B":
        t_ret0 = time.perf_counter()
        decomposed_query = " ".join(subquestions)
        retrieved = retrieve_route(decomposed_query, structured_paragraphs, k=k, route_name="DecompQuery")
        t_ret1 = time.perf_counter()
        t_gen0 = time.perf_counter()
        gen_result = generate_answer_from_evidence(question, retrieved)
        t_gen1 = time.perf_counter()
        t_end = time.perf_counter()
        return {
            "config": "Config B (QD only)",
            "question": question,
            "subquestions": subquestions,
            "evidence_passages": retrieved,
            "answer": gen_result["answer"],
            "failed": gen_result["failed"],
            "timings": {
                "decomp_sec": round(decomp_time, 4),
                "retrieval_sec": round(t_ret1 - t_ret0, 4),
                "generation_sec": round(t_gen1 - t_gen0, 4),
                "total_sec": round(t_end - t_start, 4),
            },
        }

    # 2. Multi-Query Retrieval (R0, R1, R2, R3)
    t_ret0 = time.perf_counter()
    routes_results = run_multi_query_retrieval(question, subquestions, structured_paragraphs, k=k)
    t_ret1 = time.perf_counter()
    retrieval_time = t_ret1 - t_ret0

    # If Stage C: Simple Multi-Query union by max similarity score (no RRF)
    if ablation_stage == "C":
        t_sel0 = time.perf_counter()
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
        c_passages = [doc_obj[doc_id] for doc_id, _ in sorted_docs]
        t_sel1 = time.perf_counter()
        t_gen0 = time.perf_counter()
        gen_result = generate_answer_from_evidence(question, c_passages)
        t_gen1 = time.perf_counter()
        t_end = time.perf_counter()
        return {
            "config": "Config C (QD + Multi-Query)",
            "question": question,
            "subquestions": subquestions,
            "routes_results": routes_results,
            "evidence_passages": c_passages,
            "answer": gen_result["answer"],
            "failed": gen_result["failed"],
            "timings": {
                "decomp_sec": round(decomp_time, 4),
                "retrieval_sec": round(retrieval_time, 4),
                "selection_sec": round(t_sel1 - t_sel0, 4),
                "generation_sec": round(t_gen1 - t_gen0, 4),
                "total_sec": round(t_end - t_start, 4),
            },
        }

    # 3. Reciprocal Rank Fusion (k_rrf = 60)
    t_rrf0 = time.perf_counter()
    fused_candidates = reciprocal_rank_fusion(routes_results, k_rrf=60)
    t_rrf1 = time.perf_counter()
    rrf_time = t_rrf1 - t_rrf0

    # If Stage D: Directly use top-k from RRF (no cross-encoder)
    if ablation_stage == "D":
        d_passages = fused_candidates[:k]
        t_gen0 = time.perf_counter()
        gen_result = generate_answer_from_evidence(question, d_passages)
        t_gen1 = time.perf_counter()
        t_end = time.perf_counter()
        return {
            "config": "Config D (QD + MQ + RRF)",
            "question": question,
            "subquestions": subquestions,
            "routes_results": routes_results,
            "fused_candidates": fused_candidates,
            "evidence_passages": d_passages,
            "answer": gen_result["answer"],
            "failed": gen_result["failed"],
            "timings": {
                "decomp_sec": round(decomp_time, 4),
                "retrieval_sec": round(retrieval_time, 4),
                "rrf_sec": round(rrf_time, 4),
                "generation_sec": round(t_gen1 - t_gen0, 4),
                "total_sec": round(t_end - t_start, 4),
            },
        }

    # 4. Cross-Encoder Reranking against original question (top 10 retained)
    t_rerank0 = time.perf_counter()
    top_candidates = fused_candidates[:10]
    reranked_candidates = rerank_candidates(question, top_candidates, top_n=10)
    t_rerank1 = time.perf_counter()
    rerank_time = t_rerank1 - t_rerank0

    # If Stage E: Directly use top-k from Cross-Encoder (no diversity penalty)
    if ablation_stage == "E":
        e_passages = reranked_candidates[:k]
        t_gen0 = time.perf_counter()
        gen_result = generate_answer_from_evidence(question, e_passages)
        t_gen1 = time.perf_counter()
        t_end = time.perf_counter()
        return {
            "config": "Config E (QD + MQ + RRF + CE)",
            "question": question,
            "subquestions": subquestions,
            "routes_results": routes_results,
            "fused_candidates": fused_candidates,
            "reranked_candidates": reranked_candidates,
            "evidence_passages": e_passages,
            "answer": gen_result["answer"],
            "failed": gen_result["failed"],
            "timings": {
                "decomp_sec": round(decomp_time, 4),
                "retrieval_sec": round(retrieval_time, 4),
                "rrf_sec": round(rrf_time, 4),
                "rerank_sec": round(rerank_time, 4),
                "generation_sec": round(t_gen1 - t_gen0, 4),
                "total_sec": round(t_end - t_start, 4),
            },
        }

    # 5. Evidence Diversity Selection (Stage F: Full Proposed Method)
    t_div0 = time.perf_counter()
    diverse_passages = select_diverse_evidence(reranked_candidates, top_k=k, lambda_diversity=0.15)
    t_div1 = time.perf_counter()
    div_time = t_div1 - t_div0

    # 6. LLM Generation
    t_gen0 = time.perf_counter()
    gen_result = generate_answer_from_evidence(question, diverse_passages)
    t_gen1 = time.perf_counter()
    gen_time = t_gen1 - t_gen0

    t_end = time.perf_counter()

    return {
        "config": "Config F (Full MultiRAG-QD)",
        "question": question,
        "subquestions": subquestions,
        "routes_results": routes_results,
        "fused_candidates": fused_candidates,
        "reranked_candidates": reranked_candidates,
        "evidence_passages": diverse_passages,
        "answer": gen_result["answer"],
        "failed": gen_result["failed"],
        "timings": {
            "decomp_sec": round(decomp_time, 4),
            "retrieval_sec": round(retrieval_time, 4),
            "rrf_sec": round(rrf_time, 4),
            "rerank_sec": round(rerank_time, 4),
            "diversity_sec": round(div_time, 4),
            "generation_sec": round(gen_time, 4),
            "total_sec": round(t_end - t_start, 4),
        },
    }
