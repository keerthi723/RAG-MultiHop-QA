import sys
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k, model
from knowledge_graph.graph_builder import build_graph_from_paragraphs

# Experimental initial weights — to be tuned via validation experiments later
W_RETRIEVAL = 0.4
W_SEMANTIC = 0.4
W_GRAPH = 0.2


def retrieval_confidence(top_results):
    """Signal 1: average cosine similarity of Top-K retrieved paragraphs (clipped to [0,1])."""
    scores = [max(0.0, float(score)) for _, score in top_results]
    return float(np.mean(scores)) if scores else 0.0


def semantic_confidence_detailed(question, top_paragraphs):
    """
    FIXED (Task 1, Problem 1): previously we concatenated all paragraphs into
    one long string and embedded THAT — which caused "vector dilution": as
    more paragraphs were added, the combined string's embedding drifted away
    from the question, so semantic confidence went DOWN when we added MORE
    evidence during repair. That's backwards.

    Fix: embed each paragraph individually, score each against the question,
    and use the average of the top-3 individual scores. This means adding
    more (even weak) paragraphs no longer drags the score down — only the
    best-matching paragraphs determine semantic confidence.

    Returns: (avg_top3_score, list of (paragraph, score) pairs — UNSORTED,
              same order as input, so callers can map scores back to IDs)
    """
    if not top_paragraphs:
        return 0.0, []

    q_emb = model.encode([question])
    para_embs = model.encode(top_paragraphs)
    sims = cosine_similarity(q_emb, para_embs)[0]

    paired = [(top_paragraphs[i], float(sims[i])) for i in range(len(top_paragraphs))]

    sorted_scores = sorted([s for _, s in paired], reverse=True)
    top_n = sorted_scores[:3] if len(sorted_scores) >= 3 else sorted_scores
    avg_top3 = float(np.mean(top_n)) if top_n else 0.0

    return max(0.0, avg_top3), paired


def semantic_confidence(question, top_paragraphs):
    """Backward-compatible wrapper — returns just the score, as before."""
    score, _ = semantic_confidence_detailed(question, top_paragraphs)
    return score


def graph_confidence(top_paragraphs):
    """Signal 3: fraction of retrieved paragraphs that yielded at least one extractable triple."""
    if not top_paragraphs:
        return 0.0
    contributing = 0
    for para in top_paragraphs:
        _, sub_triples = build_graph_from_paragraphs([para])
        if len(sub_triples) > 0:
            contributing += 1
    return contributing / len(top_paragraphs)


def compute_confidence(index=0, k=5, dataset_path=None):
    sample = load_question(index=index, dataset_path=dataset_path)
    question = sample["question"]
    paragraphs = extract_paragraphs(sample)
    top_results = retrieve_top_k(question, paragraphs, k=k)
    top_paragraphs = [para for para, score in top_results]

    r_conf = retrieval_confidence(top_results)
    s_conf, per_paragraph_sims = semantic_confidence_detailed(question, top_paragraphs)
    g_conf = graph_confidence(top_paragraphs)

    C = min(1.0, max(0.0, (W_RETRIEVAL * r_conf) + (W_SEMANTIC * s_conf) + (W_GRAPH * g_conf)))

    return {
        "index": index,
        "question": question,
        "retrieval_confidence": round(r_conf, 4),
        "semantic_confidence": round(s_conf, 4),
        "graph_confidence": round(g_conf, 4),
        "final_confidence": round(C, 4),
        "per_paragraph_semantic_scores": [round(s, 4) for _, s in per_paragraph_sims],
        "weights": {"w_retrieval": W_RETRIEVAL, "w_semantic": W_SEMANTIC, "w_graph": W_GRAPH,
                    "note": "Experimental initial weights — to be tuned via validation experiments"}
    }


def compute_confidence_from_paragraphs(question, top_results):
    """
    Same three-signal computation as compute_confidence(), but operates on
    an already-given (paragraph, score) list instead of re-running retrieval
    from an index. Used after repair, when the evidence set has changed.
    """
    top_paragraphs = [para for para, score in top_results]

    r_conf = retrieval_confidence(top_results)
    s_conf, per_paragraph_sims = semantic_confidence_detailed(question, top_paragraphs)
    g_conf = graph_confidence(top_paragraphs)

    C = min(1.0, max(0.0, (W_RETRIEVAL * r_conf) + (W_SEMANTIC * s_conf) + (W_GRAPH * g_conf)))

    return {
        "question": question,
        "retrieval_confidence": round(r_conf, 4),
        "semantic_confidence": round(s_conf, 4),
        "graph_confidence": round(g_conf, 4),
        "final_confidence": round(C, 4),
        "per_paragraph_semantic_scores": [round(s, 4) for _, s in per_paragraph_sims],
    }


if __name__ == "__main__":
    test_indices = [0, 1, 2, 10, 50]
    threshold = 0.60
    all_results = []

    for idx in test_indices:
        result = compute_confidence(index=idx, k=5)
        result["status"] = "HIGH" if result["final_confidence"] >= threshold else "LOW"
        all_results.append(result)

        print(f"\nQuestion: {result['question']}")
        print(f"  Retrieval confidence : {result['retrieval_confidence']}")
        print(f"  Semantic confidence  : {result['semantic_confidence']}")
        print(f"  Graph confidence     : {result['graph_confidence']}")
        print(f"  FINAL confidence (C) : {result['final_confidence']}  -> {result['status']}")

    output_dir = BASE_DIR / "results" / "confidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confidence_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")