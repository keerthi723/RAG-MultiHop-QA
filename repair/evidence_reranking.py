import sys
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k
from knowledge_graph.graph_builder import get_triples_per_paragraph, detect_conflicts

CONFLICT_PENALTY = 0.1  # Experimental initial penalty weight — to be tuned later


def rerank_for_conflict(top_results, triples_per_paragraph, conflicts):
    """
    Down-weights paragraphs that contain the MINORITY-supported version of a
    conflicting fact, promoting paragraphs that agree with the majority.
    """
    majority_object = {}
    for c in conflicts:
        key = (c["subject"], c["relation"])
        counter = Counter()
        for pid, triples in triples_per_paragraph.items():
            for t in triples:
                if t["subject"].lower() == c["subject"] and t["relation"] == c["relation"]:
                    counter[t["object"].lower()] += 1
        if counter:
            majority_object[key] = counter.most_common(1)[0][0]

    reranked = []
    for pid, (para, score) in enumerate(top_results):
        penalty = 0.0
        for t in triples_per_paragraph.get(pid, []):
            key = (t["subject"].lower(), t["relation"])
            if key in majority_object and t["object"].lower() != majority_object[key]:
                penalty += CONFLICT_PENALTY
        adjusted_score = max(0.0, float(score) - penalty)
        reranked.append({
            "paragraph_id": pid,
            "paragraph_preview": para[:100] + "...",
            "original_score": round(float(score), 4),
            "conflict_penalty": round(penalty, 4),
            "adjusted_score": round(adjusted_score, 4),
        })

    reranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return reranked


def run_reranking_for_question(index=0, k=5):
    sample = load_question(index=index)
    question = sample["question"]
    paragraphs = extract_paragraphs(sample)
    top_results = retrieve_top_k(question, paragraphs, k=k)

    top_paragraphs = [para for para, score in top_results]
    triples_per_paragraph = get_triples_per_paragraph(top_paragraphs)

    all_triples = []
    for triples in triples_per_paragraph.values():
        all_triples.extend(triples)
    conflicts = detect_conflicts(all_triples)

    reranked = rerank_for_conflict(top_results, triples_per_paragraph, conflicts)

    return {
        "index": index,
        "question": question,
        "num_conflicts": len(conflicts),
        "before_reranking": [
            {"rank": i + 1, "paragraph_id": i, "score": round(float(s), 4)}
            for i, (p, s) in enumerate(top_results)
        ],
        "after_reranking": [
            {"rank": i + 1, **entry} for i, entry in enumerate(reranked)
        ],
    }


if __name__ == "__main__":
    # Test specifically on index 0 — the question we saw flagged with LOW confidence
    result = run_reranking_for_question(index=0, k=5)

    print(f"Question: {result['question']}")
    print(f"Conflicts detected: {result['num_conflicts']}\n")

    print("BEFORE re-ranking (original order):")
    for entry in result["before_reranking"]:
        print(f"  Rank {entry['rank']}: paragraph {entry['paragraph_id']}, score {entry['score']}")

    print("\nAFTER re-ranking (conflict-adjusted order):")
    for entry in result["after_reranking"]:
        print(f"  Rank {entry['rank']}: paragraph {entry['paragraph_id']}, "
              f"original {entry['original_score']}, penalty {entry['conflict_penalty']}, "
              f"adjusted {entry['adjusted_score']}")

    output_dir = BASE_DIR / "results" / "repair"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reranking_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")


    import sys
from pathlib import Path
BASE_DIR2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR2))

import spacy
from retrieval.retrieve import retrieve_top_k, model
from sklearn.metrics.pairwise import cosine_similarity

_nlp_light = spacy.load("en_core_web_sm")


def get_key_terms(question):
    """Extracts noun chunks (key entities/concepts) from the question."""
    doc = _nlp_light(question)
    return set(chunk.text.lower().strip() for chunk in doc.noun_chunks)


def entity_overlap_score(paragraph, key_terms):
    """Fraction of the question's key terms that appear in this paragraph."""
    if not key_terms:
        return 0.0
    para_lower = paragraph.lower()
    matches = sum(1 for term in key_terms if term in para_lower)
    return matches / len(key_terms)


def rerank_for_semantic_misalignment(question, all_paragraphs, k=5):
    """
    Repair for C3_SEMANTIC_MISALIGNMENT: blends the original embedding
    similarity with a lexical entity-overlap signal, since embedding
    similarity alone can miss paragraphs that share exact question
    entities but are phrased very differently.
    """
    key_terms = get_key_terms(question)

    q_emb = model.encode([question])
    para_embs = model.encode(all_paragraphs)
    embed_scores = cosine_similarity(q_emb, para_embs)[0]

    combined = []
    for i, para in enumerate(all_paragraphs):
        overlap = entity_overlap_score(para, key_terms)
        adjusted = (0.7 * float(embed_scores[i])) + (0.3 * overlap)
        combined.append((para, adjusted))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]