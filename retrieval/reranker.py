import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import CrossEncoder

_cross_encoder = None


def get_cross_encoder():
    """Lazy load cross-encoder/ms-marco-MiniLM-L-6-v2 model."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_candidates(question, candidates, top_n=10):
    """
    Reranks candidate passages against the ORIGINAL question using the Cross-Encoder.
    
    Args:
        question: Original question string (NO gold facts or answers used).
        candidates: List of fused candidate dicts (must have 'text', 'doc_id', etc.).
        top_n: Number of reranked candidates to retain (default: 10).

    Returns:
        List of candidate dicts with added 'cross_encoder_score' and 'rerank_rank',
        sorted descending by cross_encoder_score.
    """
    if not candidates:
        return []

    ce = get_cross_encoder()
    pairs = [(question, c["text"]) for c in candidates]
    scores = ce.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["cross_encoder_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    for idx, item in enumerate(reranked, start=1):
        item["rerank_rank"] = idx

    return reranked[:top_n]
