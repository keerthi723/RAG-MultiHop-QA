from sentence_transformers import CrossEncoder

_cross_encoder = None


def get_cross_encoder():
    """Lazy-load: the model downloads once (~80MB) on first use and is cached
    locally afterward, so only the very first call will be slow."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def score_relevance(question, paragraphs):
    """
    Returns list of (paragraph, score) using a CROSS-encoder — question and
    paragraph are scored jointly by one model, not compared as two separately
    -computed embeddings (which is what plain cosine similarity does). This
    is the standard fix for bi-encoder similarity's known weakness: high
    embedding similarity does not reliably indicate that a passage actually
    supports answering the question.
    """
    ce = get_cross_encoder()
    pairs = [(question, p) for p in paragraphs]
    scores = ce.predict(pairs)
    return list(zip(paragraphs, [float(s) for s in scores]))