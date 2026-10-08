import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import retrieve_top_k


def expand_retrieval(question, all_paragraphs, expanded_k=8):
    """
    Repair for C2_KNOWLEDGE_SPARSITY: widens the retrieval pool beyond
    the original Top-K, pulling in additional evidence that was ranked
    just below the original cutoff.
    """
    expanded_k = min(expanded_k, len(all_paragraphs))
    return retrieve_top_k(question, all_paragraphs, k=expanded_k)