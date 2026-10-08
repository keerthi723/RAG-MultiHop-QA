from retrieval.passage_utils import get_gold_supporting_titles


def compute_retrieval_metrics(retrieved_items, sample):
    """
    EVALUATION ONLY: Computes retrieval quality metrics against HotpotQA gold supporting facts.
    MUST NEVER be invoked during inference.

    Args:
        retrieved_items: Ranked list of retrieved items (dicts or objects containing 'title').
        sample: Raw HotpotQA sample containing 'supporting_facts'.

    Returns:
        Dict with:
          - recall@1
          - recall@3
          - recall@5
          - recall@10
          - supporting_fact_recall@5
          - supporting_fact_recall@10
          - mrr
          - gold_titles: list of gold titles
          - retrieved_titles: list of retrieved titles in order
    """
    gold_titles = set(get_gold_supporting_titles(sample))
    num_gold = len(gold_titles)

    retrieved_titles = []
    for item in retrieved_items:
        if isinstance(item, dict):
            retrieved_titles.append(item.get("title", ""))
        elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], dict):
            retrieved_titles.append(item[0].get("title", ""))
        else:
            retrieved_titles.append(str(item))

    if num_gold == 0:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "supporting_fact_recall@5": 0.0,
            "supporting_fact_recall@10": 0.0,
            "mrr": 0.0,
            "gold_titles": [],
            "retrieved_titles": retrieved_titles,
        }

    def calc_recall_at_k(k):
        top_k_titles = set(retrieved_titles[:k])
        hits = len(top_k_titles & gold_titles)
        return float(hits / num_gold)

    def calc_all_found_at_k(k):
        top_k_titles = set(retrieved_titles[:k])
        return 1.0 if gold_titles.issubset(top_k_titles) else 0.0

    # Calculate MRR (Reciprocal Rank of the FIRST gold paragraph retrieved)
    reciprocal_rank = 0.0
    for rank_idx, title in enumerate(retrieved_titles, start=1):
        if title in gold_titles:
            reciprocal_rank = 1.0 / rank_idx
            break

    return {
        "recall@1": round(calc_recall_at_k(1), 4),
        "recall@3": round(calc_recall_at_k(3), 4),
        "recall@5": round(calc_recall_at_k(5), 4),
        "recall@10": round(calc_recall_at_k(10), 4),
        "supporting_fact_recall@5": round(calc_all_found_at_k(5), 4),
        "supporting_fact_recall@10": round(calc_all_found_at_k(10), 4),
        "mrr": round(reciprocal_rank, 4),
        "gold_titles": list(gold_titles),
        "retrieved_titles": retrieved_titles,
    }
