from collections import defaultdict


def reciprocal_rank_fusion(routes_results, k_rrf=60):
    """
    Computes Reciprocal Rank Fusion across multiple retrieval routes.
    Formula: RRF(d) = sum_{r in routes} 1 / (k_rrf + rank_r(d))
    where rank_r(d) is 1-indexed.

    Args:
        routes_results: dict mapping route_name (e.g. 'R0', 'R1') to list of result dicts,
                        where each item has at least 'doc_id', 'title', 'text', 'rank'.
        k_rrf: smoothing constant (default: 60).

    Returns:
        List of fused candidate dicts sorted by rrf_score descending.
        Each candidate includes:
          - doc_id
          - title
          - text
          - sentences
          - rrf_score
          - routes_appeared: list of route names
          - route_ranks: dict mapping route_name -> rank
          - original_rank: rank in R0 (or None if not in R0)
    """
    passage_map = {}
    rrf_scores = defaultdict(float)
    routes_appeared = defaultdict(list)
    route_ranks = defaultdict(dict)

    for route_name, results in routes_results.items():
        for item in results:
            doc_id = item["doc_id"]
            if doc_id not in passage_map:
                passage_map[doc_id] = {
                    "doc_id": doc_id,
                    "title": item["title"],
                    "text": item["text"],
                    "sentences": item.get("sentences", []),
                }
            rank = item["rank"]
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank)
            routes_appeared[doc_id].append(route_name)
            route_ranks[doc_id][route_name] = rank

    fused = []
    for doc_id, info in passage_map.items():
        fused.append({
            "doc_id": doc_id,
            "title": info["title"],
            "text": info["text"],
            "sentences": info["sentences"],
            "rrf_score": float(rrf_scores[doc_id]),
            "routes_appeared": routes_appeared[doc_id],
            "route_ranks": route_ranks[doc_id],
            "original_rank": route_ranks[doc_id].get("R0", None),
        })

    # Sort descending by fused score
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Assign fused_rank (1-indexed)
    for idx, item in enumerate(fused, start=1):
        item["fused_rank"] = idx

    return fused
