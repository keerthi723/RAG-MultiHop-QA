import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.retrieve import model


def select_diverse_evidence(candidates, top_k=5, lambda_diversity=0.15):
    """
    Greedy Maximal Marginal Relevance (MMR) style evidence diversity selection.
    
    Procedure:
      1. Start with the top-ranked candidate from the cross-encoder.
      2. For each remaining candidate, compute:
         DiversityScore(d) = NormCEScore(d) - lambda * MaxSimilarityToSelected(d)
         where NormCEScore is min-max normalized across the candidate pool to [0, 1].
      3. Greedily pick the candidate with highest DiversityScore.
      4. Repeat until top_k passages are selected (or candidates exhausted).

    Args:
        candidates: List of reranked candidate dicts (must have 'text', 'cross_encoder_score', etc.)
        top_k: Number of final passages to select (default: 5).
        lambda_diversity: Penalty weight for redundancy (default: 0.15).

    Returns:
        List of selected candidate dicts with added:
          - 'diversity_score'
          - 'max_similarity_to_selected'
          - 'final_rank' (1..top_k)
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        results = []
        for idx, c in enumerate(candidates, start=1):
            item = dict(c)
            item["diversity_score"] = item.get("cross_encoder_score", 0.0)
            item["max_similarity_to_selected"] = 0.0
            item["final_rank"] = idx
            results.append(item)
        return results

    # Pre-encode all candidate texts
    candidate_texts = [c["text"] for c in candidates]
    candidate_embeddings = model.encode(candidate_texts)

    # Min-max normalize cross-encoder scores to [0, 1] for balanced MMR
    ce_scores = [float(c["cross_encoder_score"]) for c in candidates]
    min_ce = min(ce_scores)
    max_ce = max(ce_scores)
    range_ce = max_ce - min_ce if (max_ce - min_ce) > 1e-6 else 1.0
    norm_ce_scores = [(s - min_ce) / range_ce for s in ce_scores]

    selected_indices = [0]  # Start with highest cross-encoder candidate
    selected_items = []

    first_item = dict(candidates[0])
    first_item["diversity_score"] = float(norm_ce_scores[0])
    first_item["max_similarity_to_selected"] = 0.0
    first_item["final_rank"] = 1
    selected_items.append(first_item)

    remaining_indices = list(range(1, len(candidates)))

    while len(selected_items) < top_k and remaining_indices:
        best_cand_idx = None
        best_div_score = -float("inf")
        best_max_sim = 0.0

        for idx in remaining_indices:
            cand_emb = candidate_embeddings[idx : idx + 1]
            sel_embs = candidate_embeddings[selected_indices]
            sims = cosine_similarity(cand_emb, sel_embs)[0]
            max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

            div_score = norm_ce_scores[idx] - (lambda_diversity * max_sim)

            if div_score > best_div_score:
                best_div_score = div_score
                best_cand_idx = idx
                best_max_sim = max_sim

        if best_cand_idx is None:
            break

        selected_indices.append(best_cand_idx)
        remaining_indices.remove(best_cand_idx)

        item = dict(candidates[best_cand_idx])
        item["diversity_score"] = float(best_div_score)
        item["max_similarity_to_selected"] = float(best_max_sim)
        item["final_rank"] = len(selected_items) + 1
        selected_items.append(item)

    return selected_items
