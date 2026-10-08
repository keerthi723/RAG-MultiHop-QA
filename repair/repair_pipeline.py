import sys
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k, model
from root_cause.root_cause_analysis import run_rca_for_question
from knowledge_graph.graph_builder import get_triples_per_paragraph, detect_conflicts
from confidence.confidence_assessment import compute_confidence_from_paragraphs

THETA = 0.60
CONFLICT_RELEVANCE_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Existing cause-specific repair functions — UNCHANGED logic from prior work
# ---------------------------------------------------------------------------

def _per_paragraph_similarity(query_text, paragraphs):
    if not paragraphs:
        return []
    q_emb = model.encode([query_text])
    para_embs = model.encode(paragraphs)
    sims = cosine_similarity(q_emb, para_embs)[0]
    return [(i, paragraphs[i], float(sims[i])) for i in range(len(paragraphs))]


def _conflict_relevance(question, subject, relation):
    q_emb = model.encode([question])
    c_emb = model.encode([f"{subject} {relation}"])
    return float(cosine_similarity(q_emb, c_emb)[0][0])


def repair_information_conflict(question, top_results, all_paragraphs):
    paragraphs = [p for p, s in top_results]
    triples_per_paragraph = get_triples_per_paragraph(paragraphs)
    all_triples = [t for triples in triples_per_paragraph.values() for t in triples]
    conflicts = detect_conflicts(all_triples)

    scored = _per_paragraph_similarity(question, paragraphs)
    sim_map = {i: s for i, p, s in scored}

    relevant_conflicts, irrelevant_conflicts = [], []
    for c in conflicts:
        relevance = _conflict_relevance(question, c["subject"], c["relation"])
        if relevance >= CONFLICT_RELEVANCE_THRESHOLD:
            relevant_conflicts.append(c)
        else:
            irrelevant_conflicts.append({**c, "relevance": round(relevance, 4)})

    if not relevant_conflicts:
        remaining = sorted(scored, key=lambda x: x[2], reverse=True)
        action = (f"No relevant conflicts to repair: {len(irrelevant_conflicts)} conflict(s) "
                  f"below relevance threshold — evidence preserved unchanged.")
        return [(p, s) for i, p, s in remaining], [i for i, p, s in remaining], action, {
            "conflicts_found": len(conflicts), "conflicts_repaired": 0,
            "conflicts_ignored_as_irrelevant": irrelevant_conflicts,
        }

    excluded_ids = set()
    for c in relevant_conflicts:
        pids_per_object = {}
        for pid, triples in triples_per_paragraph.items():
            for t in triples:
                if t["subject"].lower() == c["subject"] and t["relation"] == c["relation"]:
                    pids_per_object.setdefault(t["object"].lower(), []).append(pid)
        variants = []
        for obj, pids in pids_per_object.items():
            best_pid = max(pids, key=lambda pid: sim_map.get(pid, 0.0))
            variants.append((obj, best_pid, sim_map.get(best_pid, 0.0)))
        if len(variants) > 1:
            variants.sort(key=lambda x: x[2], reverse=True)
            for obj, pid, sim in variants[1:]:
                excluded_ids.add(pid)

    remaining_paragraphs = [p for i, p, s in scored if i not in excluded_ids]
    unused_paragraphs = [p for p in all_paragraphs if p not in paragraphs]
    reformulated = f"{question} " + " ".join(list({t["subject"] for triples in triples_per_paragraph.values() for t in triples})[:8])
    backfill_scored = _per_paragraph_similarity(reformulated, unused_paragraphs)
    backfill_scored.sort(key=lambda x: x[2], reverse=True)
    needed = len(excluded_ids)
    backfill = [(p, s) for i, p, s in backfill_scored[:needed]]

    final_paragraphs = remaining_paragraphs + [p for p, s in backfill]
    final_scored = _per_paragraph_similarity(question, final_paragraphs)
    final_scored.sort(key=lambda x: x[2], reverse=True)

    repaired_top_results = [(p, s) for i, p, s in final_scored]
    selected_ids = list(range(len(final_scored)))
    action = (f"Conflict-aware repair: excluded {len(excluded_ids)} paragraph(s), "
              f"backfilled with {len(backfill)} replacement(s).")
    return repaired_top_results, selected_ids, action, {
        "conflicts_found": len(conflicts), "conflicts_repaired": len(relevant_conflicts),
        "conflicts_ignored_as_irrelevant": irrelevant_conflicts, "backfilled_count": len(backfill),
    }


def repair_knowledge_sparsity(question, all_paragraphs, top_results, keep_k=5):
    from repair.relevance_scorer import score_relevance
    scored = score_relevance(question, all_paragraphs)
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = scored[:keep_k]
    repaired_top_results = [(p, s) for p, s in kept]
    selected_ids = list(range(len(kept)))
    action = (f"Preserve-and-augment: cross-encoder scored the full "
              f"{len(all_paragraphs)}-paragraph pool, selected top {len(kept)}.")
    return repaired_top_results, selected_ids, action, {"method": "cross_encoder_full_pool"}


def repair_semantic_misalignment(question, all_paragraphs, top_results, keep_k=5):
    from repair.relevance_scorer import score_relevance
    scored = score_relevance(question, all_paragraphs)
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = scored[:keep_k]
    repaired_top_results = [(p, s) for p, s in kept]
    selected_ids = list(range(len(kept)))
    action = (f"Preserve-and-augment: cross-encoder scored the full "
              f"{len(all_paragraphs)}-paragraph pool, selected top {len(kept)}.")
    return repaired_top_results, selected_ids, action, {"method": "cross_encoder_full_pool"}


def apply_repair(cause, question, all_paragraphs, top_results):
    if cause == "C1_INFORMATION_CONFLICT":
        return repair_information_conflict(question, top_results, all_paragraphs)
    elif cause == "C2_KNOWLEDGE_SPARSITY":
        return repair_knowledge_sparsity(question, all_paragraphs, top_results)
    elif cause == "C3_SEMANTIC_MISALIGNMENT":
        return repair_semantic_misalignment(question, all_paragraphs, top_results)
    else:
        return top_results, list(range(len(top_results))), "No repair action defined for this cause.", {}


# ---------------------------------------------------------------------------
# Verification Gate — NEW
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Verification Gate — NLI Entailment Scorer
# ---------------------------------------------------------------------------

_nli_model = None


def _get_nli_model():
    """Lazy-load cross-encoder/nli-deberta-v3-base model."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return _nli_model


def _build_claim_hypothesis(question, answer_text):
    ans = str(answer_text).strip() if answer_text else ""
    if not ans or ans == "GENERATION ERROR" or ans.lower() == "insufficient evidence":
        return ""
    q = str(question).strip() if question else ""
    return f'The answer to the question "{q}" is {ans}.'


def _entailment_support_score(question, answer_text, evidence_paragraphs, label="evidence") -> float:
    """
    Scores how strongly the evidence entails the question-conditioned claim hypothesis.
    Uses cross-encoder/nli-deberta-v3-base model via sentence-transformers CrossEncoder.
    Hypothesis reformulation: 'The answer to the question "{question}" is {answer_text}.'
    Passage aggregation: Max entailment probability across individual evidence paragraphs to prevent sequence truncation.
    Output: entailment class probability (float in range [0.0, 1.0]).
    Labels convention: index 0 = contradiction, index 1 = entailment, index 2 = neutral.
    This function NEVER receives or references gold answers, gold supporting facts, or ground-truth labels.
    """
    if not evidence_paragraphs or not answer_text:
        return 0.0

    hypothesis = _build_claim_hypothesis(question, answer_text)
    if not hypothesis:
        return 0.0

    model = _get_nli_model()

    pairs = [(p.strip(), hypothesis) for p in evidence_paragraphs if p.strip()]
    if not pairs:
        return 0.0

    probs = model.predict(pairs, apply_softmax=True)
    entailment_probs = [float(p[1]) for p in probs]
    max_entailment = max(entailment_probs) if entailment_probs else 0.0

    print(f"    [DEBUG NLI ({label})] claim[:100]: {hypothesis[:100]!r}")
    print(f"    [DEBUG NLI ({label})] max_prob[1]: {max_entailment:.16f}")

    return max_entailment


def verification_gate(question, original_evidence, candidate_evidence):
    from generation.answer_generator import build_prompt, call_ollama, parse_llm_json

    orig_paragraphs = [p for p, s in original_evidence]
    cand_paragraphs = [p for p, s in candidate_evidence]

    orig_triples = get_triples_per_paragraph(orig_paragraphs)
    cand_triples = get_triples_per_paragraph(cand_paragraphs)

    orig_prompt = build_prompt(question, orig_paragraphs, orig_triples)
    cand_prompt = build_prompt(question, cand_paragraphs, cand_triples)

    orig_raw = call_ollama(orig_prompt)
    cand_raw = call_ollama(cand_prompt)

    orig_parsed = parse_llm_json(orig_raw)
    cand_parsed = parse_llm_json(cand_raw)

    orig_answer = orig_parsed.get("answer", "") if orig_parsed else "GENERATION ERROR"
    cand_answer = cand_parsed.get("answer", "") if cand_parsed else "GENERATION ERROR"
    cand_insufficient = cand_parsed.get("insufficient_evidence", False) if cand_parsed else True

    print(f"\n[DEBUG GATE] Question: {question[:100]!r}")
    print(f"[DEBUG GATE] orig_answer: {orig_answer!r}")
    print(f"[DEBUG GATE] cand_answer: {cand_answer!r}")

    original_score = _entailment_support_score(question, orig_answer, orig_paragraphs, label="ORIGINAL")
    candidate_score = _entailment_support_score(question, cand_answer, cand_paragraphs, label="CANDIDATE")

    orig_supported = original_score >= 0.5
    cand_supported = candidate_score >= 0.5

    if cand_insufficient:
        decision, final_ev, final_ans, case = "REJECT", original_evidence, orig_answer, "D"
        reason = "candidate abstained (insufficient_evidence)"
    elif cand_supported and not orig_supported:
        decision, final_ev, final_ans, case = "ACCEPT", candidate_evidence, cand_answer, "A"
        reason = f"candidate gained support original lacked (orig={original_score:.4f}, cand={candidate_score:.4f})"
    elif cand_supported and orig_supported:
        decision, final_ev, final_ans, case = "ACCEPT", candidate_evidence, cand_answer, "B"
        reason = f"both supported (orig={original_score:.4f}, cand={candidate_score:.4f}); prefer repair"
    elif not cand_supported and orig_supported:
        decision, final_ev, final_ans, case = "REJECT", original_evidence, orig_answer, "C"
        reason = f"repair lost support original had (orig={original_score:.4f}, cand={candidate_score:.4f})"
    else:
        decision, final_ev, final_ans, case = "REJECT", original_evidence, orig_answer, "D"
        reason = f"neither sufficiently supported (orig={original_score:.4f}, cand={candidate_score:.4f}); default to original"

    return {
        "decision": decision, "case": case, "reason": reason,
        "final_evidence": final_ev, "final_answer": final_ans,
        "original_answer": orig_answer, "candidate_answer": cand_answer,
        "original_score": original_score, "candidate_score": candidate_score,
        "original_supported": orig_supported, "candidate_supported": cand_supported,
    }


# ---------------------------------------------------------------------------
# Main entry point — dataset_path threaded through + gate wired in
# ---------------------------------------------------------------------------

def get_final_evidence(index=0, k=5, dataset_path=None):
    sample = load_question(index=index, dataset_path=dataset_path)
    question = sample["question"]
    all_paragraphs = extract_paragraphs(sample)
    original_top_results = retrieve_top_k(question, all_paragraphs, k=k)

    rca_result = run_rca_for_question(index=index, k=k, dataset_path=dataset_path)
    cause = rca_result["diagnosed_cause"]
    confidence_before = rca_result["final_confidence"]

    if confidence_before >= THETA:
        return {
            "question": question, "decision": "PASS", "diagnosed_cause": None,
            "repair_action": "No repair — confidence already sufficient.",
            "final_evidence": original_top_results, "confidence_before": confidence_before,
            "gate_result": None, "final_answer": None,
        }

    candidate_top_results, _, repair_action, _ = apply_repair(cause, question, all_paragraphs, original_top_results)
    gate_result = verification_gate(question, original_top_results, candidate_top_results)

    return {
        "question": question, "decision": "TRIGGER_RCA", "diagnosed_cause": cause,
        "repair_action": repair_action, "final_evidence": gate_result["final_evidence"],
        "confidence_before": confidence_before, "gate_result": gate_result,
        "final_answer": gate_result["final_answer"],
    }