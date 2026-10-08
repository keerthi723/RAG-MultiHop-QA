"""
Enhanced 30-question validation gate diagnostic.
Indices 101-130 (hotpotqa_val_30.json).

Captures for every question:
  - question index and text
  - RCA cause
  - original evidence IDs and texts
  - candidate evidence IDs and texts
  - whether evidence actually changed
  - original provisional answer
  - candidate provisional answer
  - FULL original claim sent to NLI (not truncated)
  - FULL candidate claim sent to NLI (not truncated)
  - original entailment score (max over passages)
  - candidate entailment score (max over passages)
  - per-passage NLI scores for original and candidate
  - decision (ACCEPT/REJECT)
  - Case (A/B/C/D)
  - final evidence IDs
  - reason for decision

Does NOT modify repair_pipeline.py or any algorithm.
Imports pipeline sub-functions to orchestrate with extra logging.
No gold answers or supporting_facts accessed during inference.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---- imports from existing pipeline (no modifications) ----
from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k
from root_cause.root_cause_analysis import run_rca_for_question
from repair.repair_pipeline import (
    apply_repair, THETA,
    _build_claim_hypothesis, _get_nli_model,
)
from generation.answer_generator import build_prompt, call_ollama, parse_llm_json
from knowledge_graph.graph_builder import get_triples_per_paragraph

VAL_FILE = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_val_30.json"
OUT_DIR = BASE_DIR / "results" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Enhanced 30-question validation gate run")
print(f"Dataset: {VAL_FILE.name}  |  N={len(val_data)}")
print(f"THETA={THETA}  |  NLI_threshold=0.5  |  TAU=0.0  |  Aggregation=MAX")
print(f"No gold answers or supporting_facts accessed during inference.")
print("=" * 72)


def score_entailment_with_detail(question, answer_text, evidence_paragraphs, label):
    """
    Like _entailment_support_score but returns per-passage breakdown too.
    Does NOT modify the algorithm — same logic, extra return values.
    """
    if not evidence_paragraphs or not answer_text:
        return 0.0, [], ""

    hypothesis = _build_claim_hypothesis(question, answer_text)
    if not hypothesis:
        return 0.0, [], hypothesis

    nli_model = _get_nli_model()
    pairs = [(p.strip(), hypothesis) for p in evidence_paragraphs if p.strip()]
    if not pairs:
        return 0.0, [], hypothesis

    probs = nli_model.predict(pairs, apply_softmax=True)
    entailment_probs = [float(p[1]) for p in probs]   # index 1 = entailment
    max_ent = max(entailment_probs) if entailment_probs else 0.0

    return max_ent, entailment_probs, hypothesis


# ---- main loop ----
case_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
pass_count = 0
logged_results = []
all_nli_scores = []   # flat list of all individual NLI scores for distribution report

for i in range(len(val_data)):
    print(f"\n{'='*72}")
    print(f"[{i}] Question: {val_data[i]['question']}")

    # ---- retrieval (uses only question + distractor paragraphs) ----
    sample = load_question(index=i, dataset_path=VAL_FILE)
    question = sample["question"]
    all_paragraphs = extract_paragraphs(sample)
    original_top_results = retrieve_top_k(question, all_paragraphs, k=5)
    orig_paragraphs = [p for p, s in original_top_results]

    # ---- RCA ----
    rca_result = run_rca_for_question(index=i, k=5, dataset_path=VAL_FILE)
    cause = rca_result["diagnosed_cause"]
    conf_before = rca_result["final_confidence"]

    if conf_before >= THETA:
        print(f"  PASS (confidence={conf_before:.4f} >= THETA={THETA})")
        pass_count += 1
        logged_results.append({
            "question_index": i,
            "question": question,
            "decision": "PASS",
            "diagnosed_cause": None,
            "confidence_before": round(conf_before, 4),
            "gate_case": None,
            "gate_decision": "PASS",
            "reason": "confidence sufficient — no repair triggered",
        })
        continue

    print(f"  Triggered RCA | cause={cause} | conf={conf_before:.4f}")

    # ---- repair ----
    candidate_top_results, _, repair_action, _ = apply_repair(
        cause, question, all_paragraphs, original_top_results
    )
    cand_paragraphs = [p for p, s in candidate_top_results]

    # did evidence actually change?
    evidence_changed = (orig_paragraphs != cand_paragraphs)
    print(f"  Repair action: {repair_action[:80]}")
    print(f"  Evidence changed: {evidence_changed}")
    print(f"  Orig para count: {len(orig_paragraphs)}  |  Cand para count: {len(cand_paragraphs)}")

    # ---- provisional answers ----
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

    print(f"  orig_answer: {orig_answer!r}")
    print(f"  cand_answer: {cand_answer!r}")

    # ---- NLI scoring with full per-passage detail ----
    orig_score, orig_per_para, orig_claim = score_entailment_with_detail(
        question, orig_answer, orig_paragraphs, "ORIGINAL"
    )
    cand_score, cand_per_para, cand_claim = score_entailment_with_detail(
        question, cand_answer, cand_paragraphs, "CANDIDATE"
    )

    all_nli_scores.extend(orig_per_para)
    all_nli_scores.extend(cand_per_para)

    print(f"  orig_claim : {orig_claim}")
    print(f"  cand_claim : {cand_claim}")
    print(f"  orig NLI per-para: {[round(x,4) for x in orig_per_para]}")
    print(f"  cand NLI per-para: {[round(x,4) for x in cand_per_para]}")
    print(f"  orig_score (max): {orig_score:.6f}  |  cand_score (max): {cand_score:.6f}")

    # ---- gate decision (same logic as verification_gate, no modification) ----
    orig_supported = orig_score >= 0.5
    cand_supported = cand_score >= 0.5

    if cand_insufficient:
        decision, case = "REJECT", "D"
        final_evidence = original_top_results
        final_answer = orig_answer
        reason = "candidate abstained (insufficient_evidence)"
    elif cand_supported and not orig_supported:
        decision, case = "ACCEPT", "A"
        final_evidence = candidate_top_results
        final_answer = cand_answer
        reason = f"candidate gained support original lacked (orig={orig_score:.4f}, cand={cand_score:.4f})"
    elif cand_supported and orig_supported:
        decision, case = "ACCEPT", "B"
        final_evidence = candidate_top_results
        final_answer = cand_answer
        reason = f"both supported (orig={orig_score:.4f}, cand={cand_score:.4f}); prefer repair"
    elif not cand_supported and orig_supported:
        decision, case = "REJECT", "C"
        final_evidence = original_top_results
        final_answer = orig_answer
        reason = f"repair lost support original had (orig={orig_score:.4f}, cand={cand_score:.4f})"
    else:
        decision, case = "REJECT", "D"
        final_evidence = original_top_results
        final_answer = orig_answer
        reason = f"neither sufficiently supported (orig={orig_score:.4f}, cand={cand_score:.4f})"

    case_counts[case] += 1
    final_paras = [p for p, s in final_evidence]

    print(f"  Gate: {decision} (Case {case})  —  {reason}")
    print(f"  Final evidence == candidate: {final_paras == cand_paragraphs}")

    logged_results.append({
        "question_index": i,
        "question": question,
        "decision": "TRIGGER_RCA",
        "diagnosed_cause": cause,
        "confidence_before": round(conf_before, 4),
        "evidence_changed": evidence_changed,
        "orig_evidence_snippets": [p[:100] for p in orig_paragraphs],
        "cand_evidence_snippets": [p[:100] for p in cand_paragraphs],
        "final_evidence_is_candidate": (final_paras == cand_paragraphs),
        "orig_answer": orig_answer,
        "cand_answer": cand_answer,
        "orig_claim": orig_claim,
        "cand_claim": cand_claim,
        "orig_nli_per_para": [round(x, 6) for x in orig_per_para],
        "cand_nli_per_para": [round(x, 6) for x in cand_per_para],
        "original_score": round(orig_score, 6),
        "candidate_score": round(cand_score, 6),
        "orig_supported": orig_supported,
        "cand_supported": cand_supported,
        "gate_case": case,
        "gate_decision": decision,
        "reason": reason,
        "cand_insufficient": cand_insufficient,
        "final_answer": final_answer,
    })

# ---- summary ----
print("\n" + "=" * 72)
print("VALIDATION SUMMARY (indices 101-130)")
print("=" * 72)
total = pass_count + sum(case_counts.values())
accepted = case_counts["A"] + case_counts["B"]
rejected = case_counts["C"] + case_counts["D"]
triggered = [r for r in logged_results if r["decision"] == "TRIGGER_RCA"]
changed_ev = sum(1 for r in triggered if r.get("evidence_changed"))
unchanged_ev = sum(1 for r in triggered if not r.get("evidence_changed"))

print(f"PASS (no repair needed):  {pass_count}")
print(f"Case A (accept, gained):  {case_counts['A']}")
print(f"Case B (accept, both):    {case_counts['B']}")
print(f"Case C (reject, lost):    {case_counts['C']}")
print(f"Case D (reject, neither): {case_counts['D']}")
print(f"")
print(f"Total processed:          {total} / {len(val_data)} expected")
print(f"Accepted repairs:         {accepted}")
print(f"Rejected repairs:         {rejected}")
print(f"Changed evidence sets:    {changed_ev}")
print(f"Unchanged evidence sets:  {unchanged_ev}")

# NLI score distribution
if all_nli_scores:
    import statistics
    buckets = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0}
    for s in all_nli_scores:
        if s < 0.1:
            buckets["<0.1"] += 1
        elif s < 0.3:
            buckets["0.1-0.3"] += 1
        elif s < 0.5:
            buckets["0.3-0.5"] += 1
        elif s < 0.7:
            buckets["0.5-0.7"] += 1
        elif s < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets[">=0.9"] += 1
    print(f"\nNLI Score Distribution (all per-passage scores, n={len(all_nli_scores)}):")
    for bucket, cnt in buckets.items():
        bar = "#" * cnt
        print(f"  {bucket:10s}: {cnt:4d}  {bar}")
    print(f"  mean={statistics.mean(all_nli_scores):.4f}  "
          f"median={statistics.median(all_nli_scores):.4f}  "
          f"max={max(all_nli_scores):.4f}  "
          f"min={min(all_nli_scores):.4f}")

# ---- save detailed results ----
out_file = OUT_DIR / "validation_gate_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(logged_results, f, indent=2, ensure_ascii=False)
print(f"\nDetailed results saved to: {out_file}")

if total != len(val_data):
    print("WARNING: totals do not match — investigate before proceeding.")
else:
    print("Totals match. All 30 questions processed.")
