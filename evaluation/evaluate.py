import sys
import json
import re
import string
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k
from repair.repair_pipeline import get_final_evidence, THETA

HOTPOT_100 = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_100.json"
OUT_DIR = BASE_DIR / "results" / "evaluation"


def to_answer_string(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def normalize_text(s):
    s = to_answer_string(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def run_baseline(index, k=5):
    from generation.answer_generator import build_prompt, call_ollama, parse_llm_json
    sample = load_question(index=index)
    question = sample["question"]
    ground_truth = sample["answer"]
    all_paragraphs = extract_paragraphs(sample)
    top_results = retrieve_top_k(question, all_paragraphs, k=k)
    evidence_paragraphs = [p for p, s in top_results]

    empty_triples = {i: [] for i in range(len(evidence_paragraphs))}
    prompt = build_prompt(question, evidence_paragraphs, empty_triples)
    raw = call_ollama(prompt)
    parsed = parse_llm_json(raw)

    if parsed is None:
        return {"index": index, "question": question, "ground_truth": ground_truth,
                "answer": "GENERATION ERROR", "evidence_paragraphs": evidence_paragraphs, "failed": True}

    answer = to_answer_string(parsed.get("answer", ""))
    return {"index": index, "question": question, "ground_truth": ground_truth,
            "answer": answer, "evidence_paragraphs": evidence_paragraphs, "failed": False}


def run_rca_rag(index, k=5):
    sample = load_question(index=index)
    ground_truth = sample["answer"]

    result = get_final_evidence(index=index, k=k)
    question = result["question"]
    final_evidence_paragraphs = [p for p, s in result["final_evidence"]]

    if result["decision"] == "PASS":
        from generation.answer_generator import build_prompt, call_ollama, parse_llm_json
        from knowledge_graph.graph_builder import get_triples_per_paragraph
        triples = get_triples_per_paragraph(final_evidence_paragraphs)
        prompt = build_prompt(question, final_evidence_paragraphs, triples)
        raw = call_ollama(prompt)
        parsed = parse_llm_json(raw)
        final_answer = to_answer_string(parsed.get("answer", "")) if parsed else "GENERATION ERROR"
        candidate_answer = None
        gate_case = None
        gate_decision = None
    else:
        final_answer = to_answer_string(result["final_answer"])
        candidate_answer = to_answer_string(result["gate_result"]["candidate_answer"])
        gate_case = result["gate_result"]["case"]
        gate_decision = result["gate_result"]["decision"]

    return {
        "index": index, "question": question, "ground_truth": ground_truth,
        "answer": final_answer, "evidence_paragraphs": final_evidence_paragraphs,
        "failed": final_answer == "GENERATION ERROR",
        "rca_triggered": result["decision"] == "TRIGGER_RCA",
        "diagnosed_cause": result["diagnosed_cause"],
        "candidate_answer": candidate_answer,
        "gate_case": gate_case,
        "gate_decision": gate_decision,
    }


def save_checkpoint(baseline_results, rca_rag_results, comparison_results, failures):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "rca_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(rca_rag_results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    if failures:
        with open(OUT_DIR / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)


def load_existing_progress():
    comp_path = OUT_DIR / "comparison_results.json"
    base_path = OUT_DIR / "baseline_results.json"
    rca_path = OUT_DIR / "rca_rag_results.json"
    if comp_path.exists() and base_path.exists() and rca_path.exists():
        with open(comp_path, encoding="utf-8") as f:
            comparison_results = json.load(f)
        with open(base_path, encoding="utf-8") as f:
            baseline_results = json.load(f)
        with open(rca_path, encoding="utf-8") as f:
            rca_rag_results = json.load(f)
        done_indices = set(c["index"] for c in comparison_results)
        return baseline_results, rca_rag_results, comparison_results, done_indices
    return [], [], [], set()


def run_evaluation(limit=100, k=5, resume=True, checkpoint_every=5):
    with open(HOTPOT_100, encoding="utf-8") as f:
        data = json.load(f)
    n = min(limit, len(data))

    if resume:
        baseline_results, rca_rag_results, comparison_results, done_indices = load_existing_progress()
        if done_indices:
            print(f"Resuming: {len(done_indices)} question(s) already completed, skipping them.")
    else:
        baseline_results, rca_rag_results, comparison_results, done_indices = [], [], [], set()

    failures = []
    processed_this_run = 0

    for idx in range(n):
        if idx in done_indices:
            continue
        print(f"\n[{idx+1}/{n}] Processing index {idx}...")

        try:
            b = run_baseline(idx, k=k)
        except Exception as e:
            print(f"  BASELINE FAILED: {e}")
            failures.append({"index": idx, "system": "baseline", "error": str(e)})
            b = {"index": idx, "question": "N/A", "ground_truth": "N/A", "answer": "ERROR",
                 "evidence_paragraphs": [], "failed": True}

        try:
            r = run_rca_rag(idx, k=k)
        except Exception as e:
            print(f"  RCA-RAG FAILED: {e}")
            failures.append({"index": idx, "system": "rca_rag", "error": str(e)})
            r = {"index": idx, "question": b["question"], "ground_truth": b["ground_truth"],
                 "answer": "ERROR", "evidence_paragraphs": [], "failed": True,
                 "rca_triggered": False, "diagnosed_cause": None,
                 "candidate_answer": None, "gate_case": None, "gate_decision": None}

        baseline_results.append(b)
        rca_rag_results.append(r)

        b_em = exact_match(b["answer"], b["ground_truth"]) if not b["failed"] else 0
        b_f1 = token_f1(b["answer"], b["ground_truth"]) if not b["failed"] else 0.0
        r_em = exact_match(r["answer"], r["ground_truth"]) if not r["failed"] else 0
        r_f1 = token_f1(r["answer"], r["ground_truth"]) if not r["failed"] else 0.0

        cand_em, cand_f1 = None, None
        if r.get("candidate_answer") is not None:
            cand_em = exact_match(r["candidate_answer"], r["ground_truth"])
            cand_f1 = token_f1(r["candidate_answer"], r["ground_truth"])

        comparison_results.append({
            "index": idx, "question": b["question"], "ground_truth": b["ground_truth"],
            "baseline_answer": b["answer"], "rca_rag_answer": r["answer"],
            "baseline_evidence": b["evidence_paragraphs"], "rca_rag_evidence": r["evidence_paragraphs"],
            "baseline_em": b_em, "rca_rag_em": r_em,
            "baseline_f1": round(b_f1, 4), "rca_rag_f1": round(r_f1, 4),
            "candidate_em": cand_em, "candidate_f1": round(cand_f1, 4) if cand_f1 is not None else None,
            "rca_triggered": r["rca_triggered"], "diagnosed_cause": r["diagnosed_cause"],
            "gate_case": r["gate_case"], "gate_decision": r["gate_decision"],
        })

        print(f"  Baseline EM/F1: {b_em}/{round(b_f1,3)}  |  RCA-RAG EM/F1: {r_em}/{round(r_f1,3)}  |  Gate: {r['gate_decision']} (Case {r['gate_case']})")

        processed_this_run += 1
        if processed_this_run % checkpoint_every == 0:
            save_checkpoint(baseline_results, rca_rag_results, comparison_results, failures)
            print(f"  [checkpoint saved — {len(comparison_results)}/{n} total questions done]")

    save_checkpoint(baseline_results, rca_rag_results, comparison_results, failures)

    n_total = len(comparison_results)
    baseline_em_avg = sum(c["baseline_em"] for c in comparison_results) / n_total
    baseline_f1_avg = sum(c["baseline_f1"] for c in comparison_results) / n_total
    rca_em_avg = sum(c["rca_rag_em"] for c in comparison_results) / n_total
    rca_f1_avg = sum(c["rca_rag_f1"] for c in comparison_results) / n_total

    cand_scored = [c for c in comparison_results if c["candidate_em"] is not None]
    cand_em_avg = sum(c["candidate_em"] for c in cand_scored) / len(cand_scored) if cand_scored else None
    cand_f1_avg = sum(c["candidate_f1"] for c in cand_scored) / len(cand_scored) if cand_scored else None

    triggered = [c for c in comparison_results if c["rca_triggered"]]
    case_counts = Counter(c["gate_case"] for c in triggered if c["gate_case"])
    accept_count = sum(1 for c in triggered if c["gate_decision"] == "ACCEPT")
    reject_count = sum(1 for c in triggered if c["gate_decision"] == "REJECT")

    summary = {
        "num_questions_evaluated": n_total, "num_failures": len(failures),
        "exact_match": {"baseline": round(baseline_em_avg, 4), "rca_rag": round(rca_em_avg, 4),
                         "repair_candidate": round(cand_em_avg, 4) if cand_em_avg is not None else None},
        "token_f1": {"baseline": round(baseline_f1_avg, 4), "rca_rag": round(rca_f1_avg, 4),
                     "repair_candidate": round(cand_f1_avg, 4) if cand_f1_avg is not None else None},
        "gate": {"accept_count": accept_count, "reject_count": reject_count,
                 "accept_rate": round(accept_count / len(triggered), 4) if triggered else None,
                 "case_A_count": case_counts.get("A", 0), "case_B_count": case_counts.get("B", 0),
                 "case_C_count": case_counts.get("C", 0), "case_D_count": case_counts.get("D", 0)},
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (gated repair)")
    print("=" * 70)
    print(f"Baseline EM/F1:         {summary['exact_match']['baseline']}/{summary['token_f1']['baseline']}")
    print(f"RCA-RAG (gated) EM/F1:  {summary['exact_match']['rca_rag']}/{summary['token_f1']['rca_rag']}")
    print(f"Repair candidate EM/F1: {summary['exact_match']['repair_candidate']}/{summary['token_f1']['repair_candidate']}")
    print(f"Gate accept/reject:     {accept_count}/{reject_count}  (accept rate: {summary['gate']['accept_rate']})")
    print(f"Case A/B/C/D:           {summary['gate']['case_A_count']}/{summary['gate']['case_B_count']}/{summary['gate']['case_C_count']}/{summary['gate']['case_D_count']}")
    print(f"Failures: {len(failures)}")
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, resume=not args.no_resume)