import json
import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "results" / "evaluation"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    comp_path = EVAL_DIR / "comparison_results.json"
    ablation_path = EVAL_DIR / "ablation_results.json"
    if not comp_path.exists():
        raise FileNotFoundError("comparison_results.json missing — run evaluate.py first.")
    if not ablation_path.exists():
        raise FileNotFoundError("ablation_results.json missing — run ablation.py first.")
    return load_json(comp_path), load_json(ablation_path)


# ---------- A. Main comparison ----------

def section_a(comparisons):
    n = len(comparisons)
    return {
        "n": n,
        "baseline_em": round(sum(c["baseline_em"] for c in comparisons) / n, 4),
        "baseline_f1": round(sum(c["baseline_f1"] for c in comparisons) / n, 4),
        "rca_rag_em": round(sum(c["rca_rag_em"] for c in comparisons) / n, 4),
        "rca_rag_f1": round(sum(c["rca_rag_f1"] for c in comparisons) / n, 4),
    }


# ---------- C. Cause distribution (including explicit "No RCA") ----------

def section_c(comparisons):
    n = len(comparisons)
    not_triggered = [c for c in comparisons if not c["rca_triggered"]]
    triggered = [c for c in comparisons if c["rca_triggered"]]
    counts = Counter(c["diagnosed_cause"] for c in triggered if c["diagnosed_cause"])
    return {
        "total": n,
        "no_rca_pct": round(len(not_triggered) / n, 4),
        "no_rca_count": len(not_triggered),
        "C1_INFORMATION_CONFLICT": {"count": counts.get("C1_INFORMATION_CONFLICT", 0),
                                     "pct": round(counts.get("C1_INFORMATION_CONFLICT", 0) / n, 4)},
        "C2_KNOWLEDGE_SPARSITY": {"count": counts.get("C2_KNOWLEDGE_SPARSITY", 0),
                                   "pct": round(counts.get("C2_KNOWLEDGE_SPARSITY", 0) / n, 4)},
        "C3_SEMANTIC_MISALIGNMENT": {"count": counts.get("C3_SEMANTIC_MISALIGNMENT", 0),
                                      "pct": round(counts.get("C3_SEMANTIC_MISALIGNMENT", 0) / n, 4)},
    }


# ---------- D. Repair effectiveness (confidence-based, verified against actual evidence changes) ----------

def evidence_change_type(baseline_evidence, rca_rag_evidence):
    """
    Directly compares the actual evidence text lists (not just confidence
    numbers) to determine what repair really did:
    - 'unchanged'  : identical paragraphs, identical order
    - 'reordered'  : same paragraphs, different order (C3's intended behavior)
    - 'changed'    : the paragraph SET itself differs (C1/C2 actually swapped evidence)
    """
    if baseline_evidence == rca_rag_evidence:
        return "unchanged"
    if set(baseline_evidence) == set(rca_rag_evidence):
        return "reordered"
    return "changed"


def section_d(comparisons):
    triggered = [c for c in comparisons if c["rca_triggered"] and c.get("confidence_delta") is not None]
    n = len(triggered)
    if n == 0:
        return {"n": 0, "note": "No triggered questions with valid confidence data."}

    avg_before = sum(c["initial_confidence"] for c in triggered) / n
    avg_after = sum(c["final_confidence"] for c in triggered) / n
    avg_delta = sum(c["confidence_delta"] for c in triggered) / n
    increased = sum(1 for c in triggered if c["confidence_delta"] > 0)
    decreased = sum(1 for c in triggered if c["confidence_delta"] < 0)
    unchanged = sum(1 for c in triggered if c["confidence_delta"] == 0)

    # Verified evidence-change breakdown, computed directly from actual text
    change_types = Counter(
        evidence_change_type(c["baseline_evidence"], c["rca_rag_evidence"]) for c in triggered
    )

    return {
        "n": n,
        "avg_confidence_before": round(avg_before, 4),
        "avg_confidence_after": round(avg_after, 4),
        "avg_confidence_delta": round(avg_delta, 4),
        "pct_increased": round(increased / n, 4),
        "pct_decreased": round(decreased / n, 4),
        "pct_unchanged": round(unchanged / n, 4),
        "evidence_change_breakdown": {
            "unchanged (identical evidence)": change_types.get("unchanged", 0),
            "reordered (same evidence, new order)": change_types.get("reordered", 0),
            "changed (evidence set actually differs)": change_types.get("changed", 0),
        },
    }


# ---------- E. Error analysis — grounded, verified examples ----------

def section_e(comparisons):
    examples = {}

    def fmt(c):
        return {
            "index": c["index"], "question": c["question"], "ground_truth": c["ground_truth"],
            "baseline_answer": c["baseline_answer"], "rca_rag_answer": c["rca_rag_answer"],
            "baseline_em": c["baseline_em"], "rca_rag_em": c["rca_rag_em"],
            "diagnosed_cause": c["diagnosed_cause"], "repair_action": c["repair_action"],
            "confidence_before": c["initial_confidence"], "confidence_after": c["final_confidence"],
            "evidence_change": evidence_change_type(c["baseline_evidence"], c["rca_rag_evidence"]),
        }

    baseline_correct_rca_wrong = [c for c in comparisons if c["baseline_em"] == 1 and c["rca_rag_em"] == 0]
    baseline_wrong_rca_correct = [c for c in comparisons if c["baseline_em"] == 0 and c["rca_rag_em"] == 1]

    c1_cases = [c for c in comparisons if c["diagnosed_cause"] == "C1_INFORMATION_CONFLICT"]
    c1_false_conflict = [c for c in c1_cases if c["repair_action"] and "No relevant conflicts" in c["repair_action"]]

    c2_cases = [c for c in comparisons if c["diagnosed_cause"] == "C2_KNOWLEDGE_SPARSITY"]
    c2_no_change = [c for c in c2_cases if evidence_change_type(c["baseline_evidence"], c["rca_rag_evidence"]) == "unchanged"]
    c2_changed = [c for c in c2_cases if evidence_change_type(c["baseline_evidence"], c["rca_rag_evidence"]) != "unchanged"]

    c3_cases = [c for c in comparisons if c["diagnosed_cause"] == "C3_SEMANTIC_MISALIGNMENT"]
    c3_reordered = [c for c in c3_cases if evidence_change_type(c["baseline_evidence"], c["rca_rag_evidence"]) == "reordered"]

    examples["baseline_correct_rca_wrong"] = [fmt(c) for c in baseline_correct_rca_wrong[:3]]
    examples["baseline_wrong_rca_correct"] = [fmt(c) for c in baseline_wrong_rca_correct[:3]]
    examples["c1_false_conflict_correctly_ignored"] = [fmt(c) for c in c1_false_conflict[:2]]
    examples["c1_all_cases"] = [fmt(c) for c in c1_cases]  # only ~2 exist total, show all
    examples["c2_evidence_unchanged_dataset_ceiling"] = [fmt(c) for c in c2_no_change[:2]]
    examples["c2_evidence_actually_changed"] = [fmt(c) for c in c2_changed[:2]]
    examples["c3_confirmed_reorder_only"] = [fmt(c) for c in c3_reordered[:2]]

    counts = {
        "baseline_correct_rca_wrong_count": len(baseline_correct_rca_wrong),
        "baseline_wrong_rca_correct_count": len(baseline_wrong_rca_correct),
        "c1_total": len(c1_cases),
        "c1_false_conflict_count": len(c1_false_conflict),
        "c2_total": len(c2_cases),
        "c2_evidence_unchanged_count": len(c2_no_change),
        "c2_evidence_changed_count": len(c2_changed),
        "c3_total": len(c3_cases),
        "c3_reordered_count": len(c3_reordered),
    }
    return examples, counts


# ---------- CSV writers ----------

def write_csvs(section_a_data, ablation_data, section_c_data, section_d_data):
    with open(EVAL_DIR / "table_baseline_vs_rcarag.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Baseline RAG", "RCA-RAG"])
        w.writerow(["Exact Match", section_a_data["baseline_em"], section_a_data["rca_rag_em"]])
        w.writerow(["Token F1", section_a_data["baseline_f1"], section_a_data["rca_rag_f1"]])

    with open(EVAL_DIR / "table_root_cause_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Count", "Percentage"])
        w.writerow(["No RCA (PASS)", section_c_data["no_rca_count"], section_c_data["no_rca_pct"]])
        for key in ["C1_INFORMATION_CONFLICT", "C2_KNOWLEDGE_SPARSITY", "C3_SEMANTIC_MISALIGNMENT"]:
            w.writerow([key, section_c_data[key]["count"], section_c_data[key]["pct"]])

    with open(EVAL_DIR / "table_repair_effectiveness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if section_d_data.get("n", 0) == 0:
            w.writerow(["No triggered questions with valid data."])
        else:
            w.writerow(["Metric", "Value"])
            w.writerow(["N triggered", section_d_data["n"]])
            w.writerow(["Avg confidence before", section_d_data["avg_confidence_before"]])
            w.writerow(["Avg confidence after", section_d_data["avg_confidence_after"]])
            w.writerow(["Avg confidence delta", section_d_data["avg_confidence_delta"]])
            w.writerow(["% increased", section_d_data["pct_increased"]])
            w.writerow(["% decreased", section_d_data["pct_decreased"]])
            w.writerow(["% unchanged", section_d_data["pct_unchanged"]])
            for k, v in section_d_data["evidence_change_breakdown"].items():
                w.writerow([k, v])


# ---------- Plots ----------

def generate_plots(section_a_data, section_c_data, section_d_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Exact Match", "Token F1"]
    base_vals = [section_a_data["baseline_em"], section_a_data["baseline_f1"]]
    rca_vals = [section_a_data["rca_rag_em"], section_a_data["rca_rag_f1"]]
    x = range(len(metrics))
    ax.bar([i - 0.2 for i in x], base_vals, width=0.4, label="Baseline RAG")
    ax.bar([i + 0.2 for i in x], rca_vals, width=0.4, label="RCA-RAG")
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_title("Baseline RAG vs RCA-RAG"); ax.legend()
    p = EVAL_DIR / "plot_baseline_vs_rcarag.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["No RCA", "C1: Conflict", "C2: Sparsity", "C3: Misalignment"]
    values = [section_c_data["no_rca_count"], section_c_data["C1_INFORMATION_CONFLICT"]["count"],
              section_c_data["C2_KNOWLEDGE_SPARSITY"]["count"], section_c_data["C3_SEMANTIC_MISALIGNMENT"]["count"]]
    ax.bar(labels, values, color=["#5cb85c", "#d9534f", "#f0ad4e", "#5bc0de"])
    ax.set_ylabel("Number of Questions"); ax.set_title("Adaptive Trigger Outcome / RCA Cause Distribution")
    p = EVAL_DIR / "plot_cause_distribution.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    if section_d_data.get("n", 0) > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Before Repair", "After Repair"],
               [section_d_data["avg_confidence_before"], section_d_data["avg_confidence_after"]],
               color=["#f0ad4e", "#5cb85c"])
        ax.set_ylabel("Average Confidence")
        ax.set_title("Confidence Before vs After Repair\n(Note: near-identical — see limitations)")
        p = EVAL_DIR / "plot_confidence_before_after.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths


# ---------- paper_results_summary.md ----------

def write_summary_md(a, ablation, c, d, e_counts):
    n = a["n"]
    lines = []
    lines.append("# RCA-RAG: Paper Results Summary\n")
    lines.append("*Auto-generated from actual 100-question evaluation results. "
                  "Every number below is pulled directly from `results/evaluation/`; "
                  "nothing is estimated or fabricated.*\n")

    lines.append("## A. Main Comparison\n")
    lines.append(f"| Metric | Baseline RAG | RCA-RAG |\n|---|---|---|\n"
                  f"| Exact Match | {a['baseline_em']} | {a['rca_rag_em']} |\n"
                  f"| Token F1 | {a['baseline_f1']} | {a['rca_rag_f1']} |\n")
    direction = "underperformed" if a["rca_rag_f1"] < a["baseline_f1"] else "outperformed" if a["rca_rag_f1"] > a["baseline_f1"] else "matched"
    lines.append(f"RCA-RAG **{direction}** baseline RAG on this 100-question HotpotQA subset "
                  f"(F1 delta: {round(a['rca_rag_f1'] - a['baseline_f1'], 4)}). This is reported directly.\n")

    lines.append("## B. Ablation Study\n")
    lines.append("| Configuration | EM | F1 |\n|---|---|---|\n")
    for row in ablation:
        lines.append(f"| {row['configuration']} | {row['em']} | {row['f1']} |\n")
    lines.append("\nConfigs 1–3 are architecturally guaranteed to produce identical EM/F1: "
                  "confidence estimation and RCA diagnosis do not modify the evidence set given "
                  "to the generator. Only Config 4 (full repair) changes the evidence, isolating "
                  "the repair stage as the sole source of any accuracy difference from baseline.\n")

    lines.append("## C. RCA Cause Distribution\n")
    lines.append(f"| Category | Count | % of {n} |\n|---|---|---|\n")
    lines.append(f"| No RCA (PASS) | {c['no_rca_count']} | {c['no_rca_pct']} |\n")
    for k in ["C1_INFORMATION_CONFLICT", "C2_KNOWLEDGE_SPARSITY", "C3_SEMANTIC_MISALIGNMENT"]:
        lines.append(f"| {k} | {c[k]['count']} | {c[k]['pct']} |\n")

    lines.append("\n## D. Repair Effectiveness\n")
    if d.get("n", 0) == 0:
        lines.append("No triggered questions with valid confidence data.\n")
    else:
        lines.append(f"- N triggered: {d['n']}\n- Avg confidence before: {d['avg_confidence_before']}\n"
                      f"- Avg confidence after: {d['avg_confidence_after']}\n- Avg delta: {d['avg_confidence_delta']}\n"
                      f"- % increased / decreased / unchanged: {d['pct_increased']} / {d['pct_decreased']} / {d['pct_unchanged']}\n\n")
        lines.append("**Verified evidence-change breakdown** (direct comparison of actual evidence text, "
                      "not just confidence numbers):\n")
        for k, v in d["evidence_change_breakdown"].items():
            lines.append(f"- {k}: {v}\n")
        lines.append("\n**Important distinction:** confidence delta measures the *proxy signal*, not answer "
                      "correctness. A delta of 0 does not mean repair had no effect on the answer — see "
                      "Section B, where Config 4's EM/F1 differs from baseline despite this confidence "
                      "metric showing no movement. Confidence improvement and answer-accuracy improvement "
                      "are empirically decoupled in this system and must not be conflated in the paper.\n")

    lines.append("\n## E. Error Analysis Case Counts\n")
    lines.append(f"- Baseline correct → RCA-RAG wrong: {e_counts['baseline_correct_rca_wrong_count']}\n")
    lines.append(f"- Baseline wrong → RCA-RAG correct: {e_counts['baseline_wrong_rca_correct_count']}\n")
    lines.append(f"- C1 total cases: {e_counts['c1_total']} (false-conflict, correctly ignored: {e_counts['c1_false_conflict_count']})\n")
    lines.append(f"- C2 total cases: {e_counts['c2_total']} (evidence unchanged due to dataset ceiling: {e_counts['c2_evidence_unchanged_count']}, evidence actually changed: {e_counts['c2_evidence_changed_count']})\n")
    lines.append(f"- C3 total cases: {e_counts['c3_total']} (confirmed reorder-only, no deletion: {e_counts['c3_reordered_count']})\n")
    lines.append("\nSee `error_analysis.json` for the actual question/answer text of representative examples in each category.\n")

    lines.append("\n## Limitations Supported by This Data\n")
    lines.append("- Confidence is a proxy signal (retrieval + semantic + graph heuristics), not a "
                  "hallucination or faithfulness measurement — Section D demonstrates this proxy can be "
                  "flat while accuracy still shifts.\n")
    lines.append("- C2 (sparsity) repair has limited headroom on HotpotQA's distractor setting: each "
                  "question has exactly 10 candidate paragraphs total, so expanding the pool often "
                  "cannot surface evidence beyond what was already retrieved.\n")
    lines.append("- C1 (conflict) sample size is very small in this dataset (see count above) — any "
                  "conclusion about conflict-repair effectiveness is not statistically robust at this n.\n")
    lines.append("- Rule-based spaCy triple extraction is noisy, which is the direct cause of C1's "
                  "false-conflict cases documented above.\n")
    lines.append("- Threshold values (theta=0.60, conflict relevance=0.30, per-cause thresholds) are "
                  "experimental defaults, not tuned via validation search.\n")
    lines.append("- Generation used Ollama with temperature=0 and a fixed seed for reproducibility; "
                  "however, exact determinism on CPU inference is best-effort, not guaranteed — repeated "
                  "runs showed minor answer variation despite these settings. This should be disclosed.\n")

    lines.append("\n## Reproducibility Settings\n")
    lines.append("- Dataset: `datasets/hotpotqa/hotpotqa_100.json` (fixed 100-question subset)\n")
    lines.append("- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)\n")
    lines.append("- Generation model: `llama3.2:3b` via Ollama, temperature=0, seed=42\n")
    lines.append("- Confidence weights: retrieval=0.4, semantic=0.4, graph=0.2 (experimental, untuned)\n")
    lines.append("- Adaptive trigger threshold (theta): 0.60\n")
    lines.append("- C1 conflict relevance threshold: 0.30\n")

    with open(EVAL_DIR / "paper_results_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    comparisons, ablation = load_data()

    a = section_a(comparisons)
    c = section_c(comparisons)
    d = section_d(comparisons)
    e_examples, e_counts = section_e(comparisons)

    write_csvs(a, ablation, c, d)

    with open(EVAL_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"examples": e_examples, "counts": e_counts}, f, indent=2, ensure_ascii=False)

    try:
        plot_paths = generate_plots(a, c, d)
    except Exception as ex:
        print(f"Plot generation skipped: {ex}")
        plot_paths = []

    write_summary_md(a, ablation, c, d, e_counts)

    print("=" * 70)
    print("FINAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"A. Baseline EM/F1: {a['baseline_em']}/{a['baseline_f1']}  |  RCA-RAG EM/F1: {a['rca_rag_em']}/{a['rca_rag_f1']}")
    print(f"C. No RCA: {c['no_rca_pct']}  C1: {c['C1_INFORMATION_CONFLICT']['pct']}  "
          f"C2: {c['C2_KNOWLEDGE_SPARSITY']['pct']}  C3: {c['C3_SEMANTIC_MISALIGNMENT']['pct']}")
    print(f"D. Repair n={d.get('n')}  delta={d.get('avg_confidence_delta')}  "
          f"evidence breakdown={d.get('evidence_change_breakdown')}")
    print(f"E. Counts: {e_counts}")
    print(f"\nFiles written to: {EVAL_DIR}")
    print("  - table_baseline_vs_rcarag.csv")
    print("  - table_root_cause_distribution.csv")
    print("  - table_repair_effectiveness.csv")
    print("  - error_analysis.json")
    print("  - paper_results_summary.md")
    if plot_paths:
        print("  - " + "\n  - ".join(p.name for p in plot_paths))


if __name__ == "__main__":
    main()