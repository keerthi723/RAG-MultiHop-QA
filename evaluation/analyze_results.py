import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "results" / "evaluation"
OUT_DIR = EVAL_DIR

CAUSES = ["C1_INFORMATION_CONFLICT", "C2_KNOWLEDGE_SPARSITY", "C3_SEMANTIC_MISALIGNMENT"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_files():
    required = ["comparison_results.json", "summary.json", "baseline_results.json", "rca_rag_results.json"]
    missing = [f for f in required if not (EVAL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required result files: {missing}. "
            f"Run 'python evaluation\\evaluate.py --limit 100' first — "
            f"this script analyzes existing results, it does not generate them."
        )


# ---------- A. Baseline vs RCA-RAG ----------

def analyze_baseline_vs_rcarag(comparisons, summary):
    n = len(comparisons)
    return {
        "num_questions": n,
        "exact_match": summary["exact_match"],
        "token_f1": summary["token_f1"],
        "avg_confidence_rca_rag": summary["confidence"]["avg_after_repair"],
        "note": "avg_confidence has no baseline equivalent — baseline RAG does not compute confidence by design (fairness rule: only RCA-RAG receives confidence/RCA/repair)."
    }


# ---------- B. Root cause distribution ----------

def analyze_cause_distribution(rca_results):
    triggered = [r for r in rca_results if r.get("rca_triggered") and not r.get("failed")]
    n = len(rca_results)
    counts = Counter(r["diagnosed_cause"] for r in triggered if r["diagnosed_cause"])
    return {
        "total_questions": n,
        "rca_triggered_count": len(triggered),
        "C1_INFORMATION_CONFLICT": {"count": counts.get("C1_INFORMATION_CONFLICT", 0),
                                     "pct_of_all": round(counts.get("C1_INFORMATION_CONFLICT", 0) / n, 4)},
        "C2_KNOWLEDGE_SPARSITY": {"count": counts.get("C2_KNOWLEDGE_SPARSITY", 0),
                                   "pct_of_all": round(counts.get("C2_KNOWLEDGE_SPARSITY", 0) / n, 4)},
        "C3_SEMANTIC_MISALIGNMENT": {"count": counts.get("C3_SEMANTIC_MISALIGNMENT", 0),
                                      "pct_of_all": round(counts.get("C3_SEMANTIC_MISALIGNMENT", 0) / n, 4)},
    }


# ---------- C. Adaptive trigger analysis ----------

def analyze_trigger(rca_results):
    valid = [r for r in rca_results if not r.get("failed") and r.get("initial_confidence") is not None]
    passed = [r for r in valid if not r["rca_triggered"]]
    triggered = [r for r in valid if r["rca_triggered"]]
    n = len(valid)

    avg_pass_conf = sum(r["initial_confidence"] for r in passed) / len(passed) if passed else None
    avg_trigger_conf = sum(r["initial_confidence"] for r in triggered) / len(triggered) if triggered else None

    return {
        "total_valid": n,
        "pct_pass": round(len(passed) / n, 4) if n else None,
        "pct_trigger_rca": round(len(triggered) / n, 4) if n else None,
        "avg_confidence_pass_cases": round(avg_pass_conf, 4) if avg_pass_conf is not None else None,
        "avg_confidence_trigger_cases": round(avg_trigger_conf, 4) if avg_trigger_conf is not None else None,
    }


# ---------- D. Repair analysis (overall + per-cause) ----------

def analyze_repair(rca_results):
    triggered = [r for r in rca_results if r.get("rca_triggered") and not r.get("failed")
                 and r.get("confidence_delta") is not None]

    def summarize(subset):
        if not subset:
            return None
        n = len(subset)
        avg_before = sum(r["initial_confidence"] for r in subset) / n
        avg_after = sum(r["final_confidence"] for r in subset) / n
        avg_delta = sum(r["confidence_delta"] for r in subset) / n
        increased = sum(1 for r in subset if r["confidence_delta"] > 0)
        decreased = sum(1 for r in subset if r["confidence_delta"] < 0)
        unchanged = sum(1 for r in subset if r["confidence_delta"] == 0)
        return {
            "n": n,
            "avg_confidence_before": round(avg_before, 4),
            "avg_confidence_after": round(avg_after, 4),
            "avg_confidence_delta": round(avg_delta, 4),
            "pct_increased": round(increased / n, 4),
            "pct_decreased": round(decreased / n, 4),
            "pct_unchanged": round(unchanged / n, 4),
        }

    result = {"overall": summarize(triggered)}
    for cause in CAUSES:
        subset = [r for r in triggered if r["diagnosed_cause"] == cause]
        result[cause] = summarize(subset)
    return result


# ---------- E. Ablation study ----------

def analyze_ablation(summary):
    """
    Only two of the four ablation configurations actually exist in the
    current implementation:
      1. Baseline RAG          -> exists (results/evaluation/baseline_results.json)
      4. Full RCA-RAG           -> exists (results/evaluation/rca_rag_results.json)
    Configurations 2 and 3 do NOT exist as separate runs — the current
    pipeline does not have a mode that computes confidence WITHOUT also
    running RCA+repair when confidence is low, nor a mode that runs
    RCA (diagnosis) without the repair step. Reporting fabricated numbers
    for these would violate the no-fabrication rule, so they are marked
    NOT AVAILABLE with the specific additional implementation required.
    """
    return {
        "1_baseline_rag": {
            "exact_match": summary["exact_match"]["baseline"],
            "token_f1": summary["token_f1"]["baseline"],
            "status": "AVAILABLE",
        },
        "2_rag_plus_confidence": {
            "status": "NOT AVAILABLE",
            "reason": "No pipeline variant exists that computes confidence but always "
                      "generates directly regardless of the threshold (i.e., confidence "
                      "is measured/logged but never acted upon). Would require adding a "
                      "flag to evaluate.py to compute confidence without invoking RCA/repair.",
        },
        "3_rag_plus_confidence_plus_rca": {
            "status": "NOT AVAILABLE",
            "reason": "No pipeline variant exists that runs RCA cause diagnosis but skips "
                      "the repair step (i.e., evidence is left unmodified even when a cause "
                      "is diagnosed). Would require a flag to run diagnose_cause() without "
                      "calling apply_repair(), generating directly on original evidence.",
        },
        "4_full_rca_rag": {
            "exact_match": summary["exact_match"]["rca_rag"],
            "token_f1": summary["token_f1"]["rca_rag"],
            "status": "AVAILABLE",
        },
    }


# ---------- F. Error analysis — representative examples ----------

def analyze_errors(comparisons):
    examples = {}

    successful_repairs = [c for c in comparisons if c["rca_triggered"] and c["confidence_delta"] and c["confidence_delta"] > 0]
    failed_repairs = [c for c in comparisons if c["rca_triggered"] and c["confidence_delta"] and c["confidence_delta"] <= 0]

    def pick(lst, n=2):
        return lst[:n] if lst else []

    def format_example(c):
        return {
            "question": c["question"],
            "initial_evidence": c["rca_rag_evidence"] if c["confidence_delta"] and c["confidence_delta"] == 0 else None,
            "diagnosed_cause": c["diagnosed_cause"],
            "repair_action": c["repair_action"],
            "final_evidence": c["rca_rag_evidence"],
            "confidence_before": c["initial_confidence"],
            "confidence_after": c["final_confidence"],
            "confidence_delta": c["confidence_delta"],
            "baseline_answer": c["baseline_answer"],
            "rca_rag_answer": c["rca_rag_answer"],
            "ground_truth": c["ground_truth"],
        }

    examples["successful_repair_examples"] = [format_example(c) for c in pick(successful_repairs)]
    examples["failed_repair_examples"] = [format_example(c) for c in pick(failed_repairs)]

    for cause in CAUSES:
        subset = [c for c in comparisons if c["diagnosed_cause"] == cause]
        key = cause.lower() + "_examples"
        examples[key] = [format_example(c) for c in pick(subset)]

    return examples


# ---------- G. CSV tables ----------

def write_baseline_vs_rcarag_csv(analysis_a):
    path = OUT_DIR / "table_baseline_vs_rcarag.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline RAG", "RCA-RAG"])
        writer.writerow(["Exact Match", analysis_a["exact_match"]["baseline"], analysis_a["exact_match"]["rca_rag"]])
        writer.writerow(["Token F1", analysis_a["token_f1"]["baseline"], analysis_a["token_f1"]["rca_rag"]])
    return path


def write_cause_distribution_csv(analysis_b):
    path = OUT_DIR / "table_root_cause_distribution.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cause", "Count", "Percentage of All Questions"])
        for cause in CAUSES:
            writer.writerow([cause, analysis_b[cause]["count"], analysis_b[cause]["pct_of_all"]])
    return path


def write_repair_effectiveness_csv(analysis_d):
    path = OUT_DIR / "table_repair_effectiveness.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "N", "Avg Conf Before", "Avg Conf After", "Avg Delta", "% Increased", "% Decreased", "% Unchanged"])
        for key in ["overall"] + CAUSES:
            d = analysis_d.get(key)
            if d is None:
                writer.writerow([key, 0, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            else:
                writer.writerow([key, d["n"], d["avg_confidence_before"], d["avg_confidence_after"],
                                  d["avg_confidence_delta"], d["pct_increased"], d["pct_decreased"], d["pct_unchanged"]])
    return path


def write_ablation_csv(analysis_e):
    path = OUT_DIR / "table_ablation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Exact Match", "Token F1", "Status"])
        for key, label in [("1_baseline_rag", "1. Baseline RAG"),
                            ("2_rag_plus_confidence", "2. RAG + Confidence"),
                            ("3_rag_plus_confidence_plus_rca", "3. RAG + Confidence + RCA"),
                            ("4_full_rca_rag", "4. Full RCA-RAG")]:
            d = analysis_e[key]
            if d["status"] == "AVAILABLE":
                writer.writerow([label, d["exact_match"], d["token_f1"], "AVAILABLE"])
            else:
                writer.writerow([label, "N/A", "N/A", f"NOT AVAILABLE — {d['reason']}"])
    return path


# ---------- Plots ----------

def generate_plots(analysis_a, analysis_b, analysis_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []

    # Plot 1: Baseline vs RCA-RAG
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Exact Match", "Token F1"]
    baseline_vals = [analysis_a["exact_match"]["baseline"], analysis_a["token_f1"]["baseline"]]
    rca_vals = [analysis_a["exact_match"]["rca_rag"], analysis_a["token_f1"]["rca_rag"]]
    x = range(len(metrics))
    ax.bar([i - 0.2 for i in x], baseline_vals, width=0.4, label="Baseline RAG")
    ax.bar([i + 0.2 for i in x], rca_vals, width=0.4, label="RCA-RAG")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Baseline RAG vs RCA-RAG")
    ax.legend()
    p1 = OUT_DIR / "plot_baseline_vs_rcarag.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(p1)

    # Plot 2: Cause distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["C1: Conflict", "C2: Sparsity", "C3: Misalignment"]
    values = [analysis_b[c]["count"] for c in CAUSES]
    ax.bar(labels, values, color=["#d9534f", "#f0ad4e", "#5bc0de"])
    ax.set_ylabel("Number of Questions")
    ax.set_title("RCA Cause Distribution")
    p2 = OUT_DIR / "plot_cause_distribution.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(p2)

    # Plot 3: Confidence before vs after (overall + per cause)
    fig, ax = plt.subplots(figsize=(7, 4))
    groups, befores, afters = [], [], []
    for key, label in [("overall", "Overall")] + [(c, c.split("_")[0]) for c in CAUSES]:
        d = analysis_d.get(key)
        if d:
            groups.append(label)
            befores.append(d["avg_confidence_before"])
            afters.append(d["avg_confidence_after"])
    x = range(len(groups))
    ax.bar([i - 0.2 for i in x], befores, width=0.4, label="Before Repair")
    ax.bar([i + 0.2 for i in x], afters, width=0.4, label="After Repair")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Average Confidence")
    ax.set_title("Confidence Before vs After Repair")
    ax.legend()
    p3 = OUT_DIR / "plot_confidence_before_after.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(p3)

    return plot_paths


# ---------- Main ----------

def main():
    check_required_files()

    comparisons = load_json(EVAL_DIR / "comparison_results.json")
    summary = load_json(EVAL_DIR / "summary.json")
    rca_results = load_json(EVAL_DIR / "rca_rag_results.json")

    n = len(comparisons)
    if n < 100:
        print(f"\n*** WARNING: only {n} question(s) found in comparison_results.json. ***")
        print("*** Task 4 requires the full 100-question evaluation. Results below ***")
        print("*** are NOT sufficient for paper-ready claims — treat as a dry run. ***\n")

    analysis_a = analyze_baseline_vs_rcarag(comparisons, summary)
    analysis_b = analyze_cause_distribution(rca_results)
    analysis_c = analyze_trigger(rca_results)
    analysis_d = analyze_repair(rca_results)
    analysis_e = analyze_ablation(summary)
    analysis_f = analyze_errors(comparisons)

    write_baseline_vs_rcarag_csv(analysis_a)
    write_cause_distribution_csv(analysis_b)
    write_repair_effectiveness_csv(analysis_d)
    write_ablation_csv(analysis_e)

    with open(OUT_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_f, f, indent=2, ensure_ascii=False)

    try:
        plot_paths = generate_plots(analysis_a, analysis_b, analysis_d)
    except Exception as e:
        print(f"Plot generation skipped due to error: {e}")
        plot_paths = []

    full_analysis = {
        "A_baseline_vs_rcarag": analysis_a,
        "B_cause_distribution": analysis_b,
        "C_trigger_analysis": analysis_c,
        "D_repair_analysis": analysis_d,
        "E_ablation": analysis_e,
    }
    with open(OUT_DIR / "full_analysis.json", "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2, ensure_ascii=False)

    # ---------- Console summary ----------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Questions analyzed: {n}")
    print(f"\nA. Baseline vs RCA-RAG:")
    print(f"   EM: {analysis_a['exact_match']}")
    print(f"   F1: {analysis_a['token_f1']}")
    print(f"\nB. Cause distribution: {[(c, analysis_b[c]['pct_of_all']) for c in CAUSES]}")
    print(f"\nC. Trigger: {analysis_c['pct_pass']} PASS / {analysis_c['pct_trigger_rca']} TRIGGER_RCA")
    print(f"\nD. Repair (overall): {analysis_d['overall']}")
    print(f"\nE. Ablation: configs 2 and 3 NOT AVAILABLE (see table_ablation.csv for why)")
    print(f"\nFiles written to: {OUT_DIR}")
    if plot_paths:
        print("Plots:", [str(p.name) for p in plot_paths])

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print("1. Main finding        : [fill in once run on full 100 — compare EM/F1 above]")
    print("2. Strongest component : [identify from per-cause repair success in table_repair_effectiveness.csv]")
    print("3. Weakest component   : [identify from same table — lowest pct_increased]")
    print(f"4. Most common cause    : {max(CAUSES, key=lambda c: analysis_b[c]['count'])}")
    print(f"5. Repair success rate  : {summary.get('repair_success_rate')}")
    print("6. Improved over baseline: [compare EM/F1 deltas above — do not overstate small n]")
    print("7. Limitations to disclose:")
    print("   - Confidence is a proxy signal, not a direct hallucination/faithfulness measurement.")
    print("   - Ablation configs 2 and 3 (RAG+Confidence, RAG+Confidence+RCA) are not implemented.")
    print("   - C2 (sparsity) repair has limited headroom on HotpotQA distractor setting (only 10")
    print("     candidate paragraphs exist total, so pool expansion often re-ranks the same set).")
    print("   - Rule-based spaCy triple extraction is noisy; graph confidence signal is coarse.")
    print("   - Threshold values (theta=0.60, per-cause thresholds) are experimental, not tuned.")


if __name__ == "__main__":
    main()