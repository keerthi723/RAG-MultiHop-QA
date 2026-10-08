import json
import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "results" / "evaluation"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    comp_path = EVAL_DIR / "comparison_results.json"
    if not comp_path.exists():
        raise FileNotFoundError(
            "comparison_results.json not found. Run 'python evaluation\\evaluate.py --limit 100' first."
        )

    comparisons = load_json(comp_path)
    n = len(comparisons)

    # --- Config 1: Baseline RAG ---
    # Retrieval only. No confidence, no RCA, no repair.
    baseline_em = sum(c["baseline_em"] for c in comparisons) / n
    baseline_f1 = sum(c["baseline_f1"] for c in comparisons) / n

    # --- Config 2: RAG + Confidence ---
    # IMPORTANT, and this is the actual finding, not a bug: confidence is
    # COMPUTED here but never acted upon — the evidence set given to the LLM
    # is IDENTICAL to baseline's. Since nothing about the LLM's input changes,
    # EM/F1 are mathematically guaranteed to equal baseline's. What DOES
    # change is that we now have a confidence score attached to every answer.
    valid_conf = [c for c in comparisons if c.get("initial_confidence") is not None]
    avg_confidence_config2 = sum(c["initial_confidence"] for c in valid_conf) / len(valid_conf) if valid_conf else None
    config2_em = baseline_em
    config2_f1 = baseline_f1

    # --- Config 3: RAG + Confidence + RCA diagnosis, WITHOUT repair ---
    # A cause label is now diagnosed for low-confidence questions, but again,
    # the evidence set is NOT modified (repair is what modifies it). So this
    # config's EM/F1 also equal baseline's, by the same architectural logic.
    # What's new here is the cause distribution / trigger rate.
    triggered = [c for c in comparisons if c["rca_triggered"]]
    trigger_rate = len(triggered) / n
    cause_counts = Counter(c["diagnosed_cause"] for c in triggered if c["diagnosed_cause"])
    config3_em = baseline_em
    config3_f1 = baseline_f1

    # --- Config 4: Full RCA-RAG (confidence + RCA + repair) ---
    # This is the ONLY configuration where the evidence set is actually
    # modified before generation. Any EM/F1 difference from baseline is
    # therefore attributable specifically to the repair stage.
    rca_rag_em = sum(c["rca_rag_em"] for c in comparisons) / n
    rca_rag_f1 = sum(c["rca_rag_f1"] for c in comparisons) / n

    results = [
        {
            "configuration": "1. Baseline RAG",
            "retrieval": "Yes", "confidence": "No", "rca": "No", "repair": "No",
            "em": round(baseline_em, 4), "f1": round(baseline_f1, 4),
            "avg_confidence": "N/A", "trigger_rate": "N/A",
        },
        {
            "configuration": "2. RAG + Confidence",
            "retrieval": "Yes", "confidence": "Yes", "rca": "No", "repair": "No",
            "em": round(config2_em, 4), "f1": round(config2_f1, 4),
            "avg_confidence": round(avg_confidence_config2, 4) if avg_confidence_config2 else "N/A",
            "trigger_rate": "N/A",
        },
        {
            "configuration": "3. RAG + Confidence + RCA (no repair)",
            "retrieval": "Yes", "confidence": "Yes", "rca": "Yes", "repair": "No",
            "em": round(config3_em, 4), "f1": round(config3_f1, 4),
            "avg_confidence": round(avg_confidence_config2, 4) if avg_confidence_config2 else "N/A",
            "trigger_rate": round(trigger_rate, 4),
        },
        {
            "configuration": "4. Full RCA-RAG",
            "retrieval": "Yes", "confidence": "Yes", "rca": "Yes", "repair": "Yes",
            "em": round(rca_rag_em, 4), "f1": round(rca_rag_f1, 4),
            "avg_confidence": round(avg_confidence_config2, 4) if avg_confidence_config2 else "N/A",
            "trigger_rate": round(trigger_rate, 4),
        },
    ]

    # --- Write CSV ---
    csv_path = EVAL_DIR / "table_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "configuration", "retrieval", "confidence", "rca", "repair",
            "em", "f1", "avg_confidence", "trigger_rate"
        ])
        writer.writeheader()
        writer.writerows(results)

    with open(EVAL_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Print table ---
    print("=" * 90)
    print("ABLATION STUDY")
    print("=" * 90)
    print(f"{'Configuration':<40} {'EM':<8} {'F1':<8} {'Avg Conf':<10} {'Trigger Rate'}")
    print("-" * 90)
    for r in results:
        print(f"{r['configuration']:<40} {r['em']:<8} {r['f1']:<8} {str(r['avg_confidence']):<10} {r['trigger_rate']}")
    print("=" * 90)

    print("\nINTERPRETATION:")
    print(f"Configs 1-3 have IDENTICAL EM/F1 ({round(baseline_em,4)}/{round(baseline_f1,4)}) — this is")
    print("expected, not an error: confidence estimation and RCA diagnosis do not modify")
    print("the evidence set, so they architecturally cannot change the generated answer.")
    delta_f1 = round(rca_rag_f1 - baseline_f1, 4)
    direction = "decrease" if delta_f1 < 0 else "increase" if delta_f1 > 0 else "no change"
    print(f"Only Config 4 (repair) changes the evidence set, producing a {direction} of")
    print(f"{abs(delta_f1)} F1 versus baseline. This isolates the repair stage as the sole")
    print("architecturally possible source of the observed accuracy change.")

    print(f"\nSaved to:\n  {csv_path}\n  {EVAL_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()