import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from confidence.confidence_assessment import compute_confidence

THETA = 0.60  # Experimental initial threshold — to be tuned via validation experiments


def adaptive_trigger(confidence_result, theta=THETA):
    C = confidence_result["final_confidence"]
    if C >= theta:
        decision = "PASS"
        reason = f"Confidence {C} >= threshold {theta}: evidence is reliable enough to generate directly."
    else:
        decision = "TRIGGER_RCA"
        reason = f"Confidence {C} < threshold {theta}: evidence is unreliable, escalating to Root Cause Analysis."

    return {
        "index": confidence_result["index"],
        "question": confidence_result["question"],
        "confidence_score": C,
        "threshold": theta,
        "decision": decision,
        "trigger_reason": reason
    }


if __name__ == "__main__":
    test_indices = [0, 1, 2, 10, 50]
    all_trigger_results = []

    print(f"{'Question':<60} {'Confidence':<12} {'Threshold':<10} {'Decision'}")
    print("-" * 100)

    for idx in test_indices:
        conf_result = compute_confidence(index=idx, k=5)
        trigger_result = adaptive_trigger(conf_result, theta=THETA)
        all_trigger_results.append(trigger_result)

        q_short = trigger_result["question"][:55] + ("..." if len(trigger_result["question"]) > 55 else "")
        print(f"{q_short:<60} {trigger_result['confidence_score']:<12} {trigger_result['threshold']:<10} {trigger_result['decision']}")

    output_dir = BASE_DIR / "results" / "trigger"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trigger_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_trigger_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")