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

from repair.repair_pipeline import get_final_evidence

VAL_FILE = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_val_30.json"

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Running gate sanity check on {len(val_data)} validation questions (indices 101-130)...")
print("Using dataset_path parameter explicitly — no global state modified.\n")

case_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
pass_count = 0

logged_results = []

for i in range(len(val_data)):
    result = get_final_evidence(index=i, k=5, dataset_path=VAL_FILE)

    if result["decision"] == "PASS":
        pass_count += 1
        print(f"[{i}] PASS (confidence sufficient)")
        logged_results.append({
            "question_index": i,
            "diagnosed_cause": None,
            "original_score": None,
            "candidate_score": None,
            "gate_case": None,
            "gate_decision": "PASS",
            "reason": "confidence sufficient (no repair triggered)"
        })
    else:
        gate_res = result["gate_result"]
        case = gate_res["case"]
        case_counts[case] += 1
        orig_s = gate_res.get("original_score")
        cand_s = gate_res.get("candidate_score")
        print(
            f"[{i}] {result['diagnosed_cause']} -> "
            f"Gate: {gate_res['decision']} (Case {case}) "
            f"[orig_score={orig_s:.4f}, cand_score={cand_s:.4f}] — {gate_res['reason']}"
        )
        logged_results.append({
            "question_index": i,
            "diagnosed_cause": result["diagnosed_cause"],
            "original_score": orig_s,
            "candidate_score": cand_s,
            "gate_case": case,
            "gate_decision": gate_res["decision"],
            "reason": gate_res["reason"]
        })

print(f"\nPASS (no repair needed): {pass_count}")
print(f"Case A (accept, gained support): {case_counts['A']}")
print(f"Case B (accept, both supported): {case_counts['B']}")
print(f"Case C (reject, lost support): {case_counts['C']}")
print(f"Case D (reject, neither/abstain): {case_counts['D']}")

total = pass_count + sum(case_counts.values())
print(f"\nTotal processed: {total} / {len(val_data)} expected")

out_dir = BASE_DIR / "results" / "evaluation"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "validation_gate_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(logged_results, f, indent=2, ensure_ascii=False)
print(f"Saved validation gate details to {out_file}")

if total != len(val_data):
    print("WARNING: totals do not match — investigate before proceeding to the frozen 100-question run.")
else:
    print("Totals match. Mechanism confirmed working on validation subset — do not tune anything based on these numbers.")