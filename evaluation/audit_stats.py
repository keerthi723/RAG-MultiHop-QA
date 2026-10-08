import json

with open("results/evaluation/comparison_results.json", encoding="utf-8") as f:
    data = json.load(f)

n = len(data)

# A. Baseline vs RCA-RAG
improved = sum(1 for d in data if d["rca_rag_f1"] > d["baseline_f1"])
degraded = sum(1 for d in data if d["rca_rag_f1"] < d["baseline_f1"])
unchanged = n - improved - degraded
baseline_avg = sum(d["baseline_f1"] for d in data) / n
rca_avg = sum(d["rca_rag_f1"] for d in data) / n

print(f"A. n={n}  baseline_F1_avg={baseline_avg:.4f}  rca_F1_avg={rca_avg:.4f}")
print(f"   improved={improved}  degraded={degraded}  unchanged={unchanged}")
print()

# B. RCA categories
causes = {}
for d in data:
    c = d.get("diagnosed_cause") or "No_RCA"
    causes[c] = causes.get(c, 0) + 1
print("B. Cause distribution:", causes)
print("   Percentages:", {k: round(v / n, 4) for k, v in causes.items()})
print()

# C. Gate analysis
triggered = [d for d in data if d.get("rca_triggered")]
cases = {}
for d in triggered:
    c = d.get("gate_case", "None")
    cases[c] = cases.get(c, 0) + 1
print("C. Case distribution:", cases)

accepted = [d for d in triggered if d.get("gate_decision") == "ACCEPT"]
rejected = [d for d in triggered if d.get("gate_decision") == "REJECT"]

acc_helped = sum(1 for d in accepted if d["rca_rag_f1"] > d["baseline_f1"])
acc_hurt = sum(1 for d in accepted if d["rca_rag_f1"] < d["baseline_f1"])
rej_would_help = sum(1 for d in rejected if d["candidate_f1"] > d["rca_rag_f1"])
rej_would_hurt = sum(1 for d in rejected if d["candidate_f1"] < d["rca_rag_f1"])

print(f"   Accepted n={len(accepted)}: helped={acc_helped} hurt={acc_hurt}")
print(f"   Rejected n={len(rejected)}: would_have_helped={rej_would_help} would_have_hurt={rej_would_hurt}")


# D. Triggered-subset three-way comparison
triggered = [d for d in data if d.get("rca_triggered")]
n_trig = len(triggered)

baseline_on_triggered = sum(d["baseline_f1"] for d in triggered) / n_trig
candidate_on_triggered = sum(d["candidate_f1"] for d in triggered) / n_trig
gated_on_triggered = sum(d["rca_rag_f1"] for d in triggered) / n_trig

print()
print(f"D. Triggered subset n={n_trig}")
print(f"   Baseline F1 on this subset:        {baseline_on_triggered:.4f}")
print(f"   Candidate F1 on this subset:       {candidate_on_triggered:.4f}")
print(f"   Final (gated) RCA-RAG F1 on subset: {gated_on_triggered:.4f}")