"""
One-shot NLI label mapping verification for cross-encoder/nli-deberta-v3-base.
Runs three test pairs and prints softmax probabilities for each index.
Purpose: confirm which index corresponds to ENTAILMENT before the main validation run.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sentence_transformers import CrossEncoder

print("Loading cross-encoder/nli-deberta-v3-base for label verification...")
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# Pair 1: clear entailment — passage explicitly supports claim
pair_entail = (
    "George Raft starred in the 1945 film Johnny Angel as the lead actor.",
    "The answer to the question \"Who starred in Johnny Angel?\" is George Raft."
)

# Pair 2: clear contradiction — passage contradicts claim
pair_contradict = (
    "George Raft starred in the 1945 film Johnny Angel as the lead actor.",
    "The answer to the question \"Who starred in Johnny Angel?\" is Humphrey Bogart."
)

# Pair 3: neutral — passage is unrelated to the claim
pair_neutral = (
    "The Eiffel Tower is located in Paris, France.",
    "The answer to the question \"Who starred in Johnny Angel?\" is George Raft."
)

pairs = [pair_entail, pair_contradict, pair_neutral]
probs = model.predict(pairs, apply_softmax=True)

print()
print("=" * 60)
print("NLI LABEL MAPPING VERIFICATION")
print("=" * 60)
for label_idx, (pair_name, p) in enumerate(zip(
        ["ENTAILMENT pair", "CONTRADICTION pair", "NEUTRAL pair"], probs)):
    row = [round(float(x), 4) for x in p]
    max_idx = row.index(max(row))
    print(f"\n{pair_name}:")
    print(f"  Premise:    {pairs[label_idx][0][:80]}")
    print(f"  Hypothesis: {pairs[label_idx][1][:80]}")
    print(f"  Softmax probs [idx0, idx1, idx2]: {row}")
    print(f"  Max probability at index: {max_idx}  (value={max(row)})")

print()
print("=" * 60)
print("INTERPRETATION GUIDE:")
print("  For the ENTAILMENT pair, the highest prob should be at the entailment index.")
print("  For the CONTRADICTION pair, the highest prob should be at the contradiction index.")
print("  Typical nli-deberta-v3-base order: [contradiction=0, entailment=1, neutral=2]")
print("=" * 60)

# Cross-check against model config labels if available
try:
    labels = model.config.id2label
    print(f"\nModel config id2label: {labels}")
except Exception:
    print("\nModel config id2label: not accessible via this API")
