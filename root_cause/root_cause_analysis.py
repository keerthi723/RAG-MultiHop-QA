import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from confidence.confidence_assessment import compute_confidence
from knowledge_graph.graph_builder import run_for_question

# Experimental initial thresholds — to be tuned via validation experiments later
LOW_RETRIEVAL_THRESHOLD = 0.45
LOW_SEMANTIC_THRESHOLD = 0.55


def diagnose_cause(retrieval_conf, semantic_conf, graph_conf, conflicts):
    """
    Rule-based, transparent root-cause classifier.
    Priority order: conflict > sparsity > misalignment.
    This ordering is a documented design decision, not an arbitrary default —
    a detected factual conflict is treated as diagnostically stronger evidence
    than a low similarity score, since conflicts are directly observed in the
    graph rather than inferred from a threshold.
    """
    if len(conflicts) > 0:
        return "C1_INFORMATION_CONFLICT", (
            f"Detected {len(conflicts)} conflicting triple(s): the same subject+relation "
            f"has different objects across retrieved paragraphs."
        )

    if retrieval_conf < LOW_RETRIEVAL_THRESHOLD:
        return "C2_KNOWLEDGE_SPARSITY", (
            f"Retrieval confidence ({retrieval_conf}) is below threshold "
            f"({LOW_RETRIEVAL_THRESHOLD}): insufficient relevant evidence was retrieved."
        )

    if semantic_conf < LOW_SEMANTIC_THRESHOLD:
        return "C3_SEMANTIC_MISALIGNMENT", (
            f"Semantic confidence ({semantic_conf}) is below threshold "
            f"({LOW_SEMANTIC_THRESHOLD}) despite adequate retrieval: evidence is topically "
            f"present but not well-aligned to the question."
        )

    return "C3_SEMANTIC_MISALIGNMENT", (
        "Fallback: confidence was low overall but no single dominant cause "
        "(conflict or sparsity) was clearly identified."
    )


def run_rca_for_question(index=0, k=5, dataset_path=None):
    conf_result = compute_confidence(index=index, k=k, dataset_path=dataset_path)
    graph_result, G, top_paragraphs, triples = run_for_question(index=index, k=k, dataset_path=dataset_path)

    cause, reason = diagnose_cause(
        conf_result["retrieval_confidence"],
        conf_result["semantic_confidence"],
        conf_result["graph_confidence"],
        graph_result["conflicts"]
    )

    return {
        "index": index,
        "question": conf_result["question"],
        "retrieval_confidence": conf_result["retrieval_confidence"],
        "semantic_confidence": conf_result["semantic_confidence"],
        "graph_confidence": conf_result["graph_confidence"],
        "final_confidence": conf_result["final_confidence"],
        "num_conflicts": len(graph_result["conflicts"]),
        "diagnosed_cause": cause,
        "reason": reason,
    }


if __name__ == "__main__":
    test_indices = [0, 1, 2, 10, 50]
    theta = 0.60
    all_results = []

    for idx in test_indices:
        result = run_rca_for_question(index=idx, k=5)
        needed_rca = result["final_confidence"] < theta
        result["rca_was_needed"] = needed_rca
        all_results.append(result)

        print(f"\nQuestion: {result['question']}")
        print(f"  Final confidence : {result['final_confidence']} ({'LOW - RCA needed' if needed_rca else 'HIGH - RCA skipped in real pipeline'})")
        print(f"  Diagnosed cause  : {result['diagnosed_cause']}")
        print(f"  Reason           : {result['reason']}")

    output_dir = BASE_DIR / "results" / "root_cause"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rca_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")