import sys
import json
import re
from pathlib import Path
import requests
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import model
from knowledge_graph.graph_builder import get_triples_per_paragraph
from repair.repair_pipeline import get_final_evidence

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"  # configurable — change here if using a different pulled model


def build_prompt(question, evidence_paragraphs, triples_per_paragraph):
    evidence_block = ""
    for i, para in enumerate(evidence_paragraphs):
        evidence_block += f"[{i}] {para}\n"
        triples = triples_per_paragraph.get(i, [])
        if triples:
            triple_strs = [f"{t['subject']} -> {t['relation']} -> {t['object']}" for t in triples]
            evidence_block += f"    (extracted facts: {'; '.join(triple_strs)})\n"

    prompt = f"""You are a strict evidence-grounded question answering system.

RULES:
- Answer ONLY using the evidence paragraphs given below.
- Do NOT use any outside knowledge.
- Do NOT invent or assume facts not present in the evidence.
- If the evidence does not contain enough information to answer, say so explicitly.
- Be concise.

EVIDENCE:
{evidence_block}

QUESTION: {question}

Respond with ONLY valid JSON in exactly this format, nothing else before or after:
{{
  "answer": "your answer here, or 'insufficient evidence' if the evidence does not support an answer",
  "evidence_used_ids": [list of evidence numbers you actually used, e.g. [0, 2]],
  "insufficient_evidence": true or false
}}
"""
    return prompt


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "context": [],
            "options": {"temperature": 0, "seed": 42}  # determinism for reproducible evaluation
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_llm_json(raw_text):
    """LLMs sometimes wrap JSON in extra text despite instructions — extract the JSON block defensively."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def compute_generation_confidence(answer, evidence_paragraphs, evidence_used_ids, insufficient):
    """
    Transparent heuristic, NOT invented randomly: if the model declared
    insufficient evidence, confidence is 0 (nothing to be confident about).
    Otherwise, confidence = average cosine similarity between the generated
    answer and the specific evidence paragraphs the model claims it used.
    High similarity means the answer's content is well-grounded in the cited
    text; low similarity is a warning sign the answer may not truly be
    supported by what was cited.
    """
    if insufficient or not evidence_used_ids:
        return 0.0

    used_paragraphs = [evidence_paragraphs[i] for i in evidence_used_ids if i < len(evidence_paragraphs)]
    if not used_paragraphs:
        return 0.0

    ans_emb = model.encode([answer])
    ev_embs = model.encode(used_paragraphs)
    sims = cosine_similarity(ans_emb, ev_embs)[0]
    return round(float(max(0.0, sims.mean())), 4)


def generate_answer(index=0, k=5):
    pipeline_result = get_final_evidence(index=index, k=k)
    question = pipeline_result["question"]
    evidence_paragraphs = [p for p, s in pipeline_result["final_evidence"]]

    triples_per_paragraph = get_triples_per_paragraph(evidence_paragraphs)

    prompt = build_prompt(question, evidence_paragraphs, triples_per_paragraph)
    raw_response = call_ollama(prompt)
    parsed = parse_llm_json(raw_response)

    if parsed is None:
        # Model didn't return valid JSON — fail safely rather than guessing
        return {
            "question": question,
            "rca_decision": pipeline_result["decision"],
            "diagnosed_cause": pipeline_result["diagnosed_cause"],
            "final_evidence_count": len(evidence_paragraphs),
            "answer": "GENERATION ERROR: model did not return valid JSON",
            "raw_response": raw_response,
            "evidence_used": [],
            "grounded": False,
            "generation_confidence": 0.0,
        }

    answer = parsed.get("answer", "")
    evidence_used_ids = parsed.get("evidence_used_ids", [])
    insufficient = parsed.get("insufficient_evidence", False)

    gen_conf = compute_generation_confidence(answer, evidence_paragraphs, evidence_used_ids, insufficient)

    return {
        "question": question,
        "rca_decision": pipeline_result["decision"],
        "diagnosed_cause": pipeline_result["diagnosed_cause"],
        "final_evidence_count": len(evidence_paragraphs),
        "answer": answer,
        "evidence_used": evidence_used_ids,
        "grounded": not insufficient,
        "generation_confidence": gen_conf,
    }


if __name__ == "__main__":
    test_indices = [0, 1, 2, 10, 50]
    all_results = []

    for idx in test_indices:
        result = generate_answer(index=idx, k=5)
        all_results.append(result)

        print(f"\nQuestion: {result['question']}")
        print(f"  RCA decision       : {result['rca_decision']}")
        print(f"  Diagnosed cause    : {result['diagnosed_cause']}")
        print(f"  Final evidence #   : {result['final_evidence_count']}")
        print(f"  Answer             : {result['answer']}")
        print(f"  Evidence used IDs  : {result['evidence_used']}")
        print(f"  Grounded           : {result['grounded']}")
        print(f"  Generation conf.   : {result['generation_confidence']}")

    output_dir = BASE_DIR / "results" / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generation_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")