import re
import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def to_answer_string(value):
    """Converts LLM output value into a clean answer string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip()


def parse_llm_json(raw_text):
    """Defensively extracts JSON response from LLM output."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_evidence_prompt(question, evidence_items):
    """
    Builds the standard grounded QA prompt.
    Accepts evidence_items as either a list of strings or a list of dicts with 'text' and optional 'title'.
    Identical prompt template is used across all ablation configs to strictly isolate retrieval.
    """
    evidence_block = ""
    for i, item in enumerate(evidence_items):
        if isinstance(item, dict):
            title = item.get("title", "")
            text = item.get("text", "")
            if title:
                evidence_block += f"[{i}] {title}: {text}\n"
            else:
                evidence_block += f"[{i}] {text}\n"
        else:
            evidence_block += f"[{i}] {str(item)}\n"

    prompt = f"""You are a strict evidence-grounded question answering system.

RULES:
- Answer ONLY using the evidence paragraphs given below.
- Do NOT use any outside knowledge.
- Do NOT invent or assume facts not present in the evidence.
- If the evidence does not contain enough information to answer, say 'insufficient evidence'.
- Provide the shortest possible direct answer (e.g. an entity name, date, nationality, 'yes', or 'no').
- Be concise.

EVIDENCE:
{evidence_block}

QUESTION: {question}

Respond with ONLY valid JSON in exactly this format, nothing else before or after:
{{
  "answer": "your short answer here, or 'insufficient evidence' if the evidence does not support an answer",
  "evidence_used_ids": [list of evidence numbers you actually used, e.g. [0, 2]],
  "insufficient_evidence": true or false
}}
"""
    return prompt


def call_ollama(prompt, timeout=120):
    """Deterministic LLM call via Ollama API."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "context": [],
            "options": {"temperature": 0, "seed": 42},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_answer_from_evidence(question, evidence_items):
    """
    Given a question and evidence items (strings or dicts),
    invokes LLM and returns answer string and metadata.
    """
    prompt = build_evidence_prompt(question, evidence_items)
    try:
        raw = call_ollama(prompt)
        parsed = parse_llm_json(raw)
        if parsed is None:
            return {
                "answer": "GENERATION ERROR",
                "raw_response": raw,
                "evidence_used_ids": [],
                "insufficient_evidence": True,
                "failed": True,
            }
        ans = to_answer_string(parsed.get("answer", ""))
        return {
            "answer": ans,
            "raw_response": raw,
            "evidence_used_ids": parsed.get("evidence_used_ids", []),
            "insufficient_evidence": bool(parsed.get("insufficient_evidence", False)),
            "failed": False,
        }
    except Exception as e:
        return {
            "answer": f"GENERATION EXCEPTION: {e}",
            "raw_response": "",
            "evidence_used_ids": [],
            "insufficient_evidence": True,
            "failed": True,
        }
