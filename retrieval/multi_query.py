import json
import re
import requests
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from retrieval.retrieve import model

BASE_DIR = Path(__file__).resolve().parent.parent
DECOMP_PROMPT_PATH = BASE_DIR / "prompts" / "decomposition_prompt.txt"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def call_ollama(prompt, timeout=60):
    """Calls local Ollama instance deterministically with JSON format enforcement."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "context": [],
            "options": {"temperature": 0, "seed": 42},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_subquestions_json(raw_text):
    """Defensively extracts subquestions list from LLM response across multiple parsing stages."""
    if not raw_text:
        return []

    # 1. Standard JSON parse of bracketed block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            subs = data.get("subquestions", [])
            if isinstance(subs, list) and len(subs) >= 2:
                return [str(s).strip() for s in subs if str(s).strip()]
        except Exception:
            pass

    # 2. Defensive closure for truncated JSON (e.g. missing closing brace)
    clean_text = raw_text.strip()
    for suffix in ["}", "]}", "\"]}", "\"\n]}"]:
        try:
            data = json.loads(clean_text + suffix)
            subs = data.get("subquestions", [])
            if isinstance(subs, list) and len(subs) >= 2:
                return [str(s).strip() for s in subs if str(s).strip()]
        except Exception:
            pass

    # 3. Regex extraction of quoted strings within subquestions list
    sub_block = re.search(r'"subquestions"\s*:\s*\[(.*?)(\]|$)', raw_text, re.DOTALL)
    if sub_block:
        items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', sub_block.group(1))
        clean = [item.strip() for item in items if len(item.strip()) > 5]
        if len(clean) >= 2:
            return clean

    return []


def decompose_question(question):
    """
    Decomposes the original question Q into 2-3 retrieval-oriented subquestions.
    NO GOLD INFORMATION is ever supplied.
    Returns list of 2-3 strings.
    """
    with open(DECOMP_PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.replace("{question}", question)
    try:
        raw = call_ollama(prompt)
        subquestions = parse_subquestions_json(raw)
        if len(subquestions) >= 2:
            return subquestions[:3]
    except Exception as e:
        print(f"Warning: question decomposition call failed ({e}). Falling back.")

    # Fallback if decomposition fails: return original question as single subquestion
    return [question]


def retrieve_route(query, structured_paragraphs, k=5, route_name="R0"):
    """
    Dense retrieval for a single query route against the structured candidate paragraphs.
    Returns list of dicts with rank (1-indexed), similarity score, doc_id, title, text.
    """
    if not structured_paragraphs:
        return []

    texts = [p["text"] for p in structured_paragraphs]
    q_emb = model.encode([query])
    p_embs = model.encode(texts)

    scores = cosine_similarity(q_emb, p_embs)[0]
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank_idx, p_idx in enumerate(top_indices, start=1):
        p = structured_paragraphs[p_idx]
        results.append({
            "route": route_name,
            "query": query,
            "rank": rank_idx,
            "similarity_score": float(scores[p_idx]),
            "doc_id": p["doc_id"],
            "title": p["title"],
            "text": p["text"],
            "sentences": p.get("sentences", []),
        })

    return results


def run_multi_query_retrieval(question, subquestions, structured_paragraphs, k=5):
    """
    Runs multi-query retrieval across R0 (original query) and R1..R3 (subquestions).
    Returns dict mapping route_name to ranked results list.
    """
    routes = {}
    # Route R0: Original question
    routes["R0"] = retrieve_route(question, structured_paragraphs, k=k, route_name="R0")

    # Routes R1..Rm: Subquestions
    for idx, subq in enumerate(subquestions, start=1):
        route_name = f"R{idx}"
        routes[route_name] = retrieve_route(subq, structured_paragraphs, k=k, route_name=route_name)

    return routes
