import sys
import json
from collections import Counter
from pathlib import Path
import networkx as nx
import spacy

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from retrieval.retrieve import load_question, extract_paragraphs, retrieve_top_k

nlp = spacy.load("en_core_web_sm")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "this", "that",
            "who", "which", "i", "we", "you", "his", "her", "their", "its"}


def is_valid_entity(text):
    text = text.strip()
    if len(text) == 0:
        return False
    if text.lower() in PRONOUNS:
        return False
    if len(text) < 2:
        return False
    return True


def get_span_text(token):
    """Expands a token to its full noun phrase using spaCy's noun_chunks, for cleaner entity names."""
    for chunk in token.doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk.text.strip()
    return token.text.strip()


def make_triple(subj, rel, obj, paragraph_id):
    return {"subject": subj, "relation": rel, "object": obj, "paragraph_id": paragraph_id}


def extract_triples(text, paragraph_id=None):
    """
    Extracts triples using three patterns:
    1. "X is <noun> of Y"        -> e.g. "son_of" relations (copula + attr + prep-of)
    2. "X <verb> Y" (direct obj) -> general subject-verb-object
    3. "X was <verb> by Y"       -> passive voice with agent
    """
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue

            # Pattern 1: copula "is/was" + attribute + "of" -> relation_of
            if token.lemma_ == "be":
                subj, attr = None, None
                for child in token.children:
                    if "subj" in child.dep_:
                        subj = child
                    if child.dep_ == "attr":
                        attr = child
                if subj and attr:
                    subj_text = get_span_text(subj)
                    found_of = False
                    for c in attr.children:
                        if c.dep_ == "prep" and c.text.lower() == "of":
                            for pobj in c.children:
                                if pobj.dep_ == "pobj":
                                    obj_text = get_span_text(pobj)
                                    if is_valid_entity(subj_text) and is_valid_entity(obj_text) and subj_text.lower() != obj_text.lower():
                                        triples.append(make_triple(subj_text, f"{attr.lemma_}_of", obj_text, paragraph_id))
                                    found_of = True
                    if not found_of:
                        obj_text = get_span_text(attr)
                        if is_valid_entity(subj_text) and is_valid_entity(obj_text) and subj_text.lower() != obj_text.lower():
                            triples.append(make_triple(subj_text, "is_a", obj_text, paragraph_id))

            # Pattern 2: general subject-verb-direct object
            subj, obj = None, None
            for child in token.children:
                if "subj" in child.dep_:
                    subj = child
                if child.dep_ in ("dobj", "obj"):
                    obj = child
            if subj and obj:
                subj_text, obj_text = get_span_text(subj), get_span_text(obj)
                if is_valid_entity(subj_text) and is_valid_entity(obj_text) and subj_text.lower() != obj_text.lower():
                    triples.append(make_triple(subj_text, token.lemma_, obj_text, paragraph_id))

            # Pattern 3: passive voice, "X was directed by Y"
            subjpass, agent_obj = None, None
            for child in token.children:
                if child.dep_ == "nsubjpass":
                    subjpass = child
                if child.dep_ == "agent":
                    for c in child.children:
                        if c.dep_ == "pobj":
                            agent_obj = c
            if subjpass and agent_obj:
                subj_text, obj_text = get_span_text(subjpass), get_span_text(agent_obj)
                if is_valid_entity(subj_text) and is_valid_entity(obj_text) and subj_text.lower() != obj_text.lower():
                    triples.append(make_triple(subj_text, f"{token.lemma_}_by", obj_text, paragraph_id))

    # Deduplicate
    seen, deduped = set(), []
    for t in triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


def get_triples_per_paragraph(paragraphs):
    """Returns {paragraph_index: [triples]} so downstream modules know which evidence produced which facts."""
    result = {}
    for idx, para in enumerate(paragraphs):
        result[idx] = extract_triples(para, paragraph_id=idx)
    return result


def build_graph_from_paragraphs(paragraphs):
    G = nx.DiGraph()
    all_triples = []
    for idx, para in enumerate(paragraphs):
        triples = extract_triples(para, paragraph_id=idx)
        all_triples.extend(triples)
        for t in triples:
            G.add_edge(t["subject"], t["object"], relation=t["relation"], paragraph_id=t["paragraph_id"])
    return G, all_triples


def detect_conflicts(triples):
    """Finds cases where the same (subject, relation) has multiple different objects across paragraphs."""
    grouped = {}
    for t in triples:
        key = (t["subject"].lower(), t["relation"])
        grouped.setdefault(key, set()).add(t["object"].lower())
    conflicts = [{"subject": k[0], "relation": k[1], "objects": list(v)}
                 for k, v in grouped.items() if len(v) > 1]
    return conflicts


def run_for_question(index=0, k=5, dataset_path=None):
    sample = load_question(index=index, dataset_path=dataset_path)
    question = sample["question"]
    paragraphs = extract_paragraphs(sample)
    top_results = retrieve_top_k(question, paragraphs, k=k)
    top_paragraphs = [para for para, score in top_results]

    G, triples = build_graph_from_paragraphs(top_paragraphs)
    conflicts = detect_conflicts(triples)

    print(f"Question: {question}")
    print(f"Retrieved paragraphs: {len(top_paragraphs)}\n")
    print("Extracted triples:")
    for t in triples:
        print(f"[para {t['paragraph_id']}] {t['subject']} --[{t['relation']}]--> {t['object']}")
    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Conflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"  CONFLICT: {c['subject']} --[{c['relation']}]--> {c['objects']}")

    result = {
        "index": index,
        "question": question,
        "num_retrieved_paragraphs": len(top_paragraphs),
        "triples": triples,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "conflicts": conflicts,
    }
    return result, G, top_paragraphs, triples


if __name__ == "__main__":
    result, G, top_paragraphs, triples = run_for_question(index=0, k=5)

    output_dir = BASE_DIR / "results" / "knowledge_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "graph_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to:\n{output_path}")