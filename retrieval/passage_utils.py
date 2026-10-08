import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HOTPOT_100 = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_100.json"
HOTPOT_VAL_30 = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_val_30.json"


def load_question(index=0, dataset_path=None):
    """Loads a sample from either the default 100-question set or a custom path."""
    path = dataset_path if dataset_path is not None else HOTPOT_100
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[index]


def extract_structured_paragraphs(sample):
    """
    Extracts structured paragraphs with stable metadata for retrieval and evaluation.
    Each item contains:
      - doc_id: integer index within sample['context'] (0..9)
      - title: title of the Wikipedia article
      - sentences: list of sentence strings
      - text: full paragraph text
    """
    paragraphs = []
    for doc_id, (title, sentences) in enumerate(sample.get("context", [])):
        paragraph_text = " ".join(sentences).strip()
        paragraphs.append({
            "doc_id": doc_id,
            "title": title,
            "sentences": sentences,
            "text": paragraph_text,
        })
    return paragraphs


def extract_paragraphs(sample):
    """Flat list of paragraph strings for backward compatibility."""
    return [" ".join(sentences) for _, sentences in sample.get("context", [])]


def get_gold_supporting_titles(sample):
    """
    EVALUATION ONLY. Must never be called during inference.
    Extracts the unique set of article titles appearing in gold supporting_facts.
    """
    supporting_facts = sample.get("supporting_facts", [])
    return list(dict.fromkeys(title for title, _ in supporting_facts))


def get_gold_supporting_facts(sample):
    """
    EVALUATION ONLY. Must never be called during inference.
    Returns the list of (title, sentence_idx) gold facts.
    """
    return sample.get("supporting_facts", [])
