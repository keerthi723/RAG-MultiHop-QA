import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
HOTPOT_100 = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_100.json"

# Load the embedding model once, when the script starts
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_question(index=0, dataset_path=None):
    path = dataset_path if dataset_path is not None else HOTPOT_100
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[index]


def extract_paragraphs(sample):
    """
    HotpotQA's 'context' field looks like:
    [ ["Title 1", ["sentence 1", "sentence 2", ...]], ["Title 2", [...]], ... ]

    Turns this into a flat list of paragraph strings, one per title.
    """
    paragraphs = []
    for title, sentences in sample["context"]:
        paragraph_text = " ".join(sentences)
        paragraphs.append(paragraph_text)
    return paragraphs


def retrieve_top_k(question, paragraphs, k=5):
    """
    Embeds the question and all paragraphs, computes cosine similarity,
    and returns the top-k paragraphs sorted by similarity (highest first).
    """
    question_embedding = model.encode([question])
    paragraph_embeddings = model.encode(paragraphs)

    scores = cosine_similarity(question_embedding, paragraph_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append((paragraphs[idx], scores[idx]))

    return results


if __name__ == "__main__":
    test_indices = [0, 1, 2, 10, 50]
    all_results = []

    with open(HOTPOT_100, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    for idx in test_indices:
        sample = full_data[idx]
        question = sample["question"]
        paragraphs = extract_paragraphs(sample)
        top_results = retrieve_top_k(question, paragraphs, k=5)

        print(f"\n{'='*70}")
        print(f"Index {idx} — Question:")
        print(question)
        print(f"\nTop-5 retrieved paragraphs:")

        result_entry = {
            "index": idx,
            "question": question,
            "top_5": []
        }

        for i, (para, score) in enumerate(top_results, 1):
            print(f"[{i}] Score: {score:.4f} — {para[:100]}...")
            result_entry["top_5"].append({
                "rank": i,
                "score": float(score),
                "paragraph": para
            })

        all_results.append(result_entry)

    output_path = BASE_DIR / "results" / "retrieval" / "retrieval_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nSaved results for {len(test_indices)} questions to:")
    print(output_path)