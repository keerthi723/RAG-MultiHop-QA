import json
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent

HOTPOT_FILE = BASE_DIR / "datasets" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
WIKI_FILE = BASE_DIR / "datasets" / "2wikimultihop" / "dev.json"

HOTPOT_OUTPUT = BASE_DIR / "datasets" / "hotpotqa" / "hotpotqa_100.json"
WIKI_OUTPUT = BASE_DIR / "datasets" / "2wikimultihop" / "2wikimultihop_100.json"


def create_subset(input_file, output_file, number=100):
    print(f"\nReading: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total questions available: {len(data)}")

    subset = data[:number]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(subset)} questions to:")
    print(output_file)


def show_sample(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[0]

    print("\n" + "=" * 70)
    print("SAMPLE QUESTION")
    print("=" * 70)

    print("\nQuestion:")
    print(sample.get("question"))

    print("\nAnswer:")
    print(sample.get("answer"))

    print("\nContext:")

    context = sample.get("context", [])

    for item in context:
        if isinstance(item, list) and len(item) == 2:
            title, paragraphs = item
            print(f"\n[{title}]")
            for paragraph in paragraphs:
                print(paragraph)
        else:
            print(item)

    print("\n" + "=" * 70)


# Create 100-question subsets
create_subset(HOTPOT_FILE, HOTPOT_OUTPUT, 100)
create_subset(WIKI_FILE, WIKI_OUTPUT, 100)

# Display one example from each dataset
print("\n\nHOTPOTQA SAMPLE")
show_sample(HOTPOT_OUTPUT)

print("\n\n2WikiMultiHopQA SAMPLE")
show_sample(WIKI_OUTPUT)

print("\nDataset preparation completed successfully.")