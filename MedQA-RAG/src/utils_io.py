from pathlib import Path
import json

def read_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_corpus(path="data/processed/corpus.jsonl"):
    docs = []
    for r in read_jsonl(path):
        # each: {"doc_id": "...", "text": "..."}
        docs.append(r)
    return docs
