#!/usr/bin/env python3
from pathlib import Path
from src.utils_io import load_corpus
from src.bm25_index import BM25Index
from src.dense_index import DenseIndex
import pickle

CORPUS_PATH = "data/processed/corpus.jsonl"
OUT_DIR = Path("data/indexes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    corpus = load_corpus(CORPUS_PATH)
    texts = [d["text"] for d in corpus]
    ids   = [d["doc_id"] for d in corpus]

    # BM25
    print("Building BM25...")
    bm25 = BM25Index(texts, k1=1.5, b=0.75)
    bm25.save(str(OUT_DIR / "bm25.pkl"))
    print("✅ Saved BM25 → data/indexes/bm25.pkl")

    # Dense + FAISS
    print("Building Dense (bge-small) + FAISS...")
    dense = DenseIndex(model_name="BAAI/bge-small-en-v1.5")
    dense.build(texts, ids, batch_size=128)
    dense.save(str(OUT_DIR / "dense"))
    print("✅ Saved Dense → data/indexes/dense/")

    # Save corpus metadata for retrieval
    with open(OUT_DIR / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    print("✅ Saved corpus metadata")

if __name__ == "__main__":
    main()
