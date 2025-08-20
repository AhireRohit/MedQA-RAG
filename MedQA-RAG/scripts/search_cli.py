#!/usr/bin/env python3
import argparse, pickle
from src.utils_io import load_corpus
from src.bm25_index import BM25Index
from src.dense_index import DenseIndex
from src.hybrid_retriever import HybridRetriever

IDX_DIR = "data/indexes"
CORPUS_PKL = "data/indexes/corpus.pkl"

def load_indexes():
    with open(CORPUS_PKL, "rb") as f:
        corpus = pickle.load(f)
    texts = [d["text"] for d in corpus]

    bm25 = BM25Index.load(f"{IDX_DIR}/bm25.pkl", texts)
    dense = DenseIndex(model_name="BAAI/bge-small-en-v1.5")
    dense.load(f"{IDX_DIR}/dense")
    return corpus, bm25, dense

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6, help="weight for dense in [0,1]")
    ap.add_argument("--reranker", default="", help='e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"')
    args = ap.parse_args()

    corpus, bm25, dense = load_indexes()
    retr = HybridRetriever(corpus, bm25, dense, reranker_model=args.reranker if args.reranker else None)
    hits = retr.search(args.q, k=args.k, alpha=args.alpha)

    print(f"\nQuery: {args.q}\n")
    for rank, (idx, score) in enumerate(hits, 1):
        doc = corpus[idx]
        text = doc['text']
        preview = (text[:300] + "…") if len(text) > 300 else text
        print(f"[{rank}] doc_id={doc['doc_id']}  score={score:.4f}\n{preview}\n")

if __name__ == "__main__":
    main()
