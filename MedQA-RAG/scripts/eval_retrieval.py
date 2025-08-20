#!/usr/bin/env python3
import argparse, json, os, sys, pickle
from pathlib import Path
from tqdm import tqdm

# path hack so you can run `python scripts/eval_retrieval.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils_io import read_jsonl
from src.bm25_index import BM25Index
from src.dense_index import DenseIndex
from src.hybrid_retriever import HybridRetriever
from src.metrics import recall_at_k, mrr_at_k, precision_recall_f1_hits

QA_PATH = "data/processed/qa.jsonl"
CORPUS_PKL = "data/indexes/corpus.pkl"
BM25_PKL = "data/indexes/bm25.pkl"
DENSE_DIR = "data/indexes/dense"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=300, help="evaluate first N items (0=all)")
    ap.add_argument("--reranker", default="", help="optional cross-encoder name")
    ap.add_argument("--out", default="eval/retrieval_results.jsonl")
    args = ap.parse_args()

    Path("eval").mkdir(parents=True, exist_ok=True)

    with open(CORPUS_PKL, "rb") as f:
        corpus = pickle.load(f)
    texts = [d["text"] for d in corpus]

    bm25 = BM25Index.load(BM25_PKL, texts)
    dense = DenseIndex(model_name="BAAI/bge-small-en-v1.5")
    dense.load(DENSE_DIR)

    retr = HybridRetriever(corpus, bm25, dense, reranker_model=args.reranker if args.reranker else None)

    qa_iter = list(read_jsonl(QA_PATH))
    if args.limit > 0:
        qa_iter = qa_iter[: args.limit]

    out_f = open(args.out, "w", encoding="utf-8")

    recalls, mrrs, prfs = [], [], []
    for row in tqdm(qa_iter, total=len(qa_iter), desc="retrieval-eval"):
        q = row["question"]
        gold_ids = set(row.get("doc_ids", []))

        hits = retr.search(q, k=args.k, alpha=args.alpha, rerank_top=max(args.k, 20))
        ranked = []
        for idx, _score in hits:
            ranked.append(corpus[idx]["doc_id"])

        r_at_k = recall_at_k(ranked, gold_ids)
        mrr = mrr_at_k(ranked, gold_ids)
        prec, rec, f1 = precision_recall_f1_hits(ranked, gold_ids)

        recalls.append(r_at_k)
        mrrs.append(mrr)
        prfs.append((prec, rec, f1))

        out_f.write(json.dumps({
            "id": row["id"],
            "k": args.k,
            "alpha": args.alpha,
            "recall_at_k": r_at_k,
            "mrr": mrr,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }) + "\n")

    out_f.close()

    avg_recall = sum(recalls)/len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs)/len(mrrs) if mrrs else 0.0
    avg_prec = sum(x[0] for x in prfs)/len(prfs) if prfs else 0.0
    avg_rec  = sum(x[1] for x in prfs)/len(prfs) if prfs else 0.0
    avg_f1   = sum(x[2] for x in prfs)/len(prfs) if prfs else 0.0

    print("\n=== Retrieval Metrics ===")
    print(f"Items: {len(qa_iter)}")
    print(f"Recall@{args.k}: {avg_recall:.3f}")
    print(f"MRR:            {avg_mrr:.3f}")
    print(f"Prec/Rec/F1:    {avg_prec:.3f} / {avg_rec:.3f} / {avg_f1:.3f}")
    print(f"Saved per-item results → {args.out}")

if __name__ == "__main__":
    main()
