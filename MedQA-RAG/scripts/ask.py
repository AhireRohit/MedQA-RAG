#!/usr/bin/env python3
import argparse
from pprint import pprint
import sys, os
# path hack so you can run `python scripts/ask.py` directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.qa_pipeline import QAPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="question")
    ap.add_argument("--k", type=int, default=12, help="retrieval depth")
    ap.add_argument("--alpha", type=float, default=0.6, help="dense weight [0,1]")
    ap.add_argument("--model", default="google/flan-t5-base", help="HF seq2seq model")
    ap.add_argument("--reranker", default="", help="cross-encoder model (optional)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    pipe = QAPipeline(
        idx_dir="data/indexes",
        reranker_model=args.reranker if args.reranker else None,
        gen_model=args.model,
        max_ctx_passages=6
    )
    out = pipe.ask(
        args.q, k=args.k, alpha=args.alpha,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )

    print("\n=== Answer ===")
    print(out["answer"])
    print(f"\nGroundedness score: {out['groundedness']:.3f}  (0=ungrounded, 1=highly grounded)")
    print("\n=== Citations ===")
    for i, p in enumerate(out["passages"], 1):
        preview = (p["text"][:220] + "…") if len(p["text"]) > 220 else p["text"]
        print(f"[CITATION {i}] doc_id={p['doc_id']}  (retrieval_score={p['score']:.3f})\n{preview}\n")

if __name__ == "__main__":
    main()
