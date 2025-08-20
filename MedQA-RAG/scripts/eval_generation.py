#!/usr/bin/env python3
# scripts/eval_generation.py
import argparse, os, sys, csv, json
from pathlib import Path
from tqdm import tqdm

# path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils_io import read_jsonl
from src.metrics import exact_match, f1_token, rougeLsum, ROUGE_AVAILABLE
from src.qa_pipeline import QAPipeline

QA_PATH = "data/processed/qa.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out-csv", default="eval/generation_results.csv")
    ap.add_argument("--extractive-first", action="store_true")
    ap.add_argument("--no-snippets", action="store_true")
    ap.add_argument("--skip-empty-gold", action="store_true", help="(kept for compat; now we fallback instead)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    Path("eval").mkdir(parents=True, exist_ok=True)

    pipe = QAPipeline(
        idx_dir="data/indexes",
        reranker_model=None,
        gen_model=args.model,
        max_ctx_passages=4,
        per_passage_chars=600,
        max_src_tokens=480,
        use_snippets=not args.no_snippets,
        snippets_top_n=2,
        extractive_first=args.extractive_first,
        extractive_conf_threshold=0.30,
    )

    rows = list(read_jsonl(QA_PATH))
    if args.limit > 0:
        rows = rows[: args.limit]

    ems, f1s, rouges, grounds = [], [], [], []
    used_extractive_count = 0
    used_fallback_gold = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","question","pred","gold","gold_type","em","f1","rougeL","groundedness","used_extractive"])

        for idx, r in enumerate(tqdm(rows, total=len(rows), desc="gen-eval")):
            q = r["question"]
            gold = (r.get("answer_long") or "").strip()
            gold_type = "long_answer"
            if not gold:
                gold = (r.get("gold_extractive") or "").strip()
                gold_type = "gold_extractive"
                if gold:
                    used_fallback_gold += 1

            out = pipe.ask(
                q, k=args.k, alpha=args.alpha,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
            pred = out["answer"]
            gscore = out["groundedness"]
            used_extractive = getattr(pipe, "stats", {}).get("extractive_used", 0)
            used_extractive_now = 1 if used_extractive > used_extractive_count else 0
            used_extractive_count = used_extractive

            # metrics (handle empty gold gracefully)
            if gold:
                try:
                    rg = rougeLsum(pred, gold) if ROUGE_AVAILABLE else 0.0
                except RuntimeError as e:
                    print(f"[WARN] ROUGE unavailable: {e}")
                    rg = 0.0
                em = exact_match(pred, gold)
                f1 = f1_token(pred, gold)
            else:
                rg = 0.0
                em = 0
                f1 = 0.0

            ems.append(em); f1s.append(f1); rouges.append(rg); grounds.append(gscore)

            w.writerow([r["id"], q, pred, gold, gold_type, em, f1, f"{rg:.4f}", f"{gscore:.4f}", used_extractive_now])

            if args.debug and idx < 3:
                print("\n[DEBUG]")
                print("Question:", q)
                print("Gold type:", gold_type, "| len:", len(gold), "| startswith:", gold[:140].replace("\n"," "))
                print("Pred (len):", len(pred), "| startswith:", pred[:140].replace("\n"," "))
                print("EM:", em, "F1:", f1, "ROUGE-L:", rg, "Grounded:", gscore, "Used-extractive-now:", used_extractive_now)

    n = len(rows) if rows else 1
    print("\n=== Generation Metrics ===")
    print(f"Items:          {n}  (fallback gold used: {used_fallback_gold})")
    print(f"Exact Match:    {sum(ems)/n:.3f}")
    print(f"Token F1:       {sum(f1s)/n:.3f}")
    print(f"ROUGE-Lsum:     {sum(rouges)/n:.3f} {'(ROUGE enabled)' if ROUGE_AVAILABLE else '(ROUGE missing)'}")
    print(f"Groundedness:   {sum(grounds)/n:.3f}")
    print(f"Extractive used:{used_extractive_count} times total")
    print(f"Saved → {args.out_csv}")
