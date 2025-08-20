#!/usr/bin/env python3
import argparse, csv, itertools, os, sys, json
from pathlib import Path
from statistics import mean

# path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval_retrieval import main as eval_retrieval_main  # reuse code? simpler: shell out
# To avoid duplication, we'll call eval_retrieval as a subprocess for each config.

import subprocess

def run_eval(k, alpha, limit, out_file):
    cmd = [
        sys.executable, "scripts/eval_retrieval.py",
        "--k", str(k),
        "--alpha", str(alpha),
        "--limit", str(limit),
        "--out", out_file
    ]
    subprocess.run(cmd, check=True)

def parse_results(jsonl_path):
    vals = {"recall": [], "mrr": [], "prec": [], "rec": [], "f1": []}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            vals["recall"].append(r["recall_at_k"])
            vals["mrr"].append(r["mrr"])
            vals["prec"].append(r["precision"])
            vals["rec"].append(r["recall"])
            vals["f1"].append(r["f1"])
    return {k: mean(v) if v else 0.0 for k, v in vals.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--alphas", default="0.3,0.5,0.7")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out-csv", default="eval/ablation_retrieval.csv")
    args = ap.parse_args()

    Path("eval").mkdir(parents=True, exist_ok=True)

    ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    rows = []
    for k in ks:
        for a in alphas:
            tmp = f"eval/tmp_retrieval_k{k}_a{a}.jsonl"
            run_eval(k, a, args.limit, tmp)
            agg = parse_results(tmp)
            rows.append([k, a, agg["recall"], agg["mrr"], agg["prec"], agg["rec"], agg["f1"]])

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k","alpha","Recall@k","MRR","Precision","Recall","F1"])
        w.writerows(rows)

    print(f"Saved ablation table → {args.out_csv}")

if __name__ == "__main__":
    main()
