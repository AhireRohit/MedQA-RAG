#!/usr/bin/env python3
# scripts/prepare_pubmedqa.py
import argparse, hashlib, json, re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def hash_text(t: str) -> str:
    import hashlib
    return hashlib.sha1(t.strip().encode("utf-8")).hexdigest()[:16]

_sent_split = re.compile(r"(?<=[.!?])\s+")
_token_re = re.compile(r"\w+")

def best_extractive_sentence(question: str, passages: list[str]) -> str:
    """Pick one sentence from passages that best matches question (BM25 over sentences)."""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return ""  # if rank_bm25 not installed yet, just return empty

    # collect sentences
    sents = []
    for p in passages:
        for s in _sent_split.split(p):
            s = s.strip()
            if 20 <= len(s) <= 350:
                sents.append(s)
    if not sents:
        return ""

    tokenized = [_token_re.findall(s.lower()) for s in sents]
    bm25 = BM25Okapi(tokenized)
    q_toks = _token_re.findall(question.lower())
    scores = bm25.get_scores(q_toks)
    if not len(scores):
        return ""
    i = max(range(len(scores)), key=lambda k: scores[k])
    return sents[i]

def main(max_docs: int = 5000, seed: int = 42):
    out_raw = Path("data/raw/pubmedqa_pqa_labeled.jsonl")
    out_corpus = Path("data/processed/corpus.jsonl")
    out_qa = Path("data/processed/qa.jsonl")

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    # write raw (optional)
    with out_raw.open("w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({k: r[k] for k in ["pubid","question","context","long_answer","final_decision"]}, ensure_ascii=False) + "\n")

    # build corpus (unique paragraphs)
    seen = set()
    corpus = []
    for r in ds:
        ctx = r["context"]
        paras = ctx["contexts"] if isinstance(ctx, dict) and "contexts" in ctx else []
        for p in paras:
            txt = " ".join(str(p).split())
            if not txt:
                continue
            h = hash_text(txt)
            if h in seen:
                continue
            seen.add(h)
            corpus.append({"doc_id": h, "text": txt})
            if len(corpus) >= max_docs:
                break
        if len(corpus) >= max_docs:
            break

    # map QA + synthesize gold_extractive if long_answer is empty
    qa = []
    for r in ds:
        question = r["question"]
        ctx = r["context"]
        paras = ctx["contexts"] if isinstance(ctx, dict) and "contexts" in ctx else []
        doc_ids = [hash_text(" ".join(str(p).split())) for p in paras]
        long_answer = (r.get("long_answer") or "").strip()
        gold_extractive = ""
        if not long_answer:
            gold_extractive = best_extractive_sentence(question, [" ".join(str(p).split()) for p in paras])

        qa.append({
            "id": str(r["pubid"]),
            "question": question,
            "answer_long": long_answer,                  # may be empty
            "gold_extractive": gold_extractive,          # our surrogate, may be empty if BM25 not installed
            "answer_label": r.get("final_decision", None),  # yes/no/maybe
            "doc_ids": doc_ids
        })

    # write outputs
    with out_corpus.open("w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with out_qa.open("w", encoding="utf-8") as f:
        for q in qa:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(corpus)} docs to {out_corpus}")
    print(f"✅ Wrote {len(qa)} QA items to {out_qa}")
    print("Tip: install rank-bm25 before running this to populate gold_extractive.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=5000, help="cap number of unique context paragraphs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(max_docs=args.max_docs, seed=args.seed)
