# src/metrics.py
import re
from typing import Tuple
from rapidfuzz import fuzz

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def f1_token(pred: str, gold: str) -> float:
    """Approximate token-F1 using fuzzy token set ratio scaled to [0,1]."""
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return fuzz.token_set_ratio(p, g) / 100.0

# ---- ROUGE-L (fail loud if missing) ----
try:
    from rouge_score import rouge_scorer
    _rouge = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    ROUGE_AVAILABLE = True
except Exception as e:
    _rouge = None
    ROUGE_AVAILABLE = False
    _rouge_import_error = e  # for debugging

def rougeLsum(pred: str, gold: str) -> float:
    """
    ROUGE-Lsum F-measure in [0,1].
    Requires 'rouge-score' package; if missing, raise a clear error.
    """
    if not ROUGE_AVAILABLE:
        raise RuntimeError(
            f"rouge-score is not installed or failed to import: {_rouge_import_error}. "
            "Install with: pip install rouge-score"
        )
    return _rouge.score(gold, pred)["rougeLsum"].fmeasure

# ---- Retrieval metrics ----
def precision_recall_f1_hits(hits, gold_ids) -> Tuple[float, float, float]:
    if not hits:
        return 0.0, 0.0, 0.0
    hitset = set(hits)
    tp = len(hitset & gold_ids)
    fp = len(hitset - gold_ids)
    fn = len(gold_ids - hitset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def mrr_at_k(ranked_ids, gold_ids) -> float:
    for i, did in enumerate(ranked_ids, 1):
        if did in gold_ids:
            return 1.0 / i
    return 0.0

def recall_at_k(ranked_ids, gold_ids) -> float:
    return 1.0 if any(d in gold_ids for d in ranked_ids) else 0.0
