import re, numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

_tok = re.compile(r"\w+").findall

def _bigrams(tokens: List[str]) -> set[tuple]:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()

def _sent_split(text: str) -> List[str]:
    # simple splitter to avoid heavy deps
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]

class GroundednessScorer:
    def __init__(self, emb_model: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(emb_model)

    def _overlap_score(self, hyp: str, passages: List[str]) -> float:
        hyp_bi = _bigrams([t.lower() for t in _tok(hyp)])
        if not hyp_bi:
            return 0.0
        best = 0.0
        for p in passages:
            p_bi = _bigrams([t.lower() for t in _tok(p)])
            if not p_bi:
                continue
            inter = len(hyp_bi & p_bi)
            union = len(hyp_bi | p_bi)
            s = 0.0 if union == 0 else inter / union
            if s > best:
                best = s
        return float(best)

    def _semantic_score(self, hyp: str, passages: List[str]) -> float:
        reps = self.model.encode([hyp] + passages, convert_to_numpy=True, normalize_embeddings=True)
        hyp_vec = reps[0:1]
        ctx_vecs = reps[1:]
        sims = np.dot(ctx_vecs, hyp_vec.T).squeeze(axis=1)
        return float(max(sims)) if len(sims) else 0.0

    def score(self, answer: str, passages: List[str]) -> float:
        # per-sentence score, then average
        sents = _sent_split(answer)
        if not sents:
            return 0.0
        scores = []
        for s in sents:
            ov = self._overlap_score(s, passages)
            se = self._semantic_score(s, passages)
            scores.append(0.5 * ov + 0.5 * se)
        return float(np.mean(scores))
