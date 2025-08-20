from typing import List, Dict, Tuple, Optional
import numpy as np
try:
    from sentence_transformers import CrossEncoder
    CROSS_OK = True
except Exception:
    CROSS_OK = False

class HybridRetriever:
    def __init__(self, corpus, bm25, dense, reranker_model: Optional[str] = None):
        """
        corpus: list[dict] with fields: doc_id, text
        bm25: BM25Index instance
        dense: DenseIndex instance
        reranker_model: cross-encoder model name (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
        """
        self.corpus = corpus
        self.doc_id_to_pos = {d["doc_id"]: i for i, d in enumerate(corpus)}
        self.bm25 = bm25
        self.dense = dense
        self.reranker = None
        if reranker_model and CROSS_OK:
            self.reranker = CrossEncoder(reranker_model)

    def _merge(self, bm25_hits, dense_hits, alpha=0.5, k=10):
        """
        alpha ∈ [0,1]: weight for dense; (1-alpha) for bm25.
        bm25_hits: list[(idx, score)]
        dense_hits: list[(idx, score)]
        """
        scores = {}
        # min-max normalize per list to [0,1] before blending
        def normalize(hits):
            if not hits:
                return {}
            vals = np.array([s for _, s in hits], dtype="float32")
            mn, mx = float(vals.min()), float(vals.max())
            norm = {}
            for (i, s) in hits:
                ns = 0.0 if mx == mn else (s - mn) / (mx - mn)
                norm[i] = ns
            return norm

        bm = normalize(bm25_hits)
        de = normalize(dense_hits)

        for i, s in bm.items():
            scores[i] = scores.get(i, 0.0) + (1 - alpha) * s
        for i, s in de.items():
            scores[i] = scores.get(i, 0.0) + alpha * s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked  # list[(idx, blended_score)]

    def search(self, query: str, k: int = 10, alpha: float = 0.6, rerank_top: int = 20):
        bm_hits = self.bm25.search(query, k=rerank_top)
        de_hits = self.dense.search(query, k=rerank_top)
        merged = self._merge(bm_hits, de_hits, alpha=alpha, k=rerank_top)

        # Optional reranking with cross-encoder
        if self.reranker is not None:
            pairs = [(query, self.corpus[i]["text"]) for i, _ in merged]
            scores = self.reranker.predict(pairs)  # higher is better
            order = np.argsort(scores)[::-1][:k]
            final = [(merged[i][0], float(scores[i])) for i in order]
            return final

        return merged[:k]
