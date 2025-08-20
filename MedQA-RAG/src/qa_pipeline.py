# src/qa_pipeline.py
import pickle
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .prompting import build_prompt
from .groundedness import GroundednessScorer
from .bm25_index import BM25Index
from .dense_index import DenseIndex
from .hybrid_retriever import HybridRetriever

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QAPipeline:
    def __init__(
        self,
        idx_dir: str = "data/indexes",
        reranker_model: Optional[str] = None,
        gen_model: str = "google/flan-t5-base",
        max_ctx_passages: int = 4,
        per_passage_chars: int = 600,
        max_src_tokens: int = 480,
        use_snippets: bool = True,
        snippets_top_n: int = 2,
        extractive_first: bool = False,
        extractive_conf_threshold: float = 0.55,
    ):
        # ----- Load retrieval artifacts -----
        with open(f"{idx_dir}/corpus.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        texts = [d["text"] for d in self.corpus]

        self.bm25 = BM25Index.load(f"{idx_dir}/bm25.pkl", texts)
        self.dense = DenseIndex(model_name="BAAI/bge-small-en-v1.5")
        self.dense.load(f"{idx_dir}/dense")

        self.retriever = HybridRetriever(
            self.corpus, self.bm25, self.dense, reranker_model=reranker_model
        )

        # ----- Generation model -----
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(DEVICE)

        # ----- Groundedness scorer -----
        self.scorer = GroundednessScorer(emb_model="BAAI/bge-small-en-v1.5")

        # ----- Controls -----
        self.max_ctx_passages = int(max_ctx_passages)
        self.per_passage_chars = int(per_passage_chars)
        self.max_src_tokens = int(max_src_tokens)
        self.use_snippets = bool(use_snippets)
        self.snippets_top_n = int(snippets_top_n)
        self.extractive_first = bool(extractive_first)
        self.extractive_conf_threshold = float(extractive_conf_threshold)

        # Stats for debugging
        self.stats = {"extractive_used": 0}

    def retrieve(self, query: str, k: int = 12, alpha: float = 0.6) -> List[Dict[str, Any]]:
        hits = self.retriever.search(query, k=k, alpha=alpha)
        passages = []
        for i, score in hits:
            d = self.corpus[i]
            passages.append({"doc_id": d["doc_id"], "text": d["text"], "score": float(score)})
        return passages

    @staticmethod
    def _truncate_chars(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars].rstrip() + "…"

    def _extractive_snippet(self, question: str, passages: List[Dict[str, Any]], top_n: int = 2) -> List[str]:
        if top_n <= 0 or not passages:
            return []
        try:
            from rank_bm25 import BM25Okapi
            import re
            sent_split = re.compile(r"(?<=[.!?])\s+")
            token_re = re.compile(r"\w+")
            sents: List[str] = []
            for p in passages[: self.max_ctx_passages]:
                for s in sent_split.split(p["text"]):
                    s = s.strip()
                    if 20 <= len(s) <= 300:
                        sents.append(s)
            if not sents:
                return []
            tokenized = [token_re.findall(s.lower()) for s in sents]
            bm25 = BM25Okapi(tokenized)
            q_toks = token_re.findall(question.lower())
            scores = bm25.get_scores(q_toks)
            ranked = sorted(zip(sents, scores), key=lambda x: x[1], reverse=True)[:top_n]
            return [s for s, _ in ranked]
        except Exception:
            return []

    def _best_extractive_answer(self, question: str, passages: List[Dict[str, Any]]) -> tuple[str, float]:
        try:
            from rank_bm25 import BM25Okapi
            import re, numpy as np
            sent_split = re.compile(r"(?<=[.!?])\s+")
            token_re = re.compile(r"\w+")
            sents: List[str] = []
            for p in passages[: self.max_ctx_passages]:
                for s in sent_split.split(p["text"]):
                    s = s.strip()
                    if 20 <= len(s) <= 350:
                        sents.append(s)
            if not sents:
                return "", 0.0
            tokenized = [token_re.findall(s.lower()) for s in sents]
            bm25 = BM25Okapi(tokenized)
            q_toks = token_re.findall(question.lower())
            bm_scores = bm25.get_scores(q_toks)
            emb = self.scorer.model
            vecs = emb.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
            qv = emb.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
            sem_scores = (vecs @ qv)
            bm_min, bm_ptp = float(bm_scores.min()), float(bm_scores.ptp() + 1e-9)
            bm_norm = (bm_scores - bm_min) / bm_ptp
            se_min, se_ptp = float(sem_scores.min()), float(sem_scores.ptp() + 1e-9)
            se_norm = (sem_scores - se_min) / se_ptp
            scores = 0.4 * bm_norm + 0.6 * se_norm
            i = int(np.argmax(scores))
            return sents[i], float(scores[i])
        except Exception:
            return "", 0.0

    def _prepare_context_texts(self, question: str, passages: List[Dict[str, Any]]) -> List[str]:
        snippets: List[str] = []
        if self.use_snippets:
            snippets = self._extractive_snippet(question, passages, top_n=self.snippets_top_n)
        top_passages = passages[: self.max_ctx_passages]
        truncated = [self._truncate_chars(p["text"], self.per_passage_chars) for p in top_passages]
        return snippets + truncated

    def generate(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_beams: int = 1,
    ) -> Dict[str, Any]:
        if self.extractive_first:
            best_sent, conf = self._best_extractive_answer(question, passages)
            if best_sent and conf >= self.extractive_conf_threshold:
                grounded = self.scorer.score(best_sent, [p["text"] for p in passages[: self.max_ctx_passages]])
                self.stats["extractive_used"] += 1
                return {"answer": best_sent, "groundedness": float(grounded), "passages": passages[: self.max_ctx_passages]}

        ctx_texts: List[str] = self._prepare_context_texts(question, passages)
        prompt: str = build_prompt(question, ctx_texts)
        enc_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_src_tokens).to(DEVICE)
        with torch.no_grad():
            output_ids = self.model.generate(
                **enc_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature is not None and temperature > 0),
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        grounded = self.scorer.score(answer, ctx_texts)
        return {"answer": answer, "groundedness": float(grounded), "passages": passages[: self.max_ctx_passages]}

    def ask(self, question: str, k: int = 12, alpha: float = 0.6, **gen_kwargs) -> Dict[str, Any]:
        passages = self.retrieve(question, k=k, alpha=alpha)
        return self.generate(question, passages, **gen_kwargs)
