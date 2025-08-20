from rank_bm25 import BM25Okapi
import pickle, re
from typing import List, Tuple

_tokenize = re.compile(r"\w+").findall

class BM25Index:
    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus_texts = corpus_texts
        self.tokenized = [list(map(str.lower, _tokenize(t))) for t in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized, k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        q_toks = list(map(str.lower, _tokenize(query)))
        scores = self.bm25.get_scores(q_toks)
        # return top-k (index_in_corpus, score)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in idxs]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"tokenized": self.tokenized, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str, corpus_texts: List[str]):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = object.__new__(cls)
        inst.corpus_texts = corpus_texts
        inst.tokenized = obj["tokenized"]
        inst.bm25 = obj["bm25"]
        return inst
