from sentence_transformers import SentenceTransformer
import numpy as np, faiss, os
from typing import List, Tuple

class DenseIndex:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.embeddings = None  # (N, D)
        self.ids = None         # list[str] or list[int]
        self.normalize = True   # cosine via inner product on normalized vectors

    def build(self, texts: List[str], ids: List[str], batch_size: int = 64):
        embs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=self.normalize
        )
        self.embeddings = embs.astype("float32")
        self.ids = np.array(ids)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=self.normalize).astype("float32")
        D, I = self.index.search(q, k)  # inner product ~ cosine
        return [(int(i), float(s)) for i, s in zip(I[0], D[0])]

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        np.save(os.path.join(dir_path, "ids.npy"), self.ids)
        # embeddings not strictly needed if index is persisted

    def load(self, dir_path: str):
        self.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        self.ids = np.load(os.path.join(dir_path, "ids.npy"), allow_pickle=True)
