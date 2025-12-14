# src/vectorstore.py
import os
import time
import pickle
from typing import List, Any, Optional, Callable, Dict, Tuple
import numpy as np
import faiss
import psutil

class FaissVectorStore:
    def __init__(self, persist_dir: str = "E:/faiss_store"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    # -----------------------------
    # Load precomputed corpus vectors
    # -----------------------------
    def load_corpus_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        if vectors is None or vectors.size == 0:
            raise ValueError("Empty vectors provided")

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors.astype("float32"))

        if metadata is not None:
            if len(metadata) != vectors.shape[0]:
                raise ValueError("Metadata length must match number of vectors")
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(vectors.shape[0])]

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query: np.ndarray, top_k: int):
        q = query.reshape(1, -1).astype("float32")
        t0 = time.time()
        D, I = self.index.search(q, top_k)
        latency = time.time() - t0
        return I[0], latency

    def batch_search(self, queries: np.ndarray, top_k: int):
        t0 = time.time()
        D, I = self.index.search(queries.astype("float32"), top_k)
        elapsed = time.time() - t0
        qps = queries.shape[0] / elapsed if elapsed > 0 else 0.0
        return {"I": I, "D": D, "qps": qps}

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def get_index_size(self):
        return int(self.index.ntotal)

    def get_memory_usage(self):
        return psutil.Process(os.getpid()).memory_info().rss

    def get_index_disk_size(self):
        path = os.path.join(self.persist_dir, "faiss.index")
        return os.path.getsize(path) if os.path.exists(path) else 0
