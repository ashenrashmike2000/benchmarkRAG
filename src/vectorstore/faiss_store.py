import os
import time
import pickle
from typing import Any, Dict, Optional, Tuple
import numpy as np
import faiss
import psutil

from src.vectorstore.base import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, persist_dir: str = "./faiss_store"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: list[Dict[str, Any]] = []

    # -----------------------------
    # Index build
    # -----------------------------
    def load_corpus_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        if vectors is None or vectors.size == 0:
            raise ValueError("Empty vectors provided")

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors.astype("float32"))

        if metadata:
            if len(metadata) != len(vectors):
                raise ValueError("Metadata length mismatch")
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(len(vectors))]

        # persist index (optional but good for disk metrics)
        faiss.write_index(
            self.index,
            os.path.join(self.persist_dir, "faiss.index"),
        )

    # -----------------------------
    # Search
    # -----------------------------
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, float]:
        q = query_vector.reshape(1, -1).astype("float32")
        t0 = time.time()
        _, indices = self.index.search(q, top_k)
        latency = time.time() - t0
        return indices[0], latency

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
    ) -> Dict[str, Any]:
        q = query_vectors.astype("float32")
        t0 = time.time()
        _, indices = self.index.search(q, top_k)
        elapsed = time.time() - t0
        qps = len(q) / elapsed if elapsed > 0 else 0.0
        return {
            "indices": indices,
            "qps": qps,
        }

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def get_index_size(self) -> int:
        return int(self.index.ntotal)

    def get_memory_usage(self) -> int:
        return psutil.Process(os.getpid()).memory_info().rss

    def get_index_disk_size(self) -> int:
        path = os.path.join(self.persist_dir, "faiss.index")
        return os.path.getsize(path) if os.path.exists(path) else 0
