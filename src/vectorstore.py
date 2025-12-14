# src/vectorstore.py
import os
import time
import pickle
from typing import List, Any, Optional, Callable, Dict, Tuple
import numpy as np
import faiss
import psutil
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    """
    FAISS-based vector store with:
     - load_corpus_vectors for precomputed vector datasets
     - search / batch_search
     - filtered_batch_search (naive, for metadata filtering tests)
     - ingestion helpers for ingestion rate and time-to-index
     - diagnostics (memory, disk size)
    """

    def __init__(
        self,
        persist_dir: str = "E:/faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index_build_time: Optional[float] = None
        self.index_build_memory_delta: Optional[int] = None
        self.last_search_latency: Optional[float] = None

    # -----------------------------
    # Build and persist
    # -----------------------------
    def build_from_documents(self, documents: List[Any], use_embeddings: Optional[np.ndarray] = None, metadatas: Optional[List[Dict]] = None):
        mem_before = self._current_rss()
        t0 = time.time()

        if use_embeddings is None:
            emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = emb_pipe.chunk_documents(documents)
            embeddings = emb_pipe.embed_chunks(chunks).astype("float32")
            metadatas = [{"text": c.page_content} for c in chunks]
        else:
            embeddings = use_embeddings.astype("float32")
            if metadatas is None:
                metadatas = [{"text": ""} for _ in range(embeddings.shape[0])]

        self._create_index_and_add(embeddings, metadatas)
        self.save()

        t1 = time.time()
        mem_after = self._current_rss()
        self.index_build_time = t1 - t0
        self.index_build_memory_delta = mem_after - mem_before

    def _create_index_and_add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = metadatas

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"faiss index or metadata not found in {self.persist_dir}")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    # -----------------------------
    # Load precomputed vectors (fix for your error)
    # -----------------------------
    def load_corpus_vectors(self, vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Load precomputed vectors (numpy array shape (N, dim)) directly into FAISS.
        """
        if vectors is None or vectors.size == 0:
            raise ValueError("Empty vectors provided to load_corpus_vectors")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors.astype("float32"))
        if metadatas is not None:
            if len(metadatas) != vectors.shape[0]:
                # if length mismatch, pad or trim
                if len(metadatas) < vectors.shape[0]:
                    metadatas.extend([{}] * (vectors.shape[0] - len(metadatas)))
                else:
                    metadatas = metadatas[: vectors.shape[0]]
            self.metadata = metadatas
        else:
            self.metadata = [{"text": ""} for _ in range(vectors.shape[0])]
        return

    # -----------------------------
    # Search / batch search
    # -----------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        if self.index is None:
            raise ValueError("Index not built or loaded.")
        q = query_embedding.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        t0 = time.time()
        D, I = self.index.search(q, top_k)
        latency = time.time() - t0
        self.last_search_latency = latency

        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if (idx is not None and idx < len(self.metadata)) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results, latency

    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 5) -> Dict:
        if self.index is None:
            raise ValueError("Index not built or loaded.")
        q = query_embeddings.astype("float32")
        t0 = time.time()
        D, I = self.index.search(q, top_k)
        total_time = time.time() - t0
        qps = q.shape[0] / total_time if total_time > 0 else float("inf")
        return {"I": I, "D": D, "total_time": total_time, "qps": qps}

    # -----------------------------
    # Filtered search (naive)
    # -----------------------------
    def filtered_batch_search(self, query_embeddings: np.ndarray, filter_fn: Callable[[Dict[str, Any]], bool], top_k: int = 5) -> Dict:
        """
        Naive filtered search for benchmarking filter overhead.
        Returns dict with I, D, total_time, qps, selectivity.
        """
        # select indices that match
        selected_indices = [i for i, m in enumerate(self.metadata) if filter_fn(m)]
        selectivity = len(selected_indices) / max(1, len(self.metadata))

        if len(selected_indices) == 0:
            qshape = (query_embeddings.shape[0], 0)
            return {"I": np.empty(qshape, dtype="int32"), "D": np.empty(qshape, dtype="float32"), "total_time": 0.0, "qps": 0.0, "selectivity": selectivity}

        # attempt to reconstruct vectors for selected indices (may fail for some index types)
        vectors = None
        try:
            vectors = np.array([self.index.reconstruct(int(i)) for i in selected_indices], dtype="float32")
        except Exception:
            # fallback: if not possible, raise to indicate DB does not support reconstruct; user should provide direct vectors
            raise RuntimeError("Index does not support reconstruct; filtered_batch_search requires vector access.")

        # build temporary index for selected vectors
        dim = vectors.shape[1]
        tmp = faiss.IndexFlatL2(dim)
        tmp.add(vectors)
        t0 = time.time()
        D, I = tmp.search(query_embeddings.astype("float32"), top_k)
        total_time = time.time() - t0
        # convert local ids to global ids
        I_global = np.vectorize(lambda x: selected_indices[int(x)])(I)
        qps = query_embeddings.shape[0] / total_time if total_time > 0 else float("inf")
        return {"I": I_global, "D": D, "total_time": total_time, "qps": qps, "selectivity": selectivity}

    # -----------------------------
    # Ingestion / updates
    # -----------------------------
    def add_vectors(self, vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None):
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []
        self.index.add(vectors.astype("float32"))
        if metadatas is not None:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in range(vectors.shape[0])])

    def measure_ingestion_rate(self, vectors: np.ndarray, batch_size: int = 1024) -> Dict:
        total = 0
        t0 = time.time()
        for i in range(0, vectors.shape[0], batch_size):
            batch = vectors[i : i + batch_size]
            self.add_vectors(batch)
            total += batch.shape[0]
        total_time = time.time() - t0
        vps = total / total_time if total_time > 0 else float("inf")
        return {"total_vectors": total, "total_time_s": total_time, "vps": vps}

    def measure_time_to_index(self, vectors: np.ndarray, sample_query: np.ndarray, top_k: int = 1) -> float:
        t0 = time.time()
        self.add_vectors(vectors)
        # For IndexFlatL2, vectors are immediately searchable; return insertion time
        return time.time() - t0

    # -----------------------------
    # CRUD / diagnostics
    # -----------------------------
    def get_by_id(self, idx: int) -> Dict[str, Any]:
        return self.metadata[idx] if 0 <= idx < len(self.metadata) else {}

    def get_memory_usage(self) -> int:
        return self._current_rss()

    def get_index_disk_size(self) -> int:
        p = os.path.join(self.persist_dir, "faiss.index")
        return os.path.getsize(p) if os.path.exists(p) else 0

    def get_index_size(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def estimate_capacity(self, available_ram_bytes: int, dim: int, overhead_per_vector: int = 16) -> int:
        bytes_per_vector = 4 * dim + overhead_per_vector
        return int(available_ram_bytes // bytes_per_vector)

    @staticmethod
    def _current_rss() -> int:
        return psutil.Process(os.getpid()).memory_info().rss
