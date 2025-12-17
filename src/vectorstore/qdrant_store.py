import time
import requests
from typing import Any, Dict, Optional
import numpy as np
import psutil

# from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from src.vectorstore.base import VectorStore


from qdrant_client import QdrantClient

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "benchmark_vectors",
        url: str = "http://localhost:6333",
    ):
        self.collection_name = collection_name
        self.url = url  # ✅ store public URL
        self.client = QdrantClient(
            url=url,
            timeout=300.0,
            check_compatibility=False,
        )
        self._vector_dim = None


    # -----------------------------
    # Index build
    # -----------------------------
    def load_corpus_vectors(
            self,
            vectors: np.ndarray,
            metadata: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        vectors = vectors.astype("float32")
        self._vector_dim = vectors.shape[1]

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self._vector_dim,
                distance=Distance.EUCLID,
            ),
        )

        batch_size = 2_000  # VERY IMPORTANT
        total = len(vectors)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []

            for i in range(start, end):
                payload = metadata[i] if metadata else {}
                points.append(
                    PointStruct(
                        id=i,
                        vector=vectors[i].tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            time.sleep(0.05)  # give Qdrant breathing room

    # -----------------------------
    # Search
    # -----------------------------

    def search(self, query_vector: np.ndarray, top_k: int):
        t0 = time.time()

        url = f"{self.url}/collections/{self.collection_name}/points/search"

        payload = {
            "vector": query_vector.astype("float32").tolist(),
            "limit": top_k,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        latency = time.time() - t0

        hits = result["result"]
        indices = np.array([hit["id"] for hit in hits], dtype=int)

        return indices, latency

    def batch_search(self, query_vectors, top_k):
        raise NotImplementedError("Batch search not used for Qdrant")

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def get_index_size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def get_memory_usage(self) -> int:
        # local client process memory (consistent with FAISS measurement)
        return psutil.Process().memory_info().rss

    def get_index_disk_size(self) -> int:
        # Qdrant manages storage internally; not always available reliably
        return -1
