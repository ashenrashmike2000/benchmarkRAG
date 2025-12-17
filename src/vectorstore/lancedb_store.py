import time
import numpy as np
import lancedb
import os
import psutil

from src.vectorstore.base import VectorStore


class LanceDBVectorStore(VectorStore):
    def __init__(self, db_path="E:/lancedb", table_name="vectors"):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(db_path, exist_ok=True)

        self.db = lancedb.connect(db_path)
        self.table = None

    # -----------------------------
    # Index build
    # -----------------------------
    def load_corpus_vectors(self, vectors, metadata=None):
        vectors = vectors.astype("float32")

        data = []
        for i, vec in enumerate(vectors):
            row = {
                "id": i,
                "vector": vec,
            }
            if metadata:
                row.update(metadata[i])
            data.append(row)

        # overwrite table for clean benchmarking
        self.table = self.db.create_table(
            self.table_name,
            data,
            mode="overwrite",
        )

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query_vector, top_k):
        t0 = time.time()

        results = (
            self.table.search(query_vector.astype("float32"))
            .limit(top_k)
            .to_list()
        )

        latency = time.time() - t0
        indices = np.array([r["id"] for r in results], dtype=int)

        return indices, latency

    def batch_search(self, query_vectors, top_k):
        t0 = time.time()
        all_indices = []

        for q in query_vectors:
            res = (
                self.table.search(q.astype("float32"))
                .limit(top_k)
                .to_list()
            )
            all_indices.append([r["id"] for r in res])

        elapsed = time.time() - t0
        qps = len(query_vectors) / elapsed if elapsed > 0 else 0.0

        return {
            "indices": np.array(all_indices, dtype=int),
            "qps": qps,
        }

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def get_index_size(self):
        return self.table.count_rows()

    def get_memory_usage(self):
        return psutil.Process(os.getpid()).memory_info().rss

    def get_index_disk_size(self):
        return sum(
            os.path.getsize(os.path.join(self.db_path, f))
            for f in os.listdir(self.db_path)
        )
