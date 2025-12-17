import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    latency_percentiles,
)


class BenchmarkRunner:
    def __init__(self, store):
        self.store = store

    def run(self, queries, groundtruth, k=10, workers=4):
        latencies = []
        precisions, recalls, mrrs, ndcgs = [], [], [], []

        # ---- Cold start ----
        _, cold_latency = self.store.search(queries[0], k)

        # ---- Main evaluation ----
        start = time.time()

        if hasattr(self.store, "index"):
            workers = 1
            # ===== FAISS fast path (UNCHANGED behavior) =====
            D, I = self.store.index.search(queries, k)
            total_time = time.time() - start
            qps = len(queries) / total_time

            for i in range(len(queries)):
                pred = list(map(int, I[i]))
                gt = set(map(int, groundtruth[i]))

                precisions.append(precision_at_k(pred, gt, k))
                recalls.append(recall_at_k(pred, gt, k))
                mrrs.append(reciprocal_rank(pred, gt))
                ndcgs.append(ndcg_at_k(pred, gt, k))

        else:
            # ===== Generic VectorStore path (Qdrant, Milvus, etc.) =====
            total_time = 0.0

            for i, q in enumerate(queries):
                t0 = time.time()
                pred, latency = self.store.search(q, k)
                total_time += time.time() - t0

                gt = set(map(int, groundtruth[i]))

                precisions.append(precision_at_k(pred, gt, k))
                recalls.append(recall_at_k(pred, gt, k))
                mrrs.append(reciprocal_rank(pred, gt))
                ndcgs.append(ndcg_at_k(pred, gt, k))
                latencies.append(latency)

            qps = len(queries) / total_time if total_time > 0 else 0.0

        # ---- Latency sampling ----
        for q in queries[:200]:
            _, l = self.store.search(q, k)
            latencies.append(l)

        # ---- Concurrent stress ----
        def task(qs):
            for q in qs:
                self.store.search(q, k)

        parts = np.array_split(queries, workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(task, parts)

        return {
            "accuracy": {
                f"precision@{k}": float(np.mean(precisions)),
                f"recall@{k}": float(np.mean(recalls)),
                "mrr": float(np.mean(mrrs)),
                "ndcg": float(np.mean(ndcgs)),
            },
            "latency": latency_percentiles(latencies),
            "cold_start_latency": cold_latency,
            "throughput": {
                "qps": qps,
                "sustained_qps": qps,
            },
            "index": {
                "size_vectors": self.store.get_index_size(),
                "disk_bytes": self.store.get_index_disk_size(),
                "ram_bytes": self.store.get_memory_usage(),
            },
        }
