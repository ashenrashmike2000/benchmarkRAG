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

    def run(self, queries, groundtruth, k=10, workers=4, query_ids=None):
        print("DEBUG: MSMARCO groundtruth type =", type(groundtruth))

        latencies = []
        precisions, recalls, mrrs, ndcgs = [], [], [], []

        # -------------------------------------------------
        # Cold start
        # -------------------------------------------------
        _, cold_latency = self.store.search(queries[0], k)

        # -------------------------------------------------
        # Search (batch if supported, else loop)
        # -------------------------------------------------
        start = time.time()

        if hasattr(self.store, "search_batch"):
            D, I = self.store.search_batch(queries, k)
        else:
            I = []
            D = []

            for q in queries:
                ids, _ = self.store.search(q, k)
                I.append(ids)
                D.append([0.0] * len(ids))

            I = np.array(I)
            D = np.array(D)

        total_time = time.time() - start
        qps = len(queries) / total_time if total_time > 0 else 0.0

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------
        is_msmarco = isinstance(groundtruth, dict)

        for i in range(len(queries)):
            pred = list(map(int, I[i]))

            if is_msmarco:
                qid = query_ids[i]
                gt = set(map(int, groundtruth.get(qid, [])))
            else:
                gt = set(map(int, groundtruth[i]))

            precisions.append(precision_at_k(pred, gt, k))
            recalls.append(recall_at_k(pred, gt, k))
            mrrs.append(reciprocal_rank(pred, gt))
            ndcgs.append(ndcg_at_k(pred, gt, k))

        # -------------------------------------------------
        # Latency sampling
        # -------------------------------------------------
        for q in queries[:200]:
            _, l = self.store.search(q, k)
            latencies.append(l)

        # -------------------------------------------------
        # Concurrent stress
        # -------------------------------------------------
        def task(qs):
            for q in qs:
                self.store.search(q, k)

        parts = np.array_split(queries, workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.map(task, parts)

        # -------------------------------------------------
        # Results
        # -------------------------------------------------
        return {
            "accuracy": {
                f"precision@{k}": float(np.mean(precisions)),
                f"recall@{k}": float(np.mean(recalls)),
                "mrr": float(np.mean(mrrs)),
                f"ndcg@{k}": float(np.mean(ndcgs)),
            },
            "latency": latency_percentiles(latencies),
            "cold_start_latency": float(cold_latency),
            "throughput": {
                "qps": float(qps),
                "sustained_qps": float(qps),
            },
            "index": {
                "size_vectors": self.store.get_index_size(),
                "disk_bytes": self.store.get_index_disk_size(),
                "ram_bytes": self.store.get_memory_usage(),
            },
        }
