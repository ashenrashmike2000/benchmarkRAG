# src/search.py
import numpy as np
import time
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable
from src.vectorstore import FaissVectorStore

class BenchmarkRunner:
    def __init__(self, store: FaissVectorStore):
        self.store = store

    # Basic recall@k and qps
    def simple_recall_and_qps(self, queries: np.ndarray, groundtruth: np.ndarray, k: int = 10) -> Dict:
        batch = self.store.batch_search(queries, top_k=k)
        I = batch["I"]
        Q = queries.shape[0]
        recalls = []
        for i in range(Q):
            retrieved = set(int(x) for x in I[i])
            gt = set(int(x) for x in groundtruth[i] if int(x) >= 0)
            recalls.append(len(retrieved & gt) / max(len(gt), 1))
        return {"qps": batch["qps"], f"recall@{k}": float(np.mean(recalls))}

    # Throughput concurrent
    def throughput_concurrent(self, queries: np.ndarray, top_k: int, workers: int) -> Dict:
        parts = np.array_split(queries, workers)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.store.batch_search, p, top_k) for p in parts if p.size > 0]
            results = [f.result() for f in futures]
        total_time = time.time() - t0
        total_q = sum(p.shape[0] for p in parts)
        qps = total_q / total_time if total_time > 0 else float("inf")
        return {"qps": qps, "total_time": total_time}

    # Run a battery (simpler and robust)
    def run_extended_benchmark(self, queries: np.ndarray, groundtruth: np.ndarray, topk: int = 10, workers: int = 4, available_ram_bytes: int = None) -> Dict:
        out = {}
        out["index_size"] = self.store.get_index_size()
        out["index_disk_size"] = self.store.get_index_disk_size()
        out["memory_usage_bytes"] = self.store.get_memory_usage()

        # basic recall + qps
        basics = self.simple_recall_and_qps(queries, groundtruth, k=topk)
        out.update(basics)

        # throughput concurrent
        out["throughput_concurrent"] = self.throughput_concurrent(queries, top_k=topk, workers=workers)

        # ingestion sample
        sample = queries[: min(1000, queries.shape[0])]
        out["ingestion_rate_sample"] = self.store.measure_ingestion_rate(sample, batch_size=128)

        # time to index sample
        out["time_to_index_sample_s"] = self.store.measure_time_to_index(sample[:10], queries[0:1], top_k=1)

        # metadata filtered QPS (if metadata exists)
        try:
            def filter_fn(m): return m.get("category") == "A"
            filtered = self.store.filtered_batch_search(queries[:100], filter_fn, top_k=topk)
            raw = self.store.batch_search(queries[:100], top_k=topk)
            out["metadata_filter"] = {"raw_qps": raw["qps"], "filtered_qps": filtered["qps"], "selectivity": filtered.get("selectivity", None)}
        except Exception as e:
            out["metadata_filter_error"] = str(e)

        # concurrency test (median latencies) - smaller sample
        try:
            parts = np.array_split(queries[:200], min(workers, 8))
            latencies = []
            with ThreadPoolExecutor(max_workers=min(workers, 8)) as ex:
                futures = [ex.submit(self._search_collect_latencies, p, topk) for p in parts if p.size > 0]
                for f in futures:
                    latencies.extend(f.result())
            out["concurrency_median_latency_ms"] = float(np.median(latencies)) * 1000.0 if latencies else None
        except Exception as e:
            out["concurrency_error"] = str(e)

        # estimate capacity if ram provided
        if available_ram_bytes:
            dim = queries.shape[1]
            out["theoretical_max_vectors"] = self.store.estimate_capacity(available_ram_bytes, dim)

        return out

    def _search_collect_latencies(self, queries_part, top_k):
        lat = []
        for q in queries_part:
            _, l = self.store.search(q, top_k)
            lat.append(l)
        return lat
