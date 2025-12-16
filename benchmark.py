import argparse
import json

from src.vectorstore.faiss_store import FaissVectorStore
from src.dataset_registry import discover_dataset
from src.benchmark_loader import read_fvecs, read_ivecs
from src.search import BenchmarkRunner
from src.metrics import recall_at_k, reciprocal_rank, success_at_k, ndcg_at_k
from src.embedding import EmbeddingPipeline
from src.msmarco_loader import (
    load_msmarco_corpus,
    load_msmarco_queries,
    load_msmarco_qrels,
)

# -------------------------------
# MS MARCO evaluation
# -------------------------------
def run_msmarco(dataset, topk, workers):
    from time import time
    import numpy as np

    corpus = load_msmarco_corpus(dataset["corpus"])
    queries = load_msmarco_queries(dataset["queries"])
    qrels = load_msmarco_qrels(dataset["qrels"])

    embedder = EmbeddingPipeline()

    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())

    # ---------- Corpus embedding ----------
    t0 = time()
    corpus_vecs = embedder.embed_texts(corpus_texts)
    corpus_embed_time = time() - t0

    # ---------- Index build ----------
    store = FaissVectorStore(persist_dir="E:/faiss_store")
    t0 = time()
    store.load_corpus_vectors(corpus_vecs)
    index_build_time = time() - t0

    # ---------- Query embedding ----------
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    t0 = time()
    query_vecs = embedder.embed_texts(query_texts)
    query_embed_time = time() - t0

    # ---------- Evaluation ----------
    recalls_5, recalls_10, recalls_20 = [], [], []
    mrrs, successes, ndcgs = [], [], []
    latencies = []

    t0 = time()
    for qid, qvec in zip(query_ids, query_vecs):
        pred_idx, latency = store.search(qvec, topk)
        latencies.append(latency)

        pred_doc_ids = [corpus_ids[i] for i in pred_idx]
        gt = set(qrels.get(qid, []))

        recalls_5.append(recall_at_k(pred_doc_ids, gt, 5))
        recalls_10.append(recall_at_k(pred_doc_ids, gt, 10))
        recalls_20.append(recall_at_k(pred_doc_ids, gt, 20))

        mrrs.append(reciprocal_rank(pred_doc_ids, gt))
        successes.append(success_at_k(pred_doc_ids, gt, 10))
        ndcgs.append(ndcg_at_k(pred_doc_ids, gt, 10))

    total_query_time = time() - t0
    qps = len(query_ids) / total_query_time if total_query_time > 0 else 0.0

    return {
        "accuracy": {
            "mrr@10": float(np.mean(mrrs)),
            "recall@5": float(np.mean(recalls_5)),
            "recall@10": float(np.mean(recalls_10)),
            "recall@20": float(np.mean(recalls_20)),
            "success@10": float(np.mean(successes)),
            "ndcg@10": float(np.mean(ndcgs)),
        },
        "latency": {
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "throughput": {
            "qps": qps
        },
        "indexing": {
            "build_time_s": index_build_time,
            "size_vectors": store.get_index_size(),
            "ram_bytes": store.get_memory_usage(),
        },
        "embedding": {
            "corpus_time_s": corpus_embed_time,
            "query_time_s": query_embed_time,
        },
    }

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = discover_dataset(args.dataset)

    print(f"Detected dataset type: {dataset['type']}")

    if dataset["type"] == "ann":
        corpus = read_fvecs(dataset["corpus"])
        queries = read_fvecs(dataset["queries"])
        gt = read_ivecs(dataset["gt"])

        store = FaissVectorStore(persist_dir="E:/faiss_store")
        store.load_corpus_vectors(corpus)

        runner = BenchmarkRunner(store)
        results = runner.run(
            queries=queries,
            groundtruth=gt,
            k=args.topk,
            workers=args.workers,
        )

    elif dataset["type"] == "msmarco":
        results = run_msmarco(dataset, args.topk, args.workers)

    else:
        raise ValueError("Unsupported dataset type")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark complete.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
