import os
import sys

# -------------------------------------------------
# FORCE Python to ignore bytecode caches
# -------------------------------------------------
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse
import json
import time

from src.dataset_registry import discover_dataset
from src.benchmark_loader import read_fvecs, read_ivecs
from src.msmarco_loader import (
    load_msmarco_corpus,
    load_msmarco_queries,
    load_msmarco_qrels,
)
from src.search import BenchmarkRunner
from src.embedding import embed_texts

from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.qdrant_store import QdrantVectorStore
from src.vectorstore.lancedb_store import LanceDBVectorStore


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def detect_dataset_type(dataset_name: str):
    if dataset_name.lower() in {"deep1m", "sift1m"}:
        return "ann"
    elif dataset_name.lower() == "msmarco":
        return "msmarco"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    query_ids = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="deep1m | sift1m | msmarco")
    parser.add_argument("--db", choices=["faiss", "qdrant", "lancedb"], default="faiss")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    ensure_dir(args.out)

    dataset_type = detect_dataset_type(args.dataset)
    print(f"Detected dataset type: {dataset_type}")
    print(f"Using vector database: {args.db}")

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    if dataset_type == "ann":
        paths = discover_dataset(args.dataset)

        corpus = read_fvecs(paths["corpus"])
        queries = read_fvecs(paths["queries"])
        groundtruth = read_ivecs(paths["gt"])

        # -------------------------------------------------
        # LanceDB-specific subsampling (Windows-safe)
        # -------------------------------------------------
        if args.db == "lancedb" and args.dataset.lower() == "sift1m":
            corpus = corpus[:100_000]
            queries = queries[:10_000]
            groundtruth = groundtruth[:10_000]


    else:  # MSMARCO
        msmarco_dir = "data/benchmark/msmarco"
        corpus = load_msmarco_corpus(f"{msmarco_dir}/msmarco-docs-100_000.tsv")
        queries = load_msmarco_queries(f"{msmarco_dir}/msmarco-docdev-queries.tsv")
        groundtruth = load_msmarco_qrels(f"{msmarco_dir}/msmarco-docdev-qrels.tsv")

    # -------------------------------------------------
    # Create vector store
    # -------------------------------------------------
    if args.db == "faiss":
        store = FaissVectorStore()

    elif args.db == "qdrant":
        store = QdrantVectorStore(
            collection_name=f"{args.dataset}_vectors",
            url="http://localhost:6333",
        )

        # Windows safety: avoid socket exhaustion
        args.workers = 1

    elif args.db == "lancedb":
        store = LanceDBVectorStore(
            db_path=f"E:/lancedb/{args.dataset}",
            table_name="vectors",
        )

    else:
        raise ValueError(f"Unsupported DB: {args.db}")

    # -------------------------------------------------
    # Build index
    # -------------------------------------------------
    print("Building index...")
    t0 = time.time()

    # MS MARCO requires embedding (text → vectors)
    if dataset_type == "msmarco":

        corpus_texts = list(corpus.values())
        corpus_vectors = embed_texts(corpus_texts)

        store.load_corpus_vectors(corpus_vectors)

        # Also embed queries for search
        # --- MS MARCO: preserve query IDs ---
        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        query_vectors = embed_texts(query_texts)

        # pass vectors + ids separately
        queries = query_vectors


    else:
        # ANN datasets already contain vectors
        store.load_corpus_vectors(corpus)

    build_time = time.time() - t0

    # -------------------------------------------------
    # Run benchmark
    # -------------------------------------------------
    runner = BenchmarkRunner(store)
    results = runner.run(
        queries,
        groundtruth,
        k=args.topk,
        workers=args.workers,
        query_ids=query_ids
    )

    # -------------------------------------------------
    # Add metadata
    # -------------------------------------------------
    results["meta"] = {
        "dataset": args.dataset,
        "database": args.db,
        "topk": args.topk,
        "workers": args.workers,
        "index_build_time_s": build_time,
    }

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark completed. Results saved to {args.out}")


if __name__ == "__main__":
    main()
