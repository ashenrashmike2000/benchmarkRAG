import argparse
import json
import time
from src.vectorstore import FaissVectorStore
from src.benchmark_loader import read_fvecs, read_ivecs, generate_metadata
from src.search import BenchmarkRunner
from src.dataset_registry import discover_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name (e.g., deep1m, sift1m)")
    parser.add_argument("--corpus")
    parser.add_argument("--queries")
    parser.add_argument("--gt")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.dataset:
        paths = discover_dataset(args.dataset)
        args.corpus = paths["corpus"]
        args.queries = paths["queries"]
        args.gt = paths["gt"]

    if not args.corpus or not args.queries or not args.gt:
        raise ValueError(
            "You must provide either --dataset OR (--corpus, --queries, --gt)"
        )

    print("Loading data...")
    corpus = read_fvecs(args.corpus)
    queries = read_fvecs(args.queries)
    gt = read_ivecs(args.gt)

    print("Building index...")
    t0 = time.time()
    store = FaissVectorStore(persist_dir="E:/faiss_store")
    metadata = generate_metadata(len(corpus))
    store.load_corpus_vectors(corpus, metadata=metadata)
    index_build_time = time.time() - t0

    runner = BenchmarkRunner(store)
    results = runner.run(
        queries=queries,
        groundtruth=gt,
        k=args.topk,
        workers=args.workers,
    )

    results["indexing_performance"] = {
        "build_time_s": index_build_time,
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark complete.")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
