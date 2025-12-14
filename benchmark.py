import argparse
import json
import time
from src.vectorstore import FaissVectorStore
from src.data_loader import read_fvecs, read_ivecs, generate_metadata_for_corpus
from src.search import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading data...")
    corpus = read_fvecs(args.corpus)
    queries = read_fvecs(args.queries)
    gt = read_ivecs(args.gt)

    print("Building index...")
    t0 = time.time()
    store = FaissVectorStore(persist_dir="E:/faiss_store")
    metadata = generate_metadata_for_corpus(len(corpus))
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
