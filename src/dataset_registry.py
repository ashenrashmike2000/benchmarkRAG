from pathlib import Path

DATASET_ROOT = Path("data/benchmark")

def discover_dataset(dataset_name: str):
    dataset_dir = DATASET_ROOT / dataset_name.lower()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    files = list(dataset_dir.iterdir())

    corpus = next((f for f in files if f.name.lower().endswith("base.fvecs")), None)
    queries = next((f for f in files if f.name.lower().endswith("query.fvecs")), None)
    gt = next((f for f in files if f.name.lower().endswith("groundtruth.ivecs")), None)

    if not corpus or not queries or not gt:
        raise FileNotFoundError(
            f"Incomplete dataset files in {dataset_dir}\n"
            f"Found: {[f.name for f in files]}"
        )

    return {
        "corpus": str(corpus),
        "queries": str(queries),
        "gt": str(gt),
    }
