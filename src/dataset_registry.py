from pathlib import Path

DATASET_ROOT = Path("data/benchmark")

def discover_dataset(dataset_name: str):
    dataset_dir = DATASET_ROOT / dataset_name.lower()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    files = {f.name.lower(): f for f in dataset_dir.iterdir()}

    # ---------- ANN datasets (Deep1M, SIFT, etc.) ----------
    if any(name.endswith("base.fvecs") for name in files):
        corpus = next(f for f in files.values() if f.name.lower().endswith("base.fvecs"))
        queries = next(f for f in files.values() if f.name.lower().endswith("query.fvecs"))
        gt = next(f for f in files.values() if f.name.lower().endswith("groundtruth.ivecs"))

        return {
            "type": "ann",
            "corpus": str(corpus),
            "queries": str(queries),
            "gt": str(gt),
        }

    # ---------- MS MARCO (docdev) ----------
    required = {
        "msmarco-docs-100_000.tsv",
        "msmarco-docdev-queries.tsv",
        "msmarco-docdev-qrels.tsv",
    }

    if required.issubset(files.keys()):
        return {
            "type": "msmarco",
            "corpus": str(files["msmarco-docs-100_000.tsv"]),
            "queries": str(files["msmarco-docdev-queries.tsv"]),
            "qrels": str(files["msmarco-docdev-qrels.tsv"]),
        }

    raise ValueError(f"Unknown or incomplete dataset format in {dataset_dir}")
