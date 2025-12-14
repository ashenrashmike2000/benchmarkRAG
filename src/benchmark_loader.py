import os
import struct
import numpy as np
import random
from typing import List, Dict

# ---------- ANN benchmark loaders ----------

def read_fvecs(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}f", f.read(4 * dim))
            vectors.append(vec)

    return np.array(vectors, dtype="float32")


def read_ivecs(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}i", f.read(4 * dim))
            vectors.append(vec)

    return np.array(vectors, dtype="int32")


# ---------- Metadata generator (optional) ----------

def generate_metadata(n: int, categories=None, seed=42) -> List[Dict]:
    random.seed(seed)
    if categories is None:
        categories = ["A", "B", "C"]

    return [{"category": random.choice(categories)} for _ in range(n)]
