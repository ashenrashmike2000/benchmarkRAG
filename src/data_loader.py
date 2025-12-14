# src/data_loader.py
from pathlib import Path
from typing import List, Any, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
import numpy as np
import struct
import random
import os

def load_all_documents(data_dir: str) -> List[Any]:
    path = Path(data_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    docs = []
    for f in path.rglob("*"):
        try:
            if f.suffix == ".pdf":
                docs.extend(PyPDFLoader(str(f)).load())
            elif f.suffix == ".txt":
                docs.extend(TextLoader(str(f)).load())
            elif f.suffix == ".csv":
                docs.extend(CSVLoader(str(f)).load())
            elif f.suffix == ".xlsx":
                docs.extend(UnstructuredExcelLoader(str(f)).load())
            elif f.suffix == ".docx":
                docs.extend(Docx2txtLoader(str(f)).load())
            elif f.suffix == ".json":
                docs.extend(JSONLoader(str(f)).load())
        except Exception:
            # skip problematic files, but continue
            continue
    return docs

def read_fvecs(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"fvecs file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vecs = []
    size = len(data)
    while offset < size:
        dim = struct.unpack_from("i", data, offset)[0]
        offset += 4
        fmt = f"{dim}f"
        vec = struct.unpack_from(fmt, data, offset)
        offset += 4 * dim
        vecs.append(vec)
    return np.array(vecs, dtype="float32")

def read_ivecs(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ivecs file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vecs = []
    size = len(data)
    while offset < size:
        dim = struct.unpack_from("i", data, offset)[0]
        offset += 4
        fmt = f"{dim}i"
        vec = struct.unpack_from(fmt, data, offset)
        offset += 4 * dim
        vecs.append(vec)
    return np.array(vecs, dtype="int32")

def load_numpy_vectors(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"npy vectors file not found: {path}")
    return np.load(path).astype("float32")

def load_groundtruth_numpy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"npy groundtruth file not found: {path}")
    return np.load(path).astype("int32")

def generate_metadata_for_corpus(n: int, field: str = "category", categories: List[str] = None, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    if categories is None:
        categories = ["A", "B", "C", "D"]
    out = []
    for i in range(n):
        out.append({field: random.choice(categories)})
    return out
