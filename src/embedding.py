from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Load model once (shared across calls)
_model = None


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts into dense float32 vectors.
    Used for text datasets like MS MARCO.
    """
    global _model

    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings.astype("float32")
