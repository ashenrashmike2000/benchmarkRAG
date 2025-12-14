from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True).astype("float32")
