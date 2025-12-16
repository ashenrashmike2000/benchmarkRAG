from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class VectorStore(ABC):
    """
    Abstract interface for all vector databases.
    """

    @abstractmethod
    def load_corpus_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        """Build / load the vector index."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, float]:
        """Search a single query vector."""
        pass

    @abstractmethod
    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
    ) -> Dict[str, Any]:
        """Search multiple query vectors."""
        pass

    @abstractmethod
    def get_index_size(self) -> int:
        """Number of indexed vectors."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """RAM usage in bytes."""
        pass

    @abstractmethod
    def get_index_disk_size(self) -> int:
        """Disk usage in bytes."""
        pass
