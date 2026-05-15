"""EmbeddingsEngine - Vector embeddings via Ollama.

Uses ollama.embed() (current API). Falls back to zero vector on failure.
Model: locusai/all-minilm-l6-v2 (384 dimensions).
"""
import ollama
import numpy as np

from backend.config import OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


class EmbeddingsEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or OLLAMA_EMBEDDING_MODEL

    def generate(self, text: str) -> bytes:
        """Generate embedding for text, return as bytes for SQLite storage."""
        try:
            response = ollama.embed(model=self.model_name, input=text)
            # ollama.embed() returns {'embeddings': [[...]]}
            embedding_list = response['embeddings'][0]
            embedding_np = np.array(embedding_list, dtype=np.float32)
            return embedding_np.tobytes()
        except Exception as e:
            print(f"[EmbeddingsEngine] ollama.embed() failed: {e}")
            # Try legacy API as fallback
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                embedding_list = response['embedding']
                embedding_np = np.array(embedding_list, dtype=np.float32)
                return embedding_np.tobytes()
            except Exception as e2:
                print(f"[EmbeddingsEngine] Legacy API also failed: {e2}")
                return np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32).tobytes()

    def to_numpy(self, embedding_bytes: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        if not embedding_bytes:
            return np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        return np.frombuffer(embedding_bytes, dtype=np.float32)


embeddings_engine = EmbeddingsEngine()
