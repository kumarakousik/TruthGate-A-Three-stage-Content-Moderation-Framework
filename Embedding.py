from typing import List
import numpy as np
from langchain_ollama import OllamaEmbeddings

class Embeddings:
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = OllamaEmbeddings(model=self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_text_embeddings(self, text: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        embeddings = self.model.embed_documents(text)   # ✅ fixed plural
        return np.array(embeddings)

    def generate_query_embedding(self, query: str) -> List[float]:
        query = query.strip().lower()
        return self.model.embed_query(query)             # ✅ fixed self ref

Embedding_Manager = Embeddings()