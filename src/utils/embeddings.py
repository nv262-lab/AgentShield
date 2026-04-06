"""Sentence embedding utilities."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=self.batch_size,
                                  show_progress_bar=False,
                                  normalize_embeddings=normalize)
