"""Sentence embeddings using sentence-transformers."""

import os
import sys
import logging

# Disable TensorFlow to avoid import conflicts with transformers
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EmbeddingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding model wrapper using sentence-transformers."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding model.
        
        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"📥 Loading embedding model: {self.config.model_name}")
            logger.info(f"   Size: ~90MB")
            try:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device
                )
                logger.info("✅ Embedding model loaded!")
            except Exception as e:
                logger.error(f"❌ Embedding model failed: {e}")
                raise
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Make the class callable for ChromaDB compatibility.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        return self.embed_documents(texts)
