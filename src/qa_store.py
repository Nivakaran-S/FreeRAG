"""Semantic Q&A Store for FreeRAG.

Uses embeddings to find similar questions and return pre-stored answers,
avoiding model calls for frequently asked or similar questions.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SemanticQAStore:
    """Store for Q&A pairs with semantic similarity matching.
    
    When a user asks a question, we first check if a semantically similar
    question has been asked before. If so, return the cached answer.
    This dramatically reduces model calls for common questions.
    """
    
    def __init__(
        self,
        store_dir: str = "./.cache/qa_store",
        similarity_threshold: float = 0.85,
        max_entries: int = 500
    ):
        """Initialize the semantic Q&A store.
        
        Args:
            store_dir: Directory to store Q&A data.
            similarity_threshold: Minimum cosine similarity to match (0.85 = very similar).
            max_entries: Maximum number of Q&A pairs to store.
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._lock = threading.Lock()
        
        # In-memory storage
        self._qa_pairs: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model = None
        
        # Load existing data
        self._load_store()
        logger.info(f"📚 Semantic Q&A Store initialized with {len(self._qa_pairs)} entries")
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            from src.embeddings.sentence_embeddings import EmbeddingModel
            from src.config import EmbeddingConfig
            self._embedding_model = EmbeddingModel(EmbeddingConfig())
        return self._embedding_model
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text."""
        model = self._get_embedding_model()
        return np.array(model.embed_documents([text])[0])
    
    def find_similar(self, question: str) -> Optional[Tuple[str, str, float]]:
        """Find a similar question in the store.
        
        Args:
            question: The user's question.
            
        Returns:
            Tuple of (matched_question, answer, similarity_score) if found,
            None otherwise.
        """
        try:
            if not self._qa_pairs or self._embeddings is None:
                return None
            
            if len(self._embeddings) == 0:
                return None
            
            # Compute embedding for the query
            query_embedding = self._compute_embedding(question)
            
            # Handle 1D vs 2D array
            if len(self._embeddings.shape) == 1:
                self._embeddings = self._embeddings.reshape(1, -1)
            
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
            store_norms = self._embeddings / norms
            
            similarities = np.dot(store_norms, query_norm)
            best_idx = np.argmax(similarities)
            best_score = float(similarities[best_idx])
            
            if best_score >= self.similarity_threshold:
                match = self._qa_pairs[best_idx]
                logger.info(f"🎯 Semantic match found (score: {best_score:.3f})")
                return (match["question"], match["answer"], best_score)
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic search error: {e}")
            return None
    
    def add(self, question: str, answer: str, sources: Optional[List] = None) -> None:
        """Add a Q&A pair to the store.
        
        Args:
            question: The question.
            answer: The answer.
            sources: Optional list of source documents.
        """
        try:
            # Check if we already have a very similar question (outside lock)
            existing = self.find_similar(question)
            if existing and existing[2] > 0.95:
                # Already have this question, skip
                return
            
            # Compute embedding (outside lock for performance)
            embedding = self._compute_embedding(question)
            
            with self._lock:
                # Add to store
                entry = {
                    "question": question,
                    "answer": answer,
                    "sources": sources or [],
                    "hit_count": 0
                }
                self._qa_pairs.append(entry)
                
                # Update embeddings array
                if self._embeddings is None:
                    self._embeddings = embedding.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, embedding])
                
                # Evict if over capacity
                if len(self._qa_pairs) > self.max_entries:
                    self._evict_oldest()
                
                # Save to disk
                self._save_store()
                logger.info(f"💾 Added Q&A pair to semantic store (total: {len(self._qa_pairs)})")
                
        except Exception as e:
            logger.warning(f"Failed to add Q&A pair: {e}")
    
    def _evict_oldest(self) -> None:
        """Remove oldest entries to stay under capacity."""
        evict_count = len(self._qa_pairs) - self.max_entries + 50  # Remove 50 extra
        if evict_count > 0:
            self._qa_pairs = self._qa_pairs[evict_count:]
            if self._embeddings is not None:
                self._embeddings = self._embeddings[evict_count:]
            logger.info(f"♻️ Evicted {evict_count} old Q&A entries")
    
    def _save_store(self) -> None:
        """Save the store to disk."""
        try:
            # Save Q&A pairs
            qa_file = self.store_dir / "qa_pairs.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(self._qa_pairs, f, ensure_ascii=False, indent=2)
            
            # Save embeddings
            if self._embeddings is not None:
                emb_file = self.store_dir / "embeddings.npy"
                np.save(emb_file, self._embeddings)
                
        except Exception as e:
            logger.warning(f"Failed to save Q&A store: {e}")
    
    def _load_store(self) -> None:
        """Load the store from disk."""
        try:
            qa_file = self.store_dir / "qa_pairs.json"
            emb_file = self.store_dir / "embeddings.npy"
            
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    self._qa_pairs = json.load(f)
            
            if emb_file.exists():
                self._embeddings = np.load(emb_file)
                
        except Exception as e:
            logger.warning(f"Failed to load Q&A store: {e}")
            self._qa_pairs = []
            self._embeddings = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "entries": len(self._qa_pairs),
            "max_entries": self.max_entries,
            "similarity_threshold": self.similarity_threshold
        }
    
    def clear(self) -> None:
        """Clear all stored Q&A pairs."""
        with self._lock:
            self._qa_pairs = []
            self._embeddings = None
            
            # Delete files
            for f in self.store_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        
        logger.info("🗑️ Semantic Q&A store cleared")


# Global store instance
_qa_store: Optional[SemanticQAStore] = None


def get_qa_store() -> SemanticQAStore:
    """Get or create the global Q&A store."""
    global _qa_store
    if _qa_store is None:
        _qa_store = SemanticQAStore()
    return _qa_store
