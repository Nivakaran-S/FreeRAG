"""Response caching system for FreeRAG."""

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache for storing question-response pairs to avoid repeated model calls.
    
    Uses file-based storage for persistence across restarts.
    Cache entries expire after a configurable duration.
    """
    
    def __init__(
        self, 
        cache_dir: str = "./.cache/responses",
        max_entries: int = 1000,
        ttl_hours: int = 24
    ):
        """Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files.
            max_entries: Maximum number of entries to keep in memory.
            ttl_hours: Time-to-live for cache entries in hours.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Load existing cache from disk
        self._load_cache()
        logger.info(f"📦 Response cache initialized with {len(self._memory_cache)} entries")
    
    def _get_cache_key(self, question: str, context_hash: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Generate a unique cache key from question, context, and session.
        
        Args:
            question: The user's question (normalized).
            context_hash: Hash of the document context (optional).
            session_id: Session ID for per-user caching (optional).
            
        Returns:
            SHA256 hash as the cache key.
        """
        # Normalize question: lowercase, strip whitespace
        normalized = question.lower().strip()
        
        # Include session_id for per-session caching
        if session_id:
            normalized = f"{session_id}|{normalized}"
        
        # Include context hash if provided for document-specific caching
        if context_hash:
            normalized = f"{normalized}|{context_hash}"
        
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get(self, question: str, context_hash: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
        """Get cached response for a question.
        
        Args:
            question: The user's question.
            context_hash: Hash of the document context (optional).
            session_id: Session ID for per-user caching (optional).
            
        Returns:
            Cached response if found and not expired, None otherwise.
        """
        key = self._get_cache_key(question, context_hash, session_id)
        
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check expiry
                cached_at = datetime.fromisoformat(entry["cached_at"])
                if datetime.now() - cached_at < self.ttl:
                    logger.info(f"💾 Cache hit for question: '{question[:50]}...'")
                    return entry["response"]
                else:
                    # Expired, remove from cache
                    del self._memory_cache[key]
                    self._delete_from_disk(key)
        
        return None
    
    def set(
        self, 
        question: str, 
        response: str, 
        context_hash: Optional[str] = None,
        sources: Optional[list] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Store a question-response pair in cache.
        
        Args:
            question: The user's question.
            response: The generated response.
            context_hash: Hash of the document context (optional).
            sources: List of source documents used.
            session_id: Session ID for per-user caching (optional).
        """
        key = self._get_cache_key(question, context_hash, session_id)
        
        entry = {
            "question": question,
            "response": response,
            "context_hash": context_hash,
            "sources": sources or [],
            "cached_at": datetime.now().isoformat(),
            "hit_count": 0
        }
        
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._memory_cache) >= self.max_entries:
                self._evict_oldest()
            
            self._memory_cache[key] = entry
            self._save_to_disk(key, entry)
        
        logger.info(f"💾 Cached response for question: '{question[:50]}...'")
    
    def _save_to_disk(self, key: str, entry: Dict[str, Any]) -> None:
        """Save a cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache entry: {e}")
    
    def _delete_from_disk(self, key: str) -> None:
        """Delete a cache entry from disk."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete cache entry: {e}")
    
    def _load_cache(self) -> None:
        """Load cache entries from disk."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                    
                    # Check if expired
                    cached_at = datetime.fromisoformat(entry["cached_at"])
                    if datetime.now() - cached_at < self.ttl:
                        key = cache_file.stem
                        self._memory_cache[key] = entry
                    else:
                        # Expired, delete file
                        cache_file.unlink()
                        
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _evict_oldest(self) -> None:
        """Remove oldest cache entries to make room for new ones."""
        if not self._memory_cache:
            return
        
        # Sort by cached_at and remove oldest 10%
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1]["cached_at"]
        )
        
        evict_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:evict_count]:
            del self._memory_cache[key]
            self._delete_from_disk(key)
        
        logger.info(f"♻️ Evicted {evict_count} old cache entries")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            
            # Delete all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        
        logger.info("🗑️ Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._memory_cache),
            "max_entries": self.max_entries,
            "ttl_hours": self.ttl.total_seconds() / 3600,
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create the global response cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
