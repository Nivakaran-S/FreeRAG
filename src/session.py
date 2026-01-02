"""Session management for FreeRAG chat history."""

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and chat history.
    
    Each session is identified by a UUID and stores:
    - Chat history (question-answer pairs)
    - Session metadata (created_at, last_active)
    """
    
    def __init__(
        self,
        storage_dir: str = "./.cache/sessions",
        max_history: int = 6,
        session_ttl_hours: int = 24
    ):
        """Initialize session manager.
        
        Args:
            storage_dir: Directory to store session data.
            max_history: Maximum messages to keep (for context).
            session_ttl_hours: Session expiry time in hours.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        # Load existing sessions
        self._load_sessions()
        self._cleanup_expired()
        logger.info(f"📋 Session manager initialized with {len(self._sessions)} active sessions")
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "history": []
            }
            self._save_session(session_id)
        
        logger.info(f"📝 Created new session: {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID, creating if doesn't exist."""
        with self._lock:
            if session_id not in self._sessions:
                # Try to load from disk
                self._load_session(session_id)
            
            if session_id in self._sessions:
                # Update last_active
                self._sessions[session_id]["last_active"] = datetime.now().isoformat()
                return self._sessions[session_id]
        
        return None
    
    def add_message(
        self, 
        session_id: str, 
        question: str, 
        answer: str
    ) -> None:
        """Add a Q&A pair to session history."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "last_active": datetime.now().isoformat(),
                    "history": []
                }
            
            session = self._sessions[session_id]
            session["history"].append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last N messages
            if len(session["history"]) > self.max_history * 2:
                session["history"] = session["history"][-self.max_history:]
            
            session["last_active"] = datetime.now().isoformat()
            self._save_session(session_id)
    
    def get_history(self, session_id: str, limit: int = None) -> List[Tuple[str, str]]:
        """Get chat history for a session.
        
        Args:
            session_id: Session ID.
            limit: Max messages to return (default: max_history).
            
        Returns:
            List of (question, answer) tuples.
        """
        limit = limit or self.max_history
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        history = session.get("history", [])[-limit:]
        return [(h["question"], h["answer"]) for h in history]
    
    def get_history_for_prompt(self, session_id: str) -> str:
        """Get formatted history for including in prompt.
        
        Returns last 6 messages formatted for the model.
        """
        history = self.get_history(session_id, self.max_history)
        
        if not history:
            return ""
        
        formatted = []
        for q, a in history:
            # Truncate long messages
            q_short = q[:200] + "..." if len(q) > 200 else q
            a_short = a[:300] + "..." if len(a) > 300 else a
            formatted.append(f"User: {q_short}\nAssistant: {a_short}")
        
        return "\n\n".join(formatted)
    
    def clear_history(self, session_id: str) -> None:
        """Clear chat history for a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["history"] = []
                self._save_session(session_id)
    
    def _save_session(self, session_id: str) -> None:
        """Save session to disk."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self._sessions[session_id], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save session {session_id[:8]}: {e}")
    
    def _load_session(self, session_id: str) -> None:
        """Load session from disk."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    self._sessions[session_id] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session {session_id[:8]}: {e}")
    
    def _load_sessions(self) -> None:
        """Load all sessions from disk."""
        try:
            for session_file in self.storage_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        self._sessions[session["id"]] = session
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = datetime.now()
        expired = []
        
        for sid, session in self._sessions.items():
            try:
                last_active = datetime.fromisoformat(session["last_active"])
                if now - last_active > self.session_ttl:
                    expired.append(sid)
            except Exception:
                pass
        
        for sid in expired:
            self._delete_session(sid)
        
        if expired:
            logger.info(f"♻️ Cleaned up {len(expired)} expired sessions")
    
    def _delete_session(self, session_id: str) -> None:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
