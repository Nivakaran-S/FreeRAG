"""Groq LLM client with local fallback for FreeRAG."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, free model on Groq


class GroqLLM:
    """Groq-based LLM with local model fallback.
    
    Uses Groq API for fast inference, falls back to local Phi-3
    if Groq is unavailable or rate limited.
    """
    
    def __init__(self):
        """Initialize Groq client."""
        self._groq_client = None
        self._local_model = None
        self._groq_available = bool(GROQ_API_KEY)
        
        if self._groq_available:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("✅ Groq client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Groq initialization failed: {e}")
                self._groq_available = False
        else:
            logger.info("📍 No GROQ_API_KEY found, using local model only")
    
    @property
    def local_model(self):
        """Lazy load the local fallback model."""
        if self._local_model is None:
            from src.llm.phi_model import PhiModel
            from src.config import ModelConfig
            logger.info("🔄 Loading local fallback model...")
            self._local_model = PhiModel(ModelConfig())
        return self._local_model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Groq with local fallback.
        
        Args:
            prompt: User prompt/question.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated response string.
        """
        # Try Groq first if available
        if self._groq_available and self._groq_client:
            try:
                response = self._call_groq(prompt, system_prompt, max_tokens, temperature)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"⚠️ Groq API error, falling back to local: {e}")
        
        # Fallback to local model
        logger.info("🔄 Using local model for generation")
        return self._call_local(prompt, system_prompt, max_tokens)
    
    def _call_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Groq API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        result = response.choices[0].message.content
        logger.info(f"✅ Groq response generated ({len(result)} chars)")
        return result
    
    def _call_local(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int
    ) -> str:
        """Call local model."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.local_model.chat(messages, max_tokens=max_tokens)
    
    def chat_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[str] = None
    ) -> str:
        """Generate response with RAG context.
        
        Args:
            query: User's question.
            context: Retrieved context from documents.
            system_prompt: Optional system prompt.
            conversation_history: Optional conversation history.
            
        Returns:
            Generated response.
        """
        if system_prompt is None:
            system_prompt = (
                "Your name is Dragon. Always speak in only ENGLISH not any other language. "
                "You are a friendly and helpful assistant having a natural conversation. "
                "Answer questions based on the provided document context. "
                "Be conversational, warm, and helpful - like talking to a knowledgeable friend. "
                "If you can find relevant information, explain it clearly and naturally. "
                "If the context doesn't have enough information, kindly say so. "
                "Keep your responses concise but friendly."
            )
        
        # Handle empty context
        if not context or not context.strip():
            context = "No relevant documents found."
        
        # Build message with optional history
        history_section = ""
        if conversation_history and conversation_history.strip():
            history_section = f"""Previous conversation:
{conversation_history}

---
"""
        
        prompt = f"""{history_section}Here's some information from the documents:

{context}

User's current question: {query}

Please respond naturally and helpfully, considering the conversation context:"""
        
        return self.generate(prompt, system_prompt=system_prompt)


# Global Groq LLM instance
_groq_llm: Optional[GroqLLM] = None


def get_groq_llm() -> GroqLLM:
    """Get or create the global Groq LLM instance."""
    global _groq_llm
    if _groq_llm is None:
        _groq_llm = GroqLLM()
    return _groq_llm
