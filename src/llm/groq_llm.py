"""Groq LLM client with local fallback for FreeRAG."""

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

# Groq API configuration - Support for multiple API keys (up to 10)
GROQ_API_KEYS: List[str] = []

# Load primary key
_primary_key = os.environ.get("GROQ_API_KEY", "")
if _primary_key:
    GROQ_API_KEYS.append(_primary_key)

# Load additional keys (GROQ_API_KEY_2 through GROQ_API_KEY_10)
for i in range(2, 11):
    key = os.environ.get(f"GROQ_API_KEY_{i}", "")
    if key:
        GROQ_API_KEYS.append(key)

GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, free model on Groq


class GroqLLM:
    """Groq-based LLM with local model fallback.
    
    Uses Groq API for fast inference with multiple API key fallback.
    Rotates through available keys on rate limits or errors before
    falling back to local Phi-3 model.
    """
    
    def __init__(self):
        """Initialize Groq client with multiple API key support."""
        self._groq_clients: List = []
        self._local_model = None
        self._current_key_index = 0
        self._groq_available = len(GROQ_API_KEYS) > 0
        
        if self._groq_available:
            try:
                from groq import Groq
                # Initialize clients for all available API keys
                for i, api_key in enumerate(GROQ_API_KEYS):
                    try:
                        client = Groq(api_key=api_key)
                        self._groq_clients.append(client)
                        key_name = "primary" if i == 0 else f"key_{i + 1}"
                        logger.info(f"✅ Groq client initialized ({key_name})")
                    except Exception as e:
                        logger.warning(f"⚠️ Groq client {i + 1} initialization failed: {e}")
                
                if not self._groq_clients:
                    self._groq_available = False
                    logger.warning("⚠️ No valid Groq clients initialized")
                else:
                    logger.info(f"🔑 {len(self._groq_clients)} Groq API key(s) available for rotation")
            except Exception as e:
                logger.warning(f"⚠️ Groq module initialization failed: {e}")
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
        """Generate response using Groq with multi-key rotation and local fallback.
        
        Args:
            prompt: User prompt/question.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated response string.
        """
        # Try all Groq API keys before falling back to local
        if self._groq_available and self._groq_clients:
            # Try each key starting from current index
            keys_tried = 0
            total_keys = len(self._groq_clients)
            
            while keys_tried < total_keys:
                current_client = self._groq_clients[self._current_key_index]
                key_name = "primary" if self._current_key_index == 0 else f"key_{self._current_key_index + 1}"
                
                try:
                    response = self._call_groq_with_client(
                        current_client, prompt, system_prompt, max_tokens, temperature
                    )
                    if response:
                        return response
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "rate" in error_str or "limit" in error_str or "429" in error_str
                    
                    if is_rate_limit:
                        logger.warning(f"⚠️ Groq API rate limited ({key_name}), trying next key...")
                    else:
                        logger.warning(f"⚠️ Groq API error ({key_name}): {e}")
                    
                    # Move to next key
                    self._current_key_index = (self._current_key_index + 1) % total_keys
                    keys_tried += 1
            
            logger.warning(f"⚠️ All {total_keys} Groq API key(s) exhausted, falling back to local model")
        
        # Fallback to local model
        logger.info("🔄 Using local model for generation")
        return self._call_local(prompt, system_prompt, max_tokens)
    
    def _call_groq_with_client(
        self,
        client,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Groq API with a specific client."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
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
