"""LLM model wrapper using HuggingFace Transformers - Production Grade."""

import logging
import sys
import time
from typing import Optional, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
GENERATION_TIMEOUT = 60  # seconds


class ModelLoadError(Exception):
    """Custom exception for model loading failures."""
    pass


class GenerationError(Exception):
    """Custom exception for text generation failures."""
    pass


class PhiModel:
    """Production-grade LLM wrapper using HuggingFace Transformers."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model wrapper.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._is_loaded = False
    
    @property
    def model(self):
        """Lazy load the model with retry logic."""
        if self._pipeline is None:
            self._load_model_with_retry()
        return self._pipeline
    
    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded and self._pipeline is not None
    
    def _load_model_with_retry(self) -> None:
        """Load model with retry logic for production reliability."""
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"📥 Loading model (attempt {attempt}/{MAX_RETRIES}): {self.config.repo_id}")
                self._load_model()
                self._is_loaded = True
                return
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt} failed: {str(e)[:100]}")
                if attempt < MAX_RETRIES:
                    logger.info(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
        
        logger.error(f"❌ Model loading failed after {MAX_RETRIES} attempts")
        raise ModelLoadError(f"Failed to load model after {MAX_RETRIES} attempts: {last_error}")
    
    def _load_model(self) -> None:
        """Download and load the model."""
        # Load tokenizer
        logger.info("🔧 Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.repo_id,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with CPU optimizations
        logger.info("🔧 Loading model weights...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.repo_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline for text generation
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id
        )
        
        logger.info("✅ Model loaded successfully!")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text completion with error handling.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
            
        Raises:
            GenerationError: If generation fails.
        """
        if not prompt or not prompt.strip():
            return "Please provide a valid question."
        
        # Truncate very long prompts
        max_prompt_length = 3000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
            logger.warning(f"Prompt truncated to {max_prompt_length} characters")
        
        try:
            result = self.model(
                prompt,
                max_new_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                return_full_text=False
            )
            
            generated = result[0]["generated_text"].strip()
            
            # Clean up response - remove any fake dialogue continuation
            generated = self._clean_response(generated)
            
            if not generated:
                return "I couldn't generate a response. Please try rephrasing your question."
            
            return generated
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise GenerationError(f"Failed to generate response: {str(e)[:100]}")
    
    def _clean_response(self, text: str) -> str:
        """Remove fake dialogue continuations from model output.
        
        TinyLlama and similar models sometimes continue generating fake
        user/assistant dialogue. This method cuts off such continuations.
        """
        if not text:
            return text
        
        # Stop patterns - cut off if model starts generating fake dialogue
        stop_patterns = [
            "\nUser:", "\nuser:",
            "\nHuman:", "\nhuman:",
            "\nSystem:", "\nsystem:",
            "\nAssistant:", "\nassistant:",
            "\n\nUser", "\n\nHuman",
            "User's question:",
            "\n---\n",
            "<|", "[INST]", "</s>"
        ]
        
        result = text
        for pattern in stop_patterns:
            if pattern in result:
                result = result.split(pattern)[0]
        
        # Also check for repeated newlines with potential role markers
        lines = result.split("\n")
        cleaned_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            # Stop if we hit a line that looks like a role marker
            if line_lower.startswith(("user:", "human:", "system:", "assistant:")):
                break
            cleaned_lines.append(line)
        
        result = "\n".join(cleaned_lines).strip()
        
        return result
    
    def generate_safe(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text with fallback on error (never throws).
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text or fallback message.
        """
        try:
            return self.generate(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Safe generation fallback: {e}")
            return "I'm having trouble processing your request right now. Please try again in a moment."
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Assistant's response.
        """
        if not messages:
            return "Please provide a message."
        
        # Format messages for chat
        chat_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                chat_text += f"System: {content}\n\n"
            elif role == "user":
                chat_text += f"User: {content}\n\n"
            elif role == "assistant":
                chat_text += f"Assistant: {content}\n\n"
        
        chat_text += "Assistant: "
        
        return self.generate_safe(chat_text, max_tokens)
    
    def chat_with_context(
        self, 
        query: str, 
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[str] = None
    ) -> str:
        """Generate response with RAG context and conversation history.
        
        Args:
            query: User's question.
            context: Retrieved context from documents.
            system_prompt: Optional system prompt.
            conversation_history: Optional formatted conversation history (last 6 messages).
            
        Returns:
            Generated response.
        """
        if not query or not query.strip():
            return "Please ask a question."
        
        if system_prompt is None:
            system_prompt = (
                "Your name is Dragon. Always speak in only ENGLISH not any other language. "
                "You are a friendly and helpful assistant having a natural conversation. "
                "Answer questions based on the provided document context. "
                "Be conversational, warm, and helpful - like talking to a knowledgeable friend. "
                "If you can find relevant information, explain it clearly and naturally. "
                "If the context doesn't have enough information, kindly ask the user to provide "
                "more details or suggest what they might be looking for. "
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
        
        user_message = f"""{history_section}Here's some information from the documents:

{context}

User's current question: {query}

Please respond naturally and helpfully, considering the conversation context:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.chat(messages)

