"""LLM model wrapper using HuggingFace Transformers."""

import logging
import sys
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


class PhiModel:
    """Wrapper for LLM model using HuggingFace Transformers."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model wrapper.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._pipeline is None:
            self._load_model()
        return self._pipeline
    
    def _load_model(self) -> None:
        """Download and load the model with progress logging."""
        logger.info(f"📥 Loading model: {self.config.repo_id}")
        logger.info(f"   This may take a few minutes on first run...")
        
        try:
            # Load tokenizer
            logger.info("🔧 Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.repo_id,
                trust_remote_code=True
            )
            
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
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text completion.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
        """
        result = self.model(
            prompt,
            max_new_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            return_full_text=False
        )
        return result[0]["generated_text"].strip()
    
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
        # Format messages for chat
        chat_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chat_text += f"System: {content}\n\n"
            elif role == "user":
                chat_text += f"User: {content}\n\n"
            elif role == "assistant":
                chat_text += f"Assistant: {content}\n\n"
        
        chat_text += "Assistant: "
        
        return self.generate(chat_text, max_tokens)
    
    def chat_with_context(
        self, 
        query: str, 
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with RAG context.
        
        Args:
            query: User's question.
            context: Retrieved context from documents.
            system_prompt: Optional system prompt.
            
        Returns:
            Generated response.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on "
                "the provided context. If the context doesn't contain relevant "
                "information, say so honestly. Be concise and accurate."
            )
        
        user_message = f"""Context:
{context}

Question: {query}

Please answer based on the context provided above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.chat(messages)
