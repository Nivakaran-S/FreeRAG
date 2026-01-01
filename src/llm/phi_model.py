"""Phi-3.5-mini model wrapper using llama-cpp-python."""

from typing import Optional, List, Dict, Any
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.config import ModelConfig


class PhiModel:
    """Wrapper for Phi-3.5-mini model."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model wrapper.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._model: Optional[Llama] = None
        self._model_path: Optional[str] = None
    
    @property
    def model(self) -> Llama:
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Download and load the model."""
        print(f"Downloading model from {self.config.repo_id}...")
        self._model_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename
        )
        
        print("Loading model into memory...")
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            verbose=self.config.verbose
        )
        print("Model loaded successfully!")
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text completion.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
        """
        output = self.model(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            echo=False
        )
        return output["choices"][0]["text"].strip()
    
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
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature
        )
        return output["choices"][0]["message"]["content"].strip()
    
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
