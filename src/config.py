"""Configuration settings for FreeRAG."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """LLM model configuration."""
    repo_id: str = "bartowski/Phi-3.5-mini-instruct-GGUF"
    filename: str = "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 4
    max_tokens: int = 512
    temperature: float = 0.7
    verbose: bool = False


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    collection_name: str = "freerag_documents"
    persist_directory: str = "./chroma_db"
    top_k: int = 3


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    data_directory: str = "./data"
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)
        Path(self.vectorstore.persist_directory).mkdir(parents=True, exist_ok=True)
