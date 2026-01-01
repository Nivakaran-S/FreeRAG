"""ChromaDB vector store implementation."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings

from src.config import VectorStoreConfig
from src.embeddings.sentence_embeddings import EmbeddingModel
from src.document_loader.splitter import TextChunk


class VectorStore:
    """ChromaDB-based vector store for document storage and retrieval."""
    
    def __init__(
        self, 
        config: Optional[VectorStoreConfig] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """Initialize the vector store.
        
        Args:
            config: Vector store configuration.
            embedding_model: Embedding model for generating vectors.
        """
        self.config = config or VectorStoreConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        self._client: Optional[chromadb.Client] = None
        self._collection = None
    
    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Add text chunks to the vector store.
        
        Args:
            chunks: List of text chunks to add.
            
        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0
        
        # Prepare data for ChromaDB
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.embed_documents(documents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"Added {len(chunks)} chunks to vector store.")
        return len(chunks)
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of results with document, metadata, and distance.
        """
        top_k = top_k or self.config.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return formatted
    
    def get_count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.config.collection_name)
        self._collection = None
        print("Vector store cleared.")
