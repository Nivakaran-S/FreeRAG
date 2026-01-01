"""Document retriever for RAG pipeline."""

from typing import List, Dict, Any, Optional

from src.vectorstore.chroma_store import VectorStore


class Retriever:
    """Retrieve relevant documents from the vector store."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store to search.
            top_k: Number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query.
            top_k: Override default number of results.
            
        Returns:
            List of relevant documents with metadata.
        """
        return self.vector_store.search(query, top_k=top_k or self.top_k)
    
    def retrieve_text(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve and format documents as a single context string.
        
        Args:
            query: User query.
            top_k: Override default number of results.
            
        Returns:
            Formatted context string.
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("filename", "Unknown")
            content = result["content"]
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
