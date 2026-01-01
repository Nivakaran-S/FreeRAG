"""Text splitter for chunking documents."""

from dataclasses import dataclass
from typing import List, Optional

from src.config import ChunkingConfig
from src.document_loader.loader import Document


@dataclass
class TextChunk:
    """Represents a chunk of text."""
    content: str
    metadata: dict
    chunk_index: int


class TextSplitter:
    """Split text into overlapping chunks."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the text splitter.
        
        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkingConfig()
    
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Split text into chunks.
        
        Args:
            text: Text to split.
            metadata: Optional metadata to attach to chunks.
            
        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split by sentences/paragraphs first
        text = text.replace("\r\n", "\n")
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.config.chunk_size
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundaries
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            else:
                end = len(text)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end
                    },
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.config.chunk_overlap
            if start <= chunks[-1].metadata.get("start_char", 0) if chunks else 0:
                start = end  # Avoid infinite loop
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Split multiple documents into chunks.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of text chunks from all documents.
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.split_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
