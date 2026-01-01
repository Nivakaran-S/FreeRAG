"""Document loader module for FreeRAG."""

from src.document_loader.loader import DocumentLoader, Document
from src.document_loader.splitter import TextSplitter

__all__ = ["DocumentLoader", "Document", "TextSplitter"]
