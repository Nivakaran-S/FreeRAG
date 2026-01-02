"""Main RAG pipeline orchestrating all components."""

from typing import Optional, Dict, Any

from src.config import Config
from src.llm.phi_model import PhiModel
from src.embeddings.sentence_embeddings import EmbeddingModel
from src.document_loader.loader import DocumentLoader
from src.document_loader.splitter import TextSplitter
from src.vectorstore.chroma_store import VectorStore
from src.rag.retriever import Retriever


class RAGPipeline:
    """Main RAG pipeline combining all components."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the RAG pipeline.
        
        Args:
            config: Configuration. Uses defaults if not provided.
        """
        self.config = config or Config.default()
        self.config.ensure_directories()
        
        # Initialize components lazily
        self._llm: Optional[PhiModel] = None
        self._embedding_model: Optional[EmbeddingModel] = None
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None
        self._document_loader: Optional[DocumentLoader] = None
        self._text_splitter: Optional[TextSplitter] = None
    
    @property
    def llm(self) -> PhiModel:
        """Get LLM instance."""
        if self._llm is None:
            self._llm = PhiModel(self.config.model)
        return self._llm
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get embedding model instance."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(self.config.embedding)
        return self._embedding_model
    
    @property
    def vector_store(self) -> VectorStore:
        """Get vector store instance."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                self.config.vectorstore,
                self.embedding_model
            )
        return self._vector_store
    
    @property
    def retriever(self) -> Retriever:
        """Get retriever instance."""
        if self._retriever is None:
            self._retriever = Retriever(
                self.vector_store,
                top_k=self.config.vectorstore.top_k
            )
        return self._retriever
    
    @property
    def document_loader(self) -> DocumentLoader:
        """Get document loader instance."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader()
        return self._document_loader
    
    @property
    def text_splitter(self) -> TextSplitter:
        """Get text splitter instance."""
        if self._text_splitter is None:
            self._text_splitter = TextSplitter(self.config.chunking)
        return self._text_splitter
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Number of chunks added.
        """
        print(f"Loading file: {file_path}")
        document = self.document_loader.load_file(file_path)
        
        print("Splitting into chunks...")
        chunks = self.text_splitter.split_text(document.content, document.metadata)
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        return self.vector_store.add_chunks(chunks)
    
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> int:
        """Ingest all files from a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search recursively.
            
        Returns:
            Total number of chunks added.
        """
        print(f"Loading documents from: {directory_path}")
        documents = self.document_loader.load_directory(directory_path, recursive)
        
        if not documents:
            print("No documents found.")
            return 0
        
        print(f"Loaded {len(documents)} documents. Splitting into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        return self.vector_store.add_chunks(chunks)
    
    def query(self, question: str, top_k: Optional[int] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system with semantic caching and session history.
        
        Lookup order:
        1. Exact cache match (instant)
        2. Semantic similarity match (instant)
        3. Model generation with conversation history (slow)
        
        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            session_id: Optional session ID for conversation history.
            
        Returns:
            Dict with answer and sources.
        """
        from src.cache import get_response_cache
        from src.qa_store import get_qa_store
        
        cache = get_response_cache()
        qa_store = get_qa_store()
        
        # 1. Check exact cache first
        cached_response = cache.get(question)
        if cached_response:
            return {
                "question": question,
                "answer": cached_response,
                "context": "[Exact Cache]",
                "sources": [],
                "cached": True,
                "match_type": "exact"
            }
        
        # 2. Check semantic similarity
        similar = qa_store.find_similar(question)
        if similar:
            matched_q, matched_answer, score = similar
            return {
                "question": question,
                "answer": matched_answer,
                "context": f"[Semantic Match: {score:.0%} similar]",
                "sources": [],
                "cached": True,
                "match_type": "semantic",
                "matched_question": matched_q
            }
        
        # 3. Generate answer using LLM (no cache hit)
        # Get conversation history if session provided
        conversation_history = None
        if session_id:
            from src.session import get_session_manager
            session_mgr = get_session_manager()
            conversation_history = session_mgr.get_history_for_prompt(session_id)
        
        context = self.retriever.retrieve_text(question, top_k)
        sources = self.retriever.retrieve(question, top_k)
        answer = self.llm.chat_with_context(
            question, 
            context, 
            conversation_history=conversation_history
        )
        
        # Build source list
        source_list = [
            {
                "filename": s["metadata"].get("filename", "Unknown"),
                "source": s["metadata"].get("source", "Unknown"),
                "distance": s.get("distance")
            }
            for s in sources
        ]
        
        # Store in both caches for future use
        cache.set(question, answer, sources=source_list)
        qa_store.add(question, answer, sources=source_list)
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": source_list,
            "cached": False,
            "match_type": "generated"
        }
    
    def chat(self, question: str) -> str:
        """Simple chat interface that returns just the answer.
        
        Args:
            question: User's question.
            
        Returns:
            Answer string.
        """
        result = self.query(question)
        return result["answer"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dict with stats about the pipeline.
        """
        return {
            "documents_count": self.vector_store.get_count(),
            "collection_name": self.config.vectorstore.collection_name,
            "model": self.config.model.repo_id,
            "embedding_model": self.config.embedding.model_name
        }
