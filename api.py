"""REST API endpoints for FreeRAG."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(default=3, description="Number of documents to retrieve", ge=1, le=10)


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str
    answer: str
    sources: List[dict]
    cached: bool
    match_type: str


class UploadResponse(BaseModel):
    """Response model for file uploads."""
    success: bool
    message: str
    files_processed: int
    chunks_added: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    documents_count: int
    model: str


class StatsResponse(BaseModel):
    """Response model for stats."""
    documents_count: int
    collection_name: str
    model: str
    embedding_model: str


# Create FastAPI app
api = FastAPI(
    title="FreeRAG API",
    description="REST API for the FreeRAG Retrieval-Augmented Generation system",
    version="1.0.0"
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_pipeline():
    """Import and get the RAG pipeline from app.py."""
    from app import get_pipeline
    return get_pipeline()


@api.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "FreeRAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /api/query",
            "upload": "POST /api/upload",
            "stats": "GET /api/stats",
            "health": "GET /api/health"
        }
    }


@api.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the system is healthy and ready."""
    try:
        pipe = get_rag_pipeline()
        stats = pipe.get_stats()
        return HealthResponse(
            status="healthy",
            documents_count=stats["documents_count"],
            model=stats["model"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@api.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics."""
    try:
        pipe = get_rag_pipeline()
        stats = pipe.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    The system will:
    1. Check cache for exact match
    2. Check for semantically similar questions
    3. Search uploaded documents
    4. Generate answer using AI if needed
    """
    try:
        pipe = get_rag_pipeline()
        
        if pipe.vector_store.get_count() == 0:
            return QueryResponse(
                question=request.question,
                answer="No documents uploaded yet. Please upload documents first using /api/upload",
                sources=[],
                cached=False,
                match_type="error"
            )
        
        result = pipe.query(request.question, top_k=request.top_k)
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources", []),
            cached=result.get("cached", False),
            match_type=result.get("match_type", "generated")
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload documents to the RAG system.
    
    Supported formats: PDF, DOCX, TXT, MD
    Maximum file size: 10MB per file
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    
    pipe = get_rag_pipeline()
    total_chunks = 0
    processed_count = 0
    errors = []
    
    for file in files:
        try:
            # Check extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{file.filename}: Unsupported format")
                continue
            
            # Save to temp file
            content = await file.read()
            
            # Check size
            if len(content) > MAX_FILE_SIZE:
                errors.append(f"{file.filename}: File too large (max 10MB)")
                continue
            
            # Write to temp file and process
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                chunks = pipe.ingest_file(tmp_path)
                total_chunks += chunks
                processed_count += 1
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    
    message = f"Processed {processed_count} file(s), {total_chunks} chunks added"
    if errors:
        message += f". Errors: {'; '.join(errors)}"
    
    return UploadResponse(
        success=processed_count > 0,
        message=message,
        files_processed=processed_count,
        chunks_added=total_chunks
    )


@api.delete("/api/clear", tags=["Documents"])
async def clear_documents():
    """Clear all uploaded documents from the system."""
    try:
        pipe = get_rag_pipeline()
        pipe.vector_store.clear()
        return {"success": True, "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
