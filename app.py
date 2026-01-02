"""Gradio web interface for FreeRAG - designed for HuggingFace Spaces."""

import gradio as gr
from pathlib import Path
import tempfile
import os
import sys
import logging
import threading

# Configure logging for HuggingFace Spaces visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force stdout to be unbuffered for real-time logs
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

logger.info("="*50)
logger.info("🚀 FreeRAG Starting...")
logger.info("="*50)

from src.config import Config
from src.rag.pipeline import RAGPipeline

logger.info("✅ Core modules imported successfully")


# Global pipeline instance with thread lock for concurrent access
pipeline: RAGPipeline = None
pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline (thread-safe)."""
    global pipeline
    with pipeline_lock:
        if pipeline is None:
            logger.info("🔧 Initializing RAG pipeline...")
            logger.info("📥 This may take a few minutes on first run (downloading models)...")
            pipeline = RAGPipeline(Config.default())
            logger.info("✅ RAG pipeline initialized successfully!")
    return pipeline


def process_files(files):
    """Process uploaded files with production-grade validation."""
    if not files:
        return "📁 Please upload at least one file.", get_stats_text()
    
    # Constants for validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    
    pipe = get_pipeline()
    total_chunks = 0
    processed_files = []
    errors = []
    
    for file in files:
        try:
            # Get file info
            file_path = file.name if hasattr(file, 'name') else file
            file_name = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()
            
            # Validate file extension
            if file_ext not in ALLOWED_EXTENSIONS:
                errors.append(f"⚠️ {file_name}: Unsupported format. Use PDF, DOCX, TXT, or MD.")
                continue
            
            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                errors.append(f"⚠️ {file_name}: File too large ({file_size // 1024 // 1024}MB). Max is 10MB.")
                continue
            
            # Process the file
            count = pipe.ingest_file(file_path)
            total_chunks += count
            processed_files.append(file_name)
            logger.info(f"✅ Processed {file_name}: {count} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            errors.append(f"❌ {Path(file_path).name}: Processing failed")
    
    # Build response message
    messages = []
    if processed_files:
        messages.append(f"✅ Successfully processed {len(processed_files)} file(s)")
        messages.append(f"📄 Files: {', '.join(processed_files)}")
        messages.append(f"📊 Added {total_chunks} chunks to knowledge base")
    
    if errors:
        messages.extend(errors)
    
    if not processed_files and not errors:
        messages.append("❌ No files were processed. Please try again.")
    
    return "\n".join(messages), get_stats_text()


def answer_question(question, top_k, chat_history):
    """Answer a question with production-grade error handling."""
    # Input validation
    if not question or not question.strip():
        return chat_history, ""
    
    question = question.strip()
    
    # Length validation
    MAX_QUESTION_LENGTH = 1000
    if len(question) > MAX_QUESTION_LENGTH:
        response = f"⚠️ Your question is too long ({len(question)} chars). Please keep it under {MAX_QUESTION_LENGTH} characters."
        chat_history.append((question[:100] + "...", response))
        return chat_history, ""
    
    try:
        pipe = get_pipeline()
        
        if pipe.vector_store.get_count() == 0:
            response = (
                "📁 **No documents uploaded yet!**\n\n"
                "Please upload some documents using the panel on the left, "
                "then ask your questions."
            )
        else:
            try:
                result = pipe.query(question, top_k=int(top_k))
                response = result.get("answer", "I couldn't generate a response. Please try again.")
                
                # Add sources if available
                sources = result.get("sources", [])
                if sources:
                    unique_sources = list(set(s.get("filename", "Unknown") for s in sources))
                    response += f"\n\n---\n📚 *Sources: {', '.join(unique_sources)}*"
                    
            except Exception as e:
                logger.error(f"Query error: {e}")
                response = (
                    "😔 I had trouble processing your question.\n\n"
                    "**Try:**\n"
                    "- Rephrasing your question\n"
                    "- Asking something more specific\n"
                    "- Uploading more relevant documents"
                )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        response = "⚠️ The system is temporarily unavailable. Please try again in a moment."
    
    chat_history.append((question, response))
    return chat_history, ""


def get_stats_text() -> str:
    """Get stats as formatted text."""
    pipe = get_pipeline()
    stats = pipe.get_stats()
    return (
        f"📊 Documents: {stats['documents_count']} chunks\n"
        f"🤖 Model: Phi-3.5-mini\n"
        f"📐 Embeddings: {stats['embedding_model']}"
    )


def clear_knowledge_base():
    """Clear all documents from the vector store."""
    pipe = get_pipeline()
    pipe.vector_store.clear()
    return "🗑️ Knowledge base cleared.", get_stats_text()


# Custom CSS for modern dark theme
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
.chat-message {
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
}
footer {
    display: none !important;
}
"""

# Build Gradio interface
with gr.Blocks(
    title="FreeRAG - Local RAG System",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    css=custom_css
) as demo:
    
    gr.Markdown("""
    # 🚀 FreeRAG
    ### Local RAG System powered by Phi-3.5-mini
    
    Upload your documents and ask questions! Everything runs locally with no data leaving your machine.
    """)
    
    with gr.Row():
        # Left column - Document Upload
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Documents")
            
            file_upload = gr.File(
                label="Upload files (PDF, DOCX, TXT, MD)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md"]
            )
            
            upload_btn = gr.Button("📤 Process Documents", variant="primary")
            upload_status = gr.Textbox(label="Status", lines=3, interactive=False)
            
            gr.Markdown("### 📊 Knowledge Base Stats")
            stats_display = gr.Textbox(
                label="",
                value=get_stats_text,
                lines=3,
                interactive=False,
                every=5  # Refresh every 5 seconds
            )
            
            clear_btn = gr.Button("🗑️ Clear Knowledge Base", variant="secondary")
        
        # Right column - Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about your documents...",
                    scale=4,
                    show_label=False
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Sources",
                    scale=1
                )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Ask", variant="primary", scale=2)
                clear_chat_btn = gr.Button("🧹 Clear Chat", scale=1)
    
    # Event handlers
    upload_btn.click(
        fn=process_files,
        inputs=[file_upload],
        outputs=[upload_status, stats_display]
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, top_k_slider, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, top_k_slider, chatbot],
        outputs=[chatbot, question_input]
    )
    
    clear_btn.click(
        fn=clear_knowledge_base,
        outputs=[upload_status, stats_display]
    )
    
    clear_chat_btn.click(
        fn=lambda: [],
        outputs=[chatbot]
    )
    
    gr.Markdown("""
    ---
    <center>
    <p style="color: gray;">
    Built with 💙 using Phi-3.5-mini, ChromaDB, and Gradio | 
    <a href="https://github.com/yourusername/FreeRAG">GitHub</a>
    </p>
    </center>
    """)


if __name__ == "__main__":
    logger.info("="*50)
    logger.info("🌐 Launching Gradio interface...")
    logger.info(f"📍 Server: 0.0.0.0:7860")
    logger.info("="*50)
    
    # Pre-initialize pipeline to show download progress in logs
    logger.info("🔄 Pre-loading models (this may take a few minutes)...")
    try:
        get_pipeline()
        logger.info("✅ Models loaded successfully!")
    except Exception as e:
        logger.warning(f"⚠️ Model pre-load failed: {e}")
        logger.info("Models will be loaded on first query instead.")
    
    logger.info("🎉 FreeRAG is ready! Starting web server...")
    
    # Enable queue for concurrent request handling (Gradio 4.0.0 compatible)
    demo.queue(max_size=10)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
