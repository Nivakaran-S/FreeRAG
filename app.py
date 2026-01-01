"""Gradio web interface for FreeRAG - designed for HuggingFace Spaces."""

import gradio as gr
from pathlib import Path
import tempfile
import os
import sys
import logging

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


# Global pipeline instance
pipeline: RAGPipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline."""
    global pipeline
    if pipeline is None:
        logger.info("🔧 Initializing RAG pipeline...")
        logger.info("📥 This may take a few minutes on first run (downloading models)...")
        pipeline = RAGPipeline(Config.default())
        logger.info("✅ RAG pipeline initialized successfully!")
    return pipeline


def process_files(files):
    """Process uploaded files and add to vector store."""
    if not files:
        return "Please upload at least one file.", get_stats_text()
    
    pipe = get_pipeline()
    total_chunks = 0
    processed_files = []
    
    for file in files:
        try:
            # Get the file path from gradio
            file_path = file.name if hasattr(file, 'name') else file
            count = pipe.ingest_file(file_path)
            total_chunks += count
            processed_files.append(Path(file_path).name)
        except Exception as e:
            return f"Error processing file: {e}", get_stats_text()
    
    return (
        f"✅ Successfully processed {len(processed_files)} file(s)!\n"
        f"📄 Files: {', '.join(processed_files)}\n"
        f"📊 Added {total_chunks} chunks to the knowledge base.",
        get_stats_text()
    )


def answer_question(question, top_k, chat_history):
    """Answer a question using RAG."""
    if not question.strip():
        return chat_history, ""
    
    pipe = get_pipeline()
    
    if pipe.vector_store.get_count() == 0:
        response = "⚠️ No documents have been uploaded yet. Please upload some documents first."
    else:
        try:
            result = pipe.query(question, top_k=int(top_k))
            response = result["answer"]
            
            # Add sources
            if result["sources"]:
                sources = [s["filename"] for s in result["sources"]]
                response += f"\n\n---\n📚 *Sources: {', '.join(sources)}*"
        except Exception as e:
            response = f"❌ Error: {e}"
    
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
                height=400,
                show_copy_button=True
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
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
