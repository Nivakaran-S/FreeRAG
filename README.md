---
title: FreeRAG
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# FreeRAG - Local RAG System

A modular Retrieval Augmented Generation (RAG) system powered by Phi-3.5-mini.

## Features

- 📄 **Multi-format support**: PDF, DOCX, TXT, Markdown
- 🔍 **Semantic search**: ChromaDB vector store with sentence-transformers
- 🤖 **Local LLM**: Phi-3.5-mini running via llama-cpp
- 💬 **Interactive chat**: Ask questions about your documents
- 🎨 **Modern UI**: Clean Gradio interface

## Usage

1. Upload your documents using the file upload panel
2. Wait for processing to complete
3. Ask questions in the chat interface
4. Get AI-powered answers with source citations

## Tech Stack

- **LLM**: Phi-3.5-mini (GGUF via llama-cpp-python)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **UI**: Gradio
