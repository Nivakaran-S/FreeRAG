# HuggingFace Spaces Dockerfile for FreeRAG
# Uses Python 3.11 with CPU-only llama-cpp-python

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Set up cache directories with proper permissions
RUN mkdir -p /home/user/.cache/huggingface \
    && mkdir -p /home/user/app/chroma_db

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies
# Note: llama-cpp-python will be built from source for CPU
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Expose the Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Run the application
CMD ["python", "app.py"]
