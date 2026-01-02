# HuggingFace Spaces Dockerfile for FreeRAG
# Uses HuggingFace Transformers - NO compilation required

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user

USER user
WORKDIR /home/user/app

# Create cache directories
RUN mkdir -p /home/user/.cache/huggingface

# Copy requirements
COPY --chown=user:user requirements.txt .

# Install Python dependencies (all pre-built wheels, no compilation!)
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# Expose the Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
