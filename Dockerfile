# Use NVIDIA CUDA runtime with Ubuntu 22.04 as the base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip python3 python3-pip build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other dependencies. Combine pip installs for better layer caching.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create working directory
WORKDIR /app

# Download and set up the LLaMA 2-13B model (only necessary files)
RUN git clone --depth 1 https://huggingface.co/meta-llama/Llama-2-13b-hf /app/llama-2-13b

# Expose API port
EXPOSE 8000

# Copy FastAPI application code
COPY app.py /app/

# Healthcheck (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=60s CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server with Uvicorn (correct for GPU inference)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]