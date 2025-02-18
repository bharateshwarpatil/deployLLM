FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip python3 python3-pip build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

RUN git clone --depth 1 https://huggingface.co/meta-llama/Llama-2-13b-hf /app/llama-2-13b

EXPOSE 8000

COPY app.py /app/

HEALTHCHECK --interval=30s --timeout=30s --retries=5 --start-period=60s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]