# # LLaMA 2-13B Docker Deployment

## Overview
This repository provides a Dockerized environment for deploying **LLaMA 2-13B** using **FastAPI** and **vLLM** for optimized inference. This setup ensures efficient GPU utilization and fast response times for generative AI applications.

## Features
- **Uses NVIDIA CUDA Runtime (CUDA 12.1, cuDNN 8) for GPU acceleration**
- **Efficient inference with vLLM**
- **FastAPI-based API for serving requests**
- **Asynchronous processing using Uvicorn**
- **Model caching for improved performance**

---

## Prerequisites
- **NVIDIA GPU** with CUDA support
- **Docker & NVIDIA Container Toolkit** installed
- **Sufficient VRAM (Minimum 24GB recommended)**

### Install NVIDIA Container Toolkit (if not installed)
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU support in Docker
```bash
docker run --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

---

## Build & Run the Docker Container

### **1️⃣ Clone the Repository**
```bash
git clone https://huggingface.co/meta-llama/Llama-2-13b-hf llama-2-13b
cd llama-2-13b
```

### **2️⃣ Build the Docker Image**
```bash
docker build -t llama-2-13b .
```

### **3️⃣ Run the Container**
```bash
docker run --gpus all -p 8000:8000 llama-2-13b
```

> This will start the API server on **http://localhost:8000**.

---

## API Usage
The FastAPI server exposes an endpoint for text generation.

### **Generate Text**
#### **Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time...", "max_tokens": 100}'
```

#### **Response:**
```json
{
  "generated_text": "Once upon a time, in a distant kingdom..."
}
```

---

## Optimization & Scaling
### **Enable Quantization** (Reduce memory usage)
Modify `app.py` to load a **quantized model**:
```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-2-13b-hf", quantization="awq")
```

### **Deploy at Scale** (Multi-GPU / Distributed Inference)
- **Use Ray Serve** for distributed inference.
- **Use KServe** for Kubernetes-based deployment.

---

## FAQ
### **Why use Uvicorn instead of Gunicorn?**
Gunicorn is not efficient for **GPU-based inference**, as it spawns multiple processes and duplicates memory. Uvicorn is **async and works better with FastAPI and vLLM**.

### **Why vLLM?**
- Optimized for **efficient LLaMA inference**
- Supports **batching & streaming**
- Reduces **VRAM usage**

### **Minimum System Requirements?**
- **GPU:** NVIDIA A100 / RTX 4090 / H100 (24GB+ VRAM recommended)
- **RAM:** 64GB+ for optimal performance

---

## Contributing
Feel free to submit issues and pull requests for improvements!

---

## License
This project follows the **LLaMA 2 Community License Agreement**.

