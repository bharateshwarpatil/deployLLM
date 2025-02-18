from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the model and tokenizer
model_path = "/app/llama-2-13b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

@app.get("/")
def read_root():
    return {"message": "LLaMA 2-13B API is running"}

@app.post("/generate/")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}