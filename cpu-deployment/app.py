from fastapi import FastAPI,HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel
app = FastAPI()

# Load the model and tokenizer
model_path = "/app/llama-2-13b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda") #TODO:

@app.get("/")
def read_root():
    return {"message": "LLaMA 2-13B API is running"}

class GenerateTextRequest(BaseModel):
    prompt: str
    max_length: int = 200  # Default value
    temperature: float = 0.7

# @app.post("/generate/")
# def generate_text(request: GenerateTextRequest):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_length=200)
#     return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

@app.post("/generate/")
def generate_text(request: GenerateTextRequest):
    try:
        # Tokenize the input
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")

        # Generate text with more control over parameters
        with torch.no_grad():  # Prevent gradient computation for efficiency
            output = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,  # Enable sampling based on temperature
                top_p=0.95,  # Nucleus sampling
                num_return_sequences=1
            )

        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": generated_text}

    except Exception as e:
        # Log the error if you have a logging system setup
        raise HTTPException(status_code=500, detail=f"An error occurred while generating text: {str(e)}")