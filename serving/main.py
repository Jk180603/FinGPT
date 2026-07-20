"""
FinGPT-Tiny — FastAPI Serving
"""
import torch
import json
import math
import sys
sys.path.insert(0, ".")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from model.fingpt import FinGPT

app = FastAPI(
    title="FinGPT-Tiny API",
    description="GPT-style financial language model trained from scratch",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = Tokenizer.from_file("tokenizer/financial_bpe.json")
model = FinGPT(vocab_size=16000, dim=256, num_heads=8, num_layers=6, max_seq_len=256).to(device)
model.load_state_dict(torch.load("model/checkpoints/best_model.pt", map_location=device))
model.eval()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    max_new_tokens: int = Field(80, ge=10, le=300)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=5, le=200)

class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    new_tokens: int
    model: str

@app.get("/")
def root():
    results = {}
    try:
        with open("model/eval_results.json") as f:
            results = json.load(f).get("evaluation", {})
    except:
        pass
    return {
        "model": "FinGPT-Tiny",
        "parameters": "8,819,456",
        "architecture": "GPT with RoPE + RMSNorm + SwiGLU",
        "vocab_size": 16000,
        "training_corpus": "53.8MB financial text",
        "evaluation": results,
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        ids = tokenizer.encode(req.prompt).ids
        if len(ids) == 0:
            raise HTTPException(status_code=400, detail="Could not tokenize prompt")
        x = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.generate(
                x, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_k=req.top_k
            )
        full_text = tokenizer.decode(out[0].tolist())
        new_tokens = len(out[0]) - len(ids)
        return GenerateResponse(
            prompt=req.prompt,
            completion=full_text,
            new_tokens=new_tokens,
            model="FinGPT-Tiny-8.8M"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device), "model_loaded": True}

@app.get("/vocab-size")
def vocab():
    return {"vocab_size": tokenizer.get_vocab_size()}