import os
import io
import base64
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.retrieval.retriever import TextureRetriever
from src.retrieval.neural_synthesis import StyleVariator

app    = FastAPI(title="Texture Generator API")
device = "cuda" if torch.cuda.is_available() else "cpu"

# load models once at startup — not on every request
print("Loading models...")
retriever = TextureRetriever("data/processed", device=device)
variator  = StyleVariator(device=device)
print("Models ready.")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

class GenerateRequest(BaseModel):
    query           : str
    num_variations  : int = 6
    steps           : int = 80

class TextureResult(BaseModel):
    image_b64 : str          # base64 encoded PNG
    label     : str          # matched category
    score     : float        # CLIP similarity score

class GenerateResponse(BaseModel):
    query    : str
    textures : List[TextureResult]

def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/")
def root():
    return {"status": "ok", "message": "Texture Generator API is running"}

@app.get("/classes")
def get_classes():
    categories = sorted([
        d for d in os.listdir("data/processed")
        if os.path.isdir(os.path.join("data/processed", d))
    ])
    return {"classes": categories}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if req.num_variations < 1 or req.num_variations > 6:
        raise HTTPException(status_code=400, detail="num_variations must be between 1 and 6")

    if req.steps < 20 or req.steps > 200:
        raise HTTPException(status_code=400, detail="steps must be between 20 and 200")

    # retrieve top matching textures
    results = retriever.retrieve(req.query, top_k=req.num_variations)
    paths   = [r["path"] for r in results]

    # synthesize variations
    images  = variator.vary(paths, num_variations=req.num_variations, steps=req.steps)

    textures = []
    for img, result in zip(images, results):
        textures.append(TextureResult(
            image_b64 = pil_to_b64(img),
            label     = result["label"],
            score     = result["score"]
        ))

    return GenerateResponse(query=req.query, textures=textures)

@app.get("/ui")
def serve_ui():
    return FileResponse("templates/index.html")