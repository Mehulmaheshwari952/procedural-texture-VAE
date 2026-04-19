import torch
from src.retrieval.retriever import TextureRetriever
from src.retrieval.neural_synthesis import StyleVariator

device    = "cuda" if torch.cuda.is_available() else "cpu"
retriever = TextureRetriever("data/processed", device=device)
variator  = StyleVariator(device=device)

query      = "rough rocky stone surface"
results    = retriever.retrieve(query, top_k=6)
paths      = [r["path"] for r in results]

print(f"Query: '{query}'")
print(f"Retrieved {len(results)} textures, synthesizing variations...\n")

# test_retrieval.py
variations = variator.vary(paths, num_variations=6, steps=200)

for i, img in enumerate(variations):
    fname = f"test_neural_{i}.png"
    img.save("RAG_results/" + fname)
    print(f"Saved {fname}")