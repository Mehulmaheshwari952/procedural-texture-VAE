# Procedural Texture Generator for Game Assets

A deep learning powered texture generation system built for indie game developers. Describe any surface in natural language and receive 6 unique, downloadable texture variations in seconds.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Deep Learning Components](#deep-learning-components)
- [Prior Work — Conditional VAE](#prior-work--conditional-vae)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Results](#results)
- [What We Learned](#what-we-learned)
- [Tech Stack](#tech-stack)

---

## Project Overview

This project started as an attempt to train a **Conditional Variational Autoencoder (cVAE)** from scratch to generate game textures. After extensive experimentation, we encountered fundamental limitations — posterior collapse, weak label conditioning, and insufficient data per class. Rather than abandoning the project, we documented the failure analysis and pivoted to a more practical and architecturally richer approach:

**CLIP-powered semantic retrieval + VGG19 neural texture synthesis.**

The final system accepts a natural language description like _"old cracked wood planks"_ and returns 6 genuinely synthesized texture variations, each unique and downloadable as a PNG.

---

## System Architecture

```
User Query (text)
       │
       ▼
┌─────────────────────┐
│   CLIP ViT-B/32     │  ← encodes query into 512-dim semantic vector
│  (OpenAI, 151M params)│
└─────────────────────┘
       │
       │  cosine similarity search
       ▼
┌─────────────────────┐
│   CLIP Index        │  ← 2280 pre-encoded texture images (512-dim each)
│   (numpy .npz)      │
└─────────────────────┘
       │
       │  top-6 most semantically similar image paths
       ▼
┌──────────────────────────┐
│  VGG19 Neural Synthesis  │  ← for each retrieved image:
│  (torchvision, 143M params)│   1. extract gram matrices from 4 VGG layers
│                          │   2. initialize canvas from source + noise
│                          │   3. optimize 200 steps via Adam
│                          │   4. output: genuinely new synthesized texture
└──────────────────────────┘
       │
       │  6 x PIL Images → base64 PNG
       ▼
┌─────────────────────┐
│   FastAPI Backend   │  ← /generate endpoint returns JSON
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  HTML/CSS/JS UI     │  ← gallery display + download buttons
└─────────────────────┘
```

---

## Deep Learning Components

### 1. CLIP — Semantic Text-Image Retrieval

**Model:** OpenAI ViT-B/32 (Vision Transformer, 151M parameters)
**Purpose:** Map both text queries and texture images into a shared 512-dimensional embedding space where semantically similar items cluster together.

**How it works at inference:**
- At startup, all 2280 dataset images are encoded once and stored as a numpy index
- When a user submits a query, the text is encoded into the same 512-dim space
- Cosine similarity is computed between the query vector and all image vectors
- Top-6 most similar images are retrieved in milliseconds

**Why CLIP works well for textures:**
CLIP was trained on 400 million image-text pairs from the internet. It already understands concepts like "rough", "cracked", "shiny", "mossy" in the context of surfaces and materials — no fine-tuning needed.

```python
# text encoding
text_features = clip_model.encode_text(clip.tokenize(["rough rocky stone"]))
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# cosine similarity against pre-computed image index
scores = (image_features @ text_features.T).squeeze()
top_6  = image_paths[np.argsort(scores)[::-1][:6]]
```

---

### 2. VGG19 Neural Texture Synthesis

**Model:** VGG19 (torchvision pretrained, 143M parameters)
**Purpose:** Synthesize genuinely new textures by optimizing an image to match the statistical texture signature of a retrieved reference.

**How gram matrix texture synthesis works:**

A gram matrix captures the correlation between feature channels at a given VGG layer. Two images that look visually similar at a texture level will have similar gram matrices even if their pixel values differ completely. By minimizing the gram matrix difference between a canvas and a target image, we synthesize a new image that has the same texture feel without being a copy.

```
Retrieved image → VGG19 → feature maps at 4 layers → gram matrices (target)

Random noise canvas → VGG19 → gram matrices (canvas)
                    ↑
              Adam optimizer minimizes:
              loss = Σ layer_weight * MSE(gram_canvas, gram_target)
              for 200 steps
```

**Layer weights used:**
```
relu1_1  →  weight 1.0   (edges, colors)
relu2_1  →  weight 0.8   (simple patterns)
relu3_1  →  weight 0.5   (complex patterns)
relu4_1  →  weight 0.3   (high-level structure)
```

**What makes each variation unique:**
- Each of the 6 variations starts from a different retrieved image (via CLIP)
- Each synthesis starts from the source image plus noise scaled by a unique seed
- Different seeds → different random starting points → different synthesized outputs
- The result is genuinely novel — not a copy, crop, or augmentation of any training image

---

## Prior Work — Conditional VAE

Before the CLIP pipeline, we spent significant time attempting to train a generative model from scratch. This work is preserved in `presentation/cvae_results/` and is an important part of the project narrative.

### What we built

A Conditional Variational Autoencoder with:
- **Encoder:** Pretrained ResNet18 backbone (frozen early layers, fine-tuned layer3 and layer4) + label embedding injected at the bottleneck
- **Decoder:** 5-layer transposed convolution decoder with residual blocks and InstanceNorm
- **Loss:** MSE reconstruction + VGG16 perceptual loss + KL divergence with sigmoid warmup annealing
- **Conditioning:** 64-dim class label embedding injected into both encoder and decoder

### What went wrong and why

**Posterior Collapse**
The KL divergence term successfully regularized the latent space to a unit Gaussian, but in doing so caused the decoder to stop using the latent vector `z` entirely. The decoder learned to output the dataset's mean color — a valid but useless solution to the optimization objective.

**Weak label conditioning**
The 64-dim label embedding contributed too small a gradient signal compared to the dominant reconstruction loss. The model learned "generate a plausible image" far faster than it learned "generate a class-specific image", and the label signal never caught up.

**Insufficient data per class**
2280 images / 8 classes = ~285 images per class. Effective class-conditional VAEs require 1000+ samples per class to learn distinguishable per-class distributions. Our dataset was simply too small.

**What we tried to fix it**
- KL annealing (linear → sigmoid warmup over 150 epochs)
- Log variance clamping to prevent KL explosion
- Weighted random sampling to balance classes
- Perceptual loss (VGG16 feature map comparison)
- Larger label embedding (6-dim → 32-dim → 64-dim)
- Learning rate reduction (1e-3 → 1e-4 → 5e-5 → 2e-5)
- ResNet18 pretrained encoder backbone
- Residual decoder with InstanceNorm

Despite all interventions, the model consistently converged to producing similar-looking outputs regardless of class label, confirming that the dataset size was the fundamental bottleneck.

### Lessons learned

VAEs are not well suited for high-frequency texture generation at small dataset scales. The architecture that actually works for this problem at scale is either a diffusion model (SOTA) or a retrieval-augmented system (practical). Both approaches sidestep the class-conditioning problem entirely by using richer conditioning signals (text embeddings via CLIP, or iterative denoising).

---

## Dataset

**Primary:** DTD — Describable Textures Dataset (Oxford VGG Group)
- 5,640 images across 47 perceptual texture categories
- We selected 8 categories and remapped them to game-relevant labels

**Category mapping:**

| Game Label | DTD Source Categories |
|---|---|
| grass | fibrous, matted, woven |
| rock | bumpy, porous, cracked |
| wood | grooved, lined, stratified |
| metal | shiny, polished, flecked |
| fabric | knitted, braided, gauzy |
| sand | sprinkled, granular, crystalline |
| lava | bubbly, stained |
| ice | frosted, veined, marbled |

**Final processed dataset:** 2280 images at 128×128px

**Supplementary:** AmbientCG (CC0 PBR textures) — used in early experiments, 105 images across 6 categories.

---

## Project Structure

```
DL_project/
│
├── data/
│   ├── raw/                    ← downloaded source textures
│   ├── processed/              ← resized, labeled textures (128x128)
│   │   ├── grass/
│   │   ├── rock/
│   │   ├── wood/
│   │   ├── metal/
│   │   ├── fabric/
│   │   ├── sand/
│   │   ├── lava/
│   │   └── ice/
│   └── dtd/                    ← raw DTD dataset
│
├── models/
│   ├── checkpoints/            ← cVAE training checkpoints
│   └── samples/                ← cVAE epoch sample images
│
├── presentation/
│   └── cvae_results/           ← cVAE outputs + failure analysis
│
├── src/
│   ├── dataset/
│   │   ├── download.py         ← AmbientCG downloader
│   │   ├── preprocess.py       ← AmbientCG preprocessor
│   │   ├── preprocess_dtd.py   ← DTD preprocessor + label mapper
│   │   └── texture_dataset.py  ← PyTorch Dataset class
│   │
│   ├── model/
│   │   └── cvae.py             ← cVAE architecture (ResNet encoder + residual decoder)
│   │
│   ├── training/
│   │   └── trainer.py          ← training loop, KL annealing, mixed precision
│   │
│   ├── retrieval/
│   │   ├── retriever.py        ← CLIP index builder + semantic search
│   │   └── neural_synthesis.py ← VGG19 gram matrix texture synthesis
│   │
│   └── api/
│       └── main.py             ← FastAPI backend
│
├── templates/
│   └── index.html              ← frontend UI
│
├── train.py                    ← cVAE training entry point
├── test_generate.py            ← cVAE inference test
├── test_retrieval.py           ← retrieval pipeline test
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (tested on RTX 3060 6GB)
- CUDA 12.1+

### 1. Clone the repository

```bash
git clone https://github.com/Mehulmaheshwari952/procedural-texture-VAE.git
cd procedural-texture-VAE
```

### 2. Create virtual environment

```bash
python -m venv dl_env
source dl_env/bin/activate        # Linux/Mac
# dl_env\Scripts\activate         # Windows
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```bash
pip install fastapi uvicorn pillow requests tqdm matplotlib numpy
pip install python-multipart aiofiles
pip install git+https://github.com/openai/CLIP.git
```

### 5. Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should print `True` and your GPU name.

### 6. Download and preprocess dataset

```bash
# Download DTD
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz
cd ..

# Preprocess into game categories
python src/dataset/preprocess_dtd.py

# Verify
find data/processed -name "*.jpg" | wc -l   # should print ~2280
```

---

## Running the Application

### Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

On first run, CLIP will encode all 2280 images and save a `clip_index.npz` file. This takes about 15 seconds and only happens once.

### Open the UI

```
http://localhost:8000/ui
```

### Example queries to try

- `rough rocky stone surface`
- `old cracked wood planks`
- `green mossy grass field`
- `shiny metallic surface`
- `dry cracked lava ground`
- `frozen ice crystal`
- `woven fabric texture`
- `wet sand on a beach`

---

## API Reference

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "message": "Texture Generator API is running" }
```

---

### `GET /classes`
Returns available texture categories in the dataset.

**Response:**
```json
{ "classes": ["fabric", "grass", "ice", "lava", "metal", "rock", "sand", "wood"] }
```

---

### `POST /generate`
Generate texture variations for a given query.

**Request body:**
```json
{
  "query": "rough rocky stone surface",
  "num_variations": 6,
  "steps": 200
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| query | string | required | Natural language texture description |
| num_variations | int | 6 | Number of textures to generate (1–6) |
| steps | int | 200 | Neural synthesis optimization steps (20–200). Higher = better quality, slower |

**Response:**
```json
{
  "query": "rough rocky stone surface",
  "textures": [
    {
      "image_b64": "<base64 encoded PNG>",
      "label": "rock",
      "score": 0.326
    },
    ...
  ]
}
```

| Field | Description |
|---|---|
| image_b64 | Base64-encoded PNG, decode directly into an `<img>` tag |
| label | Dataset category of the retrieved reference image |
| score | CLIP cosine similarity score (higher = more semantically relevant) |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "old cracked wood planks", "num_variations": 3, "steps": 150}'
```

---

## Results

The system correctly retrieves and synthesizes textures matching diverse natural language queries:

| Query | Retrieved categories | Visual result |
|---|---|---|
| "old cracked wood planks" | wood, rock | Cracked timber grain, tree rings, aged wood |
| "rough rocky stone surface" | rock, ice | Veined stone, porous rock, stratified surface |
| "shiny metallic surface" | metal, fabric | Polished silver, brushed metal tones |
| "dry cracked lava ground" | lava, rock | Dark cracked patterns, volcanic tones |

**Inference time** (RTX 3060, 200 steps, 6 variations): ~45–60 seconds total
**CLIP index load time**: ~1 second (after first build)
**CLIP index build time**: ~15 seconds (first run only)

---

## What We Learned

**On generative models:**
Training a cVAE from scratch requires significantly more data per class than most tutorials suggest. 285 images per class is insufficient for meaningful class-conditional generation at 128×128 resolution. Posterior collapse is a real and persistent failure mode that KL annealing only partially addresses.

**On pretrained models:**
CLIP's zero-shot understanding of texture semantics is remarkably good. A 151M parameter model trained on internet data generalizes to "rough rocky stone surface" without any fine-tuning. This is a practical lesson about when to train from scratch vs when to leverage existing representations.

**On neural texture synthesis:**
Gram matrix matching via VGG features is a principled, mathematically grounded way to transfer texture statistics. Starting optimization from the source image rather than pure noise dramatically improves the quality of synthesized outputs.

**On project design:**
Documenting failure is as valuable as documenting success. The cVAE experiments taught us why retrieval-augmented generation works better than pure generation at small data scales — that understanding is the real deliverable.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning framework | PyTorch 2.5.1 + CUDA 12.1 |
| Semantic retrieval | OpenAI CLIP ViT-B/32 |
| Texture synthesis | VGG19 (torchvision pretrained) |
| Generative model (prior work) | Custom cVAE with ResNet18 encoder |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML5 / CSS3 / JavaScript |
| Dataset | DTD (Oxford VGG) + AmbientCG |
| GPU | NVIDIA GeForce RTX 3060 Mobile 6GB |
| Training | Mixed precision (torch.amp) + Adam optimizer |

---

## Acknowledgements

- [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — Oxford Visual Geometry Group
- [AmbientCG](https://ambientcg.com) — CC0 PBR texture library
- [OpenAI CLIP](https://github.com/openai/CLIP) — Radford et al., 2021
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) — Gatys et al., 2015 (gram matrix texture synthesis)
