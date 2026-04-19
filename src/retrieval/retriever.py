import os
import glob
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

class TextureRetriever:
    def __init__(self, processed_dir: str, device: str = "cuda", model_name: str = "ViT-B/32"):
        self.device       = device
        self.processed_dir = processed_dir

        print(f"Loading CLIP {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # storage
        self.image_paths    = []
        self.image_labels   = []
        self.image_features = None   # (N, 512) normalized CLIP embeddings

        # build or load index
        self.index_path = os.path.join(processed_dir, "clip_index.npz")
        if os.path.exists(self.index_path):
            print("Loading existing CLIP index...")
            self._load_index()
        else:
            print("Building CLIP index from scratch...")
            self._build_index()

    def _build_index(self):
        all_images  = []
        all_labels  = []

        categories = sorted([
            d for d in os.listdir(self.processed_dir)
            if os.path.isdir(os.path.join(self.processed_dir, d))
        ])

        for category in categories:
            cat_dir = os.path.join(self.processed_dir, category)
            images  = (
                glob.glob(os.path.join(cat_dir, "*.jpg")) +
                glob.glob(os.path.join(cat_dir, "*.png"))
            )
            for img_path in images:
                all_images.append(img_path)
                all_labels.append(category)

        print(f"Encoding {len(all_images)} images with CLIP...")
        features = []

        with torch.no_grad():
            for img_path in tqdm(all_images):
                try:
                    img     = Image.open(img_path).convert("RGB")
                    tensor  = self.preprocess(img).unsqueeze(0).to(self.device)
                    feat    = self.model.encode_image(tensor)
                    feat    = feat / feat.norm(dim=-1, keepdim=True)  # normalize
                    features.append(feat.cpu().numpy())
                except Exception as e:
                    print(f"Error encoding {img_path}: {e}")
                    all_labels.pop()
                    continue

        self.image_paths    = all_images
        self.image_labels   = all_labels
        self.image_features = np.vstack(features)  # (N, 512)

        # save index so we don't rebuild every time
        np.savez(
            self.index_path,
            features = self.image_features,
            paths    = np.array(self.image_paths),
            labels   = np.array(self.image_labels)
        )
        print(f"Index built and saved: {self.image_features.shape}")

    def _load_index(self):
        data = np.load(self.index_path, allow_pickle=True)
        self.image_features = data["features"]
        self.image_paths    = list(data["paths"])
        self.image_labels   = list(data["labels"])
        print(f"Index loaded: {self.image_features.shape}")

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 6) -> list:
        # encode text query
        text    = clip.tokenize([query]).to(self.device)
        feat    = self.model.encode_text(text)
        feat    = feat / feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy()  # (1, 512)

        # cosine similarity — both are normalized so dot product = cosine sim
        scores  = (self.image_features @ feat_np.T).squeeze()  # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "path"  : self.image_paths[idx],
                "label" : self.image_labels[idx],
                "score" : float(scores[idx])
            })

        return results