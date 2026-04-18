import os
import glob
from PIL import Image
from tqdm import tqdm

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
IMAGE_SIZE    = 128

VALID_SUFFIXES = ["Color", "color", "BaseColor", "Albedo", "albedo"]

def is_color_map(filename: str) -> bool:
    return any(s in filename for s in VALID_SUFFIXES)

def process_category(category: str):
    src_dir = os.path.join(RAW_DIR, category)
    dst_dir = os.path.join(PROCESSED_DIR, category)
    os.makedirs(dst_dir, exist_ok=True)

    all_images = (
        glob.glob(os.path.join(src_dir, "**/*.jpg"),  recursive=True) +
        glob.glob(os.path.join(src_dir, "**/*.jpeg"), recursive=True) +
        glob.glob(os.path.join(src_dir, "**/*.png"),  recursive=True)
    )

    color_maps = [f for f in all_images if is_color_map(os.path.basename(f))]

    print(f"  {category}: found {len(color_maps)} color maps")

    for img_path in tqdm(color_maps, desc=f"  Processing {category}"):
        try:
            img      = Image.open(img_path).convert("RGB")
            img      = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            out_name = os.path.basename(img_path)
            img.save(os.path.join(dst_dir, out_name))
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

def preprocess_all():
    print("Preprocessing dataset...\n")
    categories = os.listdir(RAW_DIR)
    for category in categories:
        if os.path.isdir(os.path.join(RAW_DIR, category)):
            process_category(category)
    print("\nPreprocessing complete.")

if __name__ == "__main__":
    preprocess_all()