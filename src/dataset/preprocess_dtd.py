import os
import shutil
import glob
from PIL import Image
from tqdm import tqdm

DTD_DIR       = "data/dtd/images"
PROCESSED_DIR = "data/processed"
IMAGE_SIZE    = 128

# map DTD categories to game-friendly names
# each game label maps to one or more DTD source folders
CATEGORY_MAP = {
    "grass"  : ["fibrous", "matted", "woven"],
    "rock"   : ["bumpy",   "porous", "cracked"],
    "wood"   : ["grooved", "lined",  "stratified"],
    "metal"  : ["shiny",   "polished", "flecked"],
    "fabric" : ["knitted", "braided", "gauzy"],
    "sand"   : ["sprinkled", "granular", "crystalline"],
    "lava"   : ["bubbly",  "molten",  "stained"],
    "ice"    : ["frosted", "veined",  "marbled"],
}

def preprocess_dtd():
    print("Processing DTD dataset...\n")

    for game_label, dtd_categories in CATEGORY_MAP.items():
        out_dir = os.path.join(PROCESSED_DIR, game_label)
        os.makedirs(out_dir, exist_ok=True)

        collected = 0
        for dtd_cat in dtd_categories:
            src_dir = os.path.join(DTD_DIR, dtd_cat)

            if not os.path.exists(src_dir):
                print(f"  Warning: DTD category '{dtd_cat}' not found, skipping")
                continue

            images = (
                glob.glob(os.path.join(src_dir, "*.jpg")) +
                glob.glob(os.path.join(src_dir, "*.png"))
            )

            for img_path in tqdm(images, desc=f"  {game_label} ← {dtd_cat}"):
                try:
                    img      = Image.open(img_path).convert("RGB")
                    img      = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                    out_name = f"{dtd_cat}_{os.path.basename(img_path)}"
                    img.save(os.path.join(out_dir, out_name))
                    collected += 1
                except Exception as e:
                    print(f"  Error: {img_path}: {e}")

        print(f"  {game_label}: {collected} images total\n")

    print("DTD preprocessing complete.")
    print(f"\nFinal counts:")
    for label in CATEGORY_MAP:
        count = len(glob.glob(os.path.join(PROCESSED_DIR, label, "*.jpg")))
        print(f"  {label}: {count}")

if __name__ == "__main__":
    preprocess_dtd()