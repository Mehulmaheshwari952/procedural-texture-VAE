import os
import requests
import zipfile
from tqdm import tqdm

RAW_DIR  = "data/raw"
BASE_URL = "https://ambientcg.com/get?file={asset_id}_1K-JPG.zip"
API_URL  = "https://ambientcg.com/api/v2/full_json?type=PhotoTexturePBR&limit=100&offset={offset}&category={category}"

CATEGORIES = {
    "grass"  : "Grass",
    "wood"   : "Wood",
    "rock"   : "Rock",
    "metal"  : "Metal",
    "fabric" : "Fabric",
    "ground" : "Ground",
}

def fetch_asset_ids(category_tag: str) -> list:
    """Query AmbientCG API to get all valid asset IDs for a category."""
    asset_ids = []
    offset    = 0

    while True:
        url      = API_URL.format(category=category_tag, offset=offset)
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"  API error for {category_tag}: {response.status_code}")
            break

        data   = response.json()
        assets = data.get("foundAssets", [])

        if not assets:
            break

        for asset in assets:
            asset_ids.append(asset["assetId"])

        # if we got fewer than 100 results, we've hit the end
        if len(assets) < 100:
            break

        offset += 100

    return asset_ids

def download_asset(asset_id: str, category: str):
    category_dir = os.path.join(RAW_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # skip if already extracted (check for any image file)
    existing = [
        f for f in os.listdir(category_dir)
        if asset_id in f and f.endswith((".jpg", ".png"))
    ]
    if existing:
        print(f"  Already exists: {asset_id}")
        return

    url      = BASE_URL.format(asset_id=asset_id)
    zip_path = os.path.join(category_dir, f"{asset_id}.zip")

    response = requests.get(url, stream=True, timeout=30)

    if response.status_code != 200:
        print(f"  Failed: {asset_id} (status {response.status_code})")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"  {asset_id}", leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(category_dir)
    except zipfile.BadZipFile:
        print(f"  Bad zip: {asset_id}")

    if os.path.exists(zip_path):
        os.remove(zip_path)

def download_all():
    print("Fetching available asset IDs from AmbientCG API...\n")

    for local_name, api_tag in CATEGORIES.items():
        print(f"\nCategory: {local_name} (tag: {api_tag})")

        asset_ids = fetch_asset_ids(api_tag)
        print(f"  Found {len(asset_ids)} assets: {asset_ids}")

        if not asset_ids:
            print(f"  Skipping — no assets found")
            continue

        for asset_id in asset_ids:
            download_asset(asset_id, local_name)

    print("\n\nDownload complete.")

if __name__ == "__main__":
    download_all()