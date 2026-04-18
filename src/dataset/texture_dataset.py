import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextureDataset(Dataset):
    def __init__(self, processed_dir: str, image_size: int = 128):
        self.image_size = image_size
        self.samples    = []   # list of (image_path, label_index)
        self.classes    = []   # list of category names

        # build class list from folder names
        self.classes = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # collect all image paths with their label index
        for category in self.classes:
            cat_dir = os.path.join(processed_dir, category)
            images  = (
                glob.glob(os.path.join(cat_dir, "*.jpg"))  +
                glob.glob(os.path.join(cat_dir, "*.jpeg")) +
                glob.glob(os.path.join(cat_dir, "*.png"))
            )
            for img_path in images:
                self.samples.append((img_path, self.class_to_idx[category]))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),           # textures are rotation-invariant
            transforms.RandomResizedCrop(                    # random zoom and crop
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),              # occasional grayscale
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std =[0.5, 0.5, 0.5]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def get_label_name(self, idx: int) -> str:
        return self.classes[idx]