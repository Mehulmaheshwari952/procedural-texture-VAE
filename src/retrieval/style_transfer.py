import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random


class StyleVariator:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def vary(self, image_paths: list, num_variations: int = 6) -> list:
        """
        Takes a list of retrieved image paths and creates
        visually distinct variations using PIL augmentations.
        Each input image becomes one variation with different
        color/contrast/sharpness treatment.
        """
        variations = []
        augment_params = [
            {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "sharpness": 1.0, "rotate": 0},
            {"brightness": 1.2, "contrast": 1.1, "saturation": 0.8, "sharpness": 1.5, "rotate": 90},
            {"brightness": 0.8, "contrast": 1.3, "saturation": 1.2, "sharpness": 0.8, "rotate": 0},
            {"brightness": 1.1, "contrast": 0.9, "saturation": 1.4, "sharpness": 1.2, "rotate": 180},
            {"brightness": 0.9, "contrast": 1.2, "saturation": 0.7, "sharpness": 2.0, "rotate": 90},
            {"brightness": 1.3, "contrast": 1.0, "saturation": 1.1, "sharpness": 0.6, "rotate": 270},
        ]

        for i in range(min(num_variations, len(image_paths))):
            img    = Image.open(image_paths[i]).convert("RGB")
            img    = img.resize((256, 256), Image.LANCZOS)
            params = augment_params[i % len(augment_params)]

            # apply augmentations
            img = ImageEnhance.Brightness(img).enhance(params["brightness"])
            img = ImageEnhance.Contrast(img).enhance(params["contrast"])
            img = ImageEnhance.Color(img).enhance(params["saturation"])
            img = ImageEnhance.Sharpness(img).enhance(params["sharpness"])

            if params["rotate"] != 0:
                img = img.rotate(params["rotate"])

            variations.append(img)

        return variations

    def _shift_hue(self, tensor, shift):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

        # denormalize
        img = tensor * std + mean
        img = img.clamp(0, 1)

        # simple hue shift via channel rotation
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        img[:, 0] = (r + shift).clamp(0, 1)
        img[:, 2] = (b - shift).clamp(0, 1)

        # renormalize
        return (img - mean) / std