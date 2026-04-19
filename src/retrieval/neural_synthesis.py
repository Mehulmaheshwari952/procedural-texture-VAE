import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # we extract from 4 layers at different depths
        # shallow layers = edges/colors
        # deep layers = complex texture patterns
        self.slice1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[18:27])# relu4_1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(feat):
    B, C, H, W = feat.shape
    feat = feat.view(B, C, -1)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (C * H * W)


class NeuralTextureSynthesizer:
    def __init__(self, device: str = "cuda"):
        self.device  = device
        self.vgg     = VGGFeatureExtractor().to(device).eval()

        self.to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

        self.denorm = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std =[1/0.229,      1/0.224,       1/0.225]
            )
        ])

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img).unsqueeze(0).to(self.device)

    def synthesize(self,
                style_path: str,
                num_steps: int = 200,
                style_weight: float = 1e6,
                variation_seed: int = 0) -> Image.Image:

        torch.manual_seed(variation_seed)

        style_tensor = self._load(style_path)

        with torch.no_grad():
            style_feats = self.vgg(style_tensor)
            style_grams = [gram_matrix(f) for f in style_feats]

        # initialize from source image + small noise
        # instead of pure random noise
        # this preserves large-scale structure while varying details
        noise  = torch.randn_like(style_tensor) * 0.1 * (variation_seed * 0.3 + 0.1)
        canvas = (style_tensor + noise).detach().requires_grad_(True)

        optimizer = optim.Adam([canvas], lr=0.02)

        for step in range(num_steps):
            optimizer.zero_grad()

            with torch.no_grad():
                canvas.data.clamp_(-2.5, 2.5)

            canvas_feats = self.vgg(canvas)
            canvas_grams = [gram_matrix(f) for f in canvas_feats]

            layer_weights = [1.0, 0.8, 0.5, 0.3]
            loss = sum(
                w * F.mse_loss(cg, sg)
                for w, cg, sg in zip(layer_weights, canvas_grams, style_grams)
            ) * style_weight

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out = canvas.squeeze(0).clamp(-2.5, 2.5)
            out = self.denorm(out).clamp(0, 1)
            out = transforms.ToPILImage()(out.cpu())

        return out


class StyleVariator:
    def __init__(self, device: str = "cuda"):
        self.device      = device
        self.synthesizer = NeuralTextureSynthesizer(device=device)

    def vary(self, image_paths: list,
            num_variations: int = 6,
            steps: int = 200) -> list:

        variations = []
        for i in range(min(num_variations, len(image_paths))):
            print(f"  Synthesizing variation {i+1}/{num_variations}...")
            img = self.synthesizer.synthesize(
                style_path     = image_paths[i],
                num_steps      = steps,
                variation_seed = i   # seed controls noise strength
            )
            variations.append(img)

        return variations