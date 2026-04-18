import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # use pretrained VGG16 as feature extractor
        # we extract features from 3 different depths:
        # shallow = edges/colors, mid = patterns, deep = structures
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.slice1 = nn.Sequential(*list(vgg.features)[:4])   # edges
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])  # patterns
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]) # structures

        # freeze VGG — we never update its weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # normalize to ImageNet stats that VGG expects
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(generated.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(generated.device)

        # convert from [-1,1] to [0,1] then normalize for VGG
        gen = ((generated + 1) / 2 - mean) / std
        tgt = ((target    + 1) / 2 - mean) / std

        loss = 0.0
        for slice_ in [self.slice1, self.slice2, self.slice3]:
            gen = slice_(gen)
            tgt = slice_(tgt)
            loss += F.mse_loss(gen, tgt)

        return loss


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()

        # pretrained ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove the final classification head
        # output of this is (B, 512, 4, 4) for 128x128 input
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # freeze early layers — they already know edges and textures
        # only fine-tune the last two blocks
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # label embedding injected into bottleneck
        self.label_embedding = nn.Embedding(num_classes, 64)

        # project from ResNet features + label to latent params
        # 512 * 4 * 4 = 8192 from ResNet + 64 from label = 8256
        self.flatten_dim = 512 * 4 * 4
        self.fc_mu      = nn.Linear(self.flatten_dim + 64, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim + 64, latent_dim)

        # layer norm stabilizes the latent space
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, label):
        features  = self.backbone(x)                         # (B, 512, 4, 4)
        features  = features.view(features.size(0), -1)      # (B, 8192)

        label_emb = self.label_embedding(label)              # (B, 64)
        combined  = torch.cat([features, label_emb], dim=1)  # (B, 8256)

        mu      = self.norm(self.fc_mu(combined))
        log_var = torch.clamp(self.fc_log_var(combined), min=-4, max=4)

        return mu, log_var


class TextureDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, 64)

        # project latent + label into spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 64, 512 * 4 * 4),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            # 512 x 4 x 4 → 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256 x 8 x 8 → 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 16 x 16 → 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 32 x 32 → 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 x 64 x 64 → 3 x 128 x 128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, label):
        label_emb = self.label_embedding(label)          # (B, 64)
        z         = torch.cat([z, label_emb], dim=1)     # (B, latent_dim + 64)
        x         = self.fc(z)                           # (B, 512*4*4)
        x         = x.view(x.size(0), 512, 4, 4)        # (B, 512, 4, 4)
        x         = self.deconv(x)                       # (B, 3, 128, 128)
        return x


class CVAE(nn.Module):
    def __init__(self, latent_dim: int = 256, num_classes: int = 6, image_size: int = 128):
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        self.encoder = ResNetEncoder(latent_dim, num_classes)
        self.decoder = TextureDecoder(latent_dim, num_classes)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, label):
        mu, log_var = self.encoder(x, label)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decoder(z, label)
        return x_recon, mu, log_var

    @torch.no_grad()
    def generate(self, label, num_samples: int, device):
        self.eval()
        z     = torch.randn(num_samples, self.latent_dim).to(device)
        label = torch.tensor([label] * num_samples).to(device)
        imgs  = self.decoder(z, label)
        return imgs


# module level perceptual loss — instantiated once, reused every batch
_perceptual_loss_fn = None

def get_perceptual_loss_fn(device):
    global _perceptual_loss_fn
    if _perceptual_loss_fn is None:
        _perceptual_loss_fn = PerceptualLoss().to(device)
    return _perceptual_loss_fn

def cvae_loss(x_recon, x, mu, log_var, beta: float = 1.0, device=None):
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")

    perc_fn   = get_perceptual_loss_fn(device or x.device)
    perc_loss = perc_fn(x_recon, x) * 10.0

    kl_loss   = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # edge consistency loss — forces decoder to generate
    # uniform texture all the way to the borders, no vignetting
    top    = x_recon[:, :, :4,  :]      # (B, 3, 4, 128)
    bottom = x_recon[:, :, -4:, :]      # (B, 3, 4, 128)
    left   = x_recon[:, :, :,  :4]      # (B, 3, 128, 4)
    right  = x_recon[:, :, :, -4:]      # (B, 3, 128, 4)

    # use mean of center region as reference, not expand
    center_mean = x_recon[:, :, 62:66, 62:66].mean(dim=[2,3], keepdim=True)  # (B, 3, 1, 1)

    edge_loss = (
        F.mse_loss(top,    center_mean.expand_as(top))    +
        F.mse_loss(bottom, center_mean.expand_as(bottom)) +
        F.mse_loss(left,   center_mean.expand_as(left))   +
        F.mse_loss(right,  center_mean.expand_as(right))
    ) * 5.0

    total = recon_loss + perc_loss + beta * kl_loss + edge_loss
    return total, recon_loss, kl_loss