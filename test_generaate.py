# save as test_generate.py in project root
import torch
from src.model.cvae import CVAE
from torchvision.utils import save_image

device     = torch.device("cuda")
checkpoint = torch.load("models/checkpoints/best.pt", map_location=device)

model = CVAE(
    latent_dim  = checkpoint["config"]["latent_dim"],
    num_classes = len(checkpoint["classes"]),
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

classes = checkpoint["classes"]
print("Available classes:", classes)
print()

# generate 4 samples for each class
for idx, name in enumerate(classes):
    imgs = model.generate(label=idx, num_samples=4, device=device)
    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)
    save_image(imgs, f"test_{name}.png", nrow=4)
    print(f"Saved test_{name}.png")