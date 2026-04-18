# run in python shell
import torch
from src.model.cvae import CVAE, cvae_loss

device = torch.device("cuda")
model  = CVAE(latent_dim=256, num_classes=6).to(device)

x     = torch.randn(4, 3, 128, 128).to(device)
label = torch.zeros(4, dtype=torch.long).to(device)

x_recon, mu, log_var = model(x, label)
loss, recon, kl      = cvae_loss(x_recon, x, mu, log_var, device=device)

print(f"Output : {x_recon.shape}")
print(f"Loss   : {loss.item():.2f}")
print(f"Recon  : {recon.item():.2f}")
print(f"KL     : {kl.item():.2f}")