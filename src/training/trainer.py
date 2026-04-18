import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.model.cvae import CVAE, cvae_loss
from src.dataset.texture_dataset import TextureDataset

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # directories
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["sample_dir"],     exist_ok=True)

        # dataset
        full_dataset = TextureDataset(
            processed_dir=config["data_dir"],
            image_size=config["image_size"]
        )

        # save class info for later use in inference
        self.classes      = full_dataset.classes
        self.num_classes  = len(self.classes)
        print(f"Classes found : {self.classes}")
        print(f"Total images  : {len(full_dataset)}")

        # train / val split — 90% train, 10% val
        val_size   = max(1, int(0.1 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        labels      = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
        class_counts = np.bincount(labels)
        print(f"Class counts: { {full_dataset.classes[i]: class_counts[i] for i in range(len(class_counts))} }")

        weights     = [1.0 / class_counts[labels[i]] for i in range(len(full_dataset))]
        train_weights = [weights[i] for i in train_ds.indices]
        sampler     = WeightedRandomSampler(
            weights     = train_weights,
            num_samples = len(train_ds),
            replacement = True
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size  = config["batch_size"],
            sampler     = sampler,       # replaces shuffle=True
            num_workers = 2,
            pin_memory  = True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size  = config["batch_size"],
            shuffle     = False,
            num_workers = 2,
            pin_memory  = True
        )

        # model
        self.model = CVAE(
            latent_dim=config["latent_dim"],
            num_classes=self.num_classes,
            image_size=config["image_size"]
        ).to(self.device)

        # optimizer — Adam works well for VAEs
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=1e-5
        )

        # learning rate scheduler — reduces LR when val loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=5,
            factor=0.5,
            verbose=True
        )

        # mixed precision scaler — uses your GPU's tensor cores
        # cuts VRAM usage nearly in half, speeds up training
        self.scaler = torch.amp.GradScaler('cuda')

        # tracking
        self.train_losses = []
        self.val_losses   = []
        self.best_val     = float("inf")

    def get_beta(self, epoch: int) -> float:
        # sigmoid warmup — stays near 0 for a long time then smoothly rises
        # much more stable than linear warmup for VAEs
        import math
        warmup_epochs = self.config.get("kl_warmup_epochs", 100)
        target_beta   = self.config.get("beta", 0.2)
        # sigmoid centered at halfway point of warmup
        x = (epoch - warmup_epochs / 2) / (warmup_epochs / 10)
        scale = 1 / (1 + math.exp(-x))
        return target_beta * scale

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss  = 0
        total_recon = 0
        total_kl    = 0

        beta = self.get_beta(epoch)

        for batch_idx, (imgs, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} | beta={beta:.3f}")):
            imgs   = imgs.to(self.device,   non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                x_recon, mu, log_var = self.model(imgs, labels)
                loss, recon, kl = cvae_loss(
                    x_recon, imgs, mu, log_var,
                    beta=beta,
                    device=self.device
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()

        n = len(self.train_loader)
        return total_loss / n, total_recon / n, total_kl / n

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_loss = 0

        for imgs, labels in self.val_loader:
            imgs   = imgs.to(self.device,   non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                x_recon, mu, log_var = self.model(imgs, labels)
                loss, _, _ = cvae_loss(
                    x_recon, imgs, mu, log_var,
                    beta=self.config["beta"],
                    device=self.device
                )
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def save_samples(self, epoch: int):
        self.model.eval()
        samples = []

        # generate 4 samples per class
        for class_idx in range(self.num_classes):
            imgs = self.model.generate(
                label=class_idx,
                num_samples=4,
                device=self.device
            )
            samples.append(imgs)

        # stack all into one grid image
        all_samples = torch.cat(samples, dim=0)

        # denormalize from [-1,1] back to [0,1] for saving
        all_samples = (all_samples + 1) / 2
        all_samples = all_samples.clamp(0, 1)

        save_image(
            all_samples,
            os.path.join(self.config["sample_dir"], f"epoch_{epoch:04d}.png"),
            nrow=4
        )

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        checkpoint = {
            "epoch":       epoch,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "val_loss":    val_loss,
            "classes":     self.classes,
            "config":      self.config
        }

        # always save latest
        torch.save(checkpoint, os.path.join(self.config["checkpoint_dir"], "latest.pt"))

        # save best separately
        if is_best:
            torch.save(checkpoint, os.path.join(self.config["checkpoint_dir"], "best.pt"))
            print(f"  New best model saved (val loss: {val_loss:.2f})")

    def plot_losses(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label="train")
        plt.plot(self.val_losses,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training curves")
        plt.legend()
        plt.savefig(os.path.join(self.config["sample_dir"], "loss_curve.png"))
        plt.close()

    def train(self):
        print(f"\nStarting training for {self.config['epochs']} epochs\n")

        for epoch in range(1, self.config["epochs"] + 1):
            # train
            train_loss, recon, kl = self.train_epoch(epoch)

            # validate
            val_loss = self.val_epoch()

            # track
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # scheduler step
            self.scheduler.step(val_loss)

            # print
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.1f} | "
                f"Recon: {recon:.1f} | "
                f"KL: {kl:.1f} | "
                f"Val: {val_loss:.1f}"
            )

            # save sample images every 10 epochs
            if epoch % 10 == 0:
                self.save_samples(epoch)
                self.plot_losses()

            # save checkpoint
            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)

        # final samples
        self.save_samples(self.config["epochs"])
        self.plot_losses()
        print("\nTraining complete.")