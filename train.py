from src.training.trainer import Trainer

config = {
    "data_dir"         : "data/processed",
    "image_size"       : 128,
    "latent_dim"       : 256,
    "epochs"           : 300,
    "batch_size"       : 16,
    "lr"               : 5e-5,        # lower lr for fine-tuning phase
    "beta"             : 0.5,
    "kl_warmup_epochs" : 80,
    "resume"           : "models/checkpoints/best.pt",  # resume from best
    "checkpoint_dir"   : "models/checkpoints",
    "sample_dir"       : "models/samples",
}

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()