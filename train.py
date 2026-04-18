from src.training.trainer import Trainer

config = {
    "data_dir"         : "data/processed",
    "image_size"       : 128,
    "latent_dim"       : 256,
    "epochs"           : 200,
    "batch_size"       : 16,
    "lr"               : 1e-4,
    "beta"             : 0.5,
    "kl_warmup_epochs" : 80,
    "checkpoint_dir"   : "models/checkpoints",
    "sample_dir"       : "models/samples",
}

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()