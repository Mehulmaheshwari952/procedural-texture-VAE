# cVAE Results — Why It Didn't Work

## What we built
A Conditional Variational Autoencoder with:
- Pretrained ResNet18 encoder backbone (frozen early layers)
- Residual decoder with InstanceNorm
- Perceptual loss via VGG16 feature maps
- KL annealing with sigmoid warmup
- Trained on 2280 DTD images across 8 classes

## What went wrong and why

### 1. Posterior Collapse
The KL divergence term forced the latent space to become
a unit gaussian, causing the decoder to ignore z entirely
and output the dataset mean color.

### 2. Weak label conditioning
The class label embedding (64-dim) was too small relative
to the reconstruction signal. The model learned to generate
plausible textures before it learned to generate class-specific
ones — and never needed to improve further.

### 3. Dataset still too small per class
2280 images / 8 classes = ~285 per class after augmentation.
Effective texture VAEs need 1000+ per class for meaningful
class-conditional generation.

## What we learned
- VAEs are not well suited for high-frequency texture generation
- Perceptual loss helps sharpness but doesn't fix conditioning
- The right approach for texture generation is either:
  a) Diffusion models (SOTA)
  b) Retrieval + style transfer (practical, fast)