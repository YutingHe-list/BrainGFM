# Third-Party Notices

This repository incorporates code and/or pretrained weights from the
following open-source projects. Licenses are reproduced below in summary;
see each project's repository for the full text.

## taming-transformers — https://github.com/CompVis/taming-transformers

Used in: `losses/vqperceptual.py`, `losses/discriminator.py`,
`models/quant.py` (VectorQuantizer2).

License: MIT — Copyright (c) 2021 Patrick Esser and Robin Rombach.

## latent-diffusion / stable-diffusion — https://github.com/CompVis/latent-diffusion

Used in: `models/vae_modules.py` (Encoder/Decoder/ResnetBlock/AttnBlock),
`models/vae.py`.

License: MIT — Copyright (c) 2022 Robin Rombach and Patrick Esser.

## VAR (Visual Autoregressive Modeling) — https://github.com/FoundationVision/VAR

Used in: `models/basic_blocks.py` (FFN/SelfAttention/SABlock/AdaLNSABlock),
`models/d12.py`, `models/helpers.py` (sampling utilities).

License: MIT — Copyright (c) 2024 FoundationVision.

## LPIPS — https://github.com/richzhang/PerceptualSimilarity

Used in: `losses/lpips.py` (perceptual loss network and linear layers).
Pretrained weights (`vgg.pth`, `vgg16-397923af.pth`) are loaded locally from
`pretrained/` (no network download).

License: BSD 3-Clause — Copyright (c) 2018 Richard Zhang.
