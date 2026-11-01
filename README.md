# BrainGFM

The official code of BrainGFM — A Brain Imaging Generative Foundation Model across 7 modalities (T1 / T2 / T1ce / FLAIR / DWI / CT / Unknown).

<p align="center">
  <img src="Fig/BrainGFM.png" alt="BrainGFM overview" width="90%"/>
</p>

> 📄 If the figure above does not render, you can view the full PDF here:
> [`Fig/BrainGFM.pdf`](Fig/BrainGFM.pdf)

📢 **BrainGFM paper** — *A Brain Imaging Generative Foundation Model across 7 Modalities* (TODO: add authors and BibTeX once the paper is public).

## News

- **2026-11** — Release inference & class-routed LoRA adaptation code.
- **2026-11** — Release Stage 1 (VAE codebook) & Stage 2 (D12 generation) training code.

## Brief introduction

Brain imaging across modalities (T1, T2, T1ce, FLAIR, DWI, CT, and unknown /
unassigned) is heterogeneous, scarce, and expensive to acquire. We propose
**BrainGFM**, a two-stage generative foundation model for brain images that
can jointly generate `(mask, image)` pairs across all 7 modalities, and
adapt to downstream tasks with only a handful of slices via class-routed
LoRA — *the modality you generate is the only modality you tune*, while all
other modalities' generation distributions are strictly preserved.

BrainGFM is composed of three parts:

1. **Stage 1 — VAE codebook** (`vae_ch160v4096z32`): a multi-scale residual
   vector-quantized autoencoder (`ch=160`, `vocab=4096`, `z=32`, 10 patch
   scales `[1,2,3,4,5,6,8,10,13,16]`) that tokenizes a 256×256 brain slice
   into a 10-scale token pyramid.
2. **Stage 2 — D12 generation model** (`d12`): a bidirectional,
   class-conditioned autoregressive transformer that jointly generates a
   `(mask, image)` pair via interleaved teacher forcing, with
   classifier-free guidance.
3. **Downstream adaptation — Class-routed LoRA**: a per-modality low-rank
   adaptation scheme where "the modality you generate is the only modality
   you tune"; all other modalities' LoRA rows stay at zero (`B=0`, no
   gradient), so their generation distributions are strictly preserved. A
   two-level task conditioning (modality → task subclass) is also supported
   for one-case-per-class adaptation.

## Directory layout

```
configs/        YAML configs: train_vae.yaml / train_d12.yaml / train_lora.yaml /
                task_cases_template.yaml
datasets/       BrainGFM slice dataset + few-shot selectors + degradation simulator
losses/         VQGAN + LPIPS + discriminator + Dice losses
models/         VAE codebook, D12 generation model, class-routed LoRA, transformer blocks
utils/          LR scheduling, misc helpers, W&B tracker
Tool/           Auxiliary eval / inspection scripts
Fig/            Overview figure (PDF / PNG)
docs/           DATA_PREP.md, WEIGHTS.md
train_vae.py    Stage 1: VAE codebook training
train_d12.py    Stage 2: D12 generation model training
train_lora.py   Downstream: class-routed LoRA fine-tuning
infer_lora.py   Downstream: synthetic data generation with LoRA
infer_multi_mode.py  Conditional + unconditional generation visualization
val_probe.py    Quick val CE / accuracy probe from a checkpoint
volume_recon.py Volume-level VAE reconstruction evaluation
eval_vae_compare.py  Compare pretrained vs fine-tuned VAE reconstruction
rebuild_index_phase2.py  Build train/val index JSON files from a data directory
merge_masks.py  Pre-merge brain mask + label mask into a single uint8 index map
dist.py         Distributed-training utilities
```

## Pretrained weights

All pretrained weights are stored under a single `pretrained/` directory at
the repository root, and all scripts / configs reference them via the
relative path `pretrained/<filename>`. You can override any path with the
`--vae_pretrained_path` / `--d12_pretrained_path` CLI flags or the
`vae_pretrained_path` / `d12_pretrained_path` config keys.

| Weight | Description |
| --- | --- |
| `vae_ch160v4096z32_braingfm.pth` | Stage 1 VAE codebook checkpoint (`ch=160`, `vocab=4096`, `z=32`, 10 patch scales) |
| `d12_braingfm.pth` | Stage 2 D12 generation model (depth=12, `embed_dim=768`, `num_heads=12`) checkpoint |
| `vgg.pth`, `vgg16-397923af.pth` | LPIPS perceptual weights (see `losses/lpips.py`) |

The repository does **not** bundle model weights. Place the files under
`pretrained/` before training. See [`docs/WEIGHTS.md`](docs/WEIGHTS.md) for
the download links and checkpoint format.

## 🏃 Playground: Quick start BrainGFM

### Dependencies

- Python 3.10
- PyTorch 2.7
- CUDA 12.6

Flash-Attention / xformers are optional (auto-detected; falls back to PyTorch
SDPA).

Step 1: Clone the repository

```bash
git clone https://github.com/<your-account>/BrainGFM
cd BrainGFM/
pip install -r requirements.txt
```

Step 2: Download the pretrained weights

Download the weights from the table above and place them under `pretrained/`
so that the directory looks like:

```
pretrained/
    vae_ch160v4096z32_braingfm.pth
    d12_braingfm.pth
    vgg.pth
    vgg16-397923af.pth
```

Step 3: Prepare the data

See [`docs/DATA_PREP.md`](docs/DATA_PREP.md) for the expected directory layout
under `./data/BrainGFM` and the image / mask formats. After placing the data,
build the train / val index (one-time):

```bash
python rebuild_index_phase2.py --root ./data/BrainGFM
```

This writes `.braingfm_index_train.json` and `.braingfm_index_val.json`
under `data/BrainGFM/`. The dataset loader auto-detects these files and
loads in seconds on subsequent runs.

## 1. Stage 1: VAE codebook training

```bash
python train_vae.py --config configs/train_vae.yaml
```

## 2. Stage 2: D12 generation model training

```bash
# Build the train/val index first (one-time, see Step 3 above)
python rebuild_index_phase2.py --root ./data/BrainGFM

# Train (multi-GPU example)
torchrun --nproc_per_node=2 train_d12.py --config configs/train_d12.yaml
```

## 3. Downstream: class-routed LoRA fine-tuning

Step 1: Edit `configs/train_lora.yaml` and set `lora_modality` to the
modality you want to adapt (one of
`t1 | t2 | t1ce | flair | dwi | ct | unknow`). The default setting adapts
on 20 axial slices (`lora_num_slices: 20`), matching the paper.

Step 2: Fine-tune the class-routed LoRA:

```bash
python train_lora.py --config configs/train_lora.yaml
```

The LoRA weights of the target modality will be stored under
`experiments/lora_finetune/<run>/best_lora.pth` (only the LoRA increment,
a few MB).

Step 3 (optional): Two-level task conditioning (modality → task subclass).
Uncomment the `task_cases_config`, `lora_route_sub` keys in
`configs/train_lora.yaml` and edit `configs/task_cases_template.yaml` to
enable one-case-per-class adaptation.

## 4. Inference: synthetic data generation

### 4.1 Generate with LoRA (downstream adaptation)

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out synth/flair_100
```

### 4.2 Conditional + unconditional generation visualization

```bash
python infer_multi_mode.py
```

## 5. Evaluation & inspection scripts

| Script | Description |
| --- | --- |
| `val_probe.py` | Quick val CE / accuracy probe from a checkpoint |
| `volume_recon.py` | Volume-level VAE reconstruction evaluation |
| `eval_vae_compare.py` | Compare pretrained vs fine-tuned VAE reconstruction |
| `merge_masks.py` | Pre-merge brain mask + label mask into a single uint8 index map |

## Ongoing

- ⬜ Release BrainGFM pretrained weights on Hugging Face.
- ⬜ Add BibTeX citation once the paper is public.
- ✅ Release inference & class-routed LoRA adaptation code.
- ✅ Release Stage 1 (VAE codebook) & Stage 2 (D12 generation) training code.

## Acknowledgements

This repository reuses code from the following projects:

- [taming-transformers](https://github.com/CompVis/taming-transformers) (MIT) — `VectorQuantizer2`, VQGAN losses
- [stable-diffusion / latent-diffusion](https://github.com/CompVis/latent-diffusion) (MIT) — VQ-VAE Encoder / Decoder
- [VAR](https://github.com/FoundationVision/VAR) — transformer backbone, AdaLN-SABlock, sampling helpers
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (BSD) — perceptual loss

See [`LICENSE`](LICENSE) for the license of this repository and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream licenses.

## Citation

If you use this code, please cite the BrainGFM paper (TODO: add BibTeX once
the paper is public).

```bibtex
@article{braingfm2026,
  title  = {A Brain Imaging Generative Foundation Model across 7 Modalities},
  author = {TODO},
  journal= {TODO},
  year   = {2026}
}
```

## About

The official code of BrainGFM — A Brain Imaging Generative Foundation Model
across 7 modalities (T1 / T2 / T1ce / FLAIR / DWI / CT / Unknown).
