# BrainGFM

The official code of BrainGFM — A Brain Imaging Generative Foundation Model across 7 modalities (T1 / T2 / T1ce / FLAIR / DWI / CT / Unknown).

<p align="center">
  <img src="Fig/BrainGFM.png" alt="BrainGFM overview" width="90%"/>
</p>

> 📄 If the figure above does not render, you can view the full PDF here:
> [`Fig/BrainGFM.pdf`](Fig/BrainGFM.pdf)

📢 **BrainGFM paper** — *A Brain Imaging Generative Foundation Model across 7 Modalities* (TODO: add authors and BibTeX once the paper is public).

## News

- **2026-11** — Release pretrained weights, LoRA adaptation, and inference code.

## Brief introduction

Computational brain imaging is constrained by a mismatch between increasingly diverse tasks and scarce, heterogeneous clinical data. Although existing brain foundation models transfer pretrained representations, their knowledge remains entangled within model-specific features and fixed architectures.

We propose **BrainGFM**, a generative foundation model with a reusable and reconfigurable brain-imaging prior, developed on a corpus of about 1.3 million MRI and CT slices (BrainGFM-1.3M; T1 / T2 / T1ce / FLAIR / DWI / CT). This prior supports two complementary forms of downstream transfer: **task-aligned data synthesis** for low-data predictive learning, and **task-adapted post-training** for image reconstruction. Across 48 downstream tasks, BrainGFM improved segmentation and classification under limited supervision and achieved high-fidelity enhancement across heterogeneous imaging conditions. Multicenter validation in 4,796 stroke patients across six centers demonstrated its applicability and generalizability to clinically relevant stroke tasks. Blinded reader assessments showed that BrainGFM-generated FLAIR approached acquired FLAIR in overall diagnostic quality, and in a human–AI collaboration study, assistance yielded a 14% relative improvement in mean radiologist stroke-subtyping accuracy.

## Pretrained weights

BrainGFM pretrained weights are hosted on Hugging Face:

👉 **[wenxuyyds/BrainGFM](https://huggingface.co/wenxuyyds/BrainGFM)**

Download the weights and place them in the `pretrained/` folder at the repository root.

## 🏃 Playground: Quick start

### Dependencies

- Python 3.10
- PyTorch 2.7
- CUDA 12.6

Step 1: Clone the repository

```bash
git clone https://github.com/<your-account>/BrainGFM
cd BrainGFM/
pip install -r requirements.txt
```

Step 2: Download the pretrained weights

Download from [wenxuyyds/BrainGFM](https://huggingface.co/wenxuyyds/BrainGFM) and put the files into `pretrained/`.

Step 3: Prepare your data

Place your slices under `./data/BrainGFM` following [`docs/DATA_PREP.md`](docs/DATA_PREP.md), then build the index (one-time):

```bash
python rebuild_index_phase2.py --root ./data/BrainGFM
```

## 1. LoRA fine-tuning

Edit `configs/train_lora.yaml` and set `lora_modality` to the modality you want to adapt (`t1 | t2 | t1ce | flair | dwi | ct | unknow`). Then run:

```bash
python train_lora.py --config configs/train_lora.yaml
```

The adapted LoRA weights will be saved under `experiments/lora_finetune/`.

## 2. Generation

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out synth/flair_100
```

## Acknowledgements

This repository reuses code from the following projects:

- [taming-transformers](https://github.com/CompVis/taming-transformers) (MIT)
- [stable-diffusion / latent-diffusion](https://github.com/CompVis/latent-diffusion) (MIT)
- [VAR](https://github.com/FoundationVision/VAR)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (BSD)

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

If you use this code, please cite the BrainGFM paper (TODO: add BibTeX once the paper is public).

```bibtex
@article{braingfm2026,
  title  = {A Brain Imaging Generative Foundation Model across 7 Modalities},
  author = {TODO},
  journal= {TODO},
  year   = {2026}
}
```
