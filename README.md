# BrainGFM

The official code of BrainGFM — a generative foundation model for computational brain imaging.

<p align="center">
  <img src="Fig/BrainGFM_banner.png" alt="BrainGFM - A Brain Imaging Foundation Model. A generative foundation model with a reusable and reconfigurable brain-imaging prior."/>
</p>

## News

- **2026-11** — Release pretrained weights, LoRA adaptation, and inference code.

## Brief introduction

Computational brain imaging is constrained by a mismatch between increasingly diverse tasks and scarce, heterogeneous clinical data. Although existing brain foundation models transfer pretrained representations, their knowledge remains entangled within model-specific features and fixed architectures.

We propose **BrainGFM**, a generative foundation model with a reusable and reconfigurable brain-imaging prior, developed on a corpus of about 1.3 million MRI and CT slices (BrainGFM-1.3M; T1 / T2 / T1ce / FLAIR / DWI / CT). This prior supports two complementary forms of downstream transfer: **task-aligned data synthesis** for low-data predictive learning, and **task-adapted post-training** for image reconstruction. Across 48 downstream tasks, BrainGFM improved segmentation and classification under limited supervision and achieved high-fidelity enhancement across heterogeneous imaging conditions. Multicenter validation in 4,796 stroke patients across six centers demonstrated its applicability and generalizability to clinically relevant stroke tasks. Blinded reader assessments showed that BrainGFM-generated FLAIR approached acquired FLAIR in overall diagnostic quality, and in a human–AI collaboration study, assistance yielded a 14% relative improvement in mean radiologist stroke-subtyping accuracy.

<p align="center">
  <img src="Fig/BrainGFM.png" alt="BrainGFM overview"/>
</p>

## 🏃 Playground: Quick start

### Dependencies

- Python 3.10
- PyTorch 2.7.0
- CUDA 12.6

**Step 1**: Install BrainGFM

`train_lora.py` and `infer_lora.py` both run on CUDA. Clone the repository and install the pinned packages, including PyTorch 2.7.0:

```bash
git clone https://github.com/YutingHe-list/BrainGFM
cd BrainGFM/
pip install -r requirements.txt
```

Then check that this environment can see the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Step 2**: Download the pretrained checkpoints

`configs/train_lora.yaml` loads the stage-1 codebook and the stage-2 D12 model from `pretrained/`.

| Weight | Download | Description |
| --- | --- | --- |
| `braingfm_pretrain.pth` | [link](https://huggingface.co/wenxuyyds/BrainGFM) | Stage-1 codebook |
| `braingfm_d12.pth` | [link](https://huggingface.co/wenxuyyds/BrainGFM) | Stage-2 generation model |

Download both checkpoints into `pretrained/`.

**Step 3**: Index the slice data

Put axial slices under `./data/BrainGFM`, using the layout in [`data/BrainGFM/README.md`](data/BrainGFM/README.md): one folder each for `T1`, `T1ce`, `T2`, `FLAIR`, `DWI`, `CT`, and `Unknow`, plus `Lable/Brain_mask` and `Lable/Lable_mask`. `rebuild_index_phase2.py` scans that directory and writes the train and validation indexes once:

```bash
python rebuild_index_phase2.py
```

The indexes are saved to `data/BrainGFM/.braingfm_index_train.json` and `data/BrainGFM/.braingfm_index_val.json`. Later training reads these files instead of scanning the slices again.

**Step 4**: Adapt one modality with LoRA

In `configs/train_lora.yaml`, set `lora_modality` to the modality you want to generate (`t1 | t2 | t1ce | flair | dwi | ct | unknow`). The default is `flair`, using 20 axial slices. The codebook and the D12 weights stay frozen; only the LoRA for that modality is trained.

```bash
python train_lora.py --config configs/train_lora.yaml
```

The run is saved to `experiments/lora_finetune/<timestamp-name>/`, and the checkpoint to load later is `best_lora.pth` in that folder.

**Step 5**: Generate slices with the adapted LoRA

`infer_lora.py` loads `best_lora.pth` on top of the stage-2 model. `--mode pair` generates aligned mask–image pairs. `--mode mask_cond` keeps the real few-shot masks and generates the corresponding images.

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out_dir synth/flair_100
```

Each sample is written into `synth/flair_100/` as `pair_XXXXXX_image.png`, `pair_XXXXXX_mask.png`, and `pair_XXXXXX_mask_idx.npy`.

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
