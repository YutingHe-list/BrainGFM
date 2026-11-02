# BrainGFM

The official code of BrainGFM — a generative foundation model for computational brain imaging.

<p align="center">
  <img src="Fig/BrainGFM_banner.png" alt="BrainGFM - A Brain Imaging Foundation Model. A generative foundation model with a reusable and reconfigurable brain-imaging prior."/>
</p>

## News

- **2026-11** — Release pretrained weights, LoRA adaptation, and inference code.

## Brief introduction

Computational brain imaging is expanding toward increasingly diverse tasks such as segmentation, disease characterization, and image enhancement. However, representative MRI and CT data remain scarce and heterogeneous across clinical centers, scan protocols, and brain diseases, limiting the clinical acceptability of downstream models. We propose a Brain imaging Generative Foundation Model (**BrainGFM**) with a reusable and reconfigurable brain-imaging prior. It has been pre-trained on BrainGFM-1.3M, a large-scale corpus of about 1.3 million MRI and CT slices, and is able to support task-aligned data synthesis for low-data predictive learning and task-adapted post-training for image reconstruction. Via the adaptation with few data, it can further achieve professional performance in real-world specific scenarios.

<p align="center">
  <img src="Fig/BrainGFM.png" alt="BrainGFM overview"/>
</p>

## 🏃 Playground: Quick start BrainGFM

## 1. Clone the repository and prepare environment

### Dependencies

- Python 3.9.21
- PyTorch 2.7.0
- CUDA 12.6

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

## 2. Download the pretrained checkpoint

| Weight | Download | Description |
| --- | --- | --- |
| `braingfm_pretrain.pth` | [link](https://huggingface.co/wenxuyyds/BrainGFM) | Pre-trained generative brain-imaging foundation model |

Download the checkpoint and save it to `pretrained/braingfm_pretrain.pth`.

## 3. Index the slice data

Put axial slices under `./data/BrainGFM`, using the layout in [`data/BrainGFM/README.md`](data/BrainGFM/README.md): one folder each for `T1`, `T1ce`, `T2`, `FLAIR`, `DWI`, `CT`, and `Unknow`, plus `Lable/Brain_mask` and `Lable/Lable_mask`. `rebuild_index_phase2.py` scans that directory and writes the train and validation indexes once:

```bash
python rebuild_index_phase2.py
```

The indexes are saved to `data/BrainGFM/.braingfm_index_train.json` and `data/BrainGFM/.braingfm_index_val.json`. Later training reads these files instead of scanning the slices again.

## 4. Adapt one modality with LoRA

In `configs/train_lora.yaml`, set `lora_modality` to the modality you want to generate (`t1 | t2 | t1ce | flair | dwi | ct | unknow`). The default is `flair`, using 20 axial slices. The codebook and the D12 weights stay frozen; only the LoRA for that modality is trained.

```bash
python train_lora.py --config configs/train_lora.yaml
```

The run is saved to `experiments/lora_finetune/<timestamp-name>/`, and the checkpoint to load later is `best_lora.pth` in that folder.

## 5. Generate slices with the adapted LoRA

`infer_lora.py` loads `best_lora.pth` on top of the stage-2 model. `--mode pair` generates aligned mask–image pairs. `--mode mask_cond` keeps the real few-shot masks and generates the corresponding images.

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out_dir synth/flair_100
```

Each sample is written into `synth/flair_100/` as `pair_XXXXXX_image.png`, `pair_XXXXXX_mask.png`, and `pair_XXXXXX_mask_idx.npy`.

## Enhancement pairs

The enhancement task learns to recover a clean slice from a degraded one, so each pair has to be pixel-aligned. Routine CT and MR are easy to collect; a second low-dose or low-field scan of the same subject usually is not. The degraded image is therefore simulated from the clean slice, with the two operators in `data_preprocess/`.

Low-dose CT noise is added where the photon count drops, in the projection domain. [`simulate_low_dose_ct.py`](data_preprocess/simulate_low_dose_ct.py) converts HU to attenuation, forward-projects a fan-beam sinogram, adds a Poisson approximation, and reconstructs with FBP. `dose_percent` is the retained dose. Draw one value per slice from the low-dose range you want to match; 5–40 is a strong setting. This step needs an NVIDIA GPU and `astra-toolbox`.

Low-field MR is approximated by the SNR loss of a magnitude image. [`simulate_low_field_mr.py`](data_preprocess/simulate_low_field_mr.py) scales the nonzero foreground to `[0, 1]` and adds Rician noise, `sigma = sqrt(delta_t * 0.005)`, then scales it back. `delta_t` is in `[0, 1]`; `0.5` is a moderate setting and `0.8` is stronger. Background voxels that are exactly 0 stay 0.

From the repository root, with a clean 2D slice in memory:

```python
from data_preprocess.simulate_low_dose_ct import low_dose_create
from data_preprocess.simulate_low_field_mr import add_rician_noise

low_dose_ct = low_dose_create(clean_hu, dose_percent=20)
noisy_mr = add_rician_noise(clean_mr, delta_t=0.5)
```

Save the result next to the clean slice as `slice_XXXX_ld.nii.gz` if an enhancement run should read the degraded image from disk.

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
