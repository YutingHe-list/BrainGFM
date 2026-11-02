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

## 2. Download the pretrained checkpoints

Download the weights from [Hugging Face](https://huggingface.co/wenxuyyds/BrainGFM) and put them in a `pretrained` folder at the repository root:

```
pretrained/braingfm_pretrain.pth
pretrained/braingfm_d12.pth
```

`braingfm_pretrain.pth` is the codebook. `braingfm_d12.pth` is the generative model. Fine-tuning and inference both read these two files. This folder is not committed.

The LoRA trained in the next step is written separately to `experiments/lora_finetune/<run>/best_lora.pth`.

## 3. Adapt one modality with LoRA

Put a few axial slices of the modality you want to adapt under `data/BrainGFM`. Each slice needs a brain mask with the same file name:

```
data/BrainGFM/FLAIR/<cohort>/<case>/slice_XXXX.nii.gz
data/BrainGFM/Lable/Brain_mask/<cohort>/<case>/slice_XXXX.nii.gz
```

T1, T1ce, T2, DWI, CT, and Unknow use the same layout, with the modality folder name changed. The default run takes 20 FLAIR slices.

Remove the skull before saving these slices. MR uses [HD-BET](https://github.com/MIC-DKFZ/HD-BET) (Isensee et al., 2019). CT uses [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (Hoopes et al., 2022).

Fine-tuning then scales each slice to -1 to 1, and the inverse maps a generated image back to the original intensity. With the pretrained weights, the two scripts below are enough to fine-tune and run inference.

In `configs/train_lora.yaml`, set `lora_modality` to `t1 | t2 | t1ce | flair | dwi | ct | unknow`. The weights stay frozen; only the LoRA for that modality is trained.

```bash
python train_lora.py --config configs/train_lora.yaml
```

The checkpoint for the next step is `experiments/lora_finetune/<timestamp-name>/best_lora.pth`.

## 4. Inference

Fine-tune on a small set of labeled slices. The adapted model can then generate masks with no image input. Each of those masks is passed through the same fine-tuned weights to generate the corresponding image. Remove the brain-region label from the mask, and the remaining labels together with that image are a labeled pair.

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out_dir synth/flair_100
```

Each sample is written into the output folder as the generated mask, the generated image, and the mask index.

## Enhancement pairs

One pair is a clean slice and a degraded slice of the same cut. The degraded slice is simulated from the clean one.

**How to build a pair.** For CT, pass a 2D HU slice to [`low_dose_create`](data_preprocess/simulate_low_dose_ct.py). `dose_percent=20` keeps 20% of a routine dose; use 5–40 for a clearly low-dose image. For MR, pass a 2D magnitude slice to [`add_rician_noise`](data_preprocess/simulate_low_field_mr.py). `delta_t` runs from 0 to 1; 0.5 is a moderate setting and 0.8 is stronger.

The CT function needs an NVIDIA GPU and the ASTRA toolbox, which is not in `requirements.txt`. MR noise does not need it.

```bash
conda install -c astra-toolbox astra-toolbox
```

Install notes are in the [ASTRA toolbox documentation](https://astra-toolbox.com/docs/install.html).

```python
from data_preprocess.simulate_low_dose_ct import low_dose_create
from data_preprocess.simulate_low_field_mr import add_rician_noise

low_dose_ct = low_dose_create(clean_hu, dose_percent=20)
noisy_mr = add_rician_noise(clean_mr, delta_t=0.5)
```

Keep the clean file as `slice_XXXX.nii.gz` and save the degraded file beside it as `slice_XXXX_ld.nii.gz`.

**Why.** Enhancement training needs the two images to line up pixel for pixel, and a second low-dose or low-field scan of the same subject is rarely available. CT dose is lowered before reconstruction: fewer photons reach the detector, and filtered back-projection turns that into the streaky low-dose texture. The CT function follows this path. A low-field MR scan mainly loses signal-to-noise ratio, and a magnitude image then follows Rician noise, which is what the MR function adds inside the brain. Background voxels that are already 0 are left unchanged.

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

@article{isensee2019hdbet,
  title   = {Automated brain extraction of multisequence MRI using artificial neural networks},
  author  = {Isensee, Fabian and Schell, Marianne and Pflueger, Irada and Brugnara, Gianluca and Bonekamp, David and Neuberger, Ulf and Wick, Antje and Schlemmer, Heinz-Peter and Heiland, Sabine and Wick, Wolfgang and Bendszus, Martin and Maier-Hein, Klaus H. and Kickingereder, Philipp},
  journal = {Human Brain Mapping},
  volume  = {40},
  number  = {17},
  pages   = {4952--4964},
  year    = {2019},
  doi     = {10.1002/hbm.24750}
}

@article{hoopes2022synthstrip,
  title   = {SynthStrip: skull-stripping for any brain image},
  author  = {Hoopes, Andrew and Mora, Jocelyn S. and Dalca, Adrian V. and Fischl, Bruce and Hoffmann, Malte},
  journal = {NeuroImage},
  volume  = {260},
  pages   = {119474},
  year    = {2022},
  doi     = {10.1016/j.neuroimage.2022.119474}
}
```
