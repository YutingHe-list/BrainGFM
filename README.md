<p align="center">
  <img src="Fig/BrainGFM_banner.png" alt="BrainGFM - A Brain Imaging Foundation Model. A generative foundation model with a reusable and reconfigurable brain-imaging prior."/>
</p>

## News

- **2026-11** — Release pretrained weights, LoRA adaptation, and inference code.

## Brief introduction

Computational brain imaging is expanding toward increasingly diverse tasks such as segmentation, disease characterization, and image enhancement. However, representative MRI and CT data remain scarce and heterogeneous across clinical centers, scan protocols, and brain diseases, limiting the clinical acceptability of downstream models. We propose a Brain imaging Generative Foundation Model (**BrainGFM**) with a reusable and reconfigurable brain-imaging prior. It has been pre-trained on BrainGFM-1.3M, a large-scale corpus of about 1.3 million MRI and CT slices, and is able to support task-aligned data synthesis for low-data predictive learning and task-adapted post-training for image reconstruction. Via the adaptation with few data, it can further achieve professional performance in real-world specific scenarios.

<p align="center">
  <img src="fig/BrainGFM.png" alt="BrainGFM overview"/>
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

## 3. Data preprocessing

Each axial slice is skull-stripped, then placed on a 256 by 256 image, before it is saved.

Skull stripping uses the two tools below. Each one predicts a brain mask, and everything outside that mask is cleared.

| Tool | Reference | Use |
| --- | --- | --- |
| HD-BET | [link](https://github.com/MIC-DKFZ/HD-BET) | MR. The mask removes the skull and other tissue outside the brain. |
| SynthStrip | [link](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) | CT. The mask removes the skull on the CT slice. |

The brain is then cropped to 256 by 256. If the brain region is already larger than 256 on either side, the slice is scaled down until the whole brain fits inside 256 by 256. The brain mask is cropped and scaled with the image.

## 4. Adapt the model with LoRA

Put the slices under data/BrainGFM. Fine-tuning scales each slice to -1 to 1, and the inverse maps a generated image back to the original intensity. The three tasks are segmentation, classification, and enhancement. Segmentation and classification train a LoRA. Enhancement updates the whole generator.

### Segmentation

Use a few labeled axial slices of one modality. Each slice needs a brain mask and a label mask with the same file name:

```
data/BrainGFM/FLAIR/<cohort>/<case>/slice_XXXX.nii.gz
data/BrainGFM/Lable/Brain_mask/<cohort>/<case>/slice_XXXX.nii.gz
data/BrainGFM/Lable/Lable_mask/<cohort>/<case>/slice_XXXX.nii.gz
```

T1, T1ce, T2, DWI, CT, and Unknow use the same layout, with the modality folder name changed. The default run takes 20 FLAIR slices.

In configs/train_lora.yaml, set lora_modality to t1, t2, t1ce, flair, dwi, ct, or unknow. The pretrained weights stay frozen, and only the LoRA for that modality is trained.

```bash
python train_lora.py --config configs/train_lora.yaml
```

The checkpoint is best_lora.pth under experiments/lora_finetune. The next section uses it to build labeled pairs.

### Classification

List each class and the cases that belong to it. Start from configs/task_cases_template.yaml. Each class uses 20 slices of the chosen modality, laid out the same way as above, with a brain mask of the same file name. The LoRA then learns a condition for each class.

```bash
python train_lora.py --config configs/train_lora.yaml --task_cases_config configs/task_cases_template.yaml
```

The checkpoint is written to the same folder. Inference in the next section reads the class names in that checkpoint and generates each class in turn.

### Enhancement

Enhancement fine-tunes the whole generator on a clean slice and a degraded slice of the same cut. How to build a pair is in [Building enhancement image pairs](#building-enhancement-image-pairs).

```bash
python pretrain/train_d12.py --config configs/train_d12.yaml
```

## 5. Inference

Fine-tune on a small set of labeled slices. The adapted model can then generate masks with no image input. Each of those masks is passed through the same fine-tuned weights to generate the corresponding image. Remove the brain-region label from the mask, and the remaining labels together with that image are a labeled pair.

```bash
python infer_lora.py \
    --lora_ckpt experiments/lora_finetune/<run>/best_lora.pth \
    --mode pair --n_samples 100 --out_dir synth/flair_100
```

Each sample is written into the output folder as the generated mask, the generated image, and the mask index.

## Building enhancement image pairs

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

This simulates a low-dose CT scan and a low-field MR scan in clinical practice.

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
