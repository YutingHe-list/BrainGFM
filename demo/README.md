# BrainGFM Web Demo (Gradio)

Mask → image online demo. Default test: brain mask (no lesion) + **T1**.

## Files to upload / deploy

From the repo root, these are the demo-related additions:

| Path | Role |
| --- | --- |
| `app.py` | Gradio web UI |
| `requirements-demo.txt` | Demo dependencies (Space can copy this to `requirements.txt`) |
| `examples/BrainGFM_demo.ipynb` | Local Jupyter demo |
| `samples/mask_testing/input/brain.nii.gz` | Built-in sample mask (already in repo) |

The demo reuses existing code: `lora/infer_online.py`, `lora/train_lora.py`, `models/`, `datasets/`.

## Run

```bash
cd BrainGFM
pip install -r requirements-demo.txt
# or: pip install -r requirements.txt && pip install gradio
python app.py
```

Open `http://127.0.0.1:7860`. Needs a CUDA GPU. First run downloads weights from `wenxuyyds/BrainGFM`.

## Hugging Face Space

1. Create a Space: [https://huggingface.co/new-space](https://huggingface.co/spaces/wenxuyyds/BrainGFM)  
   - SDK: **Gradio**  
   - Hardware: **GPU** (CPU is not enough)
2. Point the Space to this GitHub repo, or upload at least:
   - `app.py`
   - project code used by inference (`lora/`, `models/`, `datasets/`, `pretrain/`, `utils/`, `samples/`)
   - set Space `requirements.txt` content from `requirements-demo.txt`
3. After build succeeds, copy the Space URL into the main README Online Demo link.

## Notes

- Default modality is `t1`; other modalities are in the dropdown.
- Leave the mask empty / use the sample file for the no-lesion brain mask test.
- Autocast uses CUDA bfloat16, same as `lora/infer_online.py`.
