import argparse
import os

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML

from datasets.braingfm import BRAINGFM_COLORMAP
from infer_lora import rgb_to_idx_map
from models import lora as L
from train_lora import build_models, device
from train_lora_25d import encode_prev, load_prev_slice_proj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--lora_ckpt", type=str, default=None)
    p.add_argument("--d12_pretrained_path", type=str, default='pretrained/braingfm_d12.pth')
    p.add_argument("--vae_pretrained_path", type=str, default='pretrained/braingfm_pretrain.pth')
    p.add_argument("--n_slices", type=int, default=64)
    p.add_argument("--n_volumes", type=int, default=1)
    p.add_argument("--cfg", type=float, nargs=3, default=[4.0, 4.0, 4.0])
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--lora_modality", type=str, default=None)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--z_channels", type=int, default=32)
    p.add_argument("--ch", type=int, default=160)
    p.add_argument("--v_patch_nums", type=int, nargs='+',
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    p.add_argument("--mask_type", type=str, default='interleave_append')
    p.add_argument("--cond_drop_rate", type=float, default=0.1)
    p.add_argument("--braingfm_modalities", type=str, nargs='+',
                   default=['t1', 't2', 't1ce', 'flair', 'dwi', 'ct', 'unknow'])

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, encoding='utf-8') as f:
            cfg = YAML(typ='safe').load(f) or {}
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    args = p.parse_args()
    if not args.lora_ckpt:
        p.error("需要 --lora_ckpt")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    ck = torch.load(args.lora_ckpt, map_location='cpu')
    meta = ck.get('meta', {})
    if 'prev_slice_proj' not in ck:
        raise SystemExit("该权重没有连续层投影，请使用 train_lora_25d.py 保存的 best_lora.pth")
    modality = args.lora_modality or meta.get('modality')
    if not modality:
        raise SystemExit("ckpt 中没有模态，请传 --lora_modality")
    args.lora_modality = modality
    mod_idx = int(meta.get('modality_idx', args.braingfm_modalities.index(modality)))
    if meta.get('d12_pretrained_path'):
        args.d12_pretrained_path = meta['d12_pretrained_path']
    if meta.get('vae_pretrained_path'):
        args.vae_pretrained_path = meta['vae_pretrained_path']

    H = args.image_size
    out_dir = args.out_dir or f'synth/{modality}_25d_n{args.n_slices}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[infer-25d] modality={modality}(cls={mod_idx}) "
          f"volumes={args.n_volumes} slices={args.n_slices} out={out_dir}", flush=True)

    vae, d12 = build_models(args, device)
    d12 = d12.to(device)
    L.freeze_base_for_lora(d12)
    L.apply_lora_to_d12(d12, num_routes=L.lora_num_routes(d12),
                        rank=meta.get('rank', 8), alpha=meta.get('alpha', 16.0),
                        dropout=meta.get('dropout', 0.05))
    rows = [meta.get('modality_idx'), args.num_classes] if meta.get('tune_class_emb') else None
    n = L.load_lora_into_d12(d12, ck['lora'], class_emb_rows=rows)
    load_prev_slice_proj(d12, ck['prev_slice_proj']).to(device)
    print(f"[infer-25d] 已加载 LoRA({n}) 与上一层投影 {args.lora_ckpt}", flush=True)
    d12.eval()
    L.set_sub_cond(None)

    vpn = tuple(args.v_patch_nums)
    for vol in range(args.n_volumes):
        vol_dir = os.path.join(out_dir, f'vol{vol:03d}')
        os.makedirs(vol_dir, exist_ok=True)
        images, idx_maps = [], []
        prev_img = torch.full((1, 3, H, H), -1.0, device=device)
        has_prev = torch.zeros(1, device=device)
        for s in range(args.n_slices):
            if float(has_prev.item()) == 0:
                d12._prev_slice_feat = torch.zeros(1, d12.Cvae, device=device)
            else:
                d12._prev_slice_feat = encode_prev(vae, prev_img, vpn)
            d12._prev_slice_mask = has_prev
            label = torch.full((1,), mod_idx, dtype=torch.long, device=device)
            ctype = torch.zeros(1, dtype=torch.long, device=device)
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    gen = d12.conditional_infer_cfg(
                        B=1, label_B=label, cfg=tuple(args.cfg),
                        top_k=args.top_k, top_p=args.top_p,
                        g_seed=args.seed * 1000003 + vol * 10007 + s,
                        cond_type=ctype, c_mask=None, c_img=None)
            finally:
                d12._prev_slice_feat = None
                d12._prev_slice_mask = None
            m = gen[:, :, :H, :].clamp(0, 1).float()
            im = gen[:, :, H:, :].clamp(0, 1).float()
            bg = m[0].amax(dim=0) <= 0.025
            im[0][:, bg] = 0
            im_cpu = im[0].cpu()
            m_cpu = m[0].cpu()
            idx_map = rgb_to_idx_map(m_cpu, BRAINGFM_COLORMAP)
            tag = f'slice_{s:04d}'
            Image.fromarray((im_cpu.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
                os.path.join(vol_dir, f'{tag}_image.png'))
            Image.fromarray((m_cpu.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
                os.path.join(vol_dir, f'{tag}_mask.png'))
            images.append(im_cpu[0].numpy())
            idx_maps.append(idx_map)
            prev_img = im * 2 - 1
            has_prev = torch.ones(1, device=device)
            print(f"[gen] vol {vol} slice {s + 1}/{args.n_slices}", flush=True)

        img_vol = np.stack(images, axis=-1).astype(np.float32)
        msk_vol = np.stack(idx_maps, axis=-1).astype(np.uint8)
        nib.save(nib.Nifti1Image(img_vol, np.eye(4)), os.path.join(vol_dir, 'image.nii.gz'))
        nib.save(nib.Nifti1Image(msk_vol, np.eye(4)), os.path.join(vol_dir, 'mask_idx.nii.gz'))
        print(f"[vol] {vol_dir}: {img_vol.shape[-1]} 层", flush=True)

    print(f"[done] 连续层结果已存 {out_dir}", flush=True)


if __name__ == '__main__':
    main()
