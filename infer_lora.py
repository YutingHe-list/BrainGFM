import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from datasets.braingfm import BRAINGFM_COLORMAP
from datasets.fewshot import build_fewshot_dataset, build_task_fewshot_dataset
from ruamel.yaml import YAML

from train_lora import build_models
from models import lora as L

device = torch.device('cuda')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--lora_ckpt", type=str, required=True)
    p.add_argument("--d12_pretrained_path", type=str, default='pretrained/braingfm_d12.pth')
    p.add_argument("--vae_pretrained_path", type=str,
                   default='pretrained/braingfm_pretrain.pth')
    p.add_argument("--mode", type=str, default='pair', choices=['pair', 'mask_cond'])
    p.add_argument("--task_cls", type=str, default='all')
    p.add_argument("--task_cases_config", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--cfg", type=float, nargs=3, default=[4.0, 4.0, 4.0])
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--lora_modality", type=str, default=None)
    p.add_argument("--lora_source", type=str, default=None)
    p.add_argument("--lora_case", type=str, default=None)
    p.add_argument("--lora_num_slices", type=int, default=20)
    p.add_argument("--lora_min_brain_ratio", type=float, default=0.05)
    p.add_argument("--data_dir", type=str, default='./data/BrainGFM')
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
    p.add_argument("--braingfm_index_json", type=str,
                   default='./data/BrainGFM/.braingfm_index_{split}.json')
    p.add_argument("--braingfm_val_ratio", type=float, default=0.05)
    p.add_argument("--braingfm_split_seed", type=int, default=42)
    p.add_argument("--braingfm_use_merged_mask", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--lora_rot90_aug", type=lambda x: str(x).lower() == 'true', default=False)

    args = p.parse_args()
    if args.config:
        with open(args.config, encoding='utf-8') as f:
            cfg = YAML(typ='safe').load(f)
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
        args = p.parse_args()
    return args


def rgb_to_idx_map(rgb01: torch.Tensor, colormap: np.ndarray) -> np.ndarray:
    cols = torch.from_numpy(colormap).float().to(rgb01.device) / 255.0
    px = rgb01.permute(1, 2, 0)                                          # (H,W,3)
    d = (px.unsqueeze(2) - cols.unsqueeze(0).unsqueeze(0))              # (H,W,n,3)
    idx = d.pow(2).sum(-1).argmin(-1)                                    # (H,W)
    return idx.cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    ck = torch.load(args.lora_ckpt, map_location='cpu')
    meta = ck.get('meta', {})
    if meta.get('prev_slice') or 'prev_slice_proj' in ck:
        print("[infer] 该权重含连续层条件，逐层生成用 python infer_lora_25d.py", flush=True)
    modality = args.lora_modality or meta.get('modality')
    assert modality, "ckpt meta 无 modality,请显式传 --lora_modality"
    args.lora_modality = modality
    mod_idx = meta.get('modality_idx', args.braingfm_modalities.index(modality))
    if args.d12_pretrained_path in (None, 'pretrained/braingfm_d12.pth') and meta.get('d12_pretrained_path'):
        args.d12_pretrained_path = meta['d12_pretrained_path']
    if meta.get('vae_pretrained_path'):
        args.vae_pretrained_path = meta['vae_pretrained_path']

    H = args.image_size
    out_dir = args.out_dir or f'synth/{modality}_{args.mode}_n{args.n_samples}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"[infer] modality={modality}(cls={mod_idx}) mode={args.mode} "
          f"n={args.n_samples} out={out_dir}", flush=True)
    print(f"[infer] base D12 = {args.d12_pretrained_path}", flush=True)

    vae, d12 = build_models(args, device)
    d12 = d12.to(device)
    L.freeze_base_for_lora(d12)
    n_task = int(meta.get('n_task_classes') or 0)
    class_names = list(meta.get('class_names') or [])
    if n_task:
        L.attach_task_cond(d12, n_task,
                           lora_route_sub=bool(meta.get('lora_route_sub', False)))
    L.apply_lora_to_d12(d12, num_routes=L.lora_num_routes(d12),
                        rank=meta.get('rank', 8), alpha=meta.get('alpha', 16.0),
                        dropout=meta.get('dropout', 0.05))
    rows = [meta.get('modality_idx'), args.num_classes] if meta.get('tune_class_emb') else None
    n = L.load_lora_into_d12(d12, ck['lora'], class_emb_rows=rows)
    print(f"[infer] 已加载 LoRA 增量({n} 个模块)自 {args.lora_ckpt}"
          + (f"; 任务子条件 {n_task} 类 {class_names}" if n_task else ""), flush=True)
    d12 = d12.to(device).eval()

    vpn = tuple(args.v_patch_nums)
    B = args.batch
    mod_idx = int(mod_idx)
    done = 0
    t0 = time.time()

    cond_masks = None
    if args.mode == 'mask_cond':
        if getattr(args, 'task_cases_config', None):
            ds, _, _, _ = build_task_fewshot_dataset(args, split='train')
        else:
            ds = build_fewshot_dataset(args, split='train')
        cond_masks = [ds[i]['mask'] for i in range(len(ds))]

    def _sub_of(sample_i):
        if not n_task:
            return None
        if args.task_cls == 'all':
            return sample_i % n_task
        return int(args.task_cls)

    with torch.no_grad():
        while done < args.n_samples:
            b = min(B, args.n_samples - done)
            label = torch.full((b,), mod_idx, dtype=torch.long, device=device)
            ctype = torch.tensor([0], device=device).expand(b).contiguous()
            c_mask = None
            if args.mode == 'mask_cond':
                msk = torch.stack([cond_masks[(done + i) % len(cond_masks)]
                                   for i in range(b)]).to(device)
                c_mask = vae.img_to_idxBl(msk, v_patch_nums=vpn)
            subs = [_sub_of(done + i) for i in range(b)]
            if n_task:
                L.set_sub_cond(torch.tensor(subs, dtype=torch.long, device=device))
            else:
                L.set_sub_cond(None)
            g_seed = args.seed * 1000003 + done
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                gen = d12.conditional_infer_cfg(
                    B=b, label_B=label, cfg=tuple(args.cfg),
                    top_k=args.top_k, top_p=args.top_p, g_seed=g_seed,
                    cond_type=ctype, c_mask=c_mask, c_img=None)
            gmask = gen[:, :, :H, :].clamp(0, 1).float().cpu()   # (b,3,H,W)
            gimg = gen[:, :, H:, :].clamp(0, 1).float().cpu()

            for i in range(b):
                m = gmask[i]
                im = gimg[i]
                # 背景置黑:脑区外不显示码本重建噪底
                bg = m.amax(dim=0) <= 0.025
                im[:, bg] = 0
                idx_map = rgb_to_idx_map(m, BRAINGFM_COLORMAP)
                tname = f"_{class_names[subs[i]]}" if n_task else ''
                tag = f'pair{tname}_{done + i:06d}'
                Image.fromarray((im.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
                    os.path.join(out_dir, f'{tag}_image.png'))
                Image.fromarray((m.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
                    os.path.join(out_dir, f'{tag}_mask.png'))
                np.save(os.path.join(out_dir, f'{tag}_mask_idx.npy'), idx_map)
                if (done + i) < 4:
                    prev = torch.cat([m, im], dim=2)
                    Image.fromarray((prev.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
                        os.path.join(out_dir, f'preview_{tag}.png'))
            done += b
            print(f"[gen] {done}/{args.n_samples} ({(time.time()-t0)/done:.1f}s/样本)", flush=True)

    print(f"[done] 合成数据已存 {out_dir}: image/mask png + mask_idx npy × {done}", flush=True)


if __name__ == '__main__':
    main()