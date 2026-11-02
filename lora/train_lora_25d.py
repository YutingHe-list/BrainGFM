import math
import os
import sys
import time
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.consecutive import build_consecutive_dataset
from models import lora as L
from pretrain.train_d12 import tokenize_pair
from train_lora import build_models, device, parse_args, save_lora_ckpt, setup_lora


def attach_prev_slice_proj(d12):
    proj = nn.Linear(d12.Cvae, d12.C)
    nn.init.zeros_(proj.weight)
    nn.init.zeros_(proj.bias)
    d12.prev_slice_proj = proj
    d12._prev_slice_feat = None
    d12._prev_slice_mask = None
    return proj


def load_prev_slice_proj(d12, state):
    proj = nn.Linear(d12.Cvae, d12.C)
    proj.load_state_dict(state)
    d12.prev_slice_proj = proj
    d12._prev_slice_feat = None
    d12._prev_slice_mask = None
    return proj


@torch.no_grad()
def encode_prev(vae, prev_image, vpn):
    idx = vae.img_to_idxBl(prev_image, v_patch_nums=vpn)[-1]
    return vae.quantize.embedding(idx.long()).mean(dim=1)


def save_ckpt(d12, args, epoch, path, mod_idx):
    save_lora_ckpt(d12, args, epoch, path, mod_idx)
    ck = torch.load(path, map_location='cpu')
    ck['prev_slice_proj'] = {k: v.detach().cpu() for k, v in d12.prev_slice_proj.state_dict().items()}
    ck['meta']['prev_slice'] = True
    torch.save(ck, path)
    return path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    run_name = args.run_name or f"lora25d_{args.lora_modality}"
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    project_dir = os.path.join(args.output_dir, f"{timestamp}-{run_name}")
    os.makedirs(project_dir, exist_ok=True)

    ds = build_consecutive_dataset(args, split='train')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False,
                        timeout=120 if args.num_workers > 0 else 0)
    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    total_steps = steps_per_epoch * args.num_epochs

    vae, d12 = build_models(args, device)
    d12 = d12.to(device)
    lora_params, emb_params, mod_idx = setup_lora(d12, args)
    proj = attach_prev_slice_proj(d12).to(device)
    if not any(p.requires_grad for p in proj.parameters()):
        raise RuntimeError("上一层投影被冻结，连续层条件无法训练")
    train_params = lora_params + (emb_params if args.tune_class_emb else []) + list(proj.parameters())
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'weight_decay': args.weight_decay},
        {'params': emb_params if args.tune_class_emb else [], 'weight_decay': 0.0},
        {'params': list(proj.parameters()), 'weight_decay': 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    warm = args.warmup_epochs * steps_per_epoch

    def lr_scale(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        t = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    print("***** 连续层 LoRA 微调 *****", flush=True)
    print(f"  模态={args.lora_modality}(cls={mod_idx})  samples={len(ds)} "
          f"batch={args.batch_size}  epochs={args.num_epochs}", flush=True)
    print(f"  out={project_dir}", flush=True)

    vpn = tuple(args.v_patch_nums)
    autocast_enabled = args.mixed_precision == 'bf16'
    gstep = 0
    best_ce = float('inf')
    t0 = time.time()
    for epoch in range(args.num_epochs):
        d12.train()
        ep_ce, ep_n = 0.0, 0
        for batch in loader:
            img = batch['image'].to(device, non_blocking=True)
            msk = batch['mask'].to(device, non_blocking=True)
            prev = batch['prev_image'].to(device, non_blocking=True)
            has_prev = batch['has_prev'].to(device, non_blocking=True)
            cls = batch['cls'].to(device, non_blocking=True)
            ctype = batch['type'].to(device, non_blocking=True)
            L.set_sub_cond(None)

            x_in, tgt, weights = tokenize_pair(
                vae, img, msk, vpn,
                mask_idx=batch.get('mask_idx'),
                lesion_interior_weight=args.lesion_interior_weight,
                lesion_boundary_weight=args.lesion_boundary_weight)
            d12._prev_slice_feat = encode_prev(vae, prev, vpn)
            d12._prev_slice_mask = has_prev

            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate * lr_scale(gstep)
            try:
                if autocast_enabled:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = d12(cls, x_in, ctype)
                else:
                    logits = d12(cls, x_in, ctype)
            finally:
                d12._prev_slice_feat = None
                d12._prev_slice_mask = None

            ce = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(),
                                 tgt.reshape(-1), reduction='none')
            if weights is not None:
                w = weights.reshape(-1)
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            gstep += 1
            ep_ce += loss.item() * tgt.numel()
            ep_n += tgt.numel()
            if gstep % args.log_interval == 0:
                print(f"[step {gstep}/{total_steps}] ce={loss.item():.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"({(time.time() - t0) / gstep:.1f}s/step)", flush=True)

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.num_epochs:
            out = save_ckpt(d12, args, epoch + 1,
                            os.path.join(project_dir, f'lora_epoch{(epoch + 1):04d}.pth'),
                            mod_idx)
            ep_avg = ep_ce / max(ep_n, 1)
            star = ''
            if ep_avg < best_ce:
                best_ce = ep_avg
                save_ckpt(d12, args, epoch + 1,
                          os.path.join(project_dir, 'best_lora.pth'), mod_idx)
                star = ' best'
            print(f"[ckpt] epoch {epoch + 1}: ep_CE={ep_avg:.4f} -> {out}{star}", flush=True)

    print(f"[done] 连续层微调完成: {os.path.join(project_dir, 'best_lora.pth')}", flush=True)


if __name__ == '__main__':
    main()
