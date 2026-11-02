import argparse
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.fewshot import build_fewshot_dataset
from models import VAE, build_d12
from models import lora as L
from ruamel.yaml import YAML

from pretrain.train_d12 import tokenize_pair

device = torch.device('cuda')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--lora_modality", type=str, default=None)
    p.add_argument("--lora_source", type=str, default=None)
    p.add_argument("--lora_case", type=str, default=None)
    p.add_argument("--lora_num_slices", type=int, default=20)
    p.add_argument("--lora_min_brain_ratio", type=float, default=0.05)
    p.add_argument("--lora_rot90_aug", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--task_cases_config", type=str, default=None)
    p.add_argument("--lora_route_sub", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--tune_uncond_class_emb", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--tune_class_emb", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--data_dir", type=str, default='./data/BrainGFM')
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--z_channels", type=int, default=32)
    p.add_argument("--ch", type=int, default=160)
    p.add_argument("--v_patch_nums", type=int, nargs='+',
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    p.add_argument("--mask_type", type=str, default='interleave_append')
    p.add_argument("--cond_drop_rate", type=float, default=0.1)
    p.add_argument("--vae_pretrained_path", type=str,
                   default='pretrained/braingfm_pretrain.pth')
    p.add_argument("--d12_pretrained_path", type=str, default='pretrained/braingfm_d12.pth')
    p.add_argument("--braingfm_modalities", type=str, nargs='+',
                   default=['t1', 't2', 't1ce', 'flair', 'dwi', 'ct', 'unknow'])
    p.add_argument("--braingfm_index_json", type=str,
                   default='./data/BrainGFM/.braingfm_index_{split}.json')
    p.add_argument("--braingfm_val_ratio", type=float, default=0.05)
    p.add_argument("--braingfm_split_seed", type=int, default=42)
    p.add_argument("--braingfm_use_merged_mask", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--lesion_weighted_loss", action="store_true", default=False)
    p.add_argument("--lesion_interior_weight", type=float, default=5.0)
    p.add_argument("--lesion_boundary_weight", type=float, default=10.0)
    p.add_argument("--num_epochs", type=int, default=250)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--mixed_precision", type=str, default='bf16')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--vis_interval", type=int, default=25)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--cfg", type=float, nargs=3, default=[4.0, 4.0, 4.0])
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.96)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default='experiments/lora_finetune')
    p.add_argument("--use_wandb", action="store_true", default=False)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, encoding='utf-8') as f:
            cfg = YAML(typ='safe').load(f) or {}
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    args = p.parse_args()
    if not args.lora_modality:
        p.error("需要 --lora_modality，或在 --config 里给出")
    return args


def build_models(args, dev):
    vpn = tuple(args.v_patch_nums)
    vae = VAE(vocab_size=args.vocab_size, z_channels=args.z_channels,
              ch=args.ch, test_mode=True, share_quant_resi=4, v_patch_nums=vpn)
    sd = torch.load(args.vae_pretrained_path, map_location='cpu')
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not unexpected, f"VAE unexpected={unexpected[:5]}"
    vae = vae.to(dev).eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)

    d12 = build_d12(
        vae=vae, depth=args.depth, patch_nums=vpn,
        mask_type=args.mask_type, cond_drop_rate=args.cond_drop_rate,
        multi_cond=True, num_classes=args.num_classes)
    if args.d12_pretrained_path:
        msd = torch.load(args.d12_pretrained_path, map_location='cpu')
        if isinstance(msd, dict) and 'model_state_dict' in msd:
            msd = msd['model_state_dict']
        msd = {k.replace('module.', ''): v for k, v in msd.items()}
        skip = [k for k, v in msd.items()
                if k in d12.state_dict() and d12.state_dict()[k].shape != v.shape]
        msd = {k: v for k, v in msd.items() if k not in skip}
        missing, unexpected = d12.load_state_dict(msd, strict=False)
        print(f"[D12] base {args.d12_pretrained_path}: 跳过形状不匹配 {skip}, "
              f"missing={sorted(missing)[:6]}", flush=True)
    return vae, d12


def setup_lora(d12, args, n_task_classes=0):
    L.freeze_base_for_lora(d12)
    n_task, class_names = 0, []
    if n_task_classes and n_task_classes > 0:
        L.attach_task_cond(d12, int(n_task_classes),
                           lora_route_sub=bool(args.lora_route_sub))
    L.apply_lora_to_d12(d12, num_routes=L.lora_num_routes(d12),
                        rank=args.lora_rank, alpha=args.lora_alpha,
                        dropout=args.lora_dropout)
    mod_idx = args.braingfm_modalities.index(args.lora_modality)
    if args.tune_class_emb:
        L.register_class_emb_row_hook(
            d12, mod_idx, tune_uncond=args.tune_uncond_class_emb)
    else:
        d12.class_emb.weight.requires_grad_(False)
    lora_params, emb_params = L.lora_trainable_parameters(d12)
    n_lora = L.count_lora_params(d12)
    msg_sub = f", 二级条件 {n_task_classes} 子类(LoRA {'模态×子类 复合路由' if args.lora_route_sub else '模态级共享'})" if n_task_classes else ""
    print(f"[LoRA] 可训练 LoRA 参数 {n_lora / 1e6:.2f}M; "
          f"目标模态 {args.lora_modality}(idx={mod_idx}) + uncond={args.tune_uncond_class_emb}{msg_sub}",
          flush=True)
    return lora_params, emb_params, mod_idx


@torch.no_grad()
def visualize(d12, vae, watch_items, args, epoch, out_dir, mod_idx, class_names=None):
    from PIL import Image, ImageDraw
    was_training = d12.training
    d12.eval()
    dev = next(d12.parameters()).device
    vpn = tuple(args.v_patch_nums)
    H = args.image_size
    rows = []
    for item in watch_items[:2]:
        img = item['image'].unsqueeze(0).to(dev)
        msk = item['mask'].unsqueeze(0).to(dev)
        cls = torch.full((1,), mod_idx, dtype=torch.long, device=dev)
        ctype = torch.tensor([0], device=dev)
        tc = item.get('task_cls')
        if tc is not None:
            L.set_sub_cond(tc.view(1).to(dev))
        else:
            L.set_sub_cond(None)
        c_cond = vae.img_to_idxBl(msk, v_patch_nums=vpn)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen = d12.conditional_infer_cfg(
                B=1, label_B=cls, cfg=tuple(args.cfg), top_k=args.top_k,
                top_p=args.top_p, g_seed=args.seed + epoch, cond_type=ctype,
                c_mask=c_cond, c_img=None)
        L.set_sub_cond(None)
        tname = f"[{class_names[tc.item()]}] " if (tc is not None and class_names) else ''
        gen_img = gen[:, :, H:, :].clamp(0, 1)[0].float().cpu()
        bg = msk[0].amax(dim=0).cpu() <= -0.95
        gen_img[:, bg] = 0
        gt = ((img[0] + 1) / 2).clamp(0, 1).float().cpu()
        cond = ((msk[0] + 1) / 2).clamp(0, 1).float().cpu()
        rows.append((f'recon e{epoch} {tname}(mask|GT|gen)', cond, gt, gen_img))

    def _gen_once(g_seed, sub=None):
        cls = torch.full((1,), mod_idx, dtype=torch.long, device=dev)
        ctype = torch.tensor([0], device=dev)
        if sub is not None:
            L.set_sub_cond(torch.tensor([sub], dtype=torch.long, device=dev))
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen = d12.conditional_infer_cfg(
                B=1, label_B=cls, cfg=tuple(args.cfg), top_k=args.top_k,
                top_p=args.top_p, g_seed=g_seed, cond_type=ctype,
                c_mask=None, c_img=None)
        L.set_sub_cond(None)
        gmask = gen[:, :, :H, :].clamp(0, 1)[0].float().cpu()
        gimg = gen[:, :, H:, :].clamp(0, 1)[0].float().cpu()
        gbg = gmask.amax(dim=0) <= 0.025
        gimg[:, gbg] = 0
        return gmask, gimg

    if class_names:
        for t, cname in enumerate(class_names):
            gmask, gimg = _gen_once(args.seed * 1000 + epoch * 10 + t, sub=t)
            rows.append((f'gen e{epoch} [{cname}]', gmask, None, gimg))
    else:
        for k in range(2):
            gmask, gimg = _gen_once(args.seed * 1000 + epoch * 10 + k)
            rows.append((f'gen e{epoch} #{k}', gmask, None, gimg))

    os.makedirs(out_dir, exist_ok=True)
    PAD, LBL = 4, 18
    n_col = 3
    W = n_col * H + (n_col + 1) * PAD
    Ht = len(rows) * (H + LBL) + PAD
    canvas = Image.new('RGB', (W, Ht), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, (label, c1, c2, c3) in enumerate(rows):
        y = PAD + r * (H + LBL)
        draw.text((PAD, y - 14), label, fill=(200, 200, 200))
        cells = [c1, c2, c3]
        for ci, t in enumerate(cells):
            if t is None:
                continue
            arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
            canvas.paste(Image.fromarray(arr), (PAD + ci * (H + PAD), y))
    out = os.path.join(out_dir, f'vis_epoch{epoch + 1:04d}.png')
    canvas.save(out)
    print(f"[vis] epoch {epoch+1}: {len(rows)} 行 -> {out}", flush=True)
    if was_training:
        d12.train()
    return out


def save_lora_ckpt(d12, args, epoch, out_path, mod_idx,
                   n_task=0, class_names=None):
    ck = {
        'lora': L.lora_state_dict(d12),
        'meta': {
            'modality': args.lora_modality,
            'modality_idx': mod_idx,
            'num_classes': args.num_classes,
            'n_task_classes': n_task,
            'class_names': class_names or [],
            'lora_route_sub': bool(args.lora_route_sub),
            'rank': args.lora_rank,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'tune_class_emb': args.tune_class_emb,
            'tune_uncond_class_emb': args.tune_uncond_class_emb,
            'epoch': epoch,
            'd12_pretrained_path': args.d12_pretrained_path,
            'vae_pretrained_path': args.vae_pretrained_path,
        },
    }
    torch.save(ck, out_path)
    return out_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    run_name = args.run_name or f"lora_{args.lora_modality}"
    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    project_dir = os.path.join(args.output_dir, f"{timestamp}-{run_name}")
    os.makedirs(project_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project="BrainGFM-LoRA", name=f"{run_name}-{timestamp}",
                   config=vars(args))

    n_task, class_names = 0, []
    if args.task_cases_config:
        from datasets.fewshot import build_task_fewshot_dataset
        ds, n_task, class_names, task_modality = build_task_fewshot_dataset(args, split='train')
        args.lora_modality = task_modality
    else:
        ds = build_fewshot_dataset(args, split='train')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False,
                        timeout=120 if args.num_workers > 0 else 0)
    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    total_steps = steps_per_epoch * args.num_epochs

    vae, d12 = build_models(args, device)
    d12 = d12.to(device)
    lora_params, emb_params, mod_idx = setup_lora(d12, args, n_task_classes=n_task)
    param_groups = [
        {'params': lora_params, 'weight_decay': args.weight_decay},
        {'params': emb_params if args.tune_class_emb else [], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate,
                                  betas=(0.9, 0.95), eps=1e-8)
    warm = args.warmup_epochs * steps_per_epoch

    def lr_scale(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        t = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    watch_items = [ds[i] for i in range(min(2, len(ds)))]

    print(f"***** LoRA 微调 *****", flush=True)
    print(f"  模态={args.lora_modality}(cls={mod_idx})  slices={len(ds)} "
          f"batch={args.batch_size}  steps/epoch={steps_per_epoch}  "
          f"epochs={args.num_epochs}  total_steps={total_steps}", flush=True)
    print(f"  lr={args.learning_rate} r={args.lora_rank} alpha={args.lora_alpha} "
          f"drop={args.lora_dropout} cond_drop={args.cond_drop_rate}", flush=True)
    print(f"  out={project_dir}", flush=True)

    vpn = tuple(args.v_patch_nums)
    autocast_enabled = args.mixed_precision == 'bf16'
    gstep = 0
    best_ce = float('inf')
    t0 = time.time()
    for epoch in range(args.num_epochs):
        d12.train()
        ep_ce, ep_acc, ep_n = 0.0, 0.0, 0
        for batch in loader:
            img = batch['image'].to(device, non_blocking=True)
            msk = batch['mask'].to(device, non_blocking=True)
            cls = batch['cls'].to(device, non_blocking=True)
            ctype = batch['type'].to(device, non_blocking=True)
            task_cls = batch.get('task_cls')
            if task_cls is not None:
                L.set_sub_cond(task_cls.to(device, non_blocking=True))
            else:
                L.set_sub_cond(None)

            x_in, tgt, weights = tokenize_pair(
                vae, img, msk, vpn,
                mask_idx=batch.get('mask_idx'),
                lesion_interior_weight=args.lesion_interior_weight,
                lesion_boundary_weight=args.lesion_boundary_weight)

            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate * lr_scale(gstep)

            if autocast_enabled:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = d12(cls, x_in, ctype)
            else:
                logits = d12(cls, x_in, ctype)

            ce = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(),
                                 tgt.reshape(-1), reduction='none')
            if weights is not None:
                w = weights.reshape(-1)
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + emb_params, args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            gstep += 1
            ep_ce += loss.item() * tgt.numel()
            ep_acc += (logits.argmax(-1) == tgt).float().sum().item()
            ep_n += tgt.numel()

            if gstep % args.log_interval == 0:
                msg = (f"[step {gstep}/{total_steps}] ce={loss.item():.4f} "
                       f"lr={optimizer.param_groups[0]['lr']:.2e} "
                       f"({(time.time() - t0) / gstep:.1f}s/step)")
                print(msg, flush=True)
                if args.use_wandb:
                    wandb.log({"lora/ce": loss.item(),
                               "lora/token_acc": (logits.argmax(-1) == tgt).float().mean().item(),
                               "lora/lr": optimizer.param_groups[0]['lr']}, step=gstep)

        if (epoch + 1) % args.vis_interval == 0 or epoch == 0:
            try:
                img_path = visualize(d12, vae, watch_items, args, epoch,
                                     os.path.join(project_dir, 'vis'), mod_idx,
                                     class_names=class_names if n_task else None)
                if args.use_wandb:
                    wandb.log({"generation": wandb.Image(img_path)}, step=gstep)
            except Exception as e:
                print(f"[vis] 失败(不影响训练): {type(e).__name__}: {e}", flush=True)
        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.num_epochs:
            out = save_lora_ckpt(d12, args, epoch + 1,
                                 os.path.join(project_dir, f'lora_epoch{(epoch + 1):04d}.pth'),
                                 mod_idx, n_task=n_task, class_names=class_names)
            ep_avg = ep_ce / max(ep_n, 1)
            star = ''
            if ep_avg < best_ce:
                best_ce = ep_avg
                save_lora_ckpt(d12, args, epoch + 1,
                               os.path.join(project_dir, 'best_lora.pth'), mod_idx,
                               n_task=n_task, class_names=class_names)
                star = ' ★best'
            print(f"[ckpt] epoch {epoch+1}: ep_CE={ep_avg:.4f} -> {out}{star}", flush=True)

    print("[done] LoRA 微调完成。生成数据: python lora/infer_lora.py "
          f"--lora_ckpt {os.path.join(project_dir, 'best_lora.pth')}", flush=True)


if __name__ == '__main__':
    main()