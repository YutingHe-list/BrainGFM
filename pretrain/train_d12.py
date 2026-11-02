"""BrainGFM 二阶段 D12 生成模型训练（teacher-forcing CE）。

数据流:
  nii.gz -> normalize -> [-1,1] image + colorized mask
    -> 冻结 VAE 码本: img_to_idxBl(image/mask 各 10 scales)
    -> idxBl_to_ar_input 双流各 9 scales，按 [mask流; image流] 交错拼 L=1358
    -> D12.forward(label, x_BLCv_wo_first_l, cond_type) -> logits (B,1360,4096)
    -> CE(logits, 交错目标 1360)

实现要点:
  1. 加载 braingfm_d12.pth 时 class_emb (1001,768) -> (8,768) 尺寸不匹配，
     load_state_dict 必须先跳过形状不匹配的 key（strict=False）。
  2. vae.img_to_idxBl 必须显式传 v_patch_nums（默认 None 会报错）。
  3. idxBl_to_ar_input 在 vae.quantize 上（vae 本身无此方法）。
  4. teacher forcing 必须双流交错：只给 image 流会 L=679 != 1358。
  5. logits L=1360 与目标整条对齐（first_l=2 的目标也由 sos 后位置预测）。

DDP 框架要点:
  - 先写本地临时盘，后台线程拷到共享存储，防网络存储抖动挂起。
  - DataLoader timeout 防网络存储挂起。
  - NCCL 超时快速失败。
"""
import argparse
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from time import sleep, time as now

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from datasets import create_dataset
from models import VAE, build_d12
from utils.lr_control import filter_params, lr_wd_annealing

device = torch.device('cuda')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    # data（train_d12.yaml 同名项覆盖默认值）
    p.add_argument("--data_dir", type=str, default='./data/BrainGFM')
    p.add_argument("--dataset_name", type=str, default="braingfm")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    # training
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="experiments/d12_braingfm")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--weight_decay_end", type=float, default=0.0)
    p.add_argument("--lr_scheduler", type=str, default='lin0')
    p.add_argument("--lr_warmup_steps", type=float, default=0.01)
    p.add_argument("--wp0", type=float, default=0.005)
    p.add_argument("--wpe", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--mixed_precision", type=str, default='bf16')
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--val_max_batches", type=int, default=64,
                   help="早停验证最多抽前 N 个 batch")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--z_channels", type=int, default=32)
    p.add_argument("--ch", type=int, default=160)
    p.add_argument("--v_patch_nums", type=int, nargs='+',
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    p.add_argument("--mask_type", type=str, default='interleave_append')
    p.add_argument("--cond_drop_rate", type=float, default=0.1)
    p.add_argument("--lesion_weighted_loss", action="store_true", default=False)
    p.add_argument("--lesion_interior_weight", type=float, default=5.0)
    p.add_argument("--lesion_boundary_weight", type=float, default=10.0)
    p.add_argument("--reset_schedule", action="store_true", default=False)
    p.add_argument("--multi_cond", action='store_true', default=True)
    p.add_argument("--vae_pretrained_path", type=str,
                   default='pretrained/braingfm_pretrain.pth')
    p.add_argument("--d12_pretrained_path", type=str, default='pretrained/braingfm_d12.pth')
    p.add_argument("--resume", type=str, default=None)
    # 生成可视化（每 epoch 结束 rank0 做：mask->图像 条件生成，GT|生成 对比图）
    p.add_argument("--cfg", type=float, nargs=3, default=[4.0, 4.0, 4.0])
    p.add_argument("--top_k", type=int, default=900)
    p.add_argument("--top_p", type=float, default=0.96)
    p.add_argument("--vis_per_mod", type=int, default=1, help="每 epoch 每模态可视化的 val 样本数")
    # braingfm dataset 透传（与 configs/train_d12.yaml 对齐）
    p.add_argument("--braingfm_modalities", type=str, nargs='+',
                   default=['t1', 't2', 't1ce', 'flair', 'dwi', 'ct', 'unknow'])
    p.add_argument("--braingfm_val_ratio", type=float, default=0.05)
    p.add_argument("--braingfm_split_seed", type=int, default=42)
    p.add_argument("--braingfm_mask_overlay", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--braingfm_min_brain_ratio", type=float, default=0.0)
    p.add_argument("--braingfm_index_json", type=str,
                   default='./data/BrainGFM/.braingfm_index_{split}.json')
    p.add_argument("--braingfm_mask_mix_prob", type=float, default=0.0)
    p.add_argument("--braingfm_max_per_modality", type=int, default=0)
    p.add_argument("--braingfm_slices_per_case", type=int, default=0)
    p.add_argument("--braingfm_labeled_weight", type=float, default=1.0)
    p.add_argument("--braingfm_exclude_val_cases", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--braingfm_use_merged_mask", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--braingfm_val_slice_cap", type=int, default=0)
    p.add_argument("--braingfm_modality_pair", type=str, nargs='+', default=None)
    p.add_argument("--braingfm_lesion_index_dir", type=str, default=None)
    p.add_argument("--separator", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--bidirectional", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--separate_decoding", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--type_pos", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--indep", type=lambda x: str(x).lower() == 'true', default=False)

    args = p.parse_args()
    if args.config:
        with open(args.config, encoding='utf-8') as f:
            cfg = YAML(typ='safe').load(f)
        # YAML 值覆盖 argparse 默认（train_vae 同款机制）：
        # 只覆盖 parser 认识的键，避免 yaml 里的推理专用键（cfg/top_k 等）报错
        known = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
        args = p.parse_args()
    return args


def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29511')
    from datetime import timedelta
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=10))


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(d12, optimizer, args, epoch=None, step=None, best=False):
    """先写本地临时盘 + 后台线程拷到共享存储，防网络存储抖动挂起。"""
    module = d12.module if hasattr(d12, 'module') else d12
    ckpt = {
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    local_dir = os.path.join(tempfile.gettempdir(), 'd12_ckpt_local')
    os.makedirs(local_dir, exist_ok=True)
    fname = 'best_checkpoint.pth' if best else f'checkpoint_step_{step:08d}.pth'
    local = os.path.join(local_dir, fname)
    torch.save(ckpt, local)
    # 滚动清理:本地临时盘只保留最新 2 个 step checkpoint(best 与最终镜像全保留)，
    # 防临时盘被 epoch 级累积的大 checkpoint 撑爆
    if not best:
        import glob as _glob
        olds = sorted(_glob.glob(os.path.join(local_dir, 'checkpoint_step_*.pth')),
                       key=os.path.getmtime, reverse=True)
        for stale in olds[2:]:
            try:
                os.remove(stale)
            except OSError:
                pass
    final = os.path.join(args.project_dir, fname)
    def _bg():
        try:
            import shutil
            shutil.copy(local, final)
        except Exception as e:
            print(f"[ckpt] 后台拷共享存储失败（本地有 {local}）: {e}", flush=True)
    threading.Thread(target=_bg, daemon=True).start()
    return local


def build_models(args, dev):
    """VAE 码本冻结 + D12 生成模型（d12 预训练，跳过形状不匹配 key）。"""
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
        bidirectional=args.bidirectional, separate_decoding=args.separate_decoding,
        separator=args.separator, type_pos=args.type_pos, indep=args.indep,
        multi_cond=True, num_classes=args.num_classes)
    if args.d12_pretrained_path:
        msd = torch.load(args.d12_pretrained_path, map_location='cpu')
        skip = [k for k, v in msd.items()
                if k in d12.state_dict() and d12.state_dict()[k].shape != v.shape]
        msd = {k: v for k, v in msd.items() if k not in skip}
        missing, unexpected = d12.load_state_dict(msd, strict=False)
        print(f"[D12] 预训练 {args.d12_pretrained_path}: 跳过形状不匹配 {skip}，"
              f"随机初始化 {sorted(missing)}", flush=True)
    return vae, d12


def tokenize_pair(vae, img, msk, vpn, mask_idx=None,
                  lesion_interior_weight=5.0, lesion_boundary_weight=10.0):
    """双流 tokenize + 交错 teacher forcing 输入。

    返回:
      x_in   (B, 1358, Cvae)  mask流⊕image流交错，scale0 去掉（first_l 由 sos 生成）
      tgt    (B, 1360)        交错目标（含 scale0 的 2 个 token）
      weights (B, 1360) 或 None   每个 token 的 CE 权重（mask token=1；image token 按类别）
    """
    with torch.no_grad():
        i_idx = vae.img_to_idxBl(img, v_patch_nums=vpn)
        m_idx = vae.img_to_idxBl(msk, v_patch_nums=vpn)
        img_tf = vae.quantize.idxBl_to_ar_input(i_idx)   # 9 scales
        msk_tf = vae.quantize.idxBl_to_ar_input(m_idx)   # 9 scales
    x_parts = []
    for si in range(1, len(vpn)):
        x_parts.append(msk_tf[si - 1])
        x_parts.append(img_tf[si - 1])
    x_in = torch.cat(x_parts, dim=1)
    t_parts = [torch.cat([m_idx[0], i_idx[0]], dim=1)]
    for si in range(1, len(vpn)):
        t_parts.append(torch.cat([m_idx[si], i_idx[si]], dim=1))
    tgt = torch.cat(t_parts, dim=1)

    weights = None
    if mask_idx is not None:
        import torch.nn.functional as _F
        B = img.shape[0]
        H = img.shape[-1]
        w_parts = [torch.ones(B, vpn[0] * vpn[0] * 2, device=img.device)]
        for si in range(1, len(vpn)):
            pn = vpn[si]
            # 把 (B,H,H) 类别图压到 (B,pn,pn)：最近邻下采样（每格取中心像素的类别）
            idx = mask_idx.to(img.device).float()
            idx_small = _F.interpolate(idx.unsqueeze(1), size=(pn, pn), mode='nearest')[:, 0]
            idx_small = idx_small.long()
            is_lesion = idx_small >= 2
            # 边界：格子和右/下邻居类别不同
            diff_r = (idx_small[:, :, 1:] != idx_small[:, :, :-1])
            diff_d = (idx_small[:, 1:, :] != idx_small[:, :-1, :])
            boundary = _F.pad(diff_r, (0, 1), value=False) | _F.pad(diff_d, (0, 0, 0, 1), value=False)
            # image token 权重：lesion 内部=interior_weight，边界=boundary_weight，其余=1
            img_w = torch.ones(B, pn * pn, device=img.device)
            img_w[is_lesion.reshape(B, -1)] = lesion_interior_weight
            img_w[boundary.reshape(B, -1)] = lesion_boundary_weight
            # mask token 权重=1
            msk_w = torch.ones(B, pn * pn, device=img.device)
            w_parts.append(torch.cat([msk_w, img_w], dim=1))
        weights = torch.cat(w_parts, dim=1)
    return x_in, tgt, weights

def train_epoch(d12, vae, dataloader, optimizer, progress_bar, rank, args):
    d12.train()
    vpn = tuple(args.v_patch_nums)
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(int(getattr(args, 'epoch', 0)))
    autocast_enabled = args.mixed_precision == 'bf16'
    warmup_it = int(args.lr_warmup_steps * args.max_train_steps)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < getattr(args, 'skip_batches', 0):
            continue
        # lr/wd 逐 step 调度（lin0 + warmup）
        lr_wd_annealing(args.lr_scheduler, optimizer, args.learning_rate,
                        args.weight_decay, args.weight_decay_end,
                        args.completed_steps, warmup_it,
                        args.max_train_steps, wp0=args.wp0, wpe=args.wpe)
        img = batch['image'].to(device, non_blocking=True)
        msk = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)
        ctype = batch['type'].to(device, non_blocking=True)

        x_in, tgt, weights = tokenize_pair(vae, img, msk, vpn,
                                            mask_idx=batch.get('mask_idx'),
                                            lesion_interior_weight=args.lesion_interior_weight,
                                            lesion_boundary_weight=args.lesion_boundary_weight)

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
        torch.nn.utils.clip_grad_norm_(d12.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        progress_bar.update(1)
        args.completed_steps += 1

        if rank == 0:
            if args.completed_steps % args.log_interval == 0:
                acc = (logits.argmax(-1) == tgt).float().mean().item()
                wandb.log({"d12/ce_loss": loss.item(),
                           "d12/token_acc": acc,
                           "d12/lr": optimizer.param_groups[0]['lr']},
                          step=args.completed_steps)
            if (isinstance(args.save_interval, int) and args.save_interval > 0
                    and args.completed_steps > 0
                    and args.completed_steps % args.save_interval == 0):
                save_checkpoint(d12, optimizer, args, args.epoch, args.completed_steps)


@torch.no_grad()
def validate(d12, vae, val_loader, args):
    """val CE + token 准确率（早停判据）。固定抽前 val_max_batches 个 batch。"""
    module = d12.module if hasattr(d12, 'module') else d12
    was_training = module.training
    module.eval()
    vpn = tuple(args.v_patch_nums)
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    autocast_enabled = args.mixed_precision == 'bf16'
    for bi, batch in enumerate(val_loader):
        if args.val_max_batches and bi >= args.val_max_batches:
            break
        img = batch['image'].to(device, non_blocking=True)
        msk = batch['mask'].to(device, non_blocking=True)
        cls = batch['cls'].to(device, non_blocking=True)
        ctype = batch['type'].to(device, non_blocking=True)
        x_in, tgt, _ = tokenize_pair(vae, img, msk, vpn)
        if autocast_enabled:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = module(cls, x_in, ctype)
        else:
            logits = module(cls, x_in, ctype)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(),
                                tgt.reshape(-1), reduction='sum')
        acc = (logits.argmax(-1) == tgt).float().sum()
        tot_loss += loss.item()
        tot_acc += acc.item()
        n += tgt.numel()
    if was_training:
        module.train()
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


def _ct_brain_window(img_path, gen01, bg):
    """CT 显示。窗宽固定 70。

    原图用测得的颅内窗位 30（HU -5 到 65）。
    生成图的整体亮度偏低，窗位改成生成值在头颅内的中位数，窗宽仍是 70。
    背景保持纯黑。返回 (gt, gen)，各为 (3,H,W) float [0,1]。
    """
    import numpy as np
    from datasets.braingfm import _load_nii, _find_bg_threshold, denormalize_slice

    raw = _load_nii(img_path).astype(np.float32)
    thresh = _find_bg_threshold(raw)
    fg = raw > thresh
    fg_vals = raw[fg]
    fmin = float(fg_vals.min()) if fg_vals.size else 0.0
    fmax = float(fg_vals.max()) if fg_vals.size else 1.0
    if fmax - fmin < 1e-8:
        fmin, fmax = 0.0, 1.0
    ww = 70.0

    def _win(hu, level):
        lo, hi = level - ww / 2.0, level + ww / 2.0
        x = np.clip((hu - lo) / (hi - lo), 0, 1).astype(np.float32)
        x[bg.numpy()] = 0
        t = torch.from_numpy(np.ascontiguousarray(x))
        return t.unsqueeze(0).repeat(3, 1, 1)

    gt = _win(raw, 30.0)
    gen_hu = denormalize_slice(gen01[0].numpy(), fmin, fmax, float(raw.min()))
    inside = gen_hu[~bg.numpy()]
    level = float(np.median(inside)) if inside.size else 30.0
    return gt, _win(gen_hu, level)


def _rotate_dwi_indices(val_dataset, rot, n_pick):
    """每个 DWI 病例留脑区最大的一张，占比至少 0.15，再按 rot 轮换。"""
    import numpy as np
    from datasets.braingfm import _load_nii

    best = {}
    for idx, s in enumerate(val_dataset.samples):
        if s[3] != 'dwi':
            continue
        raw = _load_nii(s[1])
        fg = float((np.asarray(raw) > 0).mean()) if np.asarray(raw).size else 0.0
        case = os.path.dirname(s[0])
        prev = best.get(case)
        if prev is None or fg > prev[0]:
            best[case] = (fg, idx)
    good = [(fg, idx, case) for case, (fg, idx) in best.items() if fg >= 0.15]
    good.sort(key=lambda t: t[2])
    if not good:
        return []
    start = int(rot) % len(good)
    ordered = good[start:] + good[:start]
    picked = [idx for _, idx, _ in ordered[:max(int(n_pick), 1)]]
    print(f"[vis] DWI 轮换 rot={rot} -> {picked} "
          f"（候选 {len(good)} 例，脑区占比≥0.15）", flush=True)
    return picked


@torch.no_grad()
def visualize_generation(d12, vae, val_dataset, args, epoch, save_dir):
    """每 epoch 结束：val 集每模态取 vis_per_mod 例做 mask->图像 条件生成。

    - 用当前权重（DDP 裸 module，绕过 DDP forward）
    - 生成走 conditional_infer_cfg（与 infer_braingfm.py 同一推理路径，nii.gz 出图同源）
    - 拼图两列（GT | 生成），存 {save_dir}/vis_epoch{N}.png + wandb
    - 生成图下半（gen[:,:,H:,:]）= 图像；上半 = mask（与推理脚本语义一致）
    """
    from PIL import Image, ImageDraw
    module = d12.module if hasattr(d12, 'module') else d12
    was_training = module.training
    module.eval()
    dev = next(module.parameters()).device
    vpn = tuple(args.v_patch_nums)
    H = args.image_size

    # 每模态挑 brain 占比最大的 vis_per_mod 个样本（空切片看不出生成效果）
    ratio_cache = {}
    try:
        import json as _json
        with open(os.path.join(args.data_dir, '.braingfm_brain_ratio_cache.json')) as f:
            ratio_cache = _json.load(f)
    except Exception:
        pass
    cand = {}
    for idx, s in enumerate(val_dataset.samples):
        cand.setdefault(s[3], []).append((ratio_cache.get(s[1], 0.0), idx))
    want_idx = {}
    for mod, lst in cand.items():
        if mod == 'dwi':
            # 缓存里没有 DWI，占比全是 0，排序会固定落到最后一张边缘小切片。
            # 按 mask 真实前景挑脑区够大的病例，并用 step 轮换，每次可视化换一张。
            step = int(getattr(args, 'completed_steps', 0) or 0)
            rot = (step // 2000) if step >= 2000 else int(epoch)
            dwi_idx = _rotate_dwi_indices(val_dataset, rot, args.vis_per_mod)
            if dwi_idx:
                want_idx[mod] = dwi_idx
                continue
        lst.sort(reverse=True)   # brain ratio 降序
        want_idx[mod] = [i for _, i in lst[:args.vis_per_mod]]
    picked = {mod: [(i, val_dataset[i]) for i in idxs] for mod, idxs in want_idx.items()}

    rows = []   # (label, mask_chw01, gt_chw01, gen_chw01)
    for mod in sorted(picked):
        for i, item in picked[mod]:
            img = item['image'].unsqueeze(0).to(dev)      # (1,3,H,W) [-1,1]
            msk = item['mask'].unsqueeze(0).to(dev)
            cls = torch.tensor([item['cls']], device=dev)
            ctype = torch.tensor([item['type']], device=dev)
            with torch.no_grad():
                c_cond = vae.img_to_idxBl(msk, v_patch_nums=vpn)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    gen = module.conditional_infer_cfg(
                        B=1, label_B=cls, cfg=tuple(args.cfg),
                        top_k=args.top_k, top_p=args.top_p,
                        g_seed=args.seed * 10000 + epoch * 100 + i,
                        cond_type=ctype, c_mask=c_cond, c_img=None)
            gen_img = gen[:, :, H:, :].clamp(0, 1)[0].float().cpu()   # (3,H,W) [0,1]
            img_cpu = img[0].float().cpu()
            # 原图背景（归一化后为 -1）不参与重建，生成图这些位置保持纯黑
            bg = img_cpu.amax(dim=0) <= -0.95
            gen_img[:, bg] = 0
            if mod == 'ct':
                gt, gen_img = _ct_brain_window(val_dataset.samples[i][0], gen_img, bg)
                label = f'{mod} #{i}  WW70'
            else:
                gt = ((img_cpu + 1) / 2).clamp(0, 1)
                label = f'{mod} #{i}'
            cond = ((msk + 1) / 2).clamp(0, 1)[0].float().cpu()       # 条件 mask（彩色）
            rows.append((label, cond, gt, gen_img))

    if not rows:
        return
    # PIL 拼图：三列（条件 mask | GT | 生成），每行一样本，左侧标签
    PAD, LBL = 4, 18
    W2 = 3 * H + 4 * PAD
    H2 = len(rows) * (H + LBL) + PAD
    canvas = Image.new('RGB', (W2, H2), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, (label, cond, gt, gen_img) in enumerate(rows):
        y = PAD + r * (H + LBL)
        draw.text((PAD, y - 14), label, fill=(200, 200, 200))
        for c, t in enumerate([cond, gt, gen_img]):
            arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
            canvas.paste(Image.fromarray(arr), (PAD + c * (H + PAD), y))
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'vis_epoch{epoch + 1}.png')
    canvas.save(out)
    try:
        wandb.log({"generation": wandb.Image(out)}, step=args.completed_steps)
    except Exception:
        pass
    print(f"[vis] epoch {epoch+1}: {len(rows)} 例生成对比图 -> {out}", flush=True)
    if was_training:
        module.train()


def process(rank, world_size, args):
    print(f"Running DDP on rank {rank}.", flush=True)
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}')

    if rank == 0:
        wandb.init(project="BrainGFM-D12")

    model_name = args.run_name or f'd12_d{args.depth}_braingfm'
    timestamp = datetime.fromtimestamp(now()).strftime('%Y-%m-%d-%H-%M-%S')
    args.project_dir = f"{args.output_dir}/{timestamp}-{model_name}"
    if rank == 0:
        os.makedirs(args.project_dir, exist_ok=True)

    print("Creating dataset", flush=True)
    dataset = create_dataset(args.dataset_name, args)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=True, timeout=120,
                            persistent_workers=args.num_workers > 0)
    val_loader = None
    if rank == 0:
        val_dataset = create_dataset(args.dataset_name, args, split="val")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               num_workers=min(args.num_workers, 4), pin_memory=True,
                               shuffle=False, drop_last=False, timeout=120,
                               persistent_workers=False)
        print(f"[val] rank0 val samples = {len(val_dataset)}", flush=True)

    args.total_batch_size = args.batch_size * args.gpus * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    vae, d12 = build_models(args, dev)
    d12 = d12.to(dev)
    d12 = DDP(d12, static_graph=False)

    # 参数组：1d/bias 不做 weight decay（D12 官方 filter_params 语义）
    names, paras, para_groups = filter_params(d12, nowd_keys=())
    optimizer = torch.optim.AdamW(para_groups, lr=args.learning_rate,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay)

    if rank == 0:
        print("***** Running training *****", flush=True)
        print(f"  Num examples = {len(dataset)}  Epochs = {args.num_epochs} "
              f"batch/gpu = {args.batch_size} total = {args.total_batch_size}", flush=True)
        print(f"  steps/epoch = {num_update_steps_per_epoch} total = {args.max_train_steps}", flush=True)
        if args.lesion_weighted_loss:
            print(f"  lesion weighted loss: interior={args.lesion_interior_weight} "
                  f"boundary={args.lesion_boundary_weight} reset_schedule={args.reset_schedule}",
                  flush=True)
    progress_bar = tqdm(range(args.max_train_steps), disable=rank != 0)
    args.completed_steps = 0
    args.starting_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        msd = ckpt['model_state_dict']
        msd = {k.replace('module.', ''): v for k, v in msd.items()}
        d12.module.load_state_dict(msd)
        _ep = ckpt.get('epoch', -1)
        if args.reset_schedule:
            args.completed_steps = 0
            args.starting_epoch = 0
            args.skip_batches = 0
            print(f"[resume] {args.resume}: 只接 step={ckpt.get('step')} 的权重，"
                  f"优化器不接，步数和学习率从头计（微调）", flush=True)
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            args.completed_steps = ckpt['step']
            if _ep is not None and _ep >= 0 and args.completed_steps % num_update_steps_per_epoch == 0:
                args.starting_epoch = _ep + 1
                args.skip_batches = 0
            else:
                args.starting_epoch = args.completed_steps // num_update_steps_per_epoch
                args.skip_batches = args.completed_steps % num_update_steps_per_epoch
            print(f"[resume] {args.resume}: step={args.completed_steps} epoch={_ep} -> "
                  f"epoch {args.starting_epoch}, skip 前 {args.skip_batches} batch", flush=True)
            progress_bar.update(args.completed_steps)
    else:
        args.skip_batches = 0

    best_val = float('inf')
    patience = 0
    for epoch in range(args.starting_epoch, args.num_epochs):
        args.epoch = epoch
        sampler.set_epoch(epoch)
        flag_path = os.path.join(tempfile.gettempdir(), 'd12_ckpt_local',
                                 f'epoch_done_{epoch}.flag')
        if os.path.isfile(flag_path):
            try:
                os.remove(flag_path)
            except OSError:
                pass
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}", flush=True)

        train_epoch(d12, vae, dataloader, optimizer, progress_bar, rank, args)
        args.skip_batches = 0

        should_stop = False
        if rank == 0:
            save_checkpoint(d12, optimizer, args, epoch, args.completed_steps)
            print(f"[ckpt] epoch {epoch+1} 已存 step={args.completed_steps}", flush=True)
            if args.early_stop_patience > 0 and val_loader is not None:
                val_ce, val_acc = validate(d12, vae, val_loader, args)
                print(f"[val] epoch {epoch+1}: val_CE={val_ce:.4f} token_acc={val_acc:.4f} "
                      f"(best={best_val:.4f}, patience={patience}/{args.early_stop_patience})", flush=True)
                if val_ce < best_val - 1e-6:
                    best_val = val_ce
                    patience = 0
                    save_checkpoint(d12, optimizer, args, epoch, args.completed_steps, best=True)
                    print(f"[val] ★ 新最佳 val_CE，已存 best_checkpoint.pth (epoch {epoch+1})", flush=True)
                else:
                    patience += 1
                    if patience >= args.early_stop_patience:
                        print(f"[val] 连续 {args.early_stop_patience} 轮不降，早停。", flush=True)
                        should_stop = True
            if val_loader is not None:
                try:
                    visualize_generation(d12, vae, val_loader.dataset, args, epoch,
                                         os.path.join(args.project_dir, 'vis'))
                except Exception as e:
                    print(f"[vis] 可视化失败（不影响训练）: {type(e).__name__}: {e}", flush=True)
            os.makedirs(os.path.dirname(flag_path), exist_ok=True)
            with open(flag_path, 'w') as f:
                f.write('1' if should_stop else '0')
        else:
            t0 = now()
            while not os.path.isfile(flag_path):
                if now() - t0 > 3600:
                    raise RuntimeError(f"rank{rank} 等 flag 超 1 小时")
                sleep(1)
            with open(flag_path) as f:
                should_stop = f.read().strip() == '1'
        if should_stop:
            break

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(process, args=(args.gpus, args), nprocs=args.gpus, join=True)

