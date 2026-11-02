"""BrainGFM 一阶段 VAE 码本训练/微调。

VAE 码本(ch160) + L1/LPIPS，判别器在 disc_start 之前不进计算图。
mask / label 由数据集 image_only 跳过。
"""
import argparse
import math
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ruamel.yaml import YAML
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from datasets import create_dataset
from losses.vqperceptual import VQLPIPSWithDiscriminator
from models.vae import VAE

device = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_vae.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=4.5e-6)
    p.add_argument("--lr_warmup_steps", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--data_dir", type=str, default="./data/BrainGFM")
    p.add_argument("--dataset_name", type=str, default="braingfm")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="experiments/vae_braingfm")
    p.add_argument("--run_name", type=str, default="vae_single")
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--z_channels", type=int, default=32)
    p.add_argument("--ch", type=int, default=160)
    p.add_argument("--v_patch_nums", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    p.add_argument("--disc_start", type=int, default=999999)
    p.add_argument("--disc_weight", type=float, default=0.8)
    p.add_argument("--codebook_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_only", action="store_true", default=False)
    p.add_argument("--braingfm_index_json", type=str, default=None)
    p.add_argument("--braingfm_modalities", type=str, nargs="+", default=None)
    p.add_argument("--braingfm_min_brain_ratio", type=float, default=0.0)
    p.add_argument("--braingfm_val_slice_cap", type=int, default=0)
    p.add_argument("--braingfm_max_per_modality", type=int, default=0)
    p.add_argument("--braingfm_mask_mix_prob", type=float, default=0.0)
    p.add_argument("--braingfm_rot90_aug", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--braingfm_val_ratio", type=float, default=0.05)
    p.add_argument("--braingfm_split_seed", type=int, default=42)
    known = {a.dest for a in p._actions}
    args = p.parse_args()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = YAML(typ="safe").load(f) or {}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
        args = p.parse_args()
    return args


def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank, world_size=world_size, timeout=timedelta(minutes=30),
    )


def save_checkpoint(vae, loss_fn, optimizer_G, scheduler_G, args, epoch, step):
    ckpt = {
        "generator_state_dict": vae.state_dict(),
        "discriminator_state_dict": loss_fn.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": {},
        "scheduler_G_state_dict": scheduler_G.state_dict(),
        "scheduler_D_state_dict": {},
        "epoch": epoch,
        "step": step,
    }
    local_dir = os.path.join(tempfile.gettempdir(), "vae_ckpt_local")
    os.makedirs(local_dir, exist_ok=True)
    fname = f"checkpoint_step_{step:08d}.pth"
    local = os.path.join(local_dir, fname)
    torch.save(ckpt, local)
    import glob as _glob
    olds = sorted(_glob.glob(os.path.join(local_dir, "checkpoint_step_*.pth")),
                  key=os.path.getmtime, reverse=True)
    for stale in olds[2:]:
        try:
            os.remove(stale)
        except OSError:
            pass
    final = os.path.join(args.project_dir, fname)

    def _copy():
        try:
            torch.save(ckpt, final)
        except Exception as e:
            print(f"[checkpoint] 拷到 {final} 失败（本地仍在 {local}）: {e}", flush=True)

    threading.Thread(target=_copy, daemon=True).start()
    print(f"[checkpoint] step={step} 已写 {local}", flush=True)


def validate(vae, val_loader, dev):
    module = vae.module if hasattr(vae, "module") else vae
    module.eval()
    total_l1, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(dev, non_blocking=True)
            recon, _, _, _ = module(images, ret_usages=True)
            l1 = torch.abs(recon.float() - images.float()).mean().item()
            mse = ((recon.float() - images.float()) ** 2).mean()
            psnr = (10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))).item()
            bs = images.size(0)
            total_l1 += l1 * bs
            total_psnr += psnr * bs
            n += bs
    module.train()
    return total_l1 / max(n, 1), total_psnr / max(n, 1), n


def train_epoch(vae, loss_fn, dataloader, optimizer_G, scheduler_G, rank, args):
    vae.train()
    loss_fn.train()
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(int(getattr(args, "epoch", 0)))
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        recon, _, vq_loss, _ = vae(images, ret_usages=True)
        aeloss, _ = loss_fn(
            vq_loss, vq_loss, images, images, recon, recon,
            0, args.completed_steps,
            last_layer=vae.module.get_last_layer(), split="train",
        )
        aeloss.backward()
        optimizer_G.step()
        optimizer_G.zero_grad(set_to_none=True)
        scheduler_G.step()
        args.completed_steps += 1
        if rank == 0 and args.completed_steps % args.log_interval == 0:
            alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"[train] step={args.completed_steps} loss={aeloss.item():.4f} "
                  f"lr={optimizer_G.param_groups[0]['lr']:.3e} peak={alloc:.2f}GiB",
                  flush=True)
        if (rank == 0 and args.save_interval > 0
                and args.completed_steps % args.save_interval == 0):
            save_checkpoint(vae, loss_fn, optimizer_G, scheduler_G,
                            args, args.epoch, args.completed_steps)


def process(rank, world_size, args):
    global device
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(args.seed + rank)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.project_dir = os.path.join(args.output_dir, f"{timestamp}-{args.run_name}")
    if rank == 0:
        os.makedirs(args.project_dir, exist_ok=True)
        print(f"[out] {args.project_dir}", flush=True)

    dataset = create_dataset(args.dataset_name, args, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if rank == 0:
        val_dataset = create_dataset(args.dataset_name, args, split="val")
        print(f"[val] rank0 val samples = {len(val_dataset)}", flush=True)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=min(args.num_workers, 4), pin_memory=True, drop_last=False,
        )
        print(f"[data] train samples = {len(dataset)}  image_only，不读 mask/label", flush=True)

    vae = VAE(
        vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch,
        test_mode=False, share_quant_resi=4, v_patch_nums=tuple(args.v_patch_nums),
    ).to(device)
    vae.train()
    for prm in vae.parameters():
        prm.requires_grad_(True)
    vae = DDP(vae, device_ids=[rank])

    loss_fn = VQLPIPSWithDiscriminator(
        disc_start=args.disc_start, disc_weight=args.disc_weight,
        codebook_weight=args.codebook_weight, disc_conditional=False, disc_in_channels=3,
    ).to(device)
    for prm in loss_fn.perceptual_loss.parameters():
        prm.requires_grad_(False)

    optimizer_G = torch.optim.Adam(vae.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    steps_per_epoch = len(dataloader)
    args.max_train_steps = args.num_epochs * steps_per_epoch

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = vae.load_state_dict(ckpt["generator_state_dict"], strict=False)
        optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
        for group in optimizer_G.param_groups:
            group["lr"] = args.learning_rate
            group["initial_lr"] = args.learning_rate
        if rank == 0:
            print(f"[resume] {args.resume}: 权重和优化器动量来自 step={ckpt.get('step')} "
                  f"epoch={ckpt.get('epoch')}，学习率重新从 {args.learning_rate:.3e} 线性衰减。"
                  f" missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing:
                print(f"[resume] missing 示例: {missing[:5]}", flush=True)
            if unexpected:
                print(f"[resume] unexpected 示例: {unexpected[:5]}", flush=True)

    def _lr_lambda(step):
        if step < 0:
            return 1.0
        return max(0.0, float(args.max_train_steps - step) / float(max(1, args.max_train_steps)))

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, _lr_lambda)

    args.completed_steps = 0
    if rank == 0:
        print(f"  steps/epoch = {steps_per_epoch}  total = {args.max_train_steps}  "
              f"batch/gpu = {args.batch_size}", flush=True)

    best_l1 = float("inf")
    patience = 0
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}", flush=True)
        train_epoch(vae, loss_fn, dataloader, optimizer_G, scheduler_G, rank, args)
        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            if rank == 0:
                val_l1, val_psnr, n = validate(vae, val_loader, device)
                print(f"[val] epoch {epoch + 1}: n={n} val_L1={val_l1:.6f} "
                      f"val_PSNR={val_psnr:.2f} dB  (best={best_l1:.6f})", flush=True)
                improved = val_l1 < best_l1 - 1e-6
                if improved:
                    best_l1 = val_l1
                    patience = 0
                    save_checkpoint(vae, loss_fn, optimizer_G, scheduler_G,
                                    args, epoch, args.completed_steps)
                else:
                    patience += 1
                stop = patience >= args.early_stop_patience > 0
            else:
                stop = False
            flag = torch.tensor([1 if stop else 0], device=device)
            dist.broadcast(flag, src=0)
            if int(flag.item()) == 1:
                if rank == 0:
                    print(f"[val] 连续 {args.early_stop_patience} 轮 val_L1 未下降，停止。",
                          flush=True)
                break
        dist.barrier()

    if rank == 0:
        save_checkpoint(vae, loss_fn, optimizer_G, scheduler_G,
                        args, args.num_epochs - 1, args.completed_steps)
        print("[done] 码本微调结束", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    world = args.gpus
    if world > 1:
        mp.spawn(process, args=(world, args), nprocs=world, join=True)
    else:
        process(0, 1, args)


if __name__ == "__main__":
    main()
