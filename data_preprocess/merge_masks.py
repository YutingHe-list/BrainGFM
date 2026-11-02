"""把脑区 mask 和标签 mask 合并成一张图。

用法: python data_preprocess/merge_masks.py
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "data", "BrainGFM")
N_COLORS = 48  # 与 datasets/braingfm.py 一致


def load(path):
    a = np.asanyarray(nib.load(path).dataobj)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a


def merge_one(msk_path):
    """返回 (msk_path, merged_uint8 或 None=跳过)"""
    parts = msk_path.split(os.sep)
    src, case, fn = parts[-3], parts[-2], parts[-1]
    out = os.path.join(ROOT, "Lable", "Merged_mask", src, case, fn)
    if os.path.isfile(out):
        return msk_path, "exists"
    try:
        brain = load(msk_path) > 0
        idx = brain.astype(np.uint8)  # 0=bg, 1=brain
        lab_dir = os.path.join(ROOT, "Lable", "Lable_mask", src, case, fn)
        if os.path.isfile(lab_dir):
            label = load(lab_dir)
            overlay = label > 0
            if overlay.any():
                lab_idx = np.clip(label.astype(np.int64) + 1, 2, N_COLORS - 1)
                idx[overlay] = lab_idx[overlay].astype(np.uint8)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        nib.save(nib.Nifti1Image(idx, np.eye(4)), out)
        return msk_path, "ok"
    except Exception as e:
        return msk_path, f"ERR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify", type=int, default=200,
                    help="随机抽 N 个合并文件与实时构造结果比对")
    args = ap.parse_args()

    # 收集全部 Brain_mask 切片
    bm_root = os.path.join(ROOT, "Lable", "Brain_mask")
    tasks = []
    for src in sorted(os.listdir(bm_root)):
        sdir = os.path.join(bm_root, src)
        if not os.path.isdir(sdir):
            continue
        for case in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, case)
            for fn in os.listdir(cdir):
                if fn.startswith("slice_") and fn.endswith(".nii.gz") and not fn.endswith("_ld.nii.gz"):
                    tasks.append(os.path.join(cdir, fn))
    print(f"[merge] 共 {len(tasks)} 个 brain mask 待合并，workers={args.workers}")

    t0, done, errs, skipped = time.time(), 0, [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(merge_one, p): p for p in tasks}
        for fut in as_completed(futs):
            _, status = fut.result()
            done += 1
            if status == "ok":
                pass
            elif status == "exists":
                skipped += 1
            else:
                errs.append((futs[fut], status))
            if done % 50000 == 0:
                rate = done / (time.time() - t0 + 1e-9)
                print(f"[merge] {done}/{len(tasks)} ({rate:.0f}/s) 错误 {len(errs)}", flush=True)

    print(f"[merge] 完成: {done} 个（新建 {done - skipped}，跳过已存在 {skipped}），错误 {len(errs)}，"
          f"耗时 {time.time() - t0:.0f}s")
    for p, s in errs[:20]:
        print(f"  ERR {p}: {s}")

    # 验证：随机抽样，合并文件 == 实时构造
    if args.verify > 0:
        import random
        rng = random.Random(42)
        sample = rng.sample(tasks, min(args.verify, len(tasks)))
        n_bad = 0
        for p in sample:
            parts = p.split(os.sep)
            src, case, fn = parts[-3], parts[-2], parts[-1]
            out = os.path.join(ROOT, "Lable", "Merged_mask", src, case, fn)
            try:
                brain = load(p) > 0
                idx = brain.astype(np.uint8)
                lab = os.path.join(ROOT, "Lable", "Lable_mask", src, case, fn)
                if os.path.isfile(lab):
                    label = load(lab)
                    ov = label > 0
                    if ov.any():
                        li = np.clip(label.astype(np.int64) + 1, 2, N_COLORS - 1)
                        idx[ov] = li[ov].astype(np.uint8)
                got = load(out)
                if not np.array_equal(got, idx):
                    n_bad += 1
                    print(f"  [verify] 不一致: {out}")
            except Exception as e:
                n_bad += 1
                print(f"  [verify] 异常: {out}: {e}")
        print(f"[verify] 抽查 {len(sample)} 个，不一致 {n_bad} 个"
              + ("  ✓ 全部一致" if n_bad == 0 else "  ✗ 有不一致！"))
        sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()