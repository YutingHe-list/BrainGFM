
from __future__ import annotations

import colorsys
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationSimulatorMR

DEFAULT_COLORMAP = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)

DEFAULT_BRAINGFM_MODALITIES = ("t1", "t2", "t1ce", "flair", "dwi", "ct", "unknow")
_MODALITY_DIRS = {
    "t1": "T1", "t2": "T2", "t1ce": "T1ce", "flair": "FLAIR",
    "dwi": "DWI", "ct": "CT", "unknow": "Unknow",
}
_N_COLORS = 48

def _build_colormap(n: int = _N_COLORS) -> np.ndarray:

    base = DEFAULT_COLORMAP.tolist()
    out = list(base)
    k = 0
    while len(out) < n:
        h = (k * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append([int(r * 255), int(g * 255), int(b * 255)])
        k += 1
    return np.array(out[:n], dtype=np.uint8)

BRAINGFM_COLORMAP = _build_colormap()

def _find_bg_threshold(x: np.ndarray) -> float:

    xmin = float(x.min())
    xmax = float(x.max())
    data_range = xmax - xmin
    if data_range < 1e-8:
        return xmin + 1e-6
    thresh = xmin + 0.01 * data_range
    fg_ratio = float(np.sum(x > thresh)) / x.size
    if fg_ratio < 0.01:
        p5 = float(np.percentile(x, 5))
        if p5 > xmin + 1e-8:
            thresh = (xmin + p5) / 2.0
        else:
            p20 = float(np.percentile(x, 20))
            if p20 > xmin + 1e-8:
                thresh = (xmin + p20) / 2.0
            else:
                thresh = xmin + data_range * 0.001
    return thresh

def normalize_slice(x: np.ndarray) -> np.ndarray:

    x = x.astype(np.float32, copy=False)
    bg_thresh = _find_bg_threshold(x)
    fg = x > bg_thresh
    if fg.sum() < 10:
        return np.full_like(x, -1.0)
    fg_vals = x[fg]
    fmin, fmax = float(fg_vals.min()), float(fg_vals.max())
    if fmax - fmin < 1e-8:
        fmin, fmax = 0.0, 1.0
    out = np.full_like(x, -1.0)
    out[fg] = np.clip((fg_vals - fmin) / (fmax - fmin) * 2.0 - 1.0, -1.0, 1.0)
    return out

def denormalize_slice(x01: np.ndarray, fg_min: float, fg_max: float, bg_fill: float) -> np.ndarray:

    x01 = np.asarray(x01, dtype=np.float32)
    raw = x01 * (fg_max - fg_min) + fg_min
    raw[x01 < 0.025] = bg_fill
    return raw

def _load_nii(path: str) -> np.ndarray:
    a = np.asanyarray(nib.load(path).dataobj)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a

def _case_hash(source: str, case: str) -> int:
    h = hashlib.md5(f"{source}/{case}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)

class BrainGFMSliceDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 256,
        v_patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        separator: bool = False,
        modalities: Tuple[str, ...] = DEFAULT_BRAINGFM_MODALITIES,
        val_ratio: float = 0.05,
        split_seed: int = 42,
        mask_overlay: bool = True,
        min_brain_ratio: float = 0.0,
        sources: Optional[Tuple[str, ...]] = None,
        colorize: bool = True,
        index_json: Optional[str] = None,
        degrade: Optional[str] = None,
        degrade_delta: float = 0.10,
        mask_mix_prob: float = 0.0,
        max_per_modality: int = 0,
        slices_per_case: int = 0,
        labeled_weight: float = 1.0,
        val_slice_cap: int = 0,
        exclude_val_cases: bool = False,
        use_merged_mask: bool = False,
        modality_pair: Optional[Tuple[str, str]] = None,
        lesion_index_dir: Optional[str] = None,
        lesion_source: str = "",
        rot90_aug: bool = False,
        return_mask_idx: bool = False,
        image_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = "val" if split in ("val", "validation") else "train"
        self.image_size = image_size
        self.v_patch_nums = tuple(v_patch_nums)
        self.separator = separator
        self.modalities = tuple(m for m in modalities if m in _MODALITY_DIRS)
        assert self.modalities, f"无有效模态: {modalities}"
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.mask_overlay = bool(mask_overlay)
        self.min_brain_ratio = float(min_brain_ratio)
        self.sources = set(sources) if sources else None
        self.colorize = colorize
        self.colormap = BRAINGFM_COLORMAP

        self.index_json = index_json.format(split=self.split) if index_json else None

        assert degrade in (None, "rician", "ld"), f"未知退化模式: {degrade}"
        self.degrade = degrade
        self.degrade_delta = float(degrade_delta)
        self._sim = DegradationSimulatorMR() if degrade else None
        self._ld_missing = 0
        self._bad_files = 0
        self.rot90_aug = bool(rot90_aug) and self.split == "train"
        self.return_mask_idx = bool(return_mask_idx)
        self.image_only = bool(image_only)
        self.max_per_modality = int(max_per_modality)
        self.slices_per_case = int(slices_per_case)

        self.labeled_weight = float(labeled_weight)

        if modality_pair is not None:
            src, dst = modality_pair
            assert src in _MODALITY_DIRS and dst in _MODALITY_DIRS, f"模态名错误: {modality_pair}"
            assert src != dst, "源/目标模态不能相同"
        self.modality_pair = tuple(modality_pair) if modality_pair else None

        self.lesion_index_dir = lesion_index_dir
        self.lesion_source = lesion_source
        self._lesion_whitelist = None

        self.mask_mix_prob = float(mask_mix_prob) if not degrade else 0.0
        self._mask_pool = None

        self.val_slice_cap = int(val_slice_cap)
        self.exclude_val_cases = bool(exclude_val_cases)
        self.use_merged_mask = bool(use_merged_mask)

        if self.index_json and os.path.isfile(self.index_json):
            self._load_index_json()
            if self.modality_pair:
                self._filter_modality_pair()
            self._compute_token_seq_len()
            if self.mask_mix_prob > 0 and self._mask_pool is None:
                self._build_mask_pool()
            return

        self.samples = []
        self.label_dirs = {}
        skipped = Counter()
        per_mod = Counter()

        for mod in self.modalities:
            cls_idx = self.modalities.index(mod)
            mod_dir = os.path.join(root, _MODALITY_DIRS[mod])
            if not os.path.isdir(mod_dir):
                skipped[f"{mod}:no_dir"] += 1
                continue
            for source in sorted(os.listdir(mod_dir)):
                src_dir = os.path.join(mod_dir, source)
                if not os.path.isdir(src_dir) or (self.sources and source not in self.sources):
                    continue
                brain_src = os.path.join(root, "Lable", "Brain_mask", source)
                label_src = os.path.join(root, "Lable", "Lable_mask", source)
                has_label = os.path.isdir(label_src) and self.mask_overlay
                for case in sorted(os.listdir(src_dir)):
                    case_dir = os.path.join(src_dir, case)
                    if not os.path.isdir(case_dir):
                        continue

                    bucket = (_case_hash(source, case) + self.split_seed) % 10000
                    is_val = bucket < int(self.val_ratio * 10000)
                    if self.split == "val" and not is_val:
                        continue
                    if self.split == "train" and self.exclude_val_cases and is_val:
                        continue
                    brain_case = os.path.join(brain_src, case)
                    label_case = os.path.join(label_src, case) if has_label else None

                    wl = self._get_lesion_whitelist(source, case)
                    for fn in sorted(os.listdir(case_dir)):
                        if not (fn.startswith("slice_") and fn.endswith(".nii.gz")):
                            continue
                        if fn.endswith("_ld.nii.gz"):
                            continue
                        if wl is not None and fn not in wl:
                            continue
                        img_p = os.path.join(case_dir, fn)
                        msk_p = os.path.join(brain_case, fn)
                        if not os.path.isfile(msk_p):
                            skipped[f"{mod}:no_brain_mask"] += 1
                            continue
                        self.samples.append((img_p, msk_p, cls_idx, mod, fn))
                        if label_case is not None and os.path.isfile(os.path.join(label_case, fn)):
                            self.label_dirs[img_p] = os.path.join(label_case, fn)
                        per_mod[mod] += 1

        if self.min_brain_ratio > 0.0 and self.samples:
            self._filter_by_brain_ratio()

        if self.max_per_modality > 0 and self.samples:
            self._cap_per_modality()
            per_mod = Counter(s[3] for s in self.samples)

        if self.modality_pair:
            n0 = len(self.samples)
            self._filter_modality_pair()
            per_mod = Counter(s[3] for s in self.samples)
            print(f"[BrainGFM] modality_pair {self.modality_pair}: {n0} -> {len(self.samples)}（有源模态配对的样本）")

        if self.split == "val" and self.val_slice_cap > 0 and len(self.samples) > self.val_slice_cap:
            n0 = len(self.samples)
            by_mod = {}
            for s in self.samples:
                by_mod.setdefault(s[3], []).append(s)
            per = max(1, self.val_slice_cap // max(1, len(by_mod)))
            keep = []
            for mod, lst in by_mod.items():
                keep.extend(lst[:per])
            self.samples = keep
            keep_paths = {s[0] for s in self.samples}
            self.label_dirs = {p: q for p, q in self.label_dirs.items() if p in keep_paths}
            per_mod = Counter(s[3] for s in self.samples)
            print(f"[BrainGFM] val_slice_cap={self.val_slice_cap}: {n0} -> {len(self.samples)}（按模态均匀截断，每模态约 {per} 片）")

        if self.index_json:
            self._dump_index_json()

        print(
            f"[BrainGFM] split={self.split} samples={len(self.samples)} "
            f"per-modality={dict(per_mod)} label_overlay={len(self.label_dirs)}"
        )
        if skipped:
            print(f"[BrainGFM] skipped={dict(skipped)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"BrainGFM: 在 {root} 下没找到样本。split={self.split} "
                f"val_ratio={self.val_ratio} sources={self.sources}"
            )

        self._compute_token_seq_len()
        if self.mask_mix_prob > 0 and self._mask_pool is None:
            self._build_mask_pool()

    def _compute_token_seq_len(self):

        L = 0
        for i, pn in enumerate(self.v_patch_nums):
            num_sp = 1 if (i != 0 and self.separator) else 0
            L += (pn * pn + num_sp) * 2
        self._token_seq_len = L

    def _sibling_path(self, img_path: str, mod: str) -> str:

        parts = img_path.split(os.sep)

        for i in range(len(parts) - 1, -1, -1):
            if parts[i] in _MODALITY_DIRS.values() and i >= 1:
                parts[i] = _MODALITY_DIRS[mod]
                return os.sep.join(parts)
        raise ValueError(f"路径中未找到模态目录: {img_path}")

    def _filter_modality_pair(self):

        src, dst = self.modality_pair
        keep = []
        for s in self.samples:
            if s[3] != dst:
                continue
            try:
                if os.path.isfile(self._sibling_path(s[0], src)):
                    keep.append(s)
            except ValueError:
                continue
        self.samples = keep
        if not keep:
            raise RuntimeError(
                f"modality_pair {self.modality_pair}: 找不到任何"
                f" {src}->{dst} 的同病例配对（检查两模态的 source/case/切片名是否对齐）")

    def _load_index_json(self):

        with open(self.index_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        n0 = len(d["samples"])
        keep = [tuple(s) for s in d["samples"]
                if os.path.isfile(s[0]) and os.path.isfile(s[1])]
        self.samples = keep
        self.label_dirs = {k: v for k, v in d.get("label_dirs", {}).items()
        if os.path.isfile(v)}
        print(f"[BrainGFM] 固化索引 {self.index_json} 加载: "
        f"{n0} -> {len(keep)}（剔除已不存在的 {n0 - len(keep)} 个文件），无需扫盘")

    def _dump_index_json(self, fresh_cutoff_sec: int = 600):

        now = time.time()
        n0 = len(self.samples)
        stable = [s for s in self.samples
        if (now - os.path.getmtime(s[0])) > fresh_cutoff_sec
        and (now - os.path.getmtime(s[1])) > fresh_cutoff_sec]
        dropped = n0 - len(stable)
        stable_paths = {s[0] for s in stable}
        label_dirs = {k: v for k, v in self.label_dirs.items() if k in stable_paths}
        d = {
            "modalities": list(self.modalities), "split": self.split,
            "val_ratio": self.val_ratio, "split_seed": self.split_seed,
            "min_brain_ratio": self.min_brain_ratio, "created": now,
            "slices_per_case": self.slices_per_case, "labeled_weight": self.labeled_weight,
            "max_per_modality": self.max_per_modality,
            "samples": [list(s) for s in stable], "label_dirs": label_dirs,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.index_json)), exist_ok=True)
        with open(self.index_json, "w", encoding="utf-8") as f:
            json.dump(d, f)
        self.samples = stable
        self.label_dirs = label_dirs
        msg = f"[BrainGFM] 索引已固化到 {self.index_json}: {n0} -> {len(stable)}"
        if dropped:
            msg += f"（剔除 {dropped} 个最近 {fresh_cutoff_sec // 60} 分钟内仍在写入的文件）"
        print(msg)

    def _cap_per_modality(self):

        cap = self.max_per_modality
        rng = random.Random(self.split_seed)

        if self.slices_per_case > 0:
            cache_path = os.path.join(self.root, ".braingfm_brain_ratio_cache.json")
            ratio_cache = {}
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    ratio_cache = json.load(f)
            except Exception:
                ratio_cache = {}
            n0 = len(self.samples)
            by_case = defaultdict(list)
            for s in self.samples:
                by_case[(s[3], os.path.basename(os.path.dirname(os.path.dirname(s[0]))),
                os.path.basename(os.path.dirname(s[0])))].append(s)
            kept = []
            for (mod, src, case), lst in by_case.items():
                if len(lst) <= self.slices_per_case:
                    kept.extend(lst)
                else:
                    lst = sorted(lst, key=lambda s: ratio_cache.get(s[1], -1.0), reverse=True)
                    kept.extend(lst[:self.slices_per_case])
            self.samples = kept
            print(f"[BrainGFM] slices_per_case={self.slices_per_case}: "
            f"{n0} -> {len(kept)}（每 case 保 brain ratio 最大的 {self.slices_per_case} 片）")

        by_mod_src_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for s in self.samples:
            src = os.path.basename(os.path.dirname(os.path.dirname(s[0])))
            case = os.path.basename(os.path.dirname(s[0]))
            by_mod_src_case[s[3]][src][case].append(s)

        keep = []
        for mod, by_src in sorted(by_mod_src_case.items()):
            src_names = sorted(by_src)
            src_sizes = {src: sum(len(v) for v in by_src[src].values()) for src in src_names}
            n_cases = {src: len(by_src[src]) for src in src_names}
            total = sum(src_sizes.values())
            if total <= cap:
                keep.extend(s for src in src_names for s in chain.from_iterable(by_src[src].values()))
                print(f"[BrainGFM] cap[{mod}]: {total} <= {cap}，全保留")
                continue

                lab = {src: any(p.startswith(os.path.join(self.root, "Lable", "Lable_mask", src) + os.sep)
                    for p in self.label_dirs.values()) for src in src_names}
            if not lab:
                lab = {src: False for src in src_names}
            w = {src: (self.labeled_weight if self.labeled_weight > 0 else 1.0) if lab[src] else 1.0
            for src in src_names}
            wsum = sum(src_sizes[src] * w[src] for src in src_names)
            quota = {src: min(src_sizes[src],
            round(cap * src_sizes[src] * w[src] / wsum)) for src in src_names}
            diff = cap - sum(quota.values())
            order = sorted(src_names, key=lambda s: -(src_sizes[s] - quota[s]))
            i = 0
            while diff != 0 and i < 10000:
                src = order[i % len(order)]
                step = 1 if diff > 0 else -1
                if 0 <= quota[src] + step <= src_sizes[src]:
                    quota[src] += step
                    diff -= step
                i += 1

            mod_keep = []
            for src in src_names:
                for c in sorted(by_src[src]):
                    rng.shuffle(by_src[src][c])
                buckets = [by_src[src][c] for c in sorted(by_src[src])]
                taken, i = [], 0
                while len(taken) < quota[src]:
                    for b in buckets:
                        if i < len(b):
                            taken.append(b[i])
                            if len(taken) >= quota[src]:
                                break
                    if not any(i < len(b) for b in buckets):
                        break
                    i += 1
                mod_keep.extend(taken[:quota[src]])
            keep.extend(mod_keep)
            detail = ", ".join(
                f"{src} {min(quota[src], src_sizes[src])}/{src_sizes[src]}"
                f"({n_cases[src]} case{'*' if lab[src] else ''})"
                for src in src_names)
            print(f"[BrainGFM] cap[{mod}]: {total} -> {len(mod_keep)} | {detail}  (*=labeled×{self.labeled_weight})")
        self.samples = keep

    def _get_lesion_whitelist(self, source, case):

        if not self.lesion_index_dir or source != self.lesion_source:
            return None
        if self._lesion_whitelist is None:
            self._lesion_whitelist = {}
            if os.path.isdir(self.lesion_index_dir):
                for c in os.listdir(self.lesion_index_dir):
                    info = os.path.join(self.lesion_index_dir, c, "info.txt")
                    if not os.path.isfile(info):
                        continue
                    try:
                        with open(info, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip().startswith("exported_z:"):
                                    zs = [int(x) for x in
                                    line.split(":", 1)[1].split(",") if x.strip()]
                                    self._lesion_whitelist[c] = {
                                        f"slice_{z:04d}.nii.gz" for z in zs}
                                    break
                    except Exception:
                        continue
                print(f"[BrainGFM] lesion 白名单加载: "
                f"{len(self._lesion_whitelist)} cases from {self.lesion_index_dir}")
        return self._lesion_whitelist.get(case)

    def _filter_by_brain_ratio(self):
        cache_path = os.path.join(self.root, ".braingfm_brain_ratio_cache.json")
        cache = {}
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        todo = [s for s in self.samples if s[1] not in cache]
        if todo:
            print(f"[BrainGFM] 扫描 brain mask 占比（{len(todo)} 个切片，首次会慢，之后有缓存）...")

            def _ratio(p):
                try:
                    m = _load_nii(p)
                    return float((m > 0).mean()) if m.size else 0.0
                except Exception:
                    return -1.0

            with ThreadPoolExecutor(max_workers=8) as ex:
                ratios = list(ex.map(_ratio, [s[1] for s in todo]))
            for s, r in zip(todo, ratios):
                cache[s[1]] = r
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception:
                pass
        keep = [
            s for s in self.samples
            if cache.get(s[1], -1.0) >= self.min_brain_ratio
        ]
        print(f"[BrainGFM] min_brain_ratio={self.min_brain_ratio}: "
        f"{len(self.samples)} -> {len(keep)}（过滤 {len(self.samples) - len(keep)} 个空/近空切片）")
        self.samples = keep

    def __len__(self) -> int:
        return len(self.samples)

    def _make_degraded(self, img_path: str, img_raw: np.ndarray, fn: str):

        seed = int(hashlib.md5(img_path.encode("utf-8")).hexdigest()[:8], 16)

        if self.degrade == "ld":
            ld_path = img_path.replace(".nii.gz", "_ld.nii.gz")
            if os.path.isfile(ld_path):
                deg_raw = _load_nii(ld_path).astype(np.float32, copy=False)
                deg = normalize_slice(deg_raw)
                params = self._raw_params(deg_raw)
                return (torch.from_numpy(np.ascontiguousarray(deg)).unsqueeze(0).repeat(3, 1, 1),
                        params)
            self._ld_missing += 1

        x = np.asarray(img_raw, dtype=np.float32)

        x_min = float(x.min())
        shift = 0.0
        if x_min < 0:
            shift = -x_min
            x = x + shift
        noisy = self._sim.degrade(x, delta_t=self.degrade_delta, seed=seed)
        if shift > 0:
            noisy = noisy - shift
        deg = normalize_slice(noisy)
        return (torch.from_numpy(np.ascontiguousarray(deg)).unsqueeze(0).repeat(3, 1, 1),
                self._raw_params(noisy))

    @staticmethod
    def _raw_params(raw: np.ndarray) -> torch.Tensor:

        raw = np.asarray(raw, dtype=np.float32)
        bg_fill = float(raw.min())
        thresh = _find_bg_threshold(raw)
        fg = raw[raw > thresh]
        if fg.size < 10:
            return torch.tensor([0.0, 1.0, bg_fill], dtype=torch.float32)
        fg_min, fg_max = float(fg.min()), float(fg.max())
        if fg_max - fg_min < 1e-8:
            fg_min, fg_max = 0.0, 1.0
        return torch.tensor([fg_min, fg_max, bg_fill], dtype=torch.float32)

    def __getitem__(self, idx: int, _retry: int = 0):

        if not (0 <= idx < len(self.samples)):
            raise IndexError(f"index {idx} out of range (len={len(self.samples)})")
        try:
            return self._load_item(idx)
        except Exception as e:
            if _retry >= 50:
                raise RuntimeError(f"连续 50 个样本损坏，数据可能有问题: {e}") from e
            self._bad_files += 1
            if self._bad_files <= 10 or self._bad_files % 100 == 0:
                print(f"[BrainGFM] 跳过损坏样本 idx={idx}: {type(e).__name__}: {e}"
                f"（累计 {self._bad_files}）")
            return self.__getitem__((idx + 1) % len(self.samples), _retry + 1)

    def _colorize_idx_map(self, idx_map: np.ndarray) -> torch.Tensor:

        if self.colorize:
            rgb = self.colormap[idx_map].astype(np.float32) / 255.0
            mask = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            return mask * 2.0 - 1.0
        return torch.from_numpy(idx_map.astype(np.float32) / 255.0 * 2.0 - 1.0).unsqueeze(0).repeat(3, 1, 1)

    def _construct_mask(self, img_path: str, msk_path: str, return_idx: bool = False):

        if self.use_merged_mask:
            merged = msk_path.replace(
                os.sep + "Brain_mask" + os.sep, os.sep + "Merged_mask" + os.sep)
            if os.path.isfile(merged):
                idx_map = _load_nii(merged).astype(np.int64)
                mask = self._colorize_idx_map(idx_map)
                return (mask, idx_map) if return_idx else mask

        brain = (_load_nii(msk_path) > 0)
        idx_map = brain.astype(np.int64)
        label_path = self.label_dirs.get(img_path)
        if label_path is not None:
            label = _load_nii(label_path)
            overlay = label > 0
            if overlay.any():

                lab_idx = np.clip(label.astype(np.int64) + 1, 2, len(self.colormap) - 1)
                idx_map[overlay] = lab_idx[overlay]
        mask = self._colorize_idx_map(idx_map)
        return (mask, idx_map) if return_idx else mask

    def _build_mask_pool(self):

        pool = defaultdict(list)
        for img_p, msk_p, _cls, _mod, _fn in self.samples:
            source = os.path.basename(os.path.dirname(os.path.dirname(img_p)))
            pool[source].append((img_p, msk_p))
        self._mask_pool = dict(pool)
        print(f"[BrainGFM] mask 均衡池: {len(pool)} 个 source，"
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(pool.items())))

    def _sample_balanced_mask(self, rng: "random.Random") -> torch.Tensor:

        if self._mask_pool is None:
            self._build_mask_pool()
        for _ in range(5):
            src = rng.choice(sorted(self._mask_pool.keys()))
            img_p, msk_p = rng.choice(self._mask_pool[src])
            try:
                return self._construct_mask(img_p, msk_p)
            except Exception:
                continue
        return self._construct_mask(*rng.choice(next(iter(self._mask_pool.values()))))

    def _load_item(self, idx: int):
        img_path, msk_path, cls_idx, modality, fn = self.samples[idx]

        img_raw = _load_nii(img_path)
        img = normalize_slice(img_raw)
        if img.shape != (self.image_size, self.image_size):
            raise ValueError(f"{img_path}: 形状 {img.shape} != {(self.image_size, self.image_size)}")
        image = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).repeat(3, 1, 1)

        if self.image_only:
            mask = torch.zeros_like(image)
            mask_idx = None
        elif self.return_mask_idx:
            mask, idx_map = self._construct_mask(img_path, msk_path, return_idx=True)
            mask_idx = torch.from_numpy(np.ascontiguousarray(idx_map.astype(np.int64)))
        else:
            mask = self._construct_mask(img_path, msk_path)
            mask_idx = None

        item = None
        if self.modality_pair:

            src, _dst = self.modality_pair
            src_path = self._sibling_path(img_path, src)
            src_img = normalize_slice(_load_nii(src_path))
            if src_img.shape != (self.image_size, self.image_size):
                raise ValueError(f"{src_path}: 形状 {src_img.shape} != {(self.image_size, self.image_size)}")
            cond_image = torch.from_numpy(np.ascontiguousarray(src_img)).unsqueeze(0).repeat(3, 1, 1)

        ignore = torch.ones(self._token_seq_len, dtype=torch.float32)
        item = {
            "image": image,
            "mask": mask,
            "cls": int(cls_idx),
            "ignore_mask": ignore,
            "ignore_mask_": ignore.clone(),
            "type": torch.tensor(0, dtype=torch.long),
            "modality": modality,
            "filename": f"{os.path.basename(os.path.dirname(img_path))}/{fn}",
        }
        if self.return_mask_idx:
            item["mask_idx"] = mask_idx
        if self.modality_pair:
            item["cond_image"] = cond_image
        if self.mask_mix_prob > 0:
            rng = random.Random()
            if rng.random() < self.mask_mix_prob:
                item["image"] = self._sample_balanced_mask(rng)
        if self.degrade:

            item["degraded"], item["degraded_params"] = self._make_degraded(img_path, img_raw, fn)
        if self.rot90_aug:
            k = random.randrange(4)
            if k:
                for key in ("image", "mask", "mask_idx", "cond_image", "degraded"):
                    t = item.get(key)
                    if t is None or not torch.is_tensor(t):
                        continue
                    if t.dim() == 3:
                        item[key] = torch.rot90(t, k, dims=(1, 2))
                    elif t.dim() == 2:
                        item[key] = torch.rot90(t, k, dims=(0, 1))
        return item
