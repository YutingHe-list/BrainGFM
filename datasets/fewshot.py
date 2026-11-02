import json
import os

import numpy as np
import torch.utils.data

from .braingfm import BrainGFMSliceDataset, DEFAULT_BRAINGFM_MODALITIES, _load_nii


def _source_of(img_path: str) -> str:
    # root/<MOD_DIR>/<source>/<case>/slice_xxxx.nii.gz
    return os.path.basename(os.path.dirname(os.path.dirname(img_path)))


def _case_of(img_path: str) -> str:
    return os.path.basename(os.path.dirname(img_path))


def _filter_by_brain_ratio(ds: BrainGFMSliceDataset, min_ratio: float):
    if min_ratio <= 0 or not ds.samples:
        return
    cache_path = os.path.join(ds.root, '.braingfm_brain_ratio_cache.json')
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    keep = []
    for s in ds.samples:
        r = cache.get(s[1])
        if r is None:                      # 兜底:缓存缺失时现算(个别文件)
            try:
                m = _load_nii(s[1])
                r = float((m > 0).mean()) if m.size else 0.0
            except Exception:
                r = -1.0
        if r >= min_ratio:
            keep.append(s)
    if len(keep) < len(ds.samples):
        print(f"[few-shot] brain_ratio>={min_ratio}: {len(ds.samples)} -> {len(keep)} 片")
    ds.samples = keep


def build_fewshot_dataset(args, split: str = "train") -> BrainGFMSliceDataset:
    modality = args.lora_modality
    assert modality in DEFAULT_BRAINGFM_MODALITIES, \
        f"lora_modality 必须是 {DEFAULT_BRAINGFM_MODALITIES} 之一, got {modality}"

    modalities = tuple(getattr(args, "braingfm_modalities", None) or DEFAULT_BRAINGFM_MODALITIES)
    ds = BrainGFMSliceDataset(
        root=args.data_dir,
        split=split,
        image_size=args.image_size,
        v_patch_nums=tuple(args.v_patch_nums),
        separator=bool(getattr(args, "separator", False)),
        modalities=modalities,
        val_ratio=float(getattr(args, "braingfm_val_ratio", 0.05)),
        split_seed=int(getattr(args, "braingfm_split_seed", 42)),
        mask_overlay=bool(getattr(args, "braingfm_mask_overlay", True)),
        min_brain_ratio=0.0,
        sources=tuple(args.braingfm_sources) if getattr(args, "braingfm_sources", None) else None,
        index_json=getattr(args, "braingfm_index_json", None),
        labeled_weight=float(getattr(args, "braingfm_labeled_weight", 1.0)),
        exclude_val_cases=bool(getattr(args, "braingfm_exclude_val_cases", False)),
        use_merged_mask=bool(getattr(args, "braingfm_use_merged_mask", True)),
        rot90_aug=bool(getattr(args, "lora_rot90_aug", True)) and split == "train",
        return_mask_idx=bool(getattr(args, "lesion_weighted_loss", False)),
    )

    # ---- 近空切片过滤(index json 路径下构造函数不做,这里基于缓存统一做) ----
    _filter_by_brain_ratio(ds, float(getattr(args, "lora_min_brain_ratio", 0.05)))

    n0 = len(ds.samples)
    # ---- 逐级过滤:模态 -> source -> case ----
    cand = [s for s in ds.samples if s[3] == modality]
    if not cand:
        raise RuntimeError(f"模态 {modality} 在数据集中没有任何样本")
    if getattr(args, "lora_source", None):
        cand = [s for s in cand if _source_of(s[0]) == args.lora_source]
        if not cand:
            raise RuntimeError(f"source={args.lora_source} 下没有 {modality} 样本")
    if getattr(args, "lora_case", None):
        cand = [s for s in cand if _case_of(s[0]) == args.lora_case]
        if not cand:
            raise RuntimeError(f"case={args.lora_case} 下没有 {modality} 样本")

    # ---- 确定性均匀采样 N 片(覆盖整个候选序列,如一个 case 的全部切片)----
    n_sel = int(getattr(args, "lora_num_slices", 20))
    if len(cand) > n_sel:
        pos = np.linspace(0, len(cand) - 1, n_sel).round().astype(int)
        sel = []
        for p in pos:            # 保序去重
            if not sel or sel[-1] != p:
                sel.append(int(p))
        cand = [cand[p] for p in sel]
    ds.samples = cand

    n_cases = len({_case_of(s[0]) for s in ds.samples})
    n_sources = len({_source_of(s[0]) for s in ds.samples})
    print(f"[few-shot] modality={modality} source={args.lora_source or '*'} "
          f"case={args.lora_case or '*'}: {n0} -> {len(ds.samples)} 片 "
          f"(来自 {n_sources} 个 source / {n_cases} 个 case, "
          f"min_brain_ratio={getattr(args, 'lora_min_brain_ratio', 0.05)})")
    return ds


# ---------------------------------------------------------------------------
# 任务子类 few-shot(二级条件:模态之下再按 task class 划分)
# ---------------------------------------------------------------------------
class TaskFewShotDataset(torch.utils.data.Dataset):
    """在 BrainGFMSliceDataset 之上叠加任务子类标签。

    pairs: [(base_idx, task_cls), ...];item 额外带 'task_cls'。
    """

    def __init__(self, ds, pairs, class_names, modality):
        self.ds = ds
        self.pairs = pairs
        self.class_names = class_names
        self.modality = modality

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        base_idx, tcls = self.pairs[i]
        item = self.ds[base_idx]
        item['task_cls'] = torch.tensor(tcls, dtype=torch.long)
        return item


def build_task_fewshot_dataset(args, split: str = "train"):
    from ruamel.yaml import YAML as _YAML

    with open(args.task_cases_config, encoding='utf-8') as f:
        cfg = _YAML(typ='safe').load(f)

    modality = cfg.get('modality') or args.lora_modality
    assert modality in DEFAULT_BRAINGFM_MODALITIES, f"任务清单模态非法: {modality}"
    n_per = int(cfg.get('slices_per_class', 20))
    classes = cfg['classes']
    class_names = list(cfg.get('class_names') or classes.keys())
    assert len(class_names) == len(classes) >= 1

    modalities = tuple(getattr(args, "braingfm_modalities", None) or DEFAULT_BRAINGFM_MODALITIES)
    ds = BrainGFMSliceDataset(
        root=args.data_dir, split=split, image_size=args.image_size,
        v_patch_nums=tuple(args.v_patch_nums),
        separator=bool(getattr(args, "separator", False)),
        modalities=modalities,
        val_ratio=float(getattr(args, "braingfm_val_ratio", 0.05)),
        split_seed=int(getattr(args, "braingfm_split_seed", 42)),
        mask_overlay=bool(getattr(args, "braingfm_mask_overlay", True)),
        min_brain_ratio=0.0,
        sources=tuple(args.braingfm_sources) if getattr(args, "braingfm_sources", None) else None,
        index_json=getattr(args, "braingfm_index_json", None),
        labeled_weight=float(getattr(args, "braingfm_labeled_weight", 1.0)),
        exclude_val_cases=bool(getattr(args, "braingfm_exclude_val_cases", False)),
        use_merged_mask=bool(getattr(args, "braingfm_use_merged_mask", True)),
        rot90_aug=bool(getattr(args, "lora_rot90_aug", True)) and split == "train",
        return_mask_idx=bool(getattr(args, "lesion_weighted_loss", False)),
    )
    _filter_by_brain_ratio(ds, float(getattr(args, "lora_min_brain_ratio", 0.05)))

    # 目标模态的候选池:(list_idx, sample)
    pool = [(i, s) for i, s in enumerate(ds.samples) if s[3] == modality]
    if not pool:
        raise RuntimeError(f"模态 {modality} 无样本")

    pairs = []
    for tcls, cname in enumerate(class_names):
        want = {(c['source'], c['case']) for c in classes[cname]}
        cand = [(i, s) for i, s in pool
                if (_source_of(s[0]), _case_of(s[0])) in want]
        if not cand:
            raise RuntimeError(f"子类 {cname} 没有可用样本: {sorted(want)}")
        if len(cand) > n_per:
            pos = np.linspace(0, len(cand) - 1, n_per).round().astype(int)
            sel = []
            for p in pos:
                if not sel or sel[-1] != p:
                    sel.append(int(p))
            cand = [cand[p] for p in sel]
        pairs.extend((i, tcls) for i, _ in cand)
        print(f"[task-few-shot] 子类 {cname}(cls={tcls}): {len(cand)} 片 "
              f"来自 {sorted(want)}")

    tds = TaskFewShotDataset(ds, pairs, class_names, modality)
    print(f"[task-few-shot] modality={modality}: 共 {len(pairs)} 片 / "
          f"{len(class_names)} 子类, 每类≈{n_per} 片")
    return tds, len(class_names), class_names, modality