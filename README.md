"""重建 BrainGFM 训练/验证索引。

全量索引（默认写到 {data_dir}/.braingfm_index_{split}.json）:
  - 只用干净切片（不含任何 _ld 加噪数据）
  - train = 干净全量 - val
  - val 含 7 模态：RICH 按比例抽；CT/DWI/unknow 各取若干病例

10 倍缩减小索引（.braingfm_index_small_{split}.json）:
  - 稀少模态尽量全保留，其余按病例轮转缩到总量

用法: python rebuild_index_phase2.py
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict

from datasets.braingfm import (
    DEFAULT_BRAINGFM_MODALITIES,
    _MODALITY_DIRS,
    _case_hash,
    _load_nii,
)

ROOT = "./data/BrainGFM"                  # 可用 --root 覆盖
SPLIT_SEED = 42
VAL_RATIO = 0.05                          # 按病例比例抽验证集，不写死切片数
SMALL_FRACTION = 0.1                      # 小索引约为全量的 1/10
# 可选：扫描时跳过的来源（按需修改/清空）
EXCLUDE_SOURCES = set()
RARE = {"ct", "dwi", "unknow"}
RICH = ("t1", "t1ce", "t2", "flair")
FULL_TRAIN_JSON = os.path.join(ROOT, ".braingfm_index_train.json")
FULL_VAL_JSON = os.path.join(ROOT, ".braingfm_index_val.json")
SMALL_TRAIN_JSON = os.path.join(ROOT, ".braingfm_index_small_train.json")
SMALL_VAL_JSON = os.path.join(ROOT, ".braingfm_index_small_val.json")


def _scan_all():
    """只扫干净切片，跳过全部 _ld 加噪文件。返回 samples, label_dirs。"""
    samples = []
    label_dirs = {}
    skipped = Counter()
    for mod in DEFAULT_BRAINGFM_MODALITIES:
        cls_idx = DEFAULT_BRAINGFM_MODALITIES.index(mod)
        mod_dir = os.path.join(ROOT, _MODALITY_DIRS[mod])
        if not os.path.isdir(mod_dir):
            continue
        for source in sorted(os.listdir(mod_dir)):
            if source in EXCLUDE_SOURCES:
                skipped[f"{mod}:excluded:{source}"] += 1
                continue
            src_dir = os.path.join(mod_dir, source)
            if not os.path.isdir(src_dir):
                continue
            brain_src = os.path.join(ROOT, "Lable", "Brain_mask", source)
            label_src = os.path.join(ROOT, "Lable", "Lable_mask", source)
            has_label = os.path.isdir(label_src)
            for case in sorted(os.listdir(src_dir)):
                case_dir = os.path.join(src_dir, case)
                if not os.path.isdir(case_dir):
                    continue
                brain_case = os.path.join(brain_src, case)
                label_case = os.path.join(label_src, case) if has_label else None
                for fn in sorted(os.listdir(case_dir)):
                    if not (fn.startswith("slice_") and fn.endswith(".nii.gz")):
                        continue
                    if fn.endswith("_ld.nii.gz"):
                        continue  # 加噪退化数据，不进生成训练
                    img_p = os.path.join(case_dir, fn)
                    msk_p = os.path.join(brain_case, fn)
                    if not os.path.isfile(msk_p):
                        skipped[f"{mod}:no_brain_mask"] += 1
                        continue
                    samples.append((img_p, msk_p, cls_idx, mod, fn))
                    if label_case is not None and os.path.isfile(os.path.join(label_case, fn)):
                        label_dirs[img_p] = os.path.join(label_case, fn)
    print(f"[scan] 干净切片 {len(samples)}（已排除全部 _ld） skipped={dict(skipped)}")
    return samples, label_dirs


def _case_key(s):
    return (s[3], os.path.basename(os.path.dirname(os.path.dirname(s[0]))),
            os.path.basename(os.path.dirname(s[0])))


def _group_by_case(samples):
    by = defaultdict(list)
    for s in samples:
        by[_case_key(s)].append(s)
    return by


def _is_val_case(source, case, val_ratio):
    bucket = (_case_hash(source, case) + SPLIT_SEED) % 10000
    return bucket < int(val_ratio * 10000)


def _dump(path, samples, label_dirs, extra):
    keep_paths = {s[0] for s in samples}
    ld = {k: v for k, v in label_dirs.items() if k in keep_paths}
    d = {
        "modalities": list(DEFAULT_BRAINGFM_MODALITIES),
        "split_seed": SPLIT_SEED,
        "created": time.time(),
        "samples": [list(s) for s in samples],
        "label_dirs": ld,
        **extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"[dump] {path}: {len(samples)}  label={len(ld)}")


def _stats(tag, samples):
    c = Counter(s[3] for s in samples)
    cases = set(_case_key(s) for s in samples)
    print(f"[{tag}] n={len(samples)} cases={len(cases)} per-mod={dict(sorted(c.items()))}")


def _pick_val_cases(by_case, n_val, seed):
    """确定性抽出 n_val 个病例进 val（按 source/case 名排序后用 seed 打乱）。"""
    keys = sorted(by_case)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = min(n_val, len(keys))
    val_keys = set(keys[:n_val])
    val, train = [], []
    for k, lst in by_case.items():
        (val if k in val_keys else train).extend(lst)
    return train, val, n_val


def build_full(samples, label_dirs):
    """RICH 模态按比例抽 val；稀少模态各抽若干病例进 val。

    只用干净切片。
    """
    rare = [s for s in samples if s[3] in RARE]
    rich = [s for s in samples if s[3] in RICH]
    print(f"[full] rare={len(rare)} rich={len(rich)} total={len(samples)}")

    val_ratio = VAL_RATIO
    by_case = _group_by_case(rich)
    val_rich, train_rich = [], []
    for (mod, src, case), lst in by_case.items():
        if _is_val_case(src, case, val_ratio):
            val_rich.extend(lst)
        else:
            train_rich.extend(lst)

    train_rare, val_rare = [], []
    for mod in sorted(RARE):
        by = _group_by_case([s for s in rare if s[3] == mod])
        n_want = max(1, int(round(len(by) * val_ratio))) if by else 0
        tr, va, n_got = _pick_val_cases(by, n_want, SPLIT_SEED + hash(mod) % 1000)
        train_rare.extend(tr)
        val_rare.extend(va)
        print(f"[full] rare-val[{mod}]: {n_got}/{len(by)} case -> val 切片 {len(va)}")

    train = train_rich + train_rare
    val = val_rich + val_rare
    _stats("full-train", train)
    _stats("full-val", val)
    _dump(FULL_TRAIN_JSON, train, label_dirs, {
        "split": "train", "kind": "full",
        "val_ratio_rich": val_ratio, "min_brain_ratio": 0.0,
        "note": "只用干净切片不含_ld；val 含7模态",
    })
    _dump(FULL_VAL_JSON, val, label_dirs, {
        "split": "val", "kind": "full",
        "val_ratio_rich": val_ratio,
    })
    return train, val


def build_small(full_train, full_val, label_dirs):
    """按 SMALL_FRACTION 缩全量索引。rare 与 rich 各占一半预算。

    若 rare 总量超过一半预算，rare 预先按病例轮转缩到一半预算，rich 用另一半。
    """
    small_target = max(1, int(len(full_train) * SMALL_FRACTION))
    rare_all = [s for s in full_train if s[3] in RARE]
    rich = [s for s in full_train if s[3] in RICH]
    rare_quota = small_target // 2
    if len(rare_all) > rare_quota:
        rare_by_case = _group_by_case(rare_all)
        rng = random.Random(SPLIT_SEED + 7)
        buckets = []
        for key in sorted(rare_by_case):
            lst = rare_by_case[key][:]
            rng.shuffle(lst)
            buckets.append(lst)
        rare, k = [], 0
        while len(rare) < rare_quota:
            progressed = False
            for b in buckets:
                if k < len(b):
                    rare.append(b[k])
                    progressed = True
                    if len(rare) >= rare_quota:
                        break
            if not progressed:
                break
            k += 1
        print(f"[small] rare {len(rare_all)} -> {len(rare)}（占预算一半，病例轮转）")
    else:
        rare = rare_all
    rare_n = len(rare)
    rich_budget = max(0, small_target - rare_n)
    print(f"[small] rare {rare_n}，rich 预算 {rich_budget}")

    rng = random.Random(SPLIT_SEED)
    by_mod_src_case = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in rich:
        src = os.path.basename(os.path.dirname(os.path.dirname(s[0])))
        case = os.path.basename(os.path.dirname(s[0]))
        by_mod_src_case[s[3]][src][case].append(s)

    # 4 个 RICH 模态平分 rich_budget
    per_mod_cap = rich_budget // max(1, len(RICH))
    leftover = rich_budget - per_mod_cap * len(RICH)
    small_rich = []
    for i, mod in enumerate(RICH):
        cap = per_mod_cap + (1 if i < leftover else 0)
        by_src = by_mod_src_case[mod]
        src_names = sorted(by_src)
        src_sizes = {src: sum(len(v) for v in by_src[src].values()) for src in src_names}
        total = sum(src_sizes.values())
        if total <= cap:
            for src in src_names:
                for lst in by_src[src].values():
                    small_rich.extend(lst)
            print(f"[small] {mod}: {total} <= {cap} 全留")
            continue
        # 源间平摊
        quota = {src: min(src_sizes[src], cap // max(1, len(src_names))) for src in src_names}
        for _ in range(len(src_names) + 2):
            if sum(quota.values()) >= cap:
                break
            deficit = [src for src in src_names if quota[src] < src_sizes[src]]
            if not deficit:
                break
            room = min(cap - sum(quota.values()),
                       min(src_sizes[src] - quota[src] for src in deficit))
            add = max(1, room // len(deficit))
            for src in deficit:
                quota[src] = min(src_sizes[src], quota[src] + add)
        # 病例轮转
        taken_mod = []
        for src in src_names:
            for c in sorted(by_src[src]):
                rng.shuffle(by_src[src][c])
            buckets = [by_src[src][c] for c in sorted(by_src[src])]
            taken, k = [], 0
            while len(taken) < quota[src]:
                progressed = False
                for b in buckets:
                    if k < len(b):
                        taken.append(b[k])
                        progressed = True
                        if len(taken) >= quota[src]:
                            break
                if not progressed:
                    break
                k += 1
            taken_mod.extend(taken[:quota[src]])
        small_rich.extend(taken_mod)
        print(f"[small] {mod}: {total} -> {len(taken_mod)}")

    small_train = small_rich + rare
    if len(small_train) > small_target:
        print(f"[small] 超出预算，从 rare 轮转裁（优先保 case 覆盖）")
        rare_by_case = _group_by_case(rare)
        rare_budget = small_target - len(small_rich)
        buckets = []
        for key in sorted(rare_by_case):
            lst = rare_by_case[key][:]
            rng.shuffle(lst)
            buckets.append(lst)
        taken, k = [], 0
        while len(taken) < rare_budget:
            progressed = False
            for b in buckets:
                if k < len(b):
                    taken.append(b[k])
                    progressed = True
                    if len(taken) >= rare_budget:
                        break
            if not progressed:
                break
            k += 1
        small_train = small_rich + taken[:rare_budget]

    # 小 val：按同样比例缩 full_val，按病例轮转抽
    small_val_target = max(1, int(len(full_val) * SMALL_FRACTION))
    by_mod = defaultdict(list)
    for s in full_val:
        by_mod[s[3]].append(s)
    n0 = sum(len(v) for v in by_mod.values())
    scale = small_val_target / max(1, n0)
    alloc = {m: max(1, round(len(v) * scale)) for m, v in by_mod.items()}
    diff = small_val_target - sum(alloc.values())
    order = sorted(alloc, key=lambda m: -len(by_mod[m]))
    i = 0
    while diff != 0 and i < 500:
        m = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if 1 <= alloc[m] + step <= len(by_mod[m]):
            alloc[m] += step
            diff -= step
        i += 1
    small_val = []
    for mod, lst in by_mod.items():
        by_c = _group_by_case(lst)
        keys = sorted(by_c)
        rng.shuffle(keys)
        buckets = [by_c[k][:] for k in keys]
        for b in buckets:
            rng.shuffle(b)
        taken, k = [], 0
        while len(taken) < alloc[mod]:
            progressed = False
            for b in buckets:
                if k < len(b):
                    taken.append(b[k])
                    progressed = True
                    if len(taken) >= alloc[mod]:
                        break
            if not progressed:
                break
            k += 1
        small_val.extend(taken[:alloc[mod]])
        print(f"[small-val] {mod}: {alloc[mod]} 切片 / {len(set(_case_key(s) for s in taken[:alloc[mod]]))} case")

    _stats("small-train", small_train)
    _stats("small-val", small_val)
    _dump(SMALL_TRAIN_JSON, small_train, label_dirs, {
        "split": "train", "kind": "small_10x",
        "small_fraction": SMALL_FRACTION, "rare_prefer_keep": True,
    })
    _dump(SMALL_VAL_JSON, small_val, label_dirs, {
        "split": "val", "kind": "small_10x",
    })
    return small_train, small_val


def main():
    t0 = time.time()
    samples, label_dirs = _scan_all()
    train, val = build_full(samples, label_dirs)
    build_small(train, val, label_dirs)
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
