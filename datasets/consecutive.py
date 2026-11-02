import os
from collections import defaultdict

import torch
import torch.utils.data

from .fewshot import _case_of, _source_of, build_fewshot_dataset


def _slice_index(name: str) -> int:
    stem = os.path.basename(name).replace('.nii.gz', '').replace('.nii', '')
    stem = stem.split('_ld')[0]
    num = stem.split('_')[-1]
    return int(num)


class ConsecutiveSliceDataset(torch.utils.data.Dataset):
    def __init__(self, base, pairs, rot90_aug=False):
        self.base = base
        self.pairs = pairs
        self.rot90_aug = rot90_aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        cur_i, prev_i = self.pairs[i]
        item = self.base[cur_i]
        if prev_i is None:
            prev = torch.full_like(item['image'], -1.0)
            has_prev = 0
        else:
            prev = self.base[prev_i]['image']
            has_prev = 1
        if self.rot90_aug:
            k = int(torch.randint(0, 4, ()).item())
            if k:
                item['image'] = torch.rot90(item['image'], k, dims=(-2, -1))
                item['mask'] = torch.rot90(item['mask'], k, dims=(-2, -1))
                prev = torch.rot90(prev, k, dims=(-2, -1))
                if 'mask_idx' in item:
                    item['mask_idx'] = torch.rot90(item['mask_idx'], k, dims=(-2, -1))
        item['prev_image'] = prev
        item['has_prev'] = torch.tensor(has_prev, dtype=torch.float32)
        return item


def build_consecutive_dataset(args, split: str = 'train'):
    saved_n = int(getattr(args, 'lora_num_slices', 20))
    saved_rot = bool(getattr(args, 'lora_rot90_aug', True))
    saved_ratio = float(getattr(args, 'lora_min_brain_ratio', 0.05))
    args.lora_num_slices = 10 ** 9
    args.lora_rot90_aug = False
    args.lora_min_brain_ratio = 0.05
    try:
        base = build_fewshot_dataset(args, split=split)
    finally:
        args.lora_num_slices = saved_n
        args.lora_rot90_aug = saved_rot
        args.lora_min_brain_ratio = saved_ratio
    base.rot90_aug = False

    groups = defaultdict(list)
    for i, sample in enumerate(base.samples):
        key = (_source_of(sample[0]), _case_of(sample[0]))
        groups[key].append((_slice_index(sample[4]), i))
    if not groups:
        raise RuntimeError('没有可用于连续层微调的切片')

    chosen = None
    want = getattr(args, 'lora_case', None)
    if want:
        matched = [k for k in groups if k[1] == want]
        if matched:
            chosen = sorted(matched)[0]
    if chosen is None:
        chosen = sorted(groups)[0]
    ordered = sorted(groups[chosen])

    pairs = [(ordered[0][1], None)]
    for (_, i0), (_, i1) in zip(ordered, ordered[1:]):
        pairs.append((i1, i0))

    source, case = chosen
    print(f"[25d] source={source} case={case}: 去掉脑区占比<5%后 {len(pairs)} 层, "
          f"按层序依次作为下一层条件", flush=True)
    return ConsecutiveSliceDataset(base, pairs, rot90_aug=saved_rot and split == 'train')
