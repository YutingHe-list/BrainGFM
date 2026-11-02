"""数据集工厂：BrainGFM（本项目唯一训练/推理数据集）。"""


def create_dataset(dataset_name, args, split="train"):
    if dataset_name == "braingfm":
        from .braingfm import BrainGFMSliceDataset

        # BrainGFM 切片数据集（7 模态：T1/T1ce/T2/FLAIR/DWI/CT/Unknow，三层目录 nii.gz）
        default_mods = ("t1", "t2", "t1ce", "flair", "dwi", "ct", "unknow")
        modalities = tuple(getattr(args, "braingfm_modalities", None) or default_mods)
        return BrainGFMSliceDataset(
            root=args.data_dir,
            split=split,
            image_size=args.image_size,
            v_patch_nums=tuple(args.v_patch_nums),
            separator=getattr(args, "separator", False),
            modalities=modalities,
            val_ratio=float(getattr(args, "braingfm_val_ratio", 0.05)),
            split_seed=int(getattr(args, "braingfm_split_seed", 42)),
            mask_overlay=bool(getattr(args, "braingfm_mask_overlay", True)),
            min_brain_ratio=float(getattr(args, "braingfm_min_brain_ratio", 0.0)),
            sources=tuple(args.braingfm_sources) if getattr(args, "braingfm_sources", None) else None,
            index_json=getattr(args, "braingfm_index_json", None),
            degrade=getattr(args, "degrade", None),
            degrade_delta=float(getattr(args, "degrade_delta", 0.10)),
            mask_mix_prob=float(getattr(args, "braingfm_mask_mix_prob", 0.0)),
            max_per_modality=int(getattr(args, "braingfm_max_per_modality", 0)),
            slices_per_case=int(getattr(args, "braingfm_slices_per_case", 0)),
            labeled_weight=float(getattr(args, "braingfm_labeled_weight", 1.0)),
            val_slice_cap=int(getattr(args, "braingfm_val_slice_cap", 0)),
            exclude_val_cases=bool(getattr(args, "braingfm_exclude_val_cases", False)),
            use_merged_mask=bool(getattr(args, "braingfm_use_merged_mask", False)),
            modality_pair=tuple(args.braingfm_modality_pair) if getattr(args, "braingfm_modality_pair", None) else None,
            lesion_index_dir=getattr(args, "braingfm_lesion_index_dir", None) or None,
            rot90_aug=bool(getattr(args, "braingfm_rot90_aug", False)),
            return_mask_idx=bool(getattr(args, "lesion_weighted_loss", False)),
            image_only=bool(getattr(args, "image_only", False)),
        )

    raise NotImplementedError(
        f"Unknown dataset: {dataset_name}（本项目仅使用 braingfm；"
        f"官方自然图像数据集分支已移除）")