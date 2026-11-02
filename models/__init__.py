from .vae import VAE
from .class_embedder import ClassEmbedder
from .d12 import D12


def build_d12(
    vae: VAE,
    depth: int,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    aln=1,
    aln_gamma_init=1e-3,
    shared_aln=False,
    layer_scale=-1,
    tau=4,
    cos_attn=False,
    flash_if_available=True,
    fused_if_available=True,
    mask_type="interleave_append",
    cond_drop_rate=0.1,
    bidirectional=False,
    separate_decoding=False,
    separator=False,
    type_pos=False,
    indep=False,
    multi_cond=False,
    num_classes=1000,
):
    if mask_type == "replace":
        mask_factor = 1
    elif mask_type == "interleave_append":
        mask_factor = 2
    else:
        raise NotImplementedError(mask_type)

    return D12(
        vae_local=vae,
        patch_nums=patch_nums,
        num_classes=num_classes,
        depth=depth,
        embed_dim=depth * 64,
        num_heads=depth,
        drop_path_rate=0.1 * depth / 24,
        aln=aln,
        aln_gamma_init=aln_gamma_init,
        shared_aln=shared_aln,
        layer_scale=layer_scale,
        tau=tau,
        cos_attn=cos_attn,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
        mask_factor=mask_factor,
        bidirectional=bidirectional,
        separate_decoding=separate_decoding,
        separator=separator,
        type_pos=type_pos,
        indep=indep,
        multi_cond=multi_cond,
    )
