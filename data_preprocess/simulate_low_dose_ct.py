"""
Low-dose CT operator for the enhancement task.

One axial slice, in Hounsfield units:
    HU -> linear attenuation -> fan-beam sinogram
    -> Poisson noise -> FBP -> HU

Noise is injected in the projection domain because that is where tube current
reduces the photon count. `dose_percent` is the retained fraction of a routine
acquisition; a smaller value gives a larger variance. Outside the head, the
clean background value is written back.

Requires an NVIDIA GPU (Astra CUDA projector and FBP_CUDA):
    conda install -c astra-toolbox astra-toolbox
"""

import math

import numpy as np


# Fan-beam geometry in Astra pixel units.
N_ANGLES = 720
DET_COUNT = 1024
DET_WIDTH = 1.685839319229126
SOURCE_ORIGIN = 6.004500331878662e+02
ORIGIN_DET = 4.851499423980713e+02
# Linear attenuation of water. HU 0 maps to this value.
MU_WATER = 0.0192
# Incident-photon scale inside the Poisson variance.
PHOTON_SCALE = 1e6


def low_dose_create(ct_slice, dose_percent):
    """
    Simulate one low-dose CT slice.

    Args:
        ct_slice: 2D array (H, W) in HU, without a +1024 offset.
        dose_percent: retained dose in (0, 100]. Example: 20 means 20% dose.

    Returns:
        2D float32 array (H, W) in HU.
    """
    try:
        import astra
    except ImportError:
        raise ImportError(
            "astra-toolbox is required: conda install -c astra-toolbox astra-toolbox"
        ) from None

    if not 0 < dose_percent <= 100:
        raise ValueError(f"dose_percent must be in (0, 100], got {dose_percent}")

    dose = dose_percent / 100.0
    image = np.squeeze(ct_slice).astype(np.float64)
    if image.ndim != 2:
        raise ValueError(f"ct_slice must be 2D after squeeze, got shape {ct_slice.shape}")
    height, width = image.shape

    # HU -> linear attenuation: mu = mu_water * (1 + HU / 1000)
    mu = image * MU_WATER / 1000.0 + MU_WATER

    vol_geom = astra.create_vol_geom([height, width])
    angles = np.linspace(math.pi, -math.pi, N_ANGLES)
    proj_geom = astra.create_proj_geom(
        "fanflat", DET_WIDTH, int(DET_COUNT), angles, SOURCE_ORIGIN, ORIGIN_DET
    )
    proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
    try:
        projector = astra.OpTomo(proj_id)
        # The 1/2 is inverted after FBP.
        sinogram = projector * np.mat(mu) / 2

        unit_noise = np.random.normal(0, 1, N_ANGLES * DET_COUNT)
        noise = unit_noise * np.sqrt((1 - dose) / dose * (np.exp(sinogram) / PHOTON_SCALE))
        sinogram_noisy = np.reshape(noise + sinogram, [N_ANGLES, -1])
        reconstructed = projector.reconstruct("FBP_CUDA", sinogram_noisy)
    finally:
        astra.projector.delete(proj_id)

    reconstructed = np.reshape(reconstructed, (height, width))
    low_dose = (reconstructed * 2 - MU_WATER) / MU_WATER * 1000

    background = float(image.min())
    low_dose[image <= background + 1] = background
    return low_dose.astype(np.float32)
