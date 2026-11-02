"""
Low-field MR operator for the enhancement task.

A lower field mainly reduces SNR. This is not a sequence simulator: it does
not model field strength, bandwidth, or relaxation. Magnitude images follow
Rician noise, so the brain foreground is scaled to [0, 1] and two Gaussian
components are combined in quadrature:

    sigma = sqrt(delta_t * 0.005)
    noisy = sqrt((image + n1)^2 + n2^2)

`delta_t` is in [0, 1]. Larger values are noisier. Voxels that are exactly 0
stay 0, and the foreground is scaled back to its original intensities.
"""

import numpy as np
import torch


class DegradationSimulator_MR:
    """Rician degradation used as the low-field MR stand-in."""

    def __init__(self, device=None, reduction_rate=0.005):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.reduction_rate = torch.tensor(reduction_rate, device=device, dtype=torch.float32)

    def LFNMR_degrade_multiple(self, magnitude, delta_t_list):
        """
        Degrade one magnitude image at several noise levels.

        Args:
            magnitude: tensor already scaled to [0, 1].
            delta_t_list: values in [0, 1].

        Returns:
            List of noisy tensors, one per delta_t.
        """
        results = []
        with torch.no_grad():
            for delta_t in delta_t_list:
                if not torch.is_tensor(delta_t):
                    delta_t = torch.tensor(
                        float(delta_t), device=magnitude.device, dtype=torch.float32
                    )
                sigma = torch.sqrt(delta_t * self.reduction_rate)
                noise_1 = torch.normal(
                    0, float(sigma), size=magnitude.shape,
                    device=magnitude.device, dtype=magnitude.dtype,
                )
                noise_2 = torch.normal(
                    0, float(sigma), size=magnitude.shape,
                    device=magnitude.device, dtype=magnitude.dtype,
                )
                results.append(torch.sqrt((magnitude + noise_1) ** 2 + noise_2 ** 2))
        return results


def add_rician_noise(clean, delta_t, simulator=None):
    """
    Add Rician noise on the nonzero foreground.

    Args:
        clean: 2D magnitude array. Background is exactly 0.
        delta_t: noise level in [0, 1].
        simulator: optional DegradationSimulator_MR.

    Returns:
        Noisy float32 array with the same shape, or None when the foreground
        is too small to scale.
    """
    if not 0 <= delta_t <= 1:
        raise ValueError(f"delta_t must be in [0, 1], got {delta_t}")
    if simulator is None:
        simulator = DegradationSimulator_MR()

    clean = np.squeeze(clean).astype(np.float32)
    if clean.ndim != 2:
        raise ValueError(f"clean must be 2D after squeeze, got shape {clean.shape}")

    background = clean == 0.0
    foreground = ~background
    values = clean[foreground]
    if values.size < 10:
        return None
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return None

    scaled = np.zeros_like(clean)
    scaled[foreground] = (clean[foreground] - vmin) / (vmax - vmin)
    tensor = torch.from_numpy(scaled).to(simulator.device)
    noisy = simulator.LFNMR_degrade_multiple(tensor, [delta_t])[0].detach().cpu().numpy()

    restored = np.zeros_like(clean)
    restored[foreground] = noisy[foreground] * (vmax - vmin) + vmin
    restored[background] = 0.0
    return restored
