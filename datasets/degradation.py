"""
    图像退化模拟器（增强任务用：构造 退化图 -> 干净图 训练对）。
"""
from typing import Optional

import numpy as np
import torch


class DegradationSimulatorMR:
    """Rician 噪声退化器。

    噪声模型：n1, n2 ~ N(0, sigma^2)，noisy = sqrt((x + n1)^2 + n2^2)
    sigma = sqrt(delta_t * reduction_rate)，reduction_rate = 0.005
    """

    def __init__(self, reduction_rate: float = 0.005):
        self.reduction_rate = float(reduction_rate)

    @torch.no_grad()
    def degrade(self, mag_img: np.ndarray, delta_t: float = 0.10,
                seed: Optional[int] = None) -> np.ndarray:
        """对 float32 幅度图加 Rician 噪声，返回同形状 float32。

        seed 给定时为确定性退化（同一切片跨 epoch/val 完全一致，可复现）。
        """
        x = torch.from_numpy(np.ascontiguousarray(mag_img)).float()
        if seed is not None:
            g = torch.Generator().manual_seed(int(seed))
            std = float(np.sqrt(delta_t * self.reduction_rate))
            n1 = torch.normal(0.0, std, size=x.shape, generator=g)
            n2 = torch.normal(0.0, std, size=x.shape, generator=g)
        else:
            std = float(np.sqrt(delta_t * self.reduction_rate))
            n1 = torch.normal(0.0, std, size=x.shape)
            n2 = torch.normal(0.0, std, size=x.shape)
        noisy = torch.sqrt((x + n1) ** 2 + n2 ** 2)
        return noisy.numpy().astype(np.float32)