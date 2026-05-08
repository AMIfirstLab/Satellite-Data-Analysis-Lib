from __future__ import annotations

import numpy as np


class BasePansharpeningModel:
    """팬샤프닝 모델의 기본 인터페이스."""

    def fuse(self, multispectral: np.ndarray, panchromatic: np.ndarray) -> np.ndarray:
        """멀티스펙트럴 영상과 PAN 밴드를 결합한다."""
        raise NotImplementedError


def brovey_fusion(multispectral: np.ndarray, panchromatic: np.ndarray) -> np.ndarray:
    """간단한 Brovey 방식 팬샤프닝을 수행한다."""
    # Sentinel-2에는 표준 PAN 밴드가 없으므로 센서에 따라 이 모듈은 적용되지 않을 수 있다.
    ms = multispectral.astype(np.float32)
    pan = panchromatic.astype(np.float32)
    if ms.ndim != 3:
        raise ValueError("multispectral must be a 3D array.")
    if pan.ndim == 3:
        pan = pan[0]
    denom = np.sum(ms[: min(3, ms.shape[0])], axis=0)
    denom = np.where(denom == 0, 1.0, denom)
    fused = ms * (pan / denom)
    return np.clip(fused, 0, np.iinfo(multispectral.dtype).max if np.issubdtype(multispectral.dtype, np.integer) else 1.0).astype(multispectral.dtype)


class ClassicalPansharpeningModel(BasePansharpeningModel):
    """brovey/ihs/pca 중 가능한 고전적 방식을 선택하는 래퍼."""

    def __init__(self, method: str = "brovey"):
        if method not in {"brovey", "ihs", "pca"}:
            raise ValueError(f"Unsupported pansharpening method: {method}")
        self.method = method

    def fuse(self, multispectral: np.ndarray, panchromatic: np.ndarray) -> np.ndarray:
        if self.method != "brovey":
            raise NotImplementedError(f"{self.method} pansharpening is planned.")
        return brovey_fusion(multispectral, panchromatic)


class PanNetModel(BasePansharpeningModel):
    """PanNet 딥러닝 팬샤프닝 자리표시자."""


class MSDCNNModel(BasePansharpeningModel):
    """MSDCNN 딥러닝 팬샤프닝 자리표시자."""


class PSGANModel(BasePansharpeningModel):
    """PSGAN 딥러닝 팬샤프닝 자리표시자."""


class PanGANModel(BasePansharpeningModel):
    """Pan-GAN 딥러닝 팬샤프닝 자리표시자."""


class ProximalPanNetModel(BasePansharpeningModel):
    """Proximal PanNet 딥러닝 팬샤프닝 자리표시자."""

