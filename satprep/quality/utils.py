from __future__ import annotations

import numpy as np


def infer_channel_layout(image: np.ndarray) -> str:
    """이미지 배열의 채널 배치를 추정한다."""
    if image.ndim == 2:
        return "hw"
    if image.ndim != 3:
        raise ValueError("image must be 2D or 3D.")
    first_is_rgb_like = image.shape[0] in {1, 3, 4}
    last_is_rgb_like = image.shape[-1] in {1, 3, 4}
    if first_is_rgb_like and last_is_rgb_like:
        return "chw" if image.shape[0] <= image.shape[-1] else "hwc"
    if last_is_rgb_like:
        return "hwc"
    if image.shape[0] <= 16:
        return "chw"
    return "hwc"


def to_channel_last(image: np.ndarray) -> np.ndarray:
    """입력을 HWC 형태로 정규화한다."""
    layout = infer_channel_layout(image)
    if layout == "hw":
        return image[..., None]
    if layout == "chw":
        return np.moveaxis(image, 0, -1)
    return image


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """품질 계산을 위해 단일 채널 float 이미지로 변환한다."""
    arr = to_channel_last(image).astype(np.float32)
    if arr.shape[-1] == 1:
        gray = arr[..., 0]
    else:
        gray = np.mean(arr[..., : min(3, arr.shape[-1])], axis=-1)
    if gray.size == 0:
        return gray
    max_value = float(np.nanmax(gray)) if np.isfinite(gray).any() else 0.0
    if max_value > 1.0:
        gray = gray / max_value
    return np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)


def normalize_score(value: float, scale: float) -> float:
    """양수 지표를 0~1 범위로 부드럽게 압축한다."""
    if scale <= 0:
        return 0.0
    value = max(0.0, float(value))
    return float(value / (value + scale))
