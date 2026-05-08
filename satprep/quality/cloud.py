from __future__ import annotations

import numpy as np

from satprep.quality.utils import to_channel_last


def _normalized_rgb(image: np.ndarray) -> np.ndarray:
    arr = to_channel_last(image).astype(np.float32)
    rgb = arr[..., : min(3, arr.shape[-1])]
    max_value = float(np.nanmax(rgb)) if rgb.size and np.isfinite(rgb).any() else 0.0
    if max_value > 1.0:
        rgb = rgb / max_value
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def calculate_brightness_stats(image: np.ndarray) -> dict[str, float]:
    """평균 밝기와 기본 통계를 계산한다."""
    rgb = _normalized_rgb(image)
    if rgb.size == 0:
        return {"mean_brightness": 0.0, "min_brightness": 0.0, "max_brightness": 0.0}
    brightness = np.mean(rgb, axis=-1)
    return {
        "mean_brightness": float(np.mean(brightness)),
        "min_brightness": float(np.min(brightness)),
        "max_brightness": float(np.max(brightness)),
    }


def calculate_saturation_ratio(image: np.ndarray) -> float:
    """포화에 가까운 밝은 픽셀 비율을 계산한다."""
    rgb = _normalized_rgb(image)
    if rgb.size == 0:
        return 0.0
    return float(np.mean(np.any(rgb >= 0.98, axis=-1)))


def calculate_cloud_score(image: np.ndarray) -> float:
    """밝고 흰색에 가까운 영역의 비율로 구름 점수를 추정한다."""
    # 단순 휴리스틱이므로 눈, 사막, 밝은 지붕을 구름으로 오인할 수 있다.
    rgb = _normalized_rgb(image)
    if rgb.size == 0:
        return 0.0
    brightness = np.mean(rgb, axis=-1)
    whiteness = 1.0 - (np.max(rgb, axis=-1) - np.min(rgb, axis=-1))
    cloud_like = (brightness > 0.72) & (whiteness > 0.78)
    return float(np.mean(cloud_like))

