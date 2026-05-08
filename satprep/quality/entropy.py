from __future__ import annotations

import numpy as np

from satprep.quality.utils import to_grayscale_float


def calculate_entropy_score(image: np.ndarray) -> float:
    """정보량을 Shannon entropy 기반 0~1 점수로 계산한다."""
    gray = np.clip(to_grayscale_float(image), 0.0, 1.0)
    if gray.size == 0:
        return 0.0
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob = prob[prob > 0] / max(1, gray.size)
    entropy = -np.sum(prob * np.log2(prob))
    return float(np.clip(entropy / 8.0, 0.0, 1.0))


def calculate_nodata_ratio(image: np.ndarray, nodata: int | float | None = None) -> float:
    """nodata 또는 모든 밴드가 0인 픽셀 비율을 계산한다."""
    arr = np.asarray(image)
    if arr.size == 0:
        return 1.0
    if arr.ndim == 3 and arr.shape[0] <= 16:
        pixel_axis = 0
    elif arr.ndim == 3:
        pixel_axis = -1
    else:
        pixel_axis = None
    if nodata is not None:
        if pixel_axis is None:
            mask = arr == nodata
        else:
            mask = np.all(arr == nodata, axis=pixel_axis)
    else:
        if pixel_axis is None:
            mask = arr == 0
        else:
            mask = np.all(arr == 0, axis=pixel_axis)
    return float(np.mean(mask))

