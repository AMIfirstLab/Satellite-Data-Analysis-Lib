from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from satprep.fusion.registration import check_alignment


@dataclass
class TemporalImage:
    """다중 시기 합성에 사용하는 영상과 품질 부가정보."""

    path: str
    image: np.ndarray
    timestamp: str | None
    quality_score: float | None = None
    cloud_mask: np.ndarray | None = None
    shadow_mask: np.ndarray | None = None


def stack_aligned_images(image_paths: list[str | Path]) -> np.ndarray:
    """정합된 래스터들을 시간축 첫 번째 배열로 쌓는다."""
    if not check_alignment(image_paths):
        raise ValueError("Input images are not aligned.")
    arrays = []
    for path in image_paths:
        with rasterio.open(Path(path)) as src:
            arrays.append(src.read())
    return np.stack(arrays, axis=0)

