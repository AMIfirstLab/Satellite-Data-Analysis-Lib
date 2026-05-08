from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def create_shadow_aware_composite(stack: np.ndarray, shadow_masks: np.ndarray) -> np.ndarray:
    """그림자 픽셀을 피해서 시간 합성을 생성한다."""
    # 모든 시기가 그림자인 픽셀은 중앙값으로 대체한다.
    if stack.shape[0] != shadow_masks.shape[0]:
        raise ValueError("shadow_masks time dimension must match stack.")
    valid = ~shadow_masks.astype(bool)
    valid_expanded = valid[:, None, :, :]
    masked = np.where(valid_expanded, stack, np.nan)
    composite = np.nanmedian(masked.astype(np.float32), axis=0)
    fallback = np.median(stack, axis=0)
    composite = np.where(np.isnan(composite), fallback, composite)
    return composite.astype(stack.dtype, copy=False)


def create_provenance_map(stack: np.ndarray, shadow_masks: np.ndarray) -> np.ndarray:
    """최종 픽셀에 기여한 시간 인덱스를 기록한다."""
    valid = ~shadow_masks.astype(bool)
    scores = np.where(valid, 1, 0)
    return np.argmax(scores, axis=0).astype(np.uint16)


def save_provenance_map(provenance: np.ndarray, output_path: str | Path) -> None:
    """provenance 배열을 단일 밴드 GeoTIFF 또는 NumPy 파일로 저장한다."""
    # 참조 메타데이터가 없을 때도 쓸 수 있도록 npy 저장을 허용한다.
    path = Path(output_path)
    if path.suffix.lower() == ".npy":
        np.save(path, provenance)
        return
    profile = {"driver": "GTiff", "height": provenance.shape[0], "width": provenance.shape[1], "count": 1, "dtype": str(provenance.dtype)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(provenance, 1)

