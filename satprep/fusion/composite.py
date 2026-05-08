from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


def create_median_composite(stack: np.ndarray) -> np.ndarray:
    """시간축 중앙값 합성을 생성한다."""
    # 입력은 (time, band, height, width) 형태라고 가정한다.
    return np.median(stack, axis=0).astype(stack.dtype, copy=False)


def create_quality_weighted_composite(stack: np.ndarray, quality_weights: np.ndarray) -> np.ndarray:
    """영상별 품질 가중치를 적용해 평균 합성을 생성한다."""
    weights = np.asarray(quality_weights, dtype=np.float32)
    if stack.shape[0] != weights.shape[0]:
        raise ValueError("quality_weights length must match stack time dimension.")
    weights = np.maximum(weights, 0)
    if float(weights.sum()) == 0.0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    view_shape = (weights.shape[0],) + (1,) * (stack.ndim - 1)
    result = np.sum(stack.astype(np.float32) * weights.reshape(view_shape), axis=0)
    return result.astype(stack.dtype, copy=False)


def create_clear_sky_composite(
    stack: np.ndarray,
    cloud_brightness_threshold: float = 0.72,
    cloud_whiteness_threshold: float = 0.78,
    shadow_threshold: float = 0.16,
    brightness_gain: float = 1.08,
) -> np.ndarray:
    """RGB 휴리스틱 기반 clear-sky quality mosaic을 생성한다."""
    # Sentinel-2 SCL 같은 공식 마스크가 없을 때 쓰는 간단한 데모용 방식이다.
    if stack.ndim != 4:
        raise ValueError("stack must have shape (time, band, height, width).")
    if stack.shape[1] < 3:
        raise ValueError("clear-sky composite requires at least 3 RGB bands.")

    work = stack.astype(np.float32)
    max_value = float(np.iinfo(stack.dtype).max) if np.issubdtype(stack.dtype, np.integer) else float(np.nanmax(work))
    max_value = max(max_value, 1.0)
    rgb = np.clip(work[:, :3] / max_value, 0.0, 1.0)

    brightness = np.mean(rgb, axis=1)
    whiteness = 1.0 - (np.max(rgb, axis=1) - np.min(rgb, axis=1))
    colorfulness = np.max(rgb, axis=1) - np.min(rgb, axis=1)
    cloud_like = (brightness > cloud_brightness_threshold) & (whiteness > cloud_whiteness_threshold)
    shadow_like = brightness < shadow_threshold

    # 구름/그림자는 강하게 감점하고, 너무 어둡지 않으면서 색 대비가 있는 픽셀을 선호한다.
    score = (
        brightness * 0.75
        + colorfulness * 0.55
        - cloud_like.astype(np.float32) * 2.5
        - shadow_like.astype(np.float32) * 0.9
    )
    best_time = np.argmax(score, axis=0)
    selector = best_time[None, None, :, :]
    composite = np.take_along_axis(work, selector, axis=0)[0]

    median = np.median(work, axis=0)
    all_cloudy = np.all(cloud_like, axis=0)
    composite = np.where(all_cloudy[None, :, :], median, composite)
    composite = composite * brightness_gain
    if np.issubdtype(stack.dtype, np.integer):
        composite = np.clip(np.rint(composite), 0, max_value)
    return composite.astype(stack.dtype, copy=False)


def save_composite_like(reference_path: str | Path, composite: np.ndarray, output_path: str | Path) -> None:
    """참조 래스터 메타데이터를 보존해 합성 GeoTIFF를 저장한다."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(count=composite.shape[0], height=composite.shape[1], width=composite.shape[2], dtype=str(composite.dtype))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(composite)


def write_fusion_report(output_path: str | Path, method: str, image_paths: list[str | Path]) -> None:
    """합성 처리의 간단한 provenance 리포트를 저장한다."""
    report = {"method": method, "input_images": [str(p) for p in image_paths], "image_count": len(image_paths)}
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
