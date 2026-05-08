from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine


_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _infer_layout(image: np.ndarray) -> str:
    """배열의 채널 배치를 추정한다."""
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


def upscale_image(image: np.ndarray, scale: int = 2, method: str = "bicubic") -> np.ndarray:
    """배열 이미지를 고전적 보간 방식으로 확대한다."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if method not in _INTERPOLATION:
        raise ValueError(f"Unsupported method: {method}")
    arr = np.asarray(image)
    layout = _infer_layout(arr)
    work = np.moveaxis(arr, 0, -1) if layout == "chw" else arr
    height, width = work.shape[:2]
    resized = cv2.resize(work, (width * scale, height * scale), interpolation=_INTERPOLATION[method])
    if layout == "hw":
        return resized.astype(arr.dtype, copy=False)
    if layout == "chw" and resized.ndim == 2:
        resized = resized[None, ...]
    elif layout == "chw":
        resized = np.moveaxis(resized, -1, 0)
    return resized.astype(arr.dtype, copy=False)


def upscale_raster(
    input_path: str | Path,
    output_path: str | Path,
    scale: int = 2,
    method: str = "bicubic",
) -> None:
    """GeoTIFF를 확대하고 픽셀 크기에 맞게 affine transform을 갱신한다."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(in_path) as src:
        data = src.read()
        upscaled = upscale_image(data, scale=scale, method=method)
        profile = src.profile.copy()
        profile.update(
            height=upscaled.shape[1],
            width=upscaled.shape[2],
            transform=src.transform * Affine.scale(1 / scale, 1 / scale),
            dtype=str(upscaled.dtype),
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(upscaled)
