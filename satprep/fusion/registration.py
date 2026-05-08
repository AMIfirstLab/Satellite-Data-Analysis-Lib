from __future__ import annotations

from pathlib import Path

import rasterio


def check_alignment(image_paths: list[str | Path]) -> bool:
    """CRS, transform, 크기, 해상도가 모두 같은지 확인한다."""
    if not image_paths:
        return False
    paths = [Path(p) for p in image_paths]
    with rasterio.open(paths[0]) as first:
        signature = (first.crs, first.transform, first.width, first.height, first.res)
    for path in paths[1:]:
        with rasterio.open(path) as src:
            if (src.crs, src.transform, src.width, src.height, src.res) != signature:
                return False
    return True

