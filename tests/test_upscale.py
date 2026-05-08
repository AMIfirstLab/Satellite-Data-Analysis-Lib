from pathlib import Path

import numpy as np
import rasterio

from satprep.restoration.upscale import upscale_image, upscale_raster
from tests.test_raster import write_test_raster


def test_classical_image_upscaling() -> None:
    image = np.ones((3, 4, 4), dtype=np.uint8)
    result = upscale_image(image, scale=2, method="nearest")
    assert result.shape == (3, 8, 8)
    assert result.dtype == image.dtype


def test_hwc_image_upscaling() -> None:
    image = np.ones((8, 8, 3), dtype=np.uint8)
    result = upscale_image(image, scale=2, method="nearest")
    assert result.shape == (16, 16, 3)
    assert result.dtype == image.dtype


def test_upscale_raster_transform(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    out_path = tmp_path / "upscaled.tif"
    write_test_raster(raster_path, np.ones((3, 4, 4), dtype=np.uint16))
    upscale_raster(raster_path, out_path, scale=2, method="nearest")
    with rasterio.open(out_path) as src:
        assert src.width == 8
        assert src.height == 8
        assert src.res == (5.0, 5.0)
