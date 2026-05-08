from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from satprep.io.raster import SatelliteImage


def write_test_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(100, 200, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:32652",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)


def test_raster_metadata_extraction(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    write_test_raster(raster_path, np.ones((3, 8, 8), dtype=np.uint16))
    with SatelliteImage.open(raster_path) as image:
        metadata = image.get_metadata()
    assert metadata.width == 8
    assert metadata.height == 8
    assert metadata.count == 3
    assert metadata.crs == "EPSG:32652"
    assert metadata.resolution == (10.0, 10.0)

