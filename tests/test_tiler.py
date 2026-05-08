from pathlib import Path

import numpy as np

from satprep.grid.tiler import GridTiler
from satprep.io.raster import SatelliteImage
from tests.test_raster import write_test_raster


def test_grid_window_generation(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    write_test_raster(raster_path, np.ones((3, 8, 8), dtype=np.uint16))
    with SatelliteImage.open(raster_path) as image:
        windows = GridTiler(chip_size=4, stride=4).generate_windows(image)
    assert len(windows) == 4
    assert windows[0].chip_id == "chip_000001"


def test_chip_metadata_creation(tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    out_dir = tmp_path / "chips"
    write_test_raster(raster_path, np.ones((3, 8, 8), dtype=np.uint16))
    with SatelliteImage.open(raster_path) as image:
        chips = GridTiler(chip_size=4, stride=4).save_chips(image, out_dir)
    assert len(chips) == 4
    assert Path(chips[0].file_path).exists()
    assert (out_dir / "chip_000001.json").exists()
    assert chips[0].width == 4

