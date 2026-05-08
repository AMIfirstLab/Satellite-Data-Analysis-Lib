from __future__ import annotations

import json
from pathlib import Path

import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from satprep.grid.chip import ChipMetadata, ChipWindow
from satprep.io.raster import SatelliteImage


class GridTiler:
    """대형 래스터를 고정 크기 칩으로 나눈다."""

    def __init__(
        self,
        chip_size: int = 512,
        stride: int | None = None,
        overlap: int = 0,
        drop_partial: bool = True,
    ):
        if chip_size <= 0:
            raise ValueError("chip_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        self.chip_size = chip_size
        self.stride = stride if stride is not None else chip_size - overlap
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        self.overlap = overlap
        self.drop_partial = drop_partial

    def generate_windows(self, image: SatelliteImage) -> list[ChipWindow]:
        """래스터 크기에 맞는 칩 윈도우 목록을 생성한다."""
        windows: list[ChipWindow] = []
        chip_index = 1
        for row in range(0, image.dataset.height, self.stride):
            for col in range(0, image.dataset.width, self.stride):
                width = min(self.chip_size, image.dataset.width - col)
                height = min(self.chip_size, image.dataset.height - row)
                if self.drop_partial and (width < self.chip_size or height < self.chip_size):
                    continue
                window = Window(col_off=col, row_off=row, width=width, height=height)
                windows.append(ChipWindow(f"chip_{chip_index:06d}", window))
                chip_index += 1
        return windows

    def save_chips(
        self,
        image: SatelliteImage,
        output_dir: str | Path,
        bands: list[int] | None = None,
    ) -> list[ChipMetadata]:
        """칩 GeoTIFF와 개별 JSON 메타데이터를 저장한다."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_bands = bands if bands is not None else list(range(1, image.dataset.count + 1))
        metadata_items: list[ChipMetadata] = []

        for chip in self.generate_windows(image):
            data = image.read_window(chip.window, selected_bands)
            chip_transform = window_transform(chip.window, image.dataset.transform)
            chip_path = out_dir / f"{chip.chip_id}.tif"
            profile = image.dataset.profile.copy()
            profile.update(
                driver="GTiff",
                height=int(chip.window.height),
                width=int(chip.window.width),
                count=len(selected_bands),
                transform=chip_transform,
                dtype=str(data.dtype),
            )
            with rasterio.open(chip_path, "w", **profile) as dst:
                dst.write(data)

            geo = window_bounds(chip.window, image.dataset.transform)
            chip_meta = ChipMetadata(
                chip_id=chip.chip_id,
                source_path=str(image.path),
                file_path=str(chip_path),
                pixel_bbox=[
                    int(chip.window.col_off),
                    int(chip.window.row_off),
                    int(chip.window.col_off + chip.window.width),
                    int(chip.window.row_off + chip.window.height),
                ],
                geo_bbox=[geo[0], geo[1], geo[2], geo[3]],
                crs=image.dataset.crs.to_string() if image.dataset.crs else None,
                transform=list(chip_transform)[:6],
                resolution=list(image.dataset.res),
                width=int(chip.window.width),
                height=int(chip.window.height),
                bands=selected_bands,
                dtype=str(data.dtype),
                nodata=image.dataset.nodata,
            )
            with (out_dir / f"{chip.chip_id}.json").open("w", encoding="utf-8") as f:
                json.dump(chip_meta.to_dict(), f, indent=2)
            metadata_items.append(chip_meta)
        return metadata_items


def tile_raster(
    input_path: str | Path,
    output_dir: str | Path,
    chip_size: int = 512,
    stride: int | None = None,
    overlap: int = 0,
    drop_partial: bool = True,
) -> list[ChipMetadata]:
    """입력 래스터를 열어 칩으로 저장한다."""
    with SatelliteImage.open(input_path) as image:
        tiler = GridTiler(chip_size=chip_size, stride=stride, overlap=overlap, drop_partial=drop_partial)
        return tiler.save_chips(image, output_dir)

