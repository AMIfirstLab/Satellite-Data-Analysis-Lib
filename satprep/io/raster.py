from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from satprep.io.metadata import RasterMetadata


class SatelliteImage:
    """GeoTIFF/COG를 지연 로딩 방식으로 다루는 래퍼."""

    def __init__(self, path: str | Path, dataset: DatasetReader):
        self.path = Path(path)
        self.dataset = dataset

    @classmethod
    def open(cls, path: str | Path) -> "SatelliteImage":
        """래스터 파일을 열고 전체 픽셀은 메모리에 올리지 않는다."""
        raster_path = Path(path)
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")
        return cls(raster_path, rasterio.open(raster_path))

    def read_window(self, window: Window, bands: list[int] | None = None) -> np.ndarray:
        """지정된 윈도우와 밴드만 읽는다."""
        indexes = bands if bands is not None else None
        return self.dataset.read(indexes=indexes, window=window, boundless=False)

    def get_metadata(self) -> RasterMetadata:
        """현재 래스터의 지리공간 메타데이터를 반환한다."""
        bounds = self.dataset.bounds
        crs = self.dataset.crs.to_string() if self.dataset.crs else None
        return RasterMetadata(
            path=str(self.path),
            crs=crs,
            transform=list(self.dataset.transform)[:6],
            width=self.dataset.width,
            height=self.dataset.height,
            count=self.dataset.count,
            dtype=self.dataset.dtypes[0],
            nodata=self.dataset.nodata,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            resolution=self.dataset.res,
        )

    def close(self) -> None:
        """래스터 핸들을 닫는다."""
        self.dataset.close()

    def __enter__(self) -> "SatelliteImage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

