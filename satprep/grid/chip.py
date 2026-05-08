from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rasterio.windows import Window


@dataclass(frozen=True)
class ChipWindow:
    """픽셀 기준 칩 윈도우를 표현한다."""

    chip_id: str
    window: Window


@dataclass(frozen=True)
class ChipMetadata:
    """칩 파일과 지리공간 정보를 함께 보관한다."""

    chip_id: str
    source_path: str
    file_path: str
    pixel_bbox: list[int]
    geo_bbox: list[float]
    crs: str | None
    transform: list[float]
    resolution: list[float]
    width: int
    height: int
    bands: list[int]
    dtype: str
    nodata: int | float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON 저장용 딕셔너리로 변환한다."""
        return asdict(self)

