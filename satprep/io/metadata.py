from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RasterMetadata:
    """래스터의 핵심 지리공간 메타데이터를 보관한다."""

    path: str
    crs: str | None
    transform: list[float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: int | float | None
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 쉬운 딕셔너리로 변환한다."""
        data = asdict(self)
        data["path"] = str(Path(self.path))
        return data

