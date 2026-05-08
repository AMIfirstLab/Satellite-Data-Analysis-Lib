from __future__ import annotations

from satprep.grid.chip import ChipMetadata


def filter_chips_by_status(chips: list[ChipMetadata], allowed_ids: set[str]) -> list[ChipMetadata]:
    """품질 결과에서 허용된 칩만 고른다."""
    return [chip for chip in chips if chip.chip_id in allowed_ids]

