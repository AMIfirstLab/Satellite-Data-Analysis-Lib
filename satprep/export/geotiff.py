from __future__ import annotations

import json
import shutil
from pathlib import Path


def export_chips_for_training(
    chip_dir: str | Path,
    report_path: str | Path,
    output_dir: str | Path,
    status_filter: list[str] | None = None,
) -> None:
    """품질 상태에 맞는 칩을 학습 폴더로 복사한다."""
    allowed = status_filter or ["usable"]
    source_dir = Path(chip_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(report_path).open("r", encoding="utf-8") as f:
        reports = json.load(f)
    for report in reports:
        if report.get("status") not in allowed:
            continue
        src = Path(report.get("file_path", ""))
        if not src.exists():
            src = source_dir / f"{report['chip_id']}.tif"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

