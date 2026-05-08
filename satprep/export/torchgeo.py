from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_torchgeo_folder(chip_paths: list[str | Path], output_dir: str | Path) -> None:
    """TorchGeo 실험용 간단한 폴더와 metadata.csv를 만든다."""
    out = Path(output_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = [{"path": str(path)} for path in chip_paths]
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)

