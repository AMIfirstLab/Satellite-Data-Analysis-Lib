from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from satprep.quality.report import QualityReport


def export_quality_reports_to_csv(reports: list[QualityReport], output_path: str | Path) -> None:
    """품질 리포트 목록을 CSV로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in reports]).to_csv(path, index=False)


def export_quality_reports_to_json(reports: list[QualityReport], output_path: str | Path) -> None:
    """품질 리포트 목록을 JSON으로 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in reports], f, indent=2)


def summarize_quality_reports(reports: list[QualityReport]) -> dict[str, float | int]:
    """품질 리포트 요약 통계를 계산한다."""
    total = len(reports)
    if total == 0:
        return {
            "total_chips": 0,
            "usable_chips": 0,
            "warning_chips": 0,
            "rejected_chips": 0,
            "average_blur_score": 0.0,
            "average_sharpness_score": 0.0,
            "average_cloud_score": 0.0,
            "average_shadow_score": 0.0,
            "average_object_visibility_score": 0.0,
        }
    return {
        "total_chips": total,
        "usable_chips": sum(r.status == "usable" for r in reports),
        "warning_chips": sum(r.status == "warning" for r in reports),
        "rejected_chips": sum(r.status == "reject" for r in reports),
        "average_blur_score": float(sum(r.blur_score for r in reports) / total),
        "average_sharpness_score": float(sum(r.sharpness_score for r in reports) / total),
        "average_cloud_score": float(sum(r.cloud_score for r in reports) / total),
        "average_shadow_score": float(sum(r.shadow_score for r in reports) / total),
        "average_object_visibility_score": float(sum(r.object_visibility_score for r in reports) / total),
    }

