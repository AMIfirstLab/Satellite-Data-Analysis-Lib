from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityThresholds:
    """품질 판정에 사용하는 기본 임계값."""

    reject_nodata_ratio: float = 0.4
    reject_cloud_score: float = 0.5
    warning_cloud_score: float = 0.2
    warning_shadow_score: float = 0.3
    warning_blur_score: float = 0.72
    reject_visibility_score: float = 0.25


def classify_chip(report: object, thresholds: QualityThresholds | None = None) -> str:
    """품질 리포트를 usable/warning/reject 중 하나로 분류한다."""
    limits = thresholds or QualityThresholds()
    if getattr(report, "nodata_ratio") > limits.reject_nodata_ratio:
        return "reject"
    if getattr(report, "cloud_score") > limits.reject_cloud_score:
        return "reject"
    if getattr(report, "object_visibility_score") < limits.reject_visibility_score:
        return "reject"
    if getattr(report, "cloud_score") > limits.warning_cloud_score:
        return "warning"
    if getattr(report, "shadow_score") > limits.warning_shadow_score:
        return "warning"
    if getattr(report, "blur_score") > limits.warning_blur_score:
        return "warning"
    return "usable"


def recommendation_for_status(status: str) -> str:
    """상태에 맞는 간단한 권고 문구를 반환한다."""
    if status == "reject":
        return "Exclude this chip from training unless manually reviewed."
    if status == "warning":
        return "Use with caution or inspect before training."
    return "Suitable for training."

