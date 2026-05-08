from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio

from satprep.quality.blur import calculate_blur_score
from satprep.quality.classifier import QualityThresholds, classify_chip, recommendation_for_status
from satprep.quality.cloud import calculate_brightness_stats, calculate_cloud_score, calculate_saturation_ratio
from satprep.quality.entropy import calculate_entropy_score, calculate_nodata_ratio
from satprep.quality.shadow import calculate_shadow_score
from satprep.quality.sharpness import calculate_sharpness_score
from satprep.quality.visibility import calculate_object_visibility_score


@dataclass
class QualityReport:
    """칩 단위 품질 분석 결과."""

    chip_id: str
    file_path: str
    blur_score: float
    sharpness_score: float
    entropy_score: float
    nodata_ratio: float
    mean_brightness: float
    saturation_ratio: float
    cloud_score: float
    shadow_score: float
    object_visibility_score: float
    status: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """JSON/CSV 저장용 딕셔너리로 변환한다."""
        return asdict(self)


def analyze_array(
    image: object,
    chip_id: str,
    file_path: str,
    nodata: int | float | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """메모리 배열 하나의 품질 리포트를 생성한다."""
    blur = calculate_blur_score(image)  # type: ignore[arg-type]
    sharpness = calculate_sharpness_score(image)  # type: ignore[arg-type]
    entropy = calculate_entropy_score(image)  # type: ignore[arg-type]
    nodata_ratio = calculate_nodata_ratio(image, nodata)  # type: ignore[arg-type]
    brightness = calculate_brightness_stats(image)  # type: ignore[arg-type]
    saturation = calculate_saturation_ratio(image)  # type: ignore[arg-type]
    cloud = calculate_cloud_score(image)  # type: ignore[arg-type]
    shadow = calculate_shadow_score(image)  # type: ignore[arg-type]
    visibility = calculate_object_visibility_score(blur, sharpness, cloud, shadow, entropy)
    report = QualityReport(
        chip_id=chip_id,
        file_path=file_path,
        blur_score=blur,
        sharpness_score=sharpness,
        entropy_score=entropy,
        nodata_ratio=nodata_ratio,
        mean_brightness=brightness["mean_brightness"],
        saturation_ratio=saturation,
        cloud_score=cloud,
        shadow_score=shadow,
        object_visibility_score=visibility,
        status="usable",
        recommendation="",
    )
    report.status = classify_chip(report, thresholds)
    report.recommendation = recommendation_for_status(report.status)
    return report


def analyze_chip_file(path: str | Path, thresholds: QualityThresholds | None = None) -> QualityReport:
    """GeoTIFF 칩 파일을 읽어 품질 리포트를 생성한다."""
    chip_path = Path(path)
    with rasterio.open(chip_path) as src:
        # 칩은 이미 작은 단위이므로 파일 전체를 읽는다.
        data = src.read()
        nodata = src.nodata
    return analyze_array(data, chip_path.stem, str(chip_path), nodata=nodata, thresholds=thresholds)

