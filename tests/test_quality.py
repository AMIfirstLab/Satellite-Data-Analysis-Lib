import numpy as np

from satprep.quality.blur import calculate_blur_score
from satprep.quality.classifier import classify_chip
from satprep.quality.cloud import calculate_cloud_score
from satprep.quality.entropy import calculate_entropy_score
from satprep.quality.report import analyze_array
from satprep.quality.shadow import calculate_shadow_score


def test_blur_and_entropy_scores() -> None:
    image = np.zeros((3, 32, 32), dtype=np.uint8)
    image[:, 8:24, 8:24] = 255
    assert 0.0 <= calculate_blur_score(image) <= 1.0
    assert calculate_entropy_score(image) > 0.0


def test_cloud_shadow_range() -> None:
    bright = np.full((3, 16, 16), 255, dtype=np.uint8)
    dark = np.zeros((3, 16, 16), dtype=np.uint8)
    assert 0.0 <= calculate_cloud_score(bright) <= 1.0
    assert 0.0 <= calculate_shadow_score(dark) <= 1.0


def test_quality_classification() -> None:
    image = np.full((3, 16, 16), 255, dtype=np.uint8)
    report = analyze_array(image, "chip_1", "chip_1.tif", nodata=None)
    assert classify_chip(report) in {"usable", "warning", "reject"}
    assert report.cloud_score > 0.5
    assert report.status == "reject"

