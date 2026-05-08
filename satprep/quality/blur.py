from __future__ import annotations

import cv2
import numpy as np

from satprep.quality.utils import to_grayscale_float


def calculate_blur_score(image: np.ndarray) -> float:
    """이미지의 흐림 정도를 0~1 범위로 추정한다."""
    # Laplacian 분산이 낮을수록 흐림이 크다고 보고 역방향 점수로 변환한다.
    gray = to_grayscale_float(image)
    if gray.size == 0:
        return 1.0
    variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return float(1.0 / (1.0 + variance * 80.0))

