from __future__ import annotations

import cv2
import numpy as np

from satprep.quality.utils import normalize_score, to_grayscale_float


def calculate_sharpness_score(image: np.ndarray) -> float:
    """경계 강도 기반 선명도 점수를 계산한다."""
    # Sobel gradient 평균은 물체 경계와 텍스처가 살아 있을수록 커진다.
    gray = to_grayscale_float(image)
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return normalize_score(float(np.mean(magnitude)), 0.08)

