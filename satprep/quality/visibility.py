from __future__ import annotations

import numpy as np


def calculate_object_visibility_score(
    blur_score: float,
    sharpness_score: float,
    cloud_score: float,
    shadow_score: float,
    entropy_score: float,
) -> float:
    """물체 식별 가능성을 여러 품질 지표로 합성한다."""
    # 구름과 그림자는 가림 현상이라 강하게 감점하고, 선명도와 entropy는 가산한다.
    score = (
        0.30 * (1.0 - blur_score)
        + 0.25 * sharpness_score
        + 0.20 * entropy_score
        + 0.15 * (1.0 - cloud_score)
        + 0.10 * (1.0 - shadow_score)
    )
    return float(np.clip(score, 0.0, 1.0))

