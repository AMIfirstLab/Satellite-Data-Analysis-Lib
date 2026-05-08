from __future__ import annotations

import numpy as np

from satprep.quality.utils import to_channel_last


def calculate_shadow_score(image: np.ndarray) -> float:
    """어두운 픽셀 비율로 그림자 가능성을 추정한다."""
    # 지형 음영과 수면도 그림자로 보일 수 있어 정밀 마스크가 아니다.
    arr = to_channel_last(image).astype(np.float32)
    if arr.size == 0:
        return 0.0
    channels = arr[..., : min(3, arr.shape[-1])]
    max_value = float(np.nanmax(channels)) if np.isfinite(channels).any() else 0.0
    if max_value > 1.0:
        channels = channels / max_value
    brightness = np.mean(np.clip(np.nan_to_num(channels), 0.0, 1.0), axis=-1)
    return float(np.mean(brightness < 0.15))

