from __future__ import annotations

import numpy as np


def deblur_placeholder(image: np.ndarray) -> np.ndarray:
    """디블러 모델 연동 전까지 입력을 그대로 반환한다."""
    # 과도한 디블러는 공간 패턴을 왜곡할 수 있어 첫 버전에서는 명시적 자리표시자로 둔다.
    return image

