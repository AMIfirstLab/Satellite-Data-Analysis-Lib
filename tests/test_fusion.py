import numpy as np

from satprep.fusion.composite import create_clear_sky_composite, create_median_composite, create_quality_weighted_composite
from satprep.fusion.shadow_fusion import create_shadow_aware_composite


def test_temporal_median_composite() -> None:
    stack = np.stack(
        [
            np.zeros((1, 2, 2), dtype=np.uint8),
            np.full((1, 2, 2), 10, dtype=np.uint8),
            np.full((1, 2, 2), 20, dtype=np.uint8),
        ],
        axis=0,
    )
    result = create_median_composite(stack)
    assert result.shape == (1, 2, 2)
    assert np.all(result == 10)


def test_quality_weighted_composite() -> None:
    stack = np.stack([np.zeros((1, 2, 2), dtype=np.uint8), np.full((1, 2, 2), 10, dtype=np.uint8)], axis=0)
    result = create_quality_weighted_composite(stack, np.array([0.0, 1.0]))
    assert np.all(result == 10)


def test_clear_sky_composite_avoids_cloud_like_pixels() -> None:
    clear = np.full((3, 2, 2), 80, dtype=np.uint8)
    cloud = np.full((3, 2, 2), 255, dtype=np.uint8)
    stack = np.stack([cloud, clear], axis=0)
    result = create_clear_sky_composite(stack, brightness_gain=1.0)
    assert result.shape == clear.shape
    assert np.all(result == clear)


def test_shadow_aware_composite() -> None:
    stack = np.stack([np.zeros((1, 2, 2), dtype=np.uint8), np.full((1, 2, 2), 10, dtype=np.uint8)], axis=0)
    shadows = np.array([[[True, True], [False, False]], [[False, False], [False, False]]])
    result = create_shadow_aware_composite(stack, shadows)
    assert result.shape == (1, 2, 2)
