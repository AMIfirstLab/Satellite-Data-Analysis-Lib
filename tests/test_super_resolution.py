import numpy as np
import pytest

from satprep.restoration.super_resolution import HANModel, SRCNNModel, TORCH_AVAILABLE, create_super_resolution_model


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_srcnn_predict_shape_with_untrained_model() -> None:
    image = np.ones((3, 8, 8), dtype=np.uint8) * 80
    model = SRCNNModel(scale=2, allow_untrained=True)
    result = model.predict(image)
    assert result.shape == (3, 16, 16)
    assert result.dtype == image.dtype


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_srcnn_predict_hwc_shape_with_untrained_model() -> None:
    image = np.ones((8, 8, 3), dtype=np.uint8) * 80
    model = SRCNNModel(scale=2, allow_untrained=True)
    result = model.predict(image)
    assert result.shape == (16, 16, 3)
    assert result.dtype == image.dtype


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_super_resolution_factory_supports_srcnn() -> None:
    model = create_super_resolution_model("srcnn", scale=2, allow_untrained=True)
    assert isinstance(model, SRCNNModel)


def test_super_resolution_factory_supports_han() -> None:
    model = create_super_resolution_model("han", scale=4)
    assert isinstance(model, HANModel)
