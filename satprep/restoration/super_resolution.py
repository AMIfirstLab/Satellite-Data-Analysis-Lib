from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - 선택 의존성 경로
    torch = None  # type: ignore[assignment]
    class _NNFallback:
        Module = object

    nn = _NNFallback()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

import rasterio
from affine import Affine


def _require_torch() -> None:
    """PyTorch 선택 의존성이 설치되어 있는지 확인한다."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for deep super-resolution. Install satprep[deep].")


def _array_to_chw_float(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """입력 배열을 CHW float32 0~1 범위로 정규화한다."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        chw = arr[None, ...]
        layout = "hw"
    elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] in {1, 3, 4}:
        if arr.shape[0] <= arr.shape[-1]:
            chw = arr
            layout = "chw"
        else:
            chw = np.moveaxis(arr, -1, 0)
            layout = "hwc"
    elif arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        chw = np.moveaxis(arr, -1, 0)
        layout = "hwc"
    elif arr.ndim == 3 and arr.shape[0] <= 16:
        chw = arr
        layout = "chw"
    elif arr.ndim == 3:
        chw = np.moveaxis(arr, -1, 0)
        layout = "hwc"
    else:
        raise ValueError("image must be a 2D, CHW, or HWC array.")

    original_dtype = arr.dtype
    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
    else:
        max_value = float(np.nanmax(chw)) if chw.size else 1.0
        max_value = max(max_value, 1.0)
    normalized = np.nan_to_num(chw.astype(np.float32) / max_value, nan=0.0, posinf=1.0, neginf=0.0)
    meta = {"layout": layout, "dtype": original_dtype, "max_value": max_value}
    return np.clip(normalized, 0.0, 1.0), meta


def _chw_float_to_array(chw: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """CHW float32 0~1 배열을 원래 dtype/layout으로 되돌린다."""
    clipped = np.clip(chw, 0.0, 1.0)
    dtype = meta["dtype"]
    scaled = clipped * float(meta["max_value"])
    if np.issubdtype(dtype, np.integer):
        restored = np.rint(scaled).astype(dtype)
    else:
        restored = scaled.astype(dtype)

    layout = meta["layout"]
    if layout == "hw":
        return restored[0]
    if layout == "hwc":
        return np.moveaxis(restored, 0, -1)
    return restored


def _chw_float_to_pil(chw: np.ndarray) -> Image.Image:
    """CHW float 배열을 RGB PIL 이미지로 변환한다."""
    if chw.shape[0] < 3:
        raise ValueError("HAN super-resolution requires at least 3 RGB channels.")
    rgb = np.moveaxis(np.clip(chw[:3], 0.0, 1.0), 0, -1)
    return Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")


def _tensor_to_chw_float(output: Any) -> np.ndarray:
    """super-image 출력 tensor를 CHW float 배열로 변환한다."""
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "detach"):
        tensor = output.detach().cpu()
    else:
        raise TypeError("Unexpected super-resolution output type.")
    if tensor.ndim == 4:
        tensor = tensor[0]
    return np.clip(tensor.numpy().astype(np.float32), 0.0, 1.0)


class _SRCNNNet(nn.Module):
    """SRCNN 논문 구조에 가까운 작은 CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.clamp(self.net(x), 0.0, 1.0)


class _ResidualBlock(nn.Module):
    """EDSR/RCAN에서 사용하는 기본 residual block."""

    def __init__(self, features: int, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return x + self.block(x) * self.residual_scale


class _EDSRNet(nn.Module):
    """가벼운 EDSR 구현."""

    def __init__(self, channels: int, scale: int, features: int = 64, blocks: int = 8):
        super().__init__()
        self.head = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[_ResidualBlock(features) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(features, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        head = self.head(x)
        body = self.body_tail(self.body(head)) + head
        return torch.clamp(self.tail(body), 0.0, 1.0)


class _ChannelAttention(nn.Module):
    """채널별 중요도를 학습하는 attention 블록."""

    def __init__(self, features: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, features // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, features, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return x * self.attention(x)


class _RCAB(nn.Module):
    """RCAN의 channel attention residual block."""

    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            _ChannelAttention(features),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return x + self.block(x) * 0.1


class _RCANNet(nn.Module):
    """가벼운 RCAN 구현."""

    def __init__(self, channels: int, scale: int, features: int = 64, blocks: int = 6):
        super().__init__()
        self.head = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[_RCAB(features) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.tail = nn.Sequential(
            nn.Conv2d(features, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        head = self.head(x)
        body = self.body_tail(self.body(head)) + head
        return torch.clamp(self.tail(body), 0.0, 1.0)


class BaseSuperResolutionModel:
    """딥러닝 초해상도 모델 래퍼의 공통 인터페이스."""

    model_name = "base"

    def __init__(self, scale: int = 2, device: str = "cpu", allow_untrained: bool = False):
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.scale = scale
        self.device = device
        self.allow_untrained = allow_untrained
        self.weights_path: Path | None = None
        self.channels: int | None = None
        self._model: Any | None = None
        self._has_weights = False

    def _build_model(self, channels: int) -> Any:
        """하위 클래스가 실제 torch module을 생성한다."""
        raise NotImplementedError

    def _ensure_model(self, channels: int) -> Any:
        """입력 밴드 수에 맞게 모델을 지연 생성한다."""
        _require_torch()
        if self._model is None or self.channels != channels:
            self.channels = channels
            self._model = self._build_model(channels).to(self.device)
        return self._model

    def load_weights(self, weights_path: str | Path) -> None:
        """저장된 PyTorch 가중치를 로드한다."""
        _require_torch()
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        channels = checkpoint.get("channels") if isinstance(checkpoint, dict) else None
        if channels is None:
            channels = self._infer_channels_from_state_dict(state_dict)
        model = self._ensure_model(int(channels))
        model.load_state_dict(state_dict)
        model.eval()
        self.weights_path = path
        self._has_weights = True

    def save_weights(self, output_path: str | Path) -> None:
        """현재 모델 가중치를 저장한다."""
        _require_torch()
        if self._model is None or self.channels is None:
            raise RuntimeError("Model has not been built yet.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "scale": self.scale,
                "channels": self.channels,
                "state_dict": self._model.state_dict(),
            },
            path,
        )

    def _infer_channels_from_state_dict(self, state_dict: dict[str, Any]) -> int:
        """state_dict의 첫 convolution weight에서 입력 채널 수를 추정한다."""
        for value in state_dict.values():
            if hasattr(value, "ndim") and value.ndim == 4:
                return int(value.shape[1])
        raise ValueError("Could not infer input channels from weights.")

    def _forward_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """정규화된 BCHW tensor를 초해상도 tensor로 변환한다."""
        model = self._ensure_model(int(tensor.shape[1]))
        if not self._has_weights and not self.allow_untrained:
            raise RuntimeError(
                "No super-resolution weights loaded. Call load_weights(), train a demo model, "
                "or instantiate with allow_untrained=True for experiments."
            )
        model.eval()
        with torch.no_grad():
            return model(tensor)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """단일 이미지 배열을 초해상도 결과로 변환한다."""
        _require_torch()
        chw, meta = _array_to_chw_float(image)
        tensor = torch.from_numpy(chw[None]).to(self.device)
        output = self._forward_tensor(tensor).detach().cpu().numpy()[0]
        return _chw_float_to_array(output, meta)

    def predict_tiled(self, image: np.ndarray, tile_size: int = 256, overlap: int = 32) -> np.ndarray:
        """큰 이미지를 타일로 나눠 초해상도 추론을 수행한다."""
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be non-negative and smaller than tile_size.")
        chw, meta = _array_to_chw_float(image)
        channels, height, width = chw.shape
        step = tile_size - overlap
        out_h = height * self.scale
        out_w = width * self.scale
        accum = np.zeros((channels, out_h, out_w), dtype=np.float32)
        weights = np.zeros((1, out_h, out_w), dtype=np.float32)

        for y in range(0, height, step):
            for x in range(0, width, step):
                y0 = min(y, max(0, height - tile_size))
                x0 = min(x, max(0, width - tile_size))
                tile = chw[:, y0 : y0 + tile_size, x0 : x0 + tile_size]
                predicted = _array_to_chw_float(self.predict(tile))[0]
                oy0 = y0 * self.scale
                ox0 = x0 * self.scale
                oy1 = oy0 + predicted.shape[1]
                ox1 = ox0 + predicted.shape[2]
                accum[:, oy0:oy1, ox0:ox1] += predicted
                weights[:, oy0:oy1, ox0:ox1] += 1.0

        result = accum / np.maximum(weights, 1.0)
        return _chw_float_to_array(result, meta)


class SRCNNModel(BaseSuperResolutionModel):
    """SRCNN: bicubic 확대 후 CNN으로 질감을 보정하는 실제 경량 모델."""

    model_name = "srcnn"

    def _build_model(self, channels: int) -> Any:
        return _SRCNNNet(channels)

    def _forward_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        upscaled = F.interpolate(tensor, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return super()._forward_tensor(torch.clamp(upscaled, 0.0, 1.0))


class EDSRModel(BaseSuperResolutionModel):
    """EDSR: residual CNN 기반 단일 이미지 초해상도 모델."""

    model_name = "edsr"

    def _build_model(self, channels: int) -> Any:
        return _EDSRNet(channels, self.scale)


class RCANModel(BaseSuperResolutionModel):
    """RCAN: residual channel attention을 사용하는 초해상도 모델."""

    model_name = "rcan"

    def _build_model(self, channels: int) -> Any:
        return _RCANNet(channels, self.scale)


class HANModel(BaseSuperResolutionModel):
    """HAN: Hugging Face eugenesiow/han pretrained 모델 래퍼."""

    model_name = "han"

    def __init__(
        self,
        scale: int = 2,
        device: str = "cpu",
        allow_untrained: bool = False,
        model_id: str = "eugenesiow/han",
    ):
        if scale not in {2, 3, 4}:
            raise ValueError("HAN supports scale 2, 3, and 4.")
        super().__init__(scale=scale, device=device, allow_untrained=allow_untrained)
        self.model_id = model_id

    def _build_model(self, channels: int) -> Any:
        raise RuntimeError("HAN is loaded through super-image from_pretrained().")

    def _ensure_han_model(self) -> Any:
        """super-image HAN pretrained 모델을 지연 로드한다."""
        _require_torch()
        if self._model is None:
            try:
                from super_image import HanModel as SuperImageHanModel
            except ImportError as exc:
                raise ImportError("HAN requires the optional super-image package. Install satprep[deep].") from exc
            try:
                self._model = SuperImageHanModel.from_pretrained(self.model_id, scale=self.scale).to(self.device)
            except OSError as exc:
                raise OSError(
                    f"Could not load {self.model_id} for scale={self.scale}. "
                    "The public eugenesiow/han repository currently exposes a 4x checkpoint; "
                    "try create_super_resolution_model('han', scale=4)."
                ) from exc
            self._model.eval()
            self._has_weights = True
        return self._model

    def load_weights(self, weights_path: str | Path) -> None:
        """로컬 super-image HAN checkpoint 또는 Hugging Face model id를 로드한다."""
        _require_torch()
        try:
            from super_image import HanModel as SuperImageHanModel
        except ImportError as exc:
            raise ImportError("HAN requires the optional super-image package. Install satprep[deep].") from exc
        source = str(weights_path)
        self._model = SuperImageHanModel.from_pretrained(source, scale=self.scale).to(self.device)
        self._model.eval()
        self.weights_path = Path(source) if Path(source).exists() else None
        self._has_weights = True

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Hugging Face HAN pretrained 모델로 RGB 이미지를 초해상도 변환한다."""
        _require_torch()
        try:
            from super_image import ImageLoader
        except ImportError as exc:
            raise ImportError("HAN requires the optional super-image package. Install satprep[deep].") from exc
        chw, meta = _array_to_chw_float(image)
        if chw.shape[0] != 3:
            raise ValueError("HAN currently supports RGB/true-color 3-band inputs only.")
        model = self._ensure_han_model()
        pil_image = _chw_float_to_pil(chw)
        inputs = ImageLoader.load_image(pil_image).to(self.device)
        with torch.no_grad():
            output = model(inputs)
        sr = _tensor_to_chw_float(output)
        return _chw_float_to_array(sr, meta)


class TorchScriptSuperResolutionModel(BaseSuperResolutionModel):
    """외부에서 export한 TorchScript 초해상도 모델을 실행하는 어댑터."""

    model_name = "torchscript"

    def _build_model(self, channels: int) -> Any:
        raise RuntimeError("TorchScript models must be loaded with load_weights().")

    def load_weights(self, weights_path: str | Path) -> None:
        """TorchScript 파일을 로드한다."""
        _require_torch()
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")
        self._model = torch.jit.load(str(path), map_location=self.device).eval()
        self.weights_path = path
        self._has_weights = True

    def _forward_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        if self._model is None:
            raise RuntimeError("Load a TorchScript model before prediction.")
        with torch.no_grad():
            output = self._model(tensor)
        return torch.clamp(output, 0.0, 1.0)


class RealESRGANModel(TorchScriptSuperResolutionModel):
    """Real-ESRGAN: 외부 TorchScript 가중치를 실행하는 어댑터."""

    model_name = "real-esrgan"


class SwinIRModel(TorchScriptSuperResolutionModel):
    """SwinIR: 외부 TorchScript 가중치를 실행하는 어댑터."""

    model_name = "swinir"


class DSen2Model(TorchScriptSuperResolutionModel):
    """DSen2: Sentinel-2 밴드 초해상도 TorchScript 어댑터."""

    model_name = "dsen2"


class HighResNetModel(TorchScriptSuperResolutionModel):
    """HighRes-net: 다중 프레임 위성 초해상도 TorchScript 어댑터."""

    model_name = "highres-net"


class DeepSUMModel(TorchScriptSuperResolutionModel):
    """DeepSUM: 비정합 다중 시기 초해상도 TorchScript 어댑터."""

    model_name = "deepsum"


def create_super_resolution_model(
    model_name: str,
    scale: int = 2,
    device: str = "cpu",
    allow_untrained: bool = False,
) -> BaseSuperResolutionModel:
    """모델 이름으로 초해상도 래퍼를 생성한다."""
    models: dict[str, type[BaseSuperResolutionModel]] = {
        "srcnn": SRCNNModel,
        "han": HANModel,
        "edsr": EDSRModel,
        "rcan": RCANModel,
        "real-esrgan": RealESRGANModel,
        "swinir": SwinIRModel,
        "dsen2": DSen2Model,
        "highres-net": HighResNetModel,
        "deepsum": DeepSUMModel,
    }
    key = model_name.lower()
    if key not in models:
        raise ValueError(f"Unsupported super-resolution model: {model_name}")
    return models[key](scale=scale, device=device, allow_untrained=allow_untrained)


def train_srcnn_on_image(
    image: np.ndarray,
    scale: int = 2,
    steps: int = 80,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> SRCNNModel:
    """데모용으로 단일 이미지에서 self-supervised SRCNN을 빠르게 학습한다."""
    # 실제 연구용 학습이 아니라 README 데모와 smoke test를 위한 아주 작은 최적화 루프다.
    _require_torch()
    if steps <= 0:
        raise ValueError("steps must be positive.")
    chw, _ = _array_to_chw_float(image)
    target = torch.from_numpy(chw[None]).to(device)
    model = SRCNNModel(scale=scale, device=device, allow_untrained=True)
    net = model._ensure_model(int(target.shape[1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(steps):
        low_res = F.interpolate(target, scale_factor=1 / scale, mode="bicubic", align_corners=False)
        upscaled = F.interpolate(low_res, scale_factor=scale, mode="bicubic", align_corners=False)
        prediction = net(torch.clamp(upscaled, 0.0, 1.0))
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    model._has_weights = True
    return model


def super_resolve_raster(
    input_path: str | Path,
    output_path: str | Path,
    model: BaseSuperResolutionModel,
    tile_size: int | None = None,
    overlap: int = 32,
) -> None:
    """초해상도 모델로 GeoTIFF를 처리하고 transform을 갱신해 저장한다."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(in_path) as src:
        data = src.read()
        if tile_size is None:
            result = model.predict(data)
        else:
            result = model.predict_tiled(data, tile_size=tile_size, overlap=overlap)
        profile = src.profile.copy()
        profile.update(
            height=result.shape[1],
            width=result.shape[2],
            count=result.shape[0],
            dtype=str(result.dtype),
            transform=src.transform * Affine.scale(1 / model.scale, 1 / model.scale),
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result)
