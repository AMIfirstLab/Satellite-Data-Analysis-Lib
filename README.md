# satprep

위성영상 딥러닝 워크플로우를 위한 **위성영상 전처리, 품질 분석, 해상도 향상, 다중시기 합성 라이브러리**입니다.

`satprep`은 큰 GeoTIFF 또는 Cloud Optimized GeoTIFF(COG) 위성영상을 딥러닝 학습에 쓰기 좋은 칩 단위 데이터셋으로 변환하고, 각 칩의 품질을 분석한 뒤, 학습에 적합한 칩만 선별할 수 있도록 돕습니다. 이 프로젝트는 모델 학습 프레임워크가 아니라 **학습 전 데이터 준비와 품질 관리에 집중한 전처리 라이브러리**입니다.

## 핵심 목적

위성영상 딥러닝에서는 원본 영상이 너무 크고, 구름, 그림자, 흐림, nodata 영역, 낮은 선명도 같은 문제가 섞여 있는 경우가 많습니다. `satprep`은 이런 문제를 다음 흐름으로 처리합니다.

```text
대형 위성영상 입력
  ↓
지리공간 메타데이터 보존
  ↓
격자 기반 칩 생성
  ↓
칩별 품질 분석
  ↓
usable / warning / reject 분류
  ↓
JSON / CSV 리포트 저장
  ↓
학습용 데이터셋으로 필터링
  ↓
필요 시 업스케일링, 초해상도, 다중시기 합성 적용
```

## 데모 개요

아래 GIF들은 [examples/create_readme_demo.py](examples/create_readme_demo.py)로 생성했습니다.

데모에는 실제 [Sentinel-2 L2A Cloud Optimized GeoTIFF](https://registry.opendata.aws/sentinel-2-l2a-cogs/) 데이터를 사용했습니다. 데이터 검색은 [Element 84 Earth Search STAC API](https://earth-search.aws.element84.com/v1)를 통해 수행합니다.

중요한 점은, 전체 Sentinel-2 장면을 다운로드하지 않는다는 것입니다. Sentinel-2 true-color COG 타일은 대략 `10980 x 10980` 픽셀 크기인데, 데모에서는 이 중 `2048 x 2048` 픽셀 window만 원격에서 읽습니다. 즉, 이 데모는 작은 패치 데이터셋이 아니라 **큰 원본 위성영상에서 필요한 부분만 읽고 직접 칩으로 자르는 실제 대형 래스터 처리 흐름**을 보여줍니다.

현재 README 데모에 사용된 대표 장면:

- Dataset: Sentinel-2 L2A Cloud Optimized GeoTIFF
- STAC collection: `sentinel-2-l2a`
- Item: `S2B_52SCG_20240819_0_L2A`
- Local demo window: `2048 x 2048`
- Chip size: `512 x 512`
- Generated chips: `16`

## 데모 1. Grid Tiling And Quality Control

![satprep grid and quality demo](docs/assets/satprep_grid_quality.gif)

이 데모는 `satprep`의 가장 중요한 기능인 **대형 위성영상 칩 생성과 칩 품질 분류**를 보여줍니다.

### 무엇을 위해 하는가

딥러닝 학습용 위성영상 데이터셋을 만들 때, 원본 장면 전체를 그대로 쓰기 어렵습니다. 보통 일정한 크기의 칩으로 잘라야 하고, 구름이나 그림자가 심한 칩은 학습에서 제외하거나 별도 검토해야 합니다.

이 데모는 다음 질문에 답합니다.

- 큰 Sentinel-2 COG에서 원하는 영역만 읽을 수 있는가?
- 지리공간 메타데이터를 유지하면서 칩으로 자를 수 있는가?
- 각 칩이 학습에 적합한지 자동으로 점수화할 수 있는가?
- 최종적으로 `usable`, `warning`, `reject`로 나눌 수 있는가?

### 내부 처리 흐름

1. Earth Search STAC API에서 Sentinel-2 L2A item을 검색합니다.
2. 선택된 item의 true-color `visual` COG URL을 가져옵니다.
3. `rasterio` window read로 `2048 x 2048` 영역만 읽습니다.
4. 읽은 window를 로컬 GeoTIFF로 저장합니다.
5. CRS, affine transform, 해상도, dtype, band metadata를 유지합니다.
6. `GridTiler(chip_size=512, stride=512)`로 16개 칩을 생성합니다.
7. 각 칩을 GeoTIFF로 저장하고, 칩별 JSON metadata를 생성합니다.
8. 각 칩에 대해 품질 지표를 계산합니다.
9. 품질 지표를 기준으로 `usable`, `warning`, `reject` 상태를 부여합니다.
10. 결과를 `quality_report.json`, `quality_report.csv`로 저장합니다.

현재 데모 결과:

```text
total_chips: 16
usable_chips: 8
warning_chips: 8
rejected_chips: 0
```

이 예시는 흰색/검은색 블록을 인위적으로 만든 극단적인 toy example이 아닙니다. 실제 Sentinel-2 장면에서 구름처럼 밝은 영역, 그림자처럼 어두운 영역, 도시/산림/하천의 질감 차이를 이용해 품질을 분류합니다.

### 사용하는 주요 품질 지표

- `blur_score`: Laplacian 기반 흐림 점수
- `sharpness_score`: Sobel gradient 기반 선명도 점수
- `entropy_score`: Shannon entropy 기반 정보량
- `nodata_ratio`: nodata 또는 빈 픽셀 비율
- `mean_brightness`: 평균 밝기
- `saturation_ratio`: 포화 픽셀 비율
- `cloud_score`: 밝고 흰색에 가까운 구름 후보 비율
- `shadow_score`: 매우 어두운 그림자 후보 비율
- `object_visibility_score`: 위 지표들을 조합한 객체 가시성 점수

## 데모 2. Classical And Deep Super-Resolution

![satprep super-resolution demo](docs/assets/satprep_super_resolution.gif)

이 데모는 저해상도 입력을 초해상도 모델로 복원하는 흐름을 보여줍니다. 왼쪽은 low-scale input이고, 오른쪽은 `satprep`이 Hugging Face pretrained HAN 모델을 통해 생성한 super-resolution 결과입니다.

### 무엇을 위해 하는가

위성영상에서는 동일 지역이라도 센서, 밴드, 시기, 데이터 소스에 따라 공간 해상도가 다를 수 있습니다. 고전적 보간으로 확대할 수도 있지만, 딥러닝 초해상도 모델을 이용하면 더 선명하게 보이는 결과를 만들 수 있습니다.

이 데모는 다음을 보여줍니다.

- 기존 `nearest`, `bilinear`, `bicubic`, `lanczos` 업스케일링과 별도로 딥러닝 초해상도 인터페이스를 사용할 수 있음
- Hugging Face의 pretrained 모델을 `satprep` wrapper에서 호출할 수 있음
- README에서 low-scale input과 초해상도 결과를 좌우 비교로 확인할 수 있음

### 사용한 모델

데모에서는 Hugging Face 모델 [eugenesiow/han](https://huggingface.co/eugenesiow/han)을 사용합니다.

- 모델명: HAN, Holistic Attention Network
- 논문: Single Image Super-Resolution via a Holistic Attention Network
- 라이브러리: `super-image`
- 사용 checkpoint: `pytorch_model_4x.pt`
- 지원 scale: 현재 데모는 `x4` 기준

현재 `eugenesiow/han` repository에는 `pytorch_model_4x.pt` checkpoint가 제공되어 있어, README 데모도 `scale=4`로 구성했습니다.

### 내부 처리 흐름

1. Sentinel-2 `2048 x 2048` window에서 여러 패치를 선택합니다.
2. 각 패치를 일부러 작은 low-scale input으로 축소합니다.
3. `create_super_resolution_model("han", scale=4)`로 HAN wrapper를 생성합니다.
4. `super-image`가 Hugging Face에서 pretrained HAN checkpoint를 로드합니다.
5. low-scale input을 HAN 모델에 넣어 x4 super-resolution 결과를 생성합니다.
6. GIF에서는 왼쪽에 low-scale input, 오른쪽에 HAN 결과를 배치합니다.

사용 예시:

```python
from satprep.restoration.super_resolution import create_super_resolution_model

model = create_super_resolution_model("han", scale=4)
result = model.predict(chip_array)
```

CLI 사용 예시:

```bash
satprep super-resolve input.tif --model han --scale 4 --out sr.tif
```

### 주의 사항

딥러닝 초해상도는 실제로 없던 세부 구조를 그럴듯하게 만들 수 있습니다. 따라서 결과를 물리적으로 검증된 ground truth처럼 취급하면 안 됩니다. 특히 과학 분석, 군사, 재난 대응, 탐지 성능 평가처럼 정확도가 중요한 작업에서는 원본 영상과 반드시 비교해야 합니다.

또한 HAN 모델은 자연영상 super-resolution 데이터셋으로 학습된 모델입니다. 위성영상에 적용할 수는 있지만, Sentinel-2 전용 물리 모델은 아닙니다. 이 프로젝트에서는 시각적 향상과 wrapper 인터페이스 데모 용도로 사용합니다.

## 데모 3. Multi-Temporal Composite

![satprep temporal composite demo](docs/assets/satprep_temporal_composite.gif)

이 데모는 같은 지역을 여러 날짜에 촬영한 Sentinel-2 영상으로 더 깨끗한 합성 영상을 만드는 과정을 보여줍니다.

### 무엇을 위해 하는가

위성영상에는 날짜마다 구름, 그림자, 연무, 밝기 차이, 계절 변화가 섞일 수 있습니다. 단일 날짜 영상만 보면 특정 지역이 구름에 가려져 있거나 너무 어두울 수 있습니다. 여러 날짜 영상을 조합하면 이런 일시적 문제를 줄일 수 있습니다.

이 데모는 다음을 보여줍니다.

- 같은 Sentinel-2 tile에서 여러 날짜 item을 선택
- 동일한 픽셀 window를 각 날짜에서 읽기
- `(time, band, height, width)` 형태로 stack 생성
- 단순 median보다 더 밝고 구름이 적은 clear-sky quality mosaic 생성

### 사용한 합성 방법

처음에는 `median composite`를 사용했지만, 날짜 수가 적을 때는 구름이 충분히 사라지지 않고 결과가 다소 어두워질 수 있습니다. 그래서 README 데모에는 `create_clear_sky_composite()`를 사용합니다.

이 방식은 RGB 기반 휴리스틱으로 각 날짜의 같은 위치 픽셀을 비교합니다.

- 너무 밝고 흰색에 가까운 픽셀은 구름 후보로 보고 감점
- 너무 어두운 픽셀은 그림자 후보로 보고 감점
- 적당히 밝고 색 대비가 살아 있는 픽셀을 선호
- 모든 날짜가 구름 후보인 경우에는 median fallback 사용
- 최종 결과에는 약간의 밝기 보정 적용

현재 데모에서는 plain median보다 다음처럼 개선되었습니다.

```text
median 평균 밝기: 97.97
clear-sky 평균 밝기: 111.90

median cloud-like bright pixel ratio: 0.060
clear-sky cloud-like bright pixel ratio: 0.030

median dark pixel ratio: 0.086
clear-sky dark pixel ratio: 0.023
```

즉, 결과가 더 밝고, 구름처럼 흰 영역과 그림자처럼 어두운 영역이 줄어듭니다.

## 데모 재생성 방법

```bash
python examples/create_readme_demo.py
```

이 스크립트는 다음 외부 접근이 필요합니다.

- Earth Search STAC API
- Sentinel-2 L2A COG 원격 읽기
- Hugging Face `eugenesiow/han` 모델 다운로드

생성되는 주요 결과:

```text
docs/assets/satprep_grid_quality.gif
docs/assets/satprep_super_resolution.gif
docs/assets/satprep_temporal_composite.gif
docs/demo_outputs/
```

`docs/assets/*.gif`는 README에 들어가는 최종 시각화 자산입니다. `docs/demo_outputs/`는 중간 산출물이므로 `.gitignore` 대상입니다.

## 주요 기능

- GeoTIFF / Cloud Optimized GeoTIFF 로딩
- 대형 래스터 window 기반 읽기
- CRS, affine transform, pixel resolution, nodata, dtype, bounds, band metadata 보존
- 고정 크기 grid chip 생성
- 칩별 GeoTIFF 저장
- 칩별 JSON metadata 저장
- blur, sharpness, entropy, nodata, brightness, saturation, cloud, shadow, object visibility 분석
- `usable`, `warning`, `reject` 품질 분류
- 품질 리포트 JSON / CSV export
- 품질 상태 기반 학습 데이터셋 필터링
- 고전적 업스케일링: `nearest`, `bilinear`, `bicubic`, `lanczos`
- 딥러닝 초해상도 wrapper: pretrained `HAN`, `SRCNN`, lightweight `EDSR`, lightweight `RCAN`
- 외부 TorchScript adapter: `RealESRGAN`, `SwinIR`, `DSen2`, `HighResNet`, `DeepSUM`
- 다중시기 합성: median, clear-sky quality mosaic, quality-weighted, shadow-aware composite
- Typer 기반 CLI

## 설치 방법

기본 설치:

```bash
pip install -e .
```

개발 및 테스트용 설치:

```bash
pip install -e ".[dev]"
python -m pytest
```

딥러닝 초해상도와 HAN 모델 사용:

```bash
pip install -e ".[deep]"
```

## 환경 조건

패키지 설정 기준:

- Python: `>=3.10`
- 기본 필수 패키지:
  - `numpy>=1.23`
  - `rasterio>=1.3`
  - `shapely>=2.0`
  - `pyproj>=3.4`
  - `pandas>=1.5`
  - `pillow>=9.0`
  - `opencv-python>=4.7`
  - `tqdm>=4.64`
  - `typer>=0.9`
  - `pydantic>=2.0`

선택 의존성:

- 개발 테스트: `pytest`, `pytest-cov`
- 딥러닝 초해상도: `torch`, `torchvision`, `super-image>=0.2.0`
- STAC 확장: `pystac-client`, `stackstac`
- 대규모 배열 처리 확장: `xarray`, `dask`

현재 개발 환경에서 검증한 버전:

```text
Python: 3.11.8
PyTorch: 2.11.0+cu128
rasterio: 1.4.4
OpenCV: 4.13.0
NumPy: 2.1.2
pandas: 2.2.3
Pillow: 10.4.0
```

GPU는 필수는 아닙니다. 다만 HAN 같은 딥러닝 초해상도 모델은 CPU에서도 실행 가능하지만 느릴 수 있습니다.

Windows 환경에서 Hugging Face 모델을 다운로드할 때 symlink cache warning이 뜰 수 있습니다. 이는 Hugging Face 캐시 방식 관련 경고이며, 모델 실행 자체가 실패했다는 뜻은 아닙니다.

## 빠른 시작

```python
from satprep.io.raster import SatelliteImage
from satprep.grid.tiler import GridTiler

with SatelliteImage.open("input.tif") as image:
    metadata = image.get_metadata()
    print(metadata.crs)

    tiler = GridTiler(chip_size=512, stride=512)
    chips = tiler.save_chips(image, "chips")
```

## CLI 사용 예시

래스터 메타데이터 확인:

```bash
satprep info input.tif
```

칩 생성:

```bash
satprep grid input.tif --chip-size 512 --stride 512 --out chips/
```

칩 품질 분석:

```bash
satprep quality chips/ --out report.json
```

`usable` 칩만 학습 폴더로 복사:

```bash
satprep filter report.json --status usable --chip-dir chips/ --out clean_chips/
```

고전적 업스케일링:

```bash
satprep upscale input.tif --scale 2 --method bicubic --out upscaled.tif
```

HAN pretrained 초해상도:

```bash
satprep super-resolve input.tif --model han --scale 4 --out sr.tif
```

다중시기 median composite:

```bash
satprep composite image1.tif image2.tif image3.tif --method median --out composite.tif
```

## Python 사용 예시

품질 리포트 생성:

```python
from pathlib import Path

from satprep.export.report import export_quality_reports_to_json
from satprep.quality.report import analyze_chip_file

reports = [analyze_chip_file(path) for path in Path("chips").glob("*.tif")]
export_quality_reports_to_json(reports, "report.json")
```

고전적 GeoTIFF 업스케일링:

```python
from satprep.restoration.upscale import upscale_raster

upscale_raster("input.tif", "upscaled.tif", scale=2, method="bicubic")
```

HAN pretrained 초해상도:

```python
from satprep.restoration.super_resolution import create_super_resolution_model

model = create_super_resolution_model("han", scale=4)
result = model.predict(chip_array)
```

Clear-sky 다중시기 합성:

```python
import numpy as np

from satprep.fusion.composite import create_clear_sky_composite

stack = np.stack([image_t1, image_t2, image_t3], axis=0)
composite = create_clear_sky_composite(stack)
```

## 현재 한계

- STAC 지원은 현재 README 데모 스크립트 중심이며, 패키지 API로는 아직 최소 구현 상태입니다.
- cloud/shadow 분석은 Sentinel-2 SCL 공식 마스크가 아니라 RGB 휴리스틱입니다.
- multi-temporal fusion은 입력 영상들이 이미 같은 CRS, transform, width, height로 정렬되어 있다고 가정합니다.
- HAN 모델은 자연영상 기반 pretrained 모델이므로 위성영상 물리 해상도 검증 모델이 아닙니다.
- super-resolution 결과는 시각적 향상 결과이며, 실제 지상 객체가 물리적으로 새로 관측된 것처럼 해석하면 안 됩니다.
- README 데모는 Sentinel-2 true-color `visual` asset을 사용합니다. NIR/SWIR 기반 정밀 구름 탐지는 아직 포함하지 않았습니다.

## Roadmap

- STAC API 정식 지원
- Sentinel-2 / Landsat preset
- Sentinel-2 SCL 기반 cloud/shadow masking
- DSen2 wrapper preset
- HAN 모델 옵션 확장
- SwinIR wrapper preset
- Real-ESRGAN wrapper preset
- HighRes-net wrapper preset
- DeepSUM wrapper preset
- Pansharpening 개선
- TorchGeo integration
- COCO / YOLO export
- Notebook 기반 시각화 예제

## 데이터 출처

- Sentinel-2 L2A Cloud Optimized GeoTIFF: [AWS Open Data Registry](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
- STAC 검색: [Element 84 Earth Search](https://earth-search.aws.element84.com/v1)
- README 데모 사용 item: `S2B_52SCG_20240819_0_L2A`
- 초해상도 모델: [Hugging Face eugenesiow/han](https://huggingface.co/eugenesiow/han)

