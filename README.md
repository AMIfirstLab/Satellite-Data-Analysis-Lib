# satprep

A satellite image preprocessing and quality control library for deep learning workflows.

`satprep` helps you turn large geospatial rasters into training-ready image chips, inspect chip quality, filter bad samples, upscale imagery, and build simple multi-temporal composites. It is designed as a preprocessing and quality-control layer, not a full model training framework.

## Demo

The GIFs below were generated with `examples/create_readme_demo.py` using real [Sentinel-2 L2A Cloud Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) discovered through [Element 84 Earth Search](https://earth-search.aws.element84.com/v1). The demo reads only a 2048 x 2048 pixel window from a much larger 10980 x 10980 Sentinel-2 tile, so it demonstrates the intended large-raster workflow without downloading a full scene.

### Grid Tiling And Quality Control

![satprep grid and quality demo](docs/assets/satprep_grid_quality.gif)

What happens:

1. A real Sentinel-2 L2A COG scene is queried from STAC.
2. A 2048 x 2048 pixel window is read from the remote COG with `rasterio`.
3. The window is saved as a local GeoTIFF while preserving CRS, transform, resolution, dtype, and band metadata.
4. The raster is split into 16 real 512 x 512 training chips.
5. Each chip is scored for blur, entropy, nodata, brightness, cloud-like regions, shadow-like regions, and object visibility.
6. Chips are classified as `usable`, `warning`, or `reject`.

The current README demo selected Sentinel-2 item `S2B_52SCG_20240819_0_L2A`. Its quality report produced 16 chips: 8 `usable`, 8 `warning`, and 0 `reject`. This is intentionally not a toy example with artificial white/black blocks; the warnings come from real cloud-like brightness, shadow-like darkness, texture, and visibility scores in the selected Sentinel-2 window.

### Classical And Deep Super-Resolution

![satprep super-resolution demo](docs/assets/satprep_super_resolution.gif)

What happens:

1. Real Sentinel-2 patches are extracted from the selected large-scene window.
2. Each patch is degraded to a lower-scale input.
3. The pretrained Hugging Face `eugenesiow/han` Holistic Attention Network x4 checkpoint is loaded through the optional `super-image` integration.
4. The GIF shows a side-by-side comparison: low-scale input on the left, HAN super-resolution on the right.

Classical upscaling methods such as `nearest`, `bilinear`, `bicubic`, and `lanczos` are still available through `satprep.restoration.upscale`; the README GIF focuses on the pretrained deep super-resolution path requested for the project demo.

### Multi-Temporal Composite

![satprep temporal composite demo](docs/assets/satprep_temporal_composite.gif)

What happens:

1. Multiple Sentinel-2 L2A COG items from the same tile are selected.
2. The exact same pixel window is read from each date.
3. The aligned images are stacked as `(time, band, height, width)`.
4. A clear-sky quality mosaic is generated with RGB heuristics: bright white cloud-like pixels and very dark shadow-like pixels are avoided, while clearer and slightly brighter pixels are preferred.

This produces a brighter, cleaner visual composite than a plain median when only a few dates are available.

To regenerate these assets:

```bash
python examples/create_readme_demo.py
```

The script queries STAC, reads remote Sentinel-2 COG windows, writes intermediate demo outputs into `docs/demo_outputs/`, and writes README GIFs into `docs/assets/`. The full Sentinel-2 scene is not downloaded.

## Main Features

- GeoTIFF and Cloud Optimized GeoTIFF loading with `rasterio`
- Window-based raster reads for large imagery
- CRS, affine transform, pixel resolution, nodata, dtype, bounds, and band metadata preservation
- Grid/chip generation with per-chip GeoTIFF and JSON metadata export
- Chip-level blur, sharpness, entropy, nodata, cloud, shadow, brightness, saturation, and object visibility scoring
- `usable`, `warning`, and `reject` chip classification
- JSON and CSV quality report export
- Training dataset filtering from quality reports
- Classical upscaling: `nearest`, `bilinear`, `bicubic`, `lanczos`
- PyTorch super-resolution interfaces: pretrained `HAN`, `SRCNN`, lightweight `EDSR`, lightweight `RCAN`, and TorchScript adapters for larger external models
- Multi-temporal median, clear-sky quality mosaic, quality-weighted, and shadow-aware composites
- CLI commands for common preprocessing workflows

## Installation

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
python -m pytest
```

For PyTorch and HAN super-resolution:

```bash
pip install -e ".[deep]"
```

## Quickstart

```python
from satprep.io.raster import SatelliteImage
from satprep.grid.tiler import GridTiler

with SatelliteImage.open("input.tif") as image:
    metadata = image.get_metadata()
    print(metadata.crs)

    tiler = GridTiler(chip_size=512, stride=512)
    chips = tiler.save_chips(image, "chips")
```

## CLI Examples

Inspect raster metadata:

```bash
satprep info input.tif
```

Create training chips:

```bash
satprep grid input.tif --chip-size 512 --stride 512 --out chips/
```

Analyze chip quality:

```bash
satprep quality chips/ --out report.json
```

Copy only usable chips:

```bash
satprep filter report.json --status usable --chip-dir chips/ --out clean_chips/
```

Classical upscaling:

```bash
satprep upscale input.tif --scale 2 --method bicubic --out upscaled.tif
```

Deep super-resolution with pretrained HAN:

```bash
satprep super-resolve input.tif --model han --scale 4 --out sr.tif
```

Median temporal composite:

```bash
satprep composite image1.tif image2.tif image3.tif --method median --out composite.tif
```

## Python Examples

Quality report generation:

```python
from pathlib import Path

from satprep.export.report import export_quality_reports_to_json
from satprep.quality.report import analyze_chip_file

reports = [analyze_chip_file(path) for path in Path("chips").glob("*.tif")]
export_quality_reports_to_json(reports, "report.json")
```

Classical raster upscaling:

```python
from satprep.restoration.upscale import upscale_raster

upscale_raster("input.tif", "upscaled.tif", scale=2, method="bicubic")
```

Pretrained HAN inference:

```python
from satprep.restoration.super_resolution import create_super_resolution_model

model = create_super_resolution_model("han", scale=4)
result = model.predict(chip_array)
```

## Super-Resolution Warning

Deep learning-based super-resolution can hallucinate fine spatial details. Outputs should not be treated as physically verified ground truth. For scientific, military, disaster response, or detection-critical tasks, always compare super-resolved outputs against original imagery and use uncertainty or confidence maps when possible.

The bundled `HANModel` wrapper uses the pretrained Hugging Face `eugenesiow/han` model through `super-image`. `SRCNN`, lightweight `EDSR`, and lightweight `RCAN` are practical PyTorch implementations for experimentation and wrapper development. `RealESRGAN`, `SwinIR`, `DSen2`, `HighResNet`, and `DeepSUM` are provided as TorchScript adapter interfaces so larger external models can be plugged in without forcing heavy dependencies into the base package.

## Current Limitations

- STAC support is minimal in the first version.
- Multi-temporal fusion assumes already aligned input images.
- Cloud and shadow detection use simple heuristics.
- README demo windows are selected from Sentinel-2 true-color COG assets, but the current cloud/shadow labels still come from simple RGB heuristics rather than official Sentinel-2 scene classification masks.
- The HAN super-resolution model was trained on natural-image super-resolution data, so satellite outputs should still be treated as visual enhancement rather than physically verified ground truth.

## Roadmap

- STAC API support
- Sentinel-2 and Landsat presets
- Better cloud/shadow masking
- DSen2 wrapper presets
- HAN model options
- SwinIR wrapper presets
- Real-ESRGAN wrapper presets
- HighRes-net wrapper presets
- DeepSUM wrapper presets
- Pansharpening improvements
- TorchGeo integration
- COCO/YOLO export
- Notebook visualization

## Dataset Reference

Sentinel-2 L2A Cloud Optimized GeoTIFFs are provided through the AWS Open Data Registry and are discoverable through Element 84 Earth Search STAC. The README demo uses the true-color `visual` asset from Sentinel-2 L2A item `S2B_52SCG_20240819_0_L2A`.
