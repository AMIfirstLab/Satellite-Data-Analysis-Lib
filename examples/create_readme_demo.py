from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from satprep.restoration.super_resolution import create_super_resolution_model

import cv2
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as window_transform

from satprep.export.report import export_quality_reports_to_csv, export_quality_reports_to_json, summarize_quality_reports
from satprep.fusion.composite import create_clear_sky_composite
from satprep.grid.tiler import GridTiler
from satprep.io.raster import SatelliteImage
from satprep.quality.cloud import calculate_brightness_stats, calculate_cloud_score
from satprep.quality.classifier import QualityThresholds
from satprep.quality.entropy import calculate_entropy_score, calculate_nodata_ratio
from satprep.quality.report import QualityReport, analyze_chip_file
from satprep.restoration.upscale import upscale_image

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
SENTINEL_COLLECTION = "sentinel-2-l2a"
DEMO_BBOX = [127.0, 37.3, 127.3, 37.6]
DEMO_DATETIME = "2024-08-01T00:00:00Z/2024-10-15T23:59:59Z"
FULL_SCENE_CROP_SIZE = 2048
CHIP_SIZE = 512


@dataclass(frozen=True)
class SentinelItem:
    """README 데모에 필요한 Sentinel-2 STAC item 정보를 보관한다."""

    item_id: str
    datetime: str
    cloud_cover: float
    visual_href: str
    thumbnail_href: str | None


@dataclass(frozen=True)
class SelectedWindow:
    """큰 Sentinel-2 장면에서 선택한 실제 픽셀 window."""

    item: SentinelItem
    window: Window
    score: float
    estimated_cloud_score: float
    estimated_nodata_ratio: float


def search_sentinel2_items(limit: int = 20) -> list[SentinelItem]:
    """Earth Search STAC API에서 Sentinel-2 L2A COG item을 찾는다."""
    payload = {
        "collections": [SENTINEL_COLLECTION],
        "bbox": DEMO_BBOX,
        "datetime": DEMO_DATETIME,
        "limit": limit,
    }
    request = urllib.request.Request(
        EARTH_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)

    items: list[SentinelItem] = []
    for feature in data.get("features", []):
        assets = feature.get("assets", {})
        if "visual" not in assets:
            continue
        properties = feature.get("properties", {})
        items.append(
            SentinelItem(
                item_id=feature["id"],
                datetime=properties.get("datetime", ""),
                cloud_cover=float(properties.get("eo:cloud_cover", 100.0)),
                visual_href=assets["visual"]["href"],
                thumbnail_href=assets.get("thumbnail", {}).get("href"),
            )
        )
    if not items:
        raise RuntimeError("No Sentinel-2 L2A items were found for the demo AOI.")
    return sorted(items, key=lambda item: item.cloud_cover)


def read_overview(href: str, overview_size: int = 1024) -> tuple[np.ndarray, tuple[int, int]]:
    """원격 COG 전체 장면을 작은 overview로 읽는다."""
    # COG는 필요한 overview block만 읽을 수 있어 전체 원본을 다운로드하지 않는다.
    with rasterio.open(href) as src:
        overview = src.read(
            out_shape=(src.count, overview_size, overview_size),
            resampling=Resampling.bilinear,
        )
        return overview.astype(np.uint8), (src.width, src.height)


def select_realistic_window(items: list[SentinelItem], crop_size: int = FULL_SCENE_CROP_SIZE) -> SelectedWindow:
    """실제 Sentinel-2 장면에서 품질 차이가 자연스럽게 보이는 window를 고른다."""
    best: SelectedWindow | None = None
    candidate_items = sorted(items, key=lambda item: abs(item.cloud_cover - 12.0))[:8]

    for item in candidate_items:
        overview, full_size = read_overview(item.visual_href)
        full_width, full_height = full_size
        scale_x = full_width / overview.shape[2]
        scale_y = full_height / overview.shape[1]
        crop_w = max(64, int(crop_size / scale_x))
        crop_h = max(64, int(crop_size / scale_y))
        step_x = max(32, crop_w // 2)
        step_y = max(32, crop_h // 2)

        for y in range(0, overview.shape[1] - crop_h, step_y):
            for x in range(0, overview.shape[2] - crop_w, step_x):
                patch = overview[:, y : y + crop_h, x : x + crop_w]
                nodata = calculate_nodata_ratio(patch, nodata=0)
                if nodata > 0.08:
                    continue
                cloud = calculate_cloud_score(patch)
                entropy = calculate_entropy_score(patch)
                brightness = calculate_brightness_stats(patch)["mean_brightness"]
                # 너무 완벽하거나 너무 망가진 예시보다 실제 장면에서 나올 법한 중간 품질을 선호한다.
                score = (
                    1.0
                    - abs(cloud - 0.12) * 2.0
                    - abs(brightness - 0.42) * 0.8
                    + entropy * 0.35
                    - nodata * 3.0
                )
                if best is None or score > best.score:
                    full_x = int(np.clip(x * scale_x, 0, full_width - crop_size))
                    full_y = int(np.clip(y * scale_y, 0, full_height - crop_size))
                    best = SelectedWindow(
                        item=item,
                        window=Window(full_x, full_y, crop_size, crop_size),
                        score=float(score),
                        estimated_cloud_score=float(cloud),
                        estimated_nodata_ratio=float(nodata),
                    )

    if best is None:
        fallback = items[0]
        with rasterio.open(fallback.visual_href) as src:
            window = Window((src.width - crop_size) // 2, (src.height - crop_size) // 2, crop_size, crop_size)
        return SelectedWindow(fallback, window, 0.0, 0.0, 0.0)
    return best


def read_cog_window_to_geotiff(item: SentinelItem, window: Window, output_path: Path) -> np.ndarray:
    """원격 Sentinel-2 COG의 일부 window만 읽고 로컬 GeoTIFF로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(item.visual_href) as src:
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=int(window.height),
            width=int(window.width),
            transform=window_transform(window, src.transform),
            dtype=str(data.dtype),
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    return data


def to_display_image(chw: np.ndarray, size: int = 512) -> Image.Image:
    """CHW 배열을 README에 맞는 RGB 이미지로 변환한다."""
    arr = np.moveaxis(chw[:3], 0, -1)
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def labeled_frame(image: Image.Image, label: str, height: int = 26) -> Image.Image:
    """GIF 프레임 상단에 짧은 설명을 넣는다."""
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, frame.width, height), fill=(12, 17, 24))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return frame


def save_gif(frames: list[Image.Image], output_path: Path, duration: int = 950) -> None:
    """프레임 목록을 GIF로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def draw_grid(image: Image.Image, chips_per_side: int = 4) -> Image.Image:
    """칩 격자를 시각화한다."""
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    for index in range(chips_per_side + 1):
        pos = int(index * frame.width / chips_per_side)
        draw.line((pos, 0, pos, frame.height), fill=(255, 221, 87), width=2)
        draw.line((0, pos, frame.width, pos), fill=(255, 221, 87), width=2)
    return frame


def draw_quality_overlay(image: Image.Image, reports: list[QualityReport], chips_per_side: int = 4) -> Image.Image:
    """품질 판정 결과를 칩 단위 색상으로 표시한다."""
    frame = image.convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {"usable": (42, 176, 106, 70), "warning": (255, 193, 7, 92), "reject": (220, 53, 69, 100)}
    cell = frame.width / chips_per_side
    for report in reports:
        chip_index = int(report.chip_id.split("_")[-1]) - 1
        row = chip_index // chips_per_side
        col = chip_index % chips_per_side
        x0 = int(col * cell)
        y0 = int(row * cell)
        x1 = int((col + 1) * cell)
        y1 = int((row + 1) * cell)
        draw.rectangle((x0, y0, x1, y1), fill=colors.get(report.status, (255, 255, 255, 60)))
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 150), width=1)
    return Image.alpha_composite(frame, overlay).convert("RGB")


def make_quality_demo(reference_tif: Path, work_dir: Path, assets_dir: Path) -> dict[str, float | int]:
    """실제 Sentinel-2 window를 grid tiling하고 품질 리포트를 만든다."""
    chip_dir = work_dir / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    thresholds = QualityThresholds(
        reject_nodata_ratio=0.4,
        reject_cloud_score=0.65,
        warning_cloud_score=0.16,
        warning_shadow_score=0.34,
        warning_blur_score=0.92,
        reject_visibility_score=0.18,
    )
    with SatelliteImage.open(reference_tif) as image:
        tiler = GridTiler(chip_size=CHIP_SIZE, stride=CHIP_SIZE)
        tiler.save_chips(image, chip_dir)
        base = to_display_image(image.dataset.read(), size=512)

    reports = [analyze_chip_file(path, thresholds=thresholds) for path in sorted(chip_dir.glob("*.tif"))]
    export_quality_reports_to_json(reports, work_dir / "quality_report.json")
    export_quality_reports_to_csv(reports, work_dir / "quality_report.csv")

    frames = [
        labeled_frame(base, "Real Sentinel-2 L2A COG window"),
        labeled_frame(draw_grid(base, chips_per_side=4), "Grid tiling: 2048 x 2048 window -> 16 chips"),
        labeled_frame(draw_quality_overlay(base, reports, chips_per_side=4), "Quality classification from real pixels"),
    ]
    save_gif(frames, assets_dir / "satprep_grid_quality.gif")
    return summarize_quality_reports(reports)


def side_by_side_frame(left: Image.Image, right: Image.Image, label: str) -> Image.Image:
    """저해상도 입력과 초해상도 결과를 좌우 비교 프레임으로 만든다."""
    width = left.width + right.width + 10
    height = left.height + 48
    frame = Image.new("RGB", (width, height), (12, 17, 24))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    frame.paste(left, (0, 48))
    frame.paste(right, (left.width + 10, 48))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    draw.text((8, 30), "Left: low-scale input", fill=(210, 220, 230), font=font)
    draw.text((left.width + 18, 30), "Right: HAN pretrained super-resolution", fill=(210, 220, 230), font=font)
    return frame


def make_super_resolution_demo(reference_tif: Path, work_dir: Path, assets_dir: Path) -> None:
    """실제 Sentinel-2 crop에서 HAN 초해상도 좌우 비교 GIF를 만든다."""
    with rasterio.open(reference_tif) as src:
        crop = src.read()

    model = create_super_resolution_model("han", scale=4)
    patch_origins = [(128, 128), (768, 256), (1152, 768), (512, 1152)]
    frames: list[Image.Image] = []
    for index, (row, col) in enumerate(patch_origins, start=1):
        hr_patch = crop[:, row : row + 256, col : col + 256]
        lr_hwc = cv2.resize(np.moveaxis(hr_patch, 0, -1), (64, 64), interpolation=cv2.INTER_AREA)
        lr = np.moveaxis(lr_hwc, -1, 0).astype(np.uint8)
        low_display = upscale_image(lr, scale=4, method="nearest")
        sr = model.predict(lr)
        frames.append(
            side_by_side_frame(
                to_display_image(low_display, size=320),
                to_display_image(sr, size=320),
                f"Sentinel-2 patch {index} - eugenesiow/han x4",
            )
        )
    save_gif(frames, assets_dir / "satprep_super_resolution.gif", duration=1100)


def make_temporal_demo(items: list[SentinelItem], window: Window, work_dir: Path, assets_dir: Path) -> list[str]:
    """같은 Sentinel-2 tile의 여러 날짜 window로 clear-sky composite GIF를 만든다."""
    selected = [items[0], items[max(1, len(items) // 4)], items[max(2, len(items) // 2)]]
    local_paths: list[Path] = []
    arrays: list[np.ndarray] = []
    for index, item in enumerate(selected, start=1):
        path = work_dir / f"temporal_{index}_{item.item_id}.tif"
        arrays.append(read_cog_window_to_geotiff(item, window, path))
        local_paths.append(path)

    stack = np.stack(arrays, axis=0)
    composite = create_clear_sky_composite(stack)
    frames = [
        labeled_frame(to_display_image(arrays[0], size=512), f"Time 1: {selected[0].datetime[:10]}"),
        labeled_frame(to_display_image(arrays[1], size=512), f"Time 2: {selected[1].datetime[:10]}"),
        labeled_frame(to_display_image(arrays[2], size=512), f"Time 3: {selected[2].datetime[:10]}"),
        labeled_frame(to_display_image(composite, size=512), "Clear-sky quality mosaic composite"),
    ]
    save_gif(frames, assets_dir / "satprep_temporal_composite.gif")
    return [str(path) for path in local_paths]


def filter_same_tile_items(items: list[SentinelItem], selected_item: SentinelItem) -> list[SentinelItem]:
    """선택된 item과 같은 Sentinel-2 tile의 날짜들을 고른다."""
    tile_key = selected_item.item_id.split("_")[1]
    same_tile = [item for item in items if item.item_id.split("_")[1] == tile_key]
    if len(same_tile) < 3:
        return items[:3]
    # 낮은 구름, 중간 구름, 높은 구름 날짜가 섞이도록 정렬된 목록을 사용한다.
    return same_tile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work_dir = root / "docs" / "demo_outputs"
    assets_dir = root / "docs" / "assets"
    if work_dir.exists():
        # 데모 산출물 폴더는 재생성 가능한 중간 결과라 실행할 때마다 깨끗하게 만든다.
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    items = search_sentinel2_items(limit=20)
    selected = select_realistic_window(items)
    reference_tif = work_dir / "sentinel2_real_window.tif"
    reference_data = read_cog_window_to_geotiff(selected.item, selected.window, reference_tif)

    summary = make_quality_demo(reference_tif, work_dir, assets_dir)
    make_super_resolution_demo(reference_tif, work_dir, assets_dir)
    temporal_items = filter_same_tile_items(items, selected.item)
    temporal_paths = make_temporal_demo(temporal_items, selected.window, work_dir, assets_dir)

    with (work_dir / "demo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "Sentinel-2 L2A Cloud Optimized GeoTIFF",
                "stac_collection": SENTINEL_COLLECTION,
                "selected_item": selected.item.__dict__,
                "selected_window": {
                    "col_off": selected.window.col_off,
                    "row_off": selected.window.row_off,
                    "width": selected.window.width,
                    "height": selected.window.height,
                    "estimated_cloud_score": selected.estimated_cloud_score,
                    "estimated_nodata_ratio": selected.estimated_nodata_ratio,
                },
                "reference_shape": list(reference_data.shape),
                "quality_summary": summary,
                "super_resolution_model": "eugenesiow/han",
                "temporal_composite_method": "clear-sky quality mosaic",
                "temporal_sources": temporal_paths,
            },
            f,
            indent=2,
        )
    print(f"Demo assets written to {assets_dir}")


if __name__ == "__main__":
    main()
