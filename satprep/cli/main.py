from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import typer

from satprep.export.geotiff import export_chips_for_training
from satprep.export.report import export_quality_reports_to_csv, export_quality_reports_to_json, summarize_quality_reports
from satprep.fusion.composite import create_median_composite, create_quality_weighted_composite, save_composite_like, write_fusion_report
from satprep.fusion.temporal_stack import stack_aligned_images
from satprep.grid.tiler import tile_raster
from satprep.io.raster import SatelliteImage
from satprep.quality.report import analyze_chip_file
from satprep.restoration.super_resolution import create_super_resolution_model, super_resolve_raster
from satprep.restoration.upscale import upscale_raster

app = typer.Typer(help="Satellite image preprocessing and quality-control toolkit.")


@app.command()
def info(input_path: Path) -> None:
    """래스터 메타데이터를 출력한다."""
    with SatelliteImage.open(input_path) as image:
        typer.echo(json.dumps(image.get_metadata().to_dict(), indent=2))


@app.command()
def grid(
    input_path: Path,
    chip_size: int = typer.Option(512, "--chip-size"),
    stride: int | None = typer.Option(None, "--stride"),
    overlap: int = typer.Option(0, "--overlap"),
    out: Path = typer.Option(..., "--out"),
    drop_partial: bool = typer.Option(True, "--drop-partial/--keep-partial"),
) -> None:
    """래스터를 격자 칩으로 저장한다."""
    chips = tile_raster(input_path, out, chip_size=chip_size, stride=stride, overlap=overlap, drop_partial=drop_partial)
    typer.echo(f"Saved {len(chips)} chips to {out}")


@app.command()
def quality(chip_dir: Path, out: Path = typer.Option(..., "--out")) -> None:
    """칩 폴더의 품질 리포트를 생성한다."""
    reports = [analyze_chip_file(path) for path in sorted(chip_dir.glob("*.tif"))]
    export_quality_reports_to_json(reports, out)
    export_quality_reports_to_csv(reports, out.with_suffix(".csv"))
    typer.echo(json.dumps(summarize_quality_reports(reports), indent=2))


@app.command("filter")
def filter_command(
    report_path: Path,
    status: list[str] = typer.Option(["usable"], "--status"),
    out: Path = typer.Option(..., "--out"),
    chip_dir: Path = typer.Option(Path("."), "--chip-dir"),
) -> None:
    """품질 상태로 칩을 걸러 학습 폴더에 복사한다."""
    export_chips_for_training(chip_dir, report_path, out, list(status))
    typer.echo(f"Filtered chips copied to {out}")


@app.command()
def upscale(
    input_path: Path,
    scale: int = typer.Option(2, "--scale"),
    method: str = typer.Option("bicubic", "--method"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """고전적 보간 방식으로 래스터를 확대한다."""
    upscale_raster(input_path, out, scale=scale, method=method)
    typer.echo(f"Saved upscaled raster to {out}")


@app.command("super-resolve")
def super_resolve(
    input_path: Path,
    model_name: str = typer.Option("srcnn", "--model"),
    weights: Path | None = typer.Option(None, "--weights"),
    scale: int = typer.Option(2, "--scale"),
    out: Path = typer.Option(..., "--out"),
    tile_size: int | None = typer.Option(None, "--tile-size"),
    overlap: int = typer.Option(32, "--overlap"),
    allow_untrained: bool = typer.Option(False, "--allow-untrained"),
) -> None:
    """딥러닝 초해상도 모델로 래스터를 처리한다."""
    model = create_super_resolution_model(model_name, scale=scale, allow_untrained=allow_untrained)
    if weights is not None:
        model.load_weights(weights)
    super_resolve_raster(input_path, out, model=model, tile_size=tile_size, overlap=overlap)
    typer.echo(f"Saved super-resolved raster to {out}")


@app.command()
def composite(
    image_paths: list[Path],
    method: str = typer.Option("median", "--method"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """이미 정합된 다중 시기 영상 합성을 생성한다."""
    stack = stack_aligned_images(image_paths)
    if method == "median":
        result = create_median_composite(stack)
    elif method == "quality-weighted":
        # CLI 첫 버전은 동일 가중치를 사용한다.
        result = create_quality_weighted_composite(stack, np.ones(stack.shape[0], dtype=np.float32))
    else:
        raise typer.BadParameter("method must be median or quality-weighted")
    save_composite_like(image_paths[0], result, out)
    write_fusion_report(out.with_suffix(".json"), method, image_paths)
    typer.echo(f"Saved composite raster to {out}")


if __name__ == "__main__":
    app()
