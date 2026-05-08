from satprep.restoration.upscale import upscale_raster


def main() -> None:
    # bicubic 보간으로 픽셀 크기를 2배 세밀하게 만든다.
    upscale_raster("input.tif", "outputs/upscaled.tif", scale=2, method="bicubic")


if __name__ == "__main__":
    main()

