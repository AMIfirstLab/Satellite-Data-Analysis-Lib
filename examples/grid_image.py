from satprep.grid.tiler import GridTiler
from satprep.io.raster import SatelliteImage


def main() -> None:
    # 입력 래스터를 열고 전체 영상은 메모리에 올리지 않는다.
    with SatelliteImage.open("input.tif") as image:
        tiler = GridTiler(chip_size=512, stride=512)
        chips = tiler.save_chips(image, "chips")
        print(f"Saved {len(chips)} chips")


if __name__ == "__main__":
    main()

