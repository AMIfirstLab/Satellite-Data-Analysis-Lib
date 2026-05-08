from satprep.fusion.composite import create_median_composite, save_composite_like
from satprep.fusion.temporal_stack import stack_aligned_images


def main() -> None:
    # 입력 영상들은 같은 CRS, transform, 크기를 가진다고 가정한다.
    paths = ["image1.tif", "image2.tif", "image3.tif"]
    stack = stack_aligned_images(paths)
    composite = create_median_composite(stack)
    save_composite_like(paths[0], composite, "outputs/clean_composite.tif")


if __name__ == "__main__":
    main()

