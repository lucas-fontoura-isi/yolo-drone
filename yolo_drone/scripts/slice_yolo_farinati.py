import argparse
from pathlib import Path
from yolo_drone.src.slicing import slice_yolo_dataset

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Apply a sliding window over images in a YOLO dataset.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the YOLO dataset",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the YOLO dataset output",
    )
    parser.add_argument(
        "-a",
        "--annotations_path",
        type=Path,
        help="Path to the YOLO annotations, defaults to input_path",
    )
    parser.add_argument(
        "-w",
        "--window_shape",
        type=lambda s: tuple(int(i) for i in s.split(",")[:2]),
        help=(
            "Specify the window shape as 'height,width'. Both values "
            "must be integers. Example: --window_shape 640,640"
        ),
        default=(640, 640),
    )
    parser.add_argument(
        "-s",
        "--step_size",
        type=lambda s: tuple(int(i) for i in s.split(",")[:2]),
        help=(
            "Specify the step_size as 'step_y,step_x'. Both values "
            "must be integers. Example: --step_size 640,640"
        ),
    )
    parser.add_argument(
        "-g",
        "--grid",
        action="store_true",
        help="Store the sliding windows as a grid image, for debugging.",
    )
    parser.add_argument(
        "-m",
        "--minimum_pixels",
        type=int,
        help=(
            "Minimum number of pixels of the bounding box "
            "to consider the image a background."
        ),
        default=0,
    )
    parser.add_argument(
        "-o",
        "--minimum_overlap",
        type=float,
        help=(
            "Minimum percentage of the bounding box's area "
            "to consider the image a background."
        ),
        default=0.0,
    )
    parser.add_argument(
        "-b",
        "--background_percentage",
        type=float,
        help="Percentage of background images in the final dataset.",
        default=0.0,
    )

    return parser.parse_args()

def main() -> None:
    """Apply sliding windows over a YOLO dataset."""
    args = parse_args()

    slice_yolo_dataset(
        args.input,
        args.output,
        args.window_shape,
        args.step_size,
        args.annotations_path,
        args.grid,
        args.minimum_pixels,
        args.minimum_overlap,
        args.background_percentage,
    )


if __name__ == "__main__":
    main()
