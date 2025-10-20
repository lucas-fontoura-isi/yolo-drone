import argparse
from pathlib import Path
from yolo_drone.src.segmentation import segment_synthetic

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Segment synthetic images using color-based segmentation. (It uses a segmentation_config.yaml for the colors ranges.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing synthetic images to be segmented."
    )
    parser.add_argument(
        "labels_dir",
        type=Path,
        help="Directory to save the generated YOLO-Seg labels."
    )
    parser.add_argument(
        "-g",
        "--convert_gimphsv_to_cv2hsv",
        action="store_true",
        help="If using GIMP to get HSV values, enable to convert them to OpenCV HSV format."
    )
    parser.add_argument(
        "-a",
        "--min_area",
        type=int,
        default=50,
        help="Minimum area (in pixels) for a blob to be considered valid."
    )
    parser.add_argument(
        "-e",
        "--epsilon_ratio",
        type=float,
        default=0.01,
        help="Ratio of the arc length used to approximate polygons."
    )
    parser.add_argument(
        "-d",
        "--draw_segmentation",
        action="store_true",
        help="If set, will draw segmentation masks on the images in a new directory called 'segmented_output'."
    )
    parser.add_argument(
        "-dt",
        "--detection_task",
        action="store_true",
        help="If set, will return bounding box masks, instead of segmentation ones."
    )


    return parser.parse_args()

def main() -> None:
    args = parse_args()
    segment_synthetic(**vars(args))

if __name__ == "__main__":
    main()