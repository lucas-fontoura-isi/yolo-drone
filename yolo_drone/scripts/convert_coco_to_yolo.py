import argparse
from pathlib import Path
from yolo_drone.src.clean_data import convert_yolo_to_coco

def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to YOLO format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "images_folder",
        type=Path,
        help="Path to the folder containing the images."
    )
    parser.add_argument(
        "coco_json",
        type=Path,
        help="Path to the input COCO json file."
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Path to the output folder where YOLO formatted files will be saved."
    )
    parser.add_argument(
        "-tr",
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of images to be used for training. The rest will be used for validation."
    )
    parser.add_argument(
        "-us",
        "--use_sahi",
        action="store_true",
        help="Whether to use SAHI instead of a manual split for creating train/val sets."
    )

    return parser.parse_args()

def main():
    args = get_args()
    convert_yolo_to_coco(**vars(args))

if __name__ == "__main__":
    main()