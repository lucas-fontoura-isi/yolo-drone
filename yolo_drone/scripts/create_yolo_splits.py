import argparse
from pathlib import Path
from yolo_drone.src.clean_data import create_yolo_splits

def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create .txt files for YOLO training and validation splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "images_dir",
        type=Path,
        help="Path to the directory containing the images."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory where splits will be saved."
    )
    parser.add_argument(
        "-e",
        "--img_extension",
        type=str,
        default=".png",
        help="Image file extension (e.g., .jpg, .png)."
    )
    parser.add_argument(
        "-r",
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (between 0 and 1)."
    )

    return parser.parse_args()

def main() -> None:
    create_yolo_splits(**vars(get_args()))

if __name__ == "__main__":
    main()