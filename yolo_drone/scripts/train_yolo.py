import argparse
from pathlib import Path
from yolo_drone.src.train import train_yolo

def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model."
    )

    parser.add_argument(
        "data",
        type=Path,
        help="Path to the .yaml config file on the YOLO-formatted dataset."
    )

    return parser.parse_args()

def main() -> None:
    args = get_args()
    train_yolo(args.data)

if __name__ == "__main__":
    main()