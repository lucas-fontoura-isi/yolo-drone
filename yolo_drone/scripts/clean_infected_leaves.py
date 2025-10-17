import argparse
from pathlib import Path
from yolo_drone.src.clean_data import clean_infected_leaves

# COCO json cleaning script for the InfectedLeaves dataset from the paper: "Bacterial-fungicidal vine disease detection with proximal aerial images"
# - Delete ids that are not in the images folder
# - Delete ids that have an area below 1
# - Rearange ids numbering to be consecutive
# - Scale bounding boxes to match images real size (3840x2160)

def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Clean COCO json file for InfectedLeaves dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "images_folder",
        type=Path,
        help="Path to the folder containing the images."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the input COCO json file."
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path to the output cleaned COCO json file."
    )
    parser.add_argument(
        "-sx",
        "--scale_x",
        type=float,
        default=3840/640,
        help="Scaling factor for the x coordinates of the bounding boxes. Default is 3840/640."
    )
    parser.add_argument(
        "-sy",
        "--scale_y",
        type=float,
        default=2160/360,
        help="Scaling factor for the y coordinates of the bounding boxes. Default is 2160/360."
    )
    parser.add_argument(
        "-ri",
        "--reassign_image_ids",
        action="store_true",
        help="Reassign image ids to be consecutive, starting from 1."
    )
    parser.add_argument(
        "-ra",
        "--reassign_annotation_ids",
        action="store_true",
        help="Reassign annotation ids to be consecutive, starting from 1."
    )
    parser.add_argument(
        "-koi",
        "--keep_original_ids",
        action="store_true",
        help="Keep original image and annotation ids in 'original_id' field."
    )
    parser.add_argument(
        "-kof",
        "--keep_original_fname",
        action="store_true",
        help="Keep original image file names in 'original_file_name' field."
    )

    return parser.parse_args()

def main() -> None:
    args = get_args()
    clean_infected_leaves(**vars(args))


if __name__ == "__main__":
    main()