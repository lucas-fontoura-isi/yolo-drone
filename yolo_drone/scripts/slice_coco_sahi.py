import argparse
from pathlib import Path
from yolo_drone.src.slicing import slice_coco_dataset

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Uses SAHI to slice COCO dataset images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "coco_annotation_file_path",
        type=Path,
        help="Path to the COCO annotation file."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing the images."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the sliced images and annotations."
    )
    parser.add_argument(
        "-sh",
        "--slice_height",
        type=int,
        default=640,
        help="Height of each slice."
    )
    parser.add_argument(
        "-sw",
        "--slice_width",
        type=int,
        default=640,
        help="Width of each slice."
    )
    parser.add_argument(
        "-hr",
        "--overlap_height_ratio",
        type=float,
        default=0.2,
        help="Overlap height ratio between slices."
    )
    parser.add_argument(
        "-wr",
        "--overlap_width_ratio",
        type=float,
        default=0.2,
        help="Overlap width ratio between slices."
    )
    parser.add_argument(
        "-ins",
        "--ignore_negative_samples",
        action="store_true",
        help="If set, slices without any annotations will be ignored."
    )
    parser.add_argument(
        "-ocan",
        "--output_coco_annotation_file_name",
        type=str,
        default="sliced_coco_annotations.json",
        help="Name of the output COCO annotation file."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    slice_coco_dataset(**vars(args))

if __name__ == "__main__":
    main()