from __future__ import annotations

import cv2
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, cast
from sahi.slicing import slice_coco

if TYPE_CHECKING:
    from typing import Final, Self

    from numpy import typing as npt


def slice_coco_dataset(
    coco_annotation_file_path: Path,
    image_dir: Path,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    ignore_negative_samples: bool,
    output_coco_annotation_file_name: str,
    output_dir: str
):
    # Path and I/O checks
    if not coco_annotation_file_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_annotation_file_path}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        ignore_negative_samples=ignore_negative_samples,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        output_dir=output_dir
    )


@dataclass(frozen=True)
class Annotation:
    """A YOLO formatted bounding box annotation."""

    class_id: int
    """The index of the class in the list of classes."""

    x_center: float
    """The x coordinate of the center of the bounding box in the image."""

    y_center: float
    """The y coordinate of the center of the bounding box in the image."""

    width: float
    """The width of the bounding box in the image."""

    height: float
    """The height of the bounding box in the image."""

    @classmethod
    def from_annotation_line(cls, line: str) -> Self:
        """Build a YOLO annotation from a single line in an annotation file."""
        class_id, x_center, y_center, width, height = line.strip().split()
        return cls(
            int(class_id),
            float(x_center),
            float(y_center),
            float(width),
            float(height),
        )

    def to_annotation_line(self) -> str:
        """Convert a YOLO annotation to a single line in an annotation file."""
        return (
            f"{self.class_id} {self.x_center} {self.y_center} "
            f"{self.width} {self.height}"
        )

    @classmethod
    def from_annotation_file(cls, path: Path) -> list[Self]:
        """Build a list YOLO annotation from an annotation file."""
        return [
            cls.from_annotation_line(line)
            for line in path.read_text().splitlines()
        ]

    def to_pixel_coordinates(
        self,
        image_height: int,
        image_width: int,
    ) -> Self:
        """Transform an annotation from percentage to pixel coordinates."""
        return self.__class__(
            self.class_id,
            self.x_center * image_width,
            self.y_center * image_height,
            self.width * image_width,
            self.height * image_height,
        )

    def to_percentage_coordinates(
        self,
        image_height: int,
        image_width: int,
    ) -> Self:
        """Transform an annotation from pixel to percentage coordinates."""
        return self.__class__(
            self.class_id,
            self.x_center / image_width,
            self.y_center / image_height,
            self.width / image_width,
            self.height / image_height,
        )

    def move(self, dx: int, dy: int) -> Self:
        """Move the bounding box by a given offset."""
        return self.__class__(
            self.class_id,
            self.x_center + dx,
            self.y_center + dy,
            self.width,
            self.height,
        )

    def to_x1y1x2y2(self) -> tuple[float, float, float, float]:
        """Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
        return (
            int(self.x_center - self.width / 2),
            int(self.y_center - self.height / 2),
            int(self.x_center + self.width / 2),
            int(self.y_center + self.height / 2),
        )

    @classmethod
    def from_x1y1x2y2(
        cls,
        class_id: int,
        box: tuple[float, float, float, float],
    ) -> Self:
        """Convert (x1, y1, x2, y2) to (x_center, y_center, width, height)."""
        x1, y1, x2, y2 = box
        return cls(
            class_id,
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
            int(x2 - x1),
            int(y2 - y1),
        )

    def shape(self) -> tuple[float, float]:
        """Return the shape of the bounding box."""
        return (self.height, self.width)

    def area(self) -> float:
        """Return the area of the bounding box."""
        return self.height * self.width

    def overlap(
        self,
        x1: int | float,
        y1: int | float,
        x2: int | float,
        y2: int | float,
    ) -> Self | None:
        """Compute the the bounding box overlap with a window."""
        box_x1, box_y1, box_x2, box_y2 = self.to_x1y1x2y2()

        overlap_x1 = max(x1, box_x1)
        overlap_y1 = max(y1, box_y1)
        overlap_x2 = min(x2, box_x2)
        overlap_y2 = min(y2, box_y2)

        if overlap_x1 > overlap_x2 or overlap_y1 > overlap_y2:
            return None

        return self.__class__(
            self.class_id,
            (overlap_x2 + overlap_x1) / 2,
            (overlap_y2 + overlap_y1) / 2,
            overlap_x2 - overlap_x1,
            overlap_y2 - overlap_y1,
        )


@dataclass
class SlidingWindow:
    """A container for each window and its annotations."""

    window: npt.NDArray[np.uint8]
    """The window of the original image."""

    annotations: list[Annotation]
    """The annotations for the window."""

    row: int
    """The row index of the window in the original image."""

    column: int
    """The column index of the window in the original image."""

    def save(self, path: Path, annotations_path: Path | None = None) -> None:
        """Save the window and annotations to disk."""
        # Save annotations on the same directory
        if annotations_path is None:
            annotations_path = path.with_suffix(".txt")
        cv2.imwrite(str(path), self.window)
        annotations_path.write_text(
            "\n".join(
                annotation.to_annotation_line()
                for annotation in self.annotations
            ),
        )

    def is_background(self) -> bool:
        """Return whether this window is a background."""
        return len(self.annotations) == 0


@dataclass
class SlidingWindows:
    """A collection of sliding windows over an annotated image."""

    sliding_windows: list[SlidingWindow]
    """Inner container for each of the sliding windows."""

    name: str
    """Name of the source image."""

    window_shape: tuple[int, int] = field(init=False)
    """The shape of windows over the source image."""

    rows: int = field(init=False)
    """The number of rows in the source image."""

    columns: int = field(init=False)
    """The number of columns in the source image."""

    def __post_init__(self) -> None:
        self.window_shape = cast(
            "tuple[int, int]",
            self.sliding_windows[0].window.shape[:2],
        )
        self.rows = (
            max(sliding_window.row for sliding_window in self.sliding_windows)
            + 1
        )
        self.columns = (
            max(
                sliding_window.column
                for sliding_window in self.sliding_windows
            )
            + 1
        )

    @classmethod
    def from_image(
        cls,
        image_path: Path,
        window_shape: tuple[int, int],
        annotations_path: Path | None = None,
        step_size: tuple[int, int] | int | None = None,
        minimum_pixels: int = 0,
        minimum_overlap: float = 0.0,
    ) -> Self:
        """Build the sliding windows over an image given by its path."""
        if annotations_path is None:
            annotations_path = image_path.with_suffix(".txt")
        if step_size is None:
            step_size = window_shape
        step_x, step_y = (
            step_size
            if isinstance(step_size, tuple)
            else (step_size, step_size)
        )

        # Load the source image
        image = cast("npt.NDArray[np.uint8]", cv2.imread(str(image_path)))
        height, width, _ = image.shape
        window_height, window_width = window_shape

        # Pad the image
        padded_image, (pad_top, _, _, pad_left) = pad_image(
            image,
            step_size,
            window_shape,
        )
        padded_height, padded_width, _ = padded_image.shape

        # Coordinate transform for the padded image
        # NOTE: The annotations are converted from percentage coordinates
        # to pixel coordinates on the original image, then converted back
        # to percentage coordinates, but from the padded image.
        annotations = [
            annotation.to_pixel_coordinates(height, width)
            .move(pad_left, pad_top)
            .to_percentage_coordinates(padded_height, padded_width)
            for annotation in Annotation.from_annotation_file(
                path=annotations_path,
            )
        ]

        # Apply sliding window over image and transform annotations
        sliding_windows: list[SlidingWindow] = []
        for (j, (y1, y2)), (i, (x1, x2)) in tqdm(
            list(
                itertools.product(
                    enumerate(
                        (
                            (y, y + window_height)
                            for y in range(
                                0,
                                padded_height - window_height + 1,
                                step_y,
                            )
                        ),
                    ),
                    enumerate(
                        (
                            (x, x + window_width)
                            for x in range(
                                0,
                                padded_width - window_width + 1,
                                step_x,
                            )
                        ),
                    ),
                ),
            ),
            desc="Generating sliding windows",
            leave=False,
        ):
            # Convert image annotations to window
            window_annotations: list[Annotation] = []
            for annotation in annotations:
                # Convert percentage coordinates to absolute
                original_annotation = annotation.to_pixel_coordinates(
                    padded_height,
                    padded_width,
                )
                new_annotation = original_annotation.overlap(x1, y1, x2, y2)

                if (
                    new_annotation is None
                    or min(new_annotation.shape()) < minimum_pixels
                    or new_annotation.area() / original_annotation.area()
                    < minimum_overlap
                ):
                    continue

                new_annotation = new_annotation.move(
                    -x1,
                    -y1,
                ).to_percentage_coordinates(
                    window_height,
                    window_width,
                )

                window_annotations.append(new_annotation)

            sliding_windows.append(
                SlidingWindow(
                    padded_image[y1:y2, x1:x2],
                    window_annotations,
                    row=j,
                    column=i,
                ),
            )

        return cls(sliding_windows, name=image_path.stem)

    def is_background_image(self) -> bool:
        """Return whether the whole image is a background image."""
        return all(
            sliding_window.is_background()
            for sliding_window in self.sliding_windows
        )

    def save(
        self,
        path: Path,
        annotations_path: Path | None = None,
        save_backgrounds: bool = False,
    ) -> None:
        """Save the sliding windows into a path with their annotations."""
        for sliding_window in self.sliding_windows:
            if not save_backgrounds and sliding_window.is_background():
                continue

            name = (
                f"{self.name}_window_"
                f"{sliding_window.row}_{sliding_window.column}.png"
            )

            sliding_window.save(
                path / name,
                (annotations_path / name).with_suffix(".txt")
                if annotations_path is not None
                else None,
            )

    def grid(
        self,
        padding: tuple[int, int] | None = None,
        bg_color: tuple[int, int, int] = (0, 0, 0),
        classes: list[str] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Display the sliding window generation results on a grid image."""
        if padding is None:
            padding = cast(
                "tuple[int, int]",
                tuple(max(dim // 10, 1) for dim in self.window_shape),
            )

        # Image shape
        img_height, img_width = self.window_shape

        # Padding
        pad_y, pad_x = padding

        # Compute the canvas size accounting for padding
        grid_height = self.rows * img_height + (self.rows - 1) * pad_y
        grid_width = self.columns * img_width + (self.columns - 1) * pad_x

        # Create a blank canvas with the specified background color
        grid_image = np.full(
            (grid_height, grid_width, 3),
            bg_color,
            dtype=np.uint8,
        )

        # Populate the canvas with images
        for idx, img in enumerate(self.sliding_windows):
            row = idx // self.columns
            col = idx % self.columns

            # Compute top-left corner coordinates for each image
            y_start = row * (img_height + pad_y)
            x_start = col * (img_width + pad_x)

            # Draw image bounding boxes
            window = draw_annotations(
                img.window.copy(),
                img.annotations,
                classes,
            )

            grid_image[
                y_start : y_start + img_height,
                x_start : x_start + img_width,
            ] = window

        return grid_image


def pad_image(
    image: npt.NDArray[np.uint8],
    step_size: tuple[int, int] | int,
    window_size: tuple[int, int],
) -> tuple[npt.NDArray[np.uint8], tuple[int, int, int, int]]:
    """Add extra pixels to the original imagem with even padding."""
    step_x, step_y = (
        step_size if isinstance(step_size, tuple) else (step_size, step_size)
    )

    height, width, _ = image.shape
    window_height, window_width = window_size

    # Determine padding/extra pixels needed for width and height
    extra_height, extra_width = (
        (step_y - (height % window_height) % step_y) % step_y,
        (step_x - (width % window_width) % step_x) % step_x,
    )

    # Calculate even padding, adds one pixel extra
    # to the bottom or right if needed (odd division rest)
    pad_top, extra_bottom = divmod(extra_height, 2)
    pad_left, extra_right = divmod(extra_width, 2)
    pad_bottom = pad_top + extra_bottom
    pad_right = pad_left + extra_right

    padded_image = cast(
        "npt.NDArray[np.uint8]",
        cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        ),
    )

    return padded_image, (pad_top, pad_bottom, pad_right, pad_left)


def draw_annotations(
    image: npt.NDArray[np.uint8],
    annotations: list[Annotation],
    classes: list[str] | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw annotations on the image as bounding boxes."""
    for annotation in annotations:
        x1, y1, x2, y2 = annotation.to_pixel_coordinates(
            *image.shape[:2],
        ).to_x1y1x2y2()
        image = cast(
            "npt.NDArray[np.uint8]",
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2),
        )
        image = draw_text(
            image,
            (x1, y1, x2, y2),
            str(annotation.class_id)
            if classes is None
            else classes[annotation.class_id],
            color=(0, 0, 255),
        )

    return image


def draw_text(
    image: npt.NDArray[np.uint8],
    xyxy: tuple[float, float, float, float],
    text: str,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.5,
    thickness: int = 2,
    vertical_delta: int = 7,
    color: tuple[int, int, int] = (0, 0, 255),
    text_on_top: bool = True,
) -> npt.NDArray[np.uint8]:
    """Add the text to the image bounded by its dimensions."""
    x1, y1, _, y2 = xyxy

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    text_xy = (
        inside_range(
            x1,
            0,
            image.shape[1] - text_width,
        ),
        inside_range(
            y1 - vertical_delta
            if text_on_top
            else y2 + text_height + vertical_delta,
            text_height,
            image.shape[0] - text_height,
        ),
    )

    # Text is inserted twice to draw a border around text
    return cast(
        "npt.NDArray[np.uint8]",
        cv2.putText(
            cv2.putText(
                image,
                text,
                text_xy,
                font,
                scale,
                (0, 0, 0),
                thickness + 1,
            ),
            text,
            text_xy,
            font,
            scale,
            color,
            thickness,
        ),
    )


def inside_range(x: float | int, min_: int, max_: int) -> int:
    """Clip a number x inside a min max range."""
    return min(max(int(x), min_), max_)


IMAGE_EXTENSIONS: Final[list[str]] = [".jpg", ".jpeg", ".png"]
"""List of image file extensions."""


@dataclass
class YOLODatasetSlicer:
    """Apply a sliding window of chosen shape to a YOLO dataset."""

    path: Path
    """The path to the YOLO dataset."""

    window_shape: tuple[int, int]
    """The shape of the window to slide over the images."""

    step_size: InitVar[tuple[int, int] | None] = None
    """The step size between windows."""

    _step_size: tuple[int, int] = field(init=False)
    """The step size between windows."""

    annotations_path: InitVar[Path | None] = field(default=None)
    """The path to the annotations of the dataset, defaults to path."""

    _annotations_path: Path = field(init=False)
    """The path to the annotations of the dataset, defaults to path."""

    classes: list[str] | None = None
    """The class mapping of the dataset."""

    minimum_pixels: int = 0
    """The minimum amount of pixels of a bounding box inside an image."""

    minimum_overlap: float = 0.0
    """The minimum overlap (% of area) of bounding box inside an image."""

    background_percentage: float = 0.0
    """The percentage of background images, 0 for none."""

    def __post_init__(
        self,
        step_size: tuple[int, int] | None,
        annotations_path: Path | None,
    ) -> None:
        self._step_size = (
            step_size if step_size is not None else self.window_shape
        )
        if annotations_path is None:
            # Assume standard yolo dataset format
            if (
                self.classes is None
                and (classes_path := self.path / "classes.txt").exists()
            ):
                self.classes = classes_path.read_text().splitlines()

            self._annotations_path = self.path / "labels"
            self.path /= "images"

        if not (self.path.exists() and self._annotations_path.exists()):
            raise ValueError(
                "Both the images and annotations path must exist, "
                f"got '{self.path}' and '{self._annotations_path}'",
            )

    def apply(self, output_path: Path) -> None:
        """Apply the sliding window to the dataset and save the results."""
        # Set images and labels path
        out_images_path = output_path / "images"
        out_labels_path = output_path / "labels"

        out_images_path.mkdir(parents=True, exist_ok=True)
        out_labels_path.mkdir(parents=True, exist_ok=True)

        images = list(
            itertools.chain(
                *[
                    self.path.glob(f"*{extension}")
                    for extension in IMAGE_EXTENSIONS
                ],
            ),
        )
        backgrounds = round(len(images) * self.background_percentage)
        for image_path in tqdm(images, desc="Applying sliding window"):
            windows = SlidingWindows.from_image(
                image_path,
                self.window_shape,
                (self._annotations_path / image_path.name).with_suffix(".txt"),
                self._step_size,
                self.minimum_pixels,
                self.minimum_overlap,
            )

            save_backgrounds = False
            if backgrounds > 0 and windows.is_background_image():
                backgrounds -= 1
                save_backgrounds = True

            windows.save(
                out_images_path,
                annotations_path=out_labels_path,
                save_backgrounds=save_backgrounds,
            )

        if self.classes is not None:
            (output_path / "classes.txt").write_text("\n".join(self.classes))

    def grid(self, output_path: Path) -> None:
        """Apply the sliding window to the dataset and debugging grids."""
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        images = list(
            itertools.chain(
                *[
                    self.path.glob(f"*{extension}")
                    for extension in IMAGE_EXTENSIONS
                ],
            ),
        )
        for image_path in tqdm(images, desc="Building grids"):
            windows = SlidingWindows.from_image(
                image_path,
                self.window_shape,
                self._annotations_path / image_path.with_suffix(".txt").name,
                self._step_size,
            )
            grid = windows.grid(classes=self.classes)
            cv2.imwrite(str(output_path / image_path.name), grid)


def slice_yolo_dataset(
    input_path: Path,
    output_path: Path,
    window_shape: tuple[int, int],
    step_size: tuple[int, int] | None = None,
    annotations_path: Path | None = None,
    grid: bool = False,
    minimum_pixels: int = 0,
    minimum_overlap: float = 0.0,
    background_percentage: float = 0.0,
) -> None:
    """Apply sliding windows over a YOLO dataset."""
    slicer = YOLODatasetSlicer(
        input_path,
        window_shape,
        step_size=step_size,
        annotations_path=annotations_path,
        minimum_pixels=minimum_pixels,
        minimum_overlap=minimum_overlap,
        background_percentage=background_percentage,
    )
    if not grid:
        slicer.apply(output_path)
    else:
        slicer.grid(output_path)
