import cv2
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load configuration file
CONFIG_PATH = Path(__file__).parent / "segmentation_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def convert_hsv_gimp_to_cv2(H_gimp: int, S_gimp: int, V_gimp: int) -> tuple[int, int, int]:
    H_cv2 = int(H_gimp / 2)
    S_cv2 = int(S_gimp * 255 / 100)
    V_cv2 = int(V_gimp * 255 / 100)
    return (H_cv2, S_cv2, V_cv2)


def segment_synthetic(images_dir: Path, labels_dir: Path, convert_gimphsv_to_cv2hsv: bool, min_area: int, epsilon_ratio: float, draw_segmentation: bool) -> None:
    # Path and I/O checks
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")

    # Load color ranges from config
    color_ranges = [
        (tuple(item["lower"]), tuple(item["upper"]), tuple(item["class_name"]))
        for item in config["color_ranges"]
    ]

    # Map class names to YOLO IDs
    class_to_id = {c[2]: i for i, c in enumerate(color_ranges)}

    # -------- PROCESS IMAGES ----------
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(sorted(images_dir.glob("*.*")), desc="Processing images..."):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        label_lines = []

        for lower, upper, name in color_ranges:
            lw, up = lower, upper
            if convert_gimphsv_to_cv2hsv:
                lw = convert_hsv_gimp_to_cv2(*lower)
                up = convert_hsv_gimp_to_cv2(*upper)

            mask = cv2.inRange(hsv, lw, up)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                # Simplify polygon
                epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) < 3:
                    continue

                # Normalize segmentation points
                norm_points = []
                for [x, y] in approx.reshape(-1, 2):
                    px = x / w
                    py = y / h
                    norm_points.extend([px, py])

                class_id = class_to_id[name]
                line = f"{class_id} " + " ".join(f"{p:.6f}" for p in norm_points)
                label_lines.append(line)

        # Save YOLO-Seg labels
        if label_lines:
            label_path = labels_dir / (img_file.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

    print(f"✅ Done! YOLO-Seg annotations saved in {labels_dir}")

    if draw_segmentation:
        output_dir = labels_dir / "segmented_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_file in sorted(images_dir.glob("*.*"))[:config["num_samples_for_visualization"]]:
            label_file = labels_dir / (img_file.stem + ".txt")
            visualize_yolo_seg(img_file, label_file, output_dir)


def visualize_yolo_seg(img_path: Path, label_path: Path, output_dir: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠ Could not read {img_path}")
        return

    h, w, _ = img.shape

    if not label_path.exists():
        print(f"⚠ No label for {img_path.name}")
        return

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Convert normalized coords back to pixel values
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])

        points = cv2.convexHull(np.array(points))  # ensure closed polygon
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Label with class name
        cx, cy = points[0][0]
        cv2.putText(img, config["color_ranges"][cls_id]["class_name"], (cx, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"✅ Saved visualization to {out_path}")