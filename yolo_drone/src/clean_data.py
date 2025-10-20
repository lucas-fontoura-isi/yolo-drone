import os
import re
import yaml
import json
import random
from tqdm import tqdm
from sahi.utils.coco import Coco
from pathlib import Path

def clean_filename(fname: str) -> str:
    return re.sub(r"_jpg\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", r".\1", fname, flags=re.IGNORECASE)

def clean_infected_leaves(
    images_folder: Path,
    input_json: Path,
    output_json: Path,
    scale_x: float,
    scale_y: float,
    reassign_image_ids: bool,
    reassign_annotation_ids: bool,
    keep_original_ids: bool,
    keep_original_fname: bool
) -> None:
    # Path and I/O checks
    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON file not found: {input_json}")
    if not images_folder.is_dir():
        raise NotADirectoryError(f"Images folder not found: {images_folder}")
    if output_json.is_dir():
        raise IsADirectoryError(f"Output JSON path is a directory: {output_json}")

    # Load COCO JSON
    with open(input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # List existing files (cleaned + lowercased)
    p = images_folder
    existing_files = {clean_filename(p_.name).lower().strip() for p_ in p.iterdir() if p_.is_file()}

    # Filter images and create mapping old_id -> new_id
    oldid_to_new_id = {}
    kept_images = []
    dropped_images = []

    next_img_id = 1
    for img in tqdm(coco.get("images", []), desc="Processing images..."):
        original_fname = os.path.basename(img.get("file_name", "")).strip()
        clean_fname = clean_filename(original_fname)

        if clean_fname.lower() in existing_files:
            img_copy = img.copy()
            if keep_original_fname:
                img_copy["original_file_name"] = img_copy["file_name"]
            img_copy["file_name"] = clean_fname

            old_id = img["id"]
            if reassign_image_ids:
                if keep_original_ids:
                    img_copy["original_id"] = old_id
                img_copy["id"] = next_img_id
                oldid_to_new_id[old_id] = next_img_id
                next_img_id += 1
            else:
                oldid_to_new_id[old_id] = old_id

            kept_images.append(img_copy)
        else:
            dropped_images.append(original_fname)

    # Filter & remap annotations
    kept_annotations = []
    dropped_annotations = []
    dropped_small_annotations = []
    next_ann_id = 1
    for ann in tqdm(coco.get("annotations", []), desc="Processing annotations..."):
        old_img_id = ann["image_id"]
        if old_img_id in oldid_to_new_id:
            if ann.get("area", 0) < 1:
                dropped_small_annotations.append(ann.get("id"))
                continue

            ann_copy = ann.copy()
            ann_copy["image_id"] = oldid_to_new_id[old_img_id]

            # Rescale bbox
            if "bbox" in ann_copy:
                x, y, w, h = ann_copy["bbox"]
                x *= scale_x
                y *= scale_y
                w *= scale_x
                h *= scale_y
                ann_copy["bbox"] = [x, y, w, h]

                # Update area as well
                ann_copy["area"] = w * h

            # optionally reassign annotation id
            if reassign_annotation_ids:
                if keep_original_ids:
                    ann_copy["original_id"] = ann["id"]
                ann_copy["id"] = next_ann_id
                next_ann_id += 1

            kept_annotations.append(ann_copy)
        else:
            dropped_annotations.append(ann.get("id"))

    # Build final COCO dict
    filtered_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": kept_images,
        "annotations": kept_annotations,
    }

    # Save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_coco, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"Images kept: {len(kept_images)}")
    print(f"Images dropped (not found): {len(dropped_images)}")
    if dropped_images:
        print(f"Dropped image files (examples): {dropped_images[:10]}")
    print(f"Annotations kept: {len(kept_annotations)}")
    print(f"Annotations dropped (missing images): {len(dropped_annotations)}")
    if dropped_annotations:
        print(f"Dropped annotation ids (examples): {dropped_annotations[:10]}")
    print(f"Annotations dropped (area < 1 px): {len(dropped_small_annotations)}")
    if dropped_small_annotations:
        print(f"Dropped small annotation ids (examples): {dropped_small_annotations[:10]}")
    print(f"✅ Rescaled bboxes saved to {output_json}")

def convert_yolo_to_coco(coco_json: Path, images_folder: Path, output_folder: Path, train_ratio: float, use_sahi: bool) -> None:
    # Path and I/O checks
    output_folder = output_folder
    images_folder = images_folder

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON file not found: {coco_json}")
    if not images_folder.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_folder}")

    
    if not use_sahi:
        # Make directories
        (output_folder / "images/train").mkdir(parents=True, exist_ok=True)
        (output_folder / "images/val").mkdir(parents=True, exist_ok=True)
        (output_folder / "labels/train").mkdir(parents=True, exist_ok=True)
        (output_folder / "labels/val").mkdir(parents=True, exist_ok=True)

        # Load COCO annotations
        with open(coco_json, "r") as f:
            coco = json.load(f)

        # Build lookup tables
        image_id_to_filename = {img["id"]: clean_filename(img["file_name"]) for img in coco["images"]}
        image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

        # Clean and deduplicate category names
        cleaned_names = []
        cat_id_to_newid = {}
        for cat in coco["categories"]:
            name = cat["name"].strip()
            if name not in cleaned_names:
                cleaned_names.append(name)
            cat_id_to_newid[cat["id"]] = cleaned_names.index(name)

        categories = cat_id_to_newid
        class_names = cleaned_names

        # Collect annotations per image
        annotations_per_image = {img_id: [] for img_id in image_id_to_filename.keys()}

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # COCO: [x_min, y_min, width, height]

            # Get image size
            img_w, img_h = image_id_to_size[img_id]

            # Convert to YOLO format
            x_min, y_min, w, h = bbox
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            w /= img_w
            h /= img_h

            class_id = categories[cat_id]
            annotations_per_image[img_id].append([class_id, x_center, y_center, w, h])

        # Shuffle and split dataset
        image_ids = list(image_id_to_filename.keys())
        random.shuffle(image_ids)
        split_idx = int(len(image_ids) * train_ratio)
        train_ids, val_ids = image_ids[:split_idx], image_ids[split_idx:]

        # Helper to copy and write annotations
        def process_split(ids, split):
            for img_id in ids:
                filename = image_id_to_filename[img_id]
                src_img = images_folder / filename
                dst_img = output_folder / f"images/{split}/{filename}"

                if not src_img.exists():
                    print(f"⚠️ Warning: Image not found {src_img}, skipping.")
                    continue

                # Symlink image
                os.symlink(src_img.absolute(), dst_img)

                # Write label
                label_file = output_folder / f"labels/{split}/{Path(filename).stem}.txt"
                with open(label_file, "w") as f:
                    for ann in annotations_per_image[img_id]:
                        f.write(" ".join([f"{a:.6f}" if isinstance(a, float) else str(a) for a in ann]) + "\n")

        # Process train and val splits
        process_split(train_ids, "train")
        process_split(val_ids, "val")

        # Write data.yaml
        yaml_content = {
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": class_names
        }

        with open(output_folder / "data.yaml", "w") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        print(f"✅ Conversion complete! YOLO dataset ready at: {output_folder}")
    else:
        # init Coco object
        coco = Coco.from_coco_dict_or_path(str(coco_json), image_dir=images_folder)

        # export converted YOLO formatted dataset into given output_folder with a 85% train/15% val split
        coco.export_as_yolo(
            output_dir=output_folder,
            train_split_rate=train_ratio
        )

def create_yolo_splits(images_dir: Path, output_dir: Path, img_extension: str, train_ratio: float) -> None:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    
    images = list(images_dir.glob(f"*{img_extension}"))
    random.shuffle(images)

    train_idx = int(len(images) * train_ratio)

    train_images = images[:train_idx]
    val_images = images[train_idx:]

    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(str(p) for p in train_images))
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(str(p) for p in val_images))