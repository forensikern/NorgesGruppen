"""
prepare_dataset.py
Converts the competition COCO dataset to YOLO format for YOLOv8 training.

Usage:
    python prepare_dataset.py \
        --coco_dir /path/to/NM_NGD_coco_dataset \
        --output_dir /path/to/yolo_dataset \
        --val_split 0.15

Structure of NM_NGD_coco_dataset (expected):
    images/
        img_00001.jpg
        ...
    annotations.json

Output structure:
    yolo_dataset/
        images/
            train/  val/
        labels/
            train/  val/
        dataset.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] (normalized 0-1)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1] to handle any annotation overflow
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", required=True, help="Path to NM_NGD_coco_dataset folder")
    parser.add_argument("--output_dir", required=True, help="Where to write the YOLO dataset")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of images for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.output_dir)
    # Actual structure: NM_NGD_coco_dataset/train/annotations.json
    #                   NM_NGD_coco_dataset/train/images/
    ann_path = coco_dir / "train" / "annotations.json"
    img_dir  = coco_dir / "train" / "images"

    assert ann_path.exists(), f"annotations.json not found at {ann_path}"
    assert img_dir.exists(), f"images/ folder not found at {img_dir}"

    print(f"Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookups
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    num_classes = len(categories)  # 357 (0-356 including unknown_product)
    print(f"  {len(images)} images, {len(coco['annotations'])} annotations, {num_classes} categories")

    # Group annotations by image_id
    ann_by_image = {}
    skipped = 0
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in images:
            skipped += 1
            continue
        ann_by_image.setdefault(iid, []).append(ann)
    if skipped:
        print(f"  Skipped {skipped} annotations with unknown image_id")

    # Train / val split (stratify by store section if possible via filename prefix)
    image_ids = list(images.keys())
    random.seed(args.seed)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * args.val_split))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])
    print(f"  Train: {len(train_ids)} images | Val: {len(val_ids)} images")

    # Create output directories
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(ids, split_name):
        copied = 0
        empty = 0
        for iid in ids:
            img_info = images[iid]
            src = img_dir / img_info["file_name"]
            if not src.exists():
                print(f"  WARNING: image not found: {src}")
                continue

            # Copy image
            dst_img = out_dir / "images" / split_name / img_info["file_name"]
            shutil.copy2(src, dst_img)

            # Write label file
            anns = ann_by_image.get(iid, [])
            label_name = Path(img_info["file_name"]).stem + ".txt"
            dst_lbl = out_dir / "labels" / split_name / label_name

            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                bbox = ann["bbox"]
                w = img_info["width"]
                h = img_info["height"]
                if w <= 0 or h <= 0:
                    continue
                cx, cy, nw, nh = coco_bbox_to_yolo(bbox, w, h)
                lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

            if not lines:
                empty += 1
            copied += 1

        print(f"  [{split_name}] {copied} images written ({empty} with no annotations)")

    process_split(train_ids, "train")
    process_split(val_ids, "val")

    # Write dataset.yaml  (use json-safe format — no yaml import needed by sandbox)
    # But we write a proper YAML manually since PyYAML is fine on the training machine
    category_names = [categories[i] for i in sorted(categories.keys())]
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {num_classes}",
        "names:",
    ]
    for name in category_names:
        # Escape any colons or quotes in product names
        safe_name = name.replace("'", "\\'")
        yaml_lines.append(f"  - '{safe_name}'")

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"\nDataset ready at: {out_dir}")
    print(f"dataset.yaml:     {yaml_path}")
    print(f"num_classes:      {num_classes}")
    print("\nNext step: python train.py --data", yaml_path)


if __name__ == "__main__":
    main()
