"""
augment_training.py
Adds product reference images into the YOLO training split so the model
sees clean, isolated product photos alongside shelf images.

Actual folder structure expected:
    NM_NGD_product_images/
        metadata.json
        70177084738/          ← barcode folders (345 of them)
            image_001.jpg
            image_002.jpg
            ...               ← varying number of images per product
        8445291513365/
            ...

Run AFTER prepare_dataset.py, BEFORE train.py.

Usage:
    python augment_training.py \
        --product_dir  ./NM_NGD_product_images \
        --annotations  ./NM_NGD_coco_dataset/train/annotations.json \
        --yolo_dir     ./yolo_dataset

Requirements: Pillow, tqdm
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Confirmed filenames — prioritise main/front (most representative views)
ANGLE_PRIORITY = ["main", "front", "back", "left", "right", "top", "bottom"]


def build_code_to_catid(annotations_path: Path) -> dict:
    """Returns {product_code: category_id} from COCO annotations."""
    with open(annotations_path, encoding="utf-8") as f:
        coco = json.load(f)
    code_to_catid = {}
    for ann in coco["annotations"]:
        code = ann.get("product_code")
        if code and code not in code_to_catid:
            code_to_catid[code] = ann["category_id"]
    return code_to_catid


def load_metadata(product_dir: Path) -> dict:
    """
    Optionally load metadata.json if present.
    Returns {product_code: {name, ...}} or empty dict.
    """
    meta_path = product_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    # metadata.json may be a list or a dict — normalise to {code: entry}
    if isinstance(data, list):
        return {str(entry.get("product_code", entry.get("barcode", ""))): entry
                for entry in data}
    elif isinstance(data, dict):
        return data
    return {}


def get_product_images(folder: Path) -> list:
    """
    Return angle images in priority order (main first, then front, back …).
    Falls back to any extra files not matching a standard angle name.
    """
    found = []
    seen  = set()
    for angle in ANGLE_PRIORITY:
        for ext in IMAGE_EXTS:
            p = folder / (angle + ext)
            if p.exists():
                found.append(p)
                seen.add(p.name)
                break
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name not in seen:
            found.append(p)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_dir", required=True,
                        help="Path to NM_NGD_product_images")
    parser.add_argument("--annotations", required=True,
                        help="Path to NM_NGD_coco_dataset/train/annotations.json")
    parser.add_argument("--yolo_dir",    required=True,
                        help="Root of YOLO dataset created by prepare_dataset.py")
    parser.add_argument("--max_per_product", type=int, default=5,
                        help="Max images to add per product (default 5). "
                             "Use -1 for all images.")
    args = parser.parse_args()

    product_dir = Path(args.product_dir)
    yolo_dir    = Path(args.yolo_dir)
    train_imgs  = yolo_dir / "images" / "train"
    train_lbls  = yolo_dir / "labels" / "train"

    assert train_imgs.exists(), f"Not found: {train_imgs}"
    assert train_lbls.exists(), f"Not found: {train_lbls}"

    # Build product_code → category_id from COCO annotations
    code_to_catid = build_code_to_catid(Path(args.annotations))
    print(f"Annotated product codes in training data: {len(code_to_catid)}")

    # Load metadata.json for logging (optional)
    metadata = load_metadata(product_dir)

    # Discover all product barcode folders
    product_folders = sorted(
        d for d in product_dir.iterdir()
        if d.is_dir() and d.name != "__MACOSX"
    )
    print(f"Product folders found in {product_dir}: {len(product_folders)}")

    added_images  = 0
    skipped_nocat = 0
    skipped_noimg = 0

    for folder in tqdm(product_folders, desc="Adding reference images"):
        product_code = folder.name

        # Only add products that appear in training annotations
        # (avoids injecting products with no known category_id)
        cat_id = code_to_catid.get(product_code)
        if cat_id is None:
            skipped_nocat += 1
            continue

        all_images = get_product_images(folder)
        if not all_images:
            tqdm.write(f"  SKIP {product_code} — no images found")
            skipped_noimg += 1
            continue

        # Take up to max_per_product images
        limit = args.max_per_product if args.max_per_product > 0 else len(all_images)
        images_to_use = all_images[:limit]

        for idx, src in enumerate(images_to_use):
            try:
                img = Image.open(src).convert("RGB")
            except Exception as e:
                tqdm.write(f"  WARNING: cannot open {src}: {e}")
                continue

            stem    = f"ref_{product_code}_{idx:03d}"
            dst_img = train_imgs / (stem + ".jpg")
            dst_lbl = train_lbls / (stem + ".txt")

            img.save(dst_img, "JPEG", quality=92)

            # YOLO label: product fills the reference image (full-frame bbox)
            # cx=0.5, cy=0.5, w=0.95, h=0.95 (slightly inset to avoid edge noise)
            with open(dst_lbl, "w") as f:
                f.write(f"{cat_id} 0.500000 0.500000 0.950000 0.950000\n")

            added_images += 1

    print(f"\n✓ Added {added_images} reference images to training set")
    print(f"  Skipped {skipped_nocat} products not in annotations")
    print(f"  Skipped {skipped_noimg} products with no images")

    total_train  = len(list(train_imgs.glob("*.jpg")))
    total_labels = len(list(train_lbls.glob("*.txt")))
    print(f"\n  Training images total: {total_train}")
    print(f"  Training labels total: {total_labels}")
    print("\nNext step: python train.py --data path/to/yolo_dataset/dataset.yaml")


if __name__ == "__main__":
    main()
