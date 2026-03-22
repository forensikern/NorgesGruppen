"""
build_reference_index.py
Extracts EfficientNet-B3 embeddings from every product reference image and
saves an index that run.py uses to re-rank YOLO's category predictions.

Actual folder structure expected:
    NM_NGD_product_images/
        metadata.json
        70177084738/          ← barcode as folder name
            image_001.jpg
            image_002.jpg     ← varying counts, no fixed angle names
        8445291513365/
            ...

Run ONCE on your training machine after step 3 (training).

Usage:
    python build_reference_index.py \
        --product_dir  ./NM_NGD_product_images \
        --annotations  ./NM_NGD_coco_dataset/train/annotations.json \
        --output_dir   ./embedding_index

Outputs (include ALL in submission zip):
    embedding_index/
        embedder.pt            — EfficientNet-B3 state_dict (~47 MB, fp16)
        reference_embeds.npy   — (N_products, 1536) float16
        reference_meta.json    — [{product_code, category_id, name, n_images}]

Requirements:
    pip install torch timm Pillow tqdm
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Confirmed angle filenames from actual data — ordered by representativeness.
# main/front are the best single-product views; back/left/right/top add coverage.
ANGLE_PRIORITY = ["main", "front", "back", "left", "right", "top", "bottom"]

# EfficientNet-B3 expects 300×300, ImageNet normalisation
IMGSZ = 300
MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Helpers ───────────────────────────────────────────────────────────────

def load_image_tensor(path: Path) -> torch.Tensor:
    """Load image as normalised (3, 300, 300) float32 tensor."""
    img = Image.open(path).convert("RGB").resize((IMGSZ, IMGSZ), Image.BILINEAR)
    t   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


def build_code_to_catid(annotations_path: Path):
    """Returns ({code: cat_id}, {code: name}) from COCO annotations."""
    with open(annotations_path, encoding="utf-8") as f:
        coco = json.load(f)
    catid_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    code_to_catid = {}
    code_to_name  = {}
    for ann in coco["annotations"]:
        code = ann.get("product_code")
        if code and code not in code_to_catid:
            code_to_catid[code] = ann["category_id"]
            code_to_name[code]  = catid_to_name.get(ann["category_id"], "unknown")
    print(f"Unique product codes in annotations: {len(code_to_catid)}")
    return code_to_catid, code_to_name


def load_metadata(product_dir: Path) -> dict:
    """Load metadata.json → {product_code: entry}. Returns {} if missing."""
    meta_path = product_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(e.get("product_code", e.get("barcode", ""))): e for e in data}
    return data if isinstance(data, dict) else {}


def get_product_images(folder: Path) -> list:
    """
    Return angle images in priority order: main, front, back, left, right,
    top, bottom. Falls back to any remaining files not matching a known angle
    name, so no image is silently dropped.
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_dir",  required=True)
    parser.add_argument("--annotations",  required=True)
    parser.add_argument("--output_dir",   default="./embedding_index")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_per_product", type=int, default=7,
                        help="Max images to embed per product (-1 = all). "
                             "Averaging more angles gives a richer embedding.")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    product_dir = Path(args.product_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device      = torch.device(args.device)
    print(f"Device: {device}")

    # ── Lookups ───────────────────────────────────────────────────────────
    code_to_catid, code_to_name = build_code_to_catid(Path(args.annotations))
    metadata = load_metadata(product_dir)

    # ── Load EfficientNet-B3 ──────────────────────────────────────────────
    print("Loading EfficientNet-B3 (pretrained) …")
    import timm
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    model.eval().to(device)
    embed_dim = model.num_features  # 1536
    print(f"  Embedding dimension: {embed_dim}")

    # ── Discover product folders ──────────────────────────────────────────
    product_folders = sorted(
        d for d in product_dir.iterdir()
        if d.is_dir() and d.name != "__MACOSX"
    )
    print(f"Product folders: {len(product_folders)}")

    # ── Extract embeddings ────────────────────────────────────────────────
    all_embeds = []
    meta_rows  = []
    skipped    = []

    # Accumulate image tensors across products for efficient batching
    pending_tensors = []   # flat list of tensors
    pending_info    = []   # (product_code, n_images) for grouping back

    def flush():
        """Run pending tensors through the model and group by product."""
        if not pending_tensors:
            return
        with torch.no_grad():
            t     = torch.stack(pending_tensors).to(device)   # (N, 3, 300, 300)
            feats = model(t)                                   # (N, embed_dim)
            feats = F.normalize(feats, dim=1)

        idx = 0
        for code, n in pending_info:
            prod_feats = feats[idx : idx + n]                 # (n, embed_dim)
            avg_feat   = F.normalize(prod_feats.mean(dim=0), dim=0)
            all_embeds.append(avg_feat.cpu().half().numpy())
            idx += n

        pending_tensors.clear()
        pending_info.clear()

    for folder in tqdm(product_folders, desc="Extracting embeddings"):
        product_code = folder.name

        images = get_product_images(folder)
        if not images:
            tqdm.write(f"  SKIP {product_code} — no images")
            skipped.append(product_code)
            continue

        # Limit images per product if requested
        limit      = args.max_per_product if args.max_per_product > 0 else len(images)
        images_use = images[:limit]

        tensors = []
        for img_path in images_use:
            try:
                tensors.append(load_image_tensor(img_path))
            except Exception as e:
                tqdm.write(f"  WARNING: {img_path}: {e}")

        if not tensors:
            tqdm.write(f"  SKIP {product_code} — all images failed to load")
            skipped.append(product_code)
            continue

        # Resolve category_id
        cat_id = code_to_catid.get(product_code)
        name   = code_to_name.get(product_code, "unknown")
        if cat_id is None:
            # Product exists in reference set but wasn't annotated in training —
            # assign unknown_product (356). Still useful: embedding can map to
            # annotated products at inference via cosine similarity.
            cat_id = 356
            name   = "unknown_product"

        meta_rows.append({
            "product_code": product_code,
            "category_id":  cat_id,
            "name":         name,
            "n_images":     len(tensors),
        })

        pending_tensors.extend(tensors)
        pending_info.append((product_code, len(tensors)))

        if len(pending_tensors) >= args.batch_size:
            flush()

    flush()  # remaining

    print(f"\nEmbedded {len(all_embeds)} products "
          f"({len(skipped)} skipped, "
          f"{sum(1 for r in meta_rows if r['category_id'] != 356)} with known category_id)")

    # ── Save outputs ──────────────────────────────────────────────────────

    # 1. Embedding matrix
    embeds_arr   = np.stack(all_embeds, axis=0)   # (N, 1536) float16
    embeds_path  = out_dir / "reference_embeds.npy"
    np.save(str(embeds_path), embeds_arr)
    print(f"Saved embeddings: {embeds_path}  "
          f"shape={embeds_arr.shape}  size={embeds_arr.nbytes / 1e6:.1f} MB")

    # 2. Metadata
    meta_path = out_dir / "reference_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata:   {meta_path}")

    # 3. Model weights (fp16 state_dict — ~47 MB)
    embedder_path = out_dir / "embedder.pt"
    torch.save(model.half().cpu().state_dict(), str(embedder_path))
    size_mb = embedder_path.stat().st_size / 1e6
    print(f"Saved embedder:   {embedder_path}  ({size_mb:.1f} MB)")

    # ── Size budget check ─────────────────────────────────────────────────
    total_index_mb = (
        embedder_path.stat().st_size +
        embeds_path.stat().st_size +
        meta_path.stat().st_size
    ) / 1e6
    print(f"\nIndex total size: {total_index_mb:.1f} MB")
    print("(YOLOv8l best.pt ≈ 87 MB → combined ≈ "
          f"{87 + total_index_mb:.0f} MB, well under 420 MB limit)")

    print("\n✓  Done.  Files to include in submission zip:")
    print(f"   run.py   best.pt   {out_dir}/")
    print(f"\n   zip -r submission.zip run.py best.pt {out_dir}/")


if __name__ == "__main__":
    main()
