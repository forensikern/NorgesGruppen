"""
run.py — NM in AI 2026 submission entry point (WBF ensemble).

Stage 1 — Ensemble detection (70% of score):
    Runs best_l.onnx (YOLOv8l, imgsz=1280) and best_x.onnx (YOLOv8x, imgsz=1280)
    then fuses predictions with Weighted Box Fusion (WBF).

Stage 2 — Embedding re-ranker (30% of score):
    EfficientNet-B3 cosine similarity vs reference index.

Executed as: python run.py --input /data/images --output /output/predictions.json

Files required in zip:
    best_l.onnx                           — YOLOv8l FP16 (84 MB)
    best_x.onnx                           — YOLOv8x FP16 (131 MB)
    embedding_index/embedder.pt
    embedding_index/reference_embeds.npy
    embedding_index/reference_meta.json
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────────────────────
YOLO_IMGSZ           = 1280
YOLO_CONF            = 0.05
YOLO_IOU             = 0.45
YOLO_MAX_DET         = 500

# WBF parameters
WBF_IOU_THR          = 0.55   # IoU threshold for box fusion
WBF_SKIP_BOX_THR     = 0.05   # discard boxes below this confidence after fusion
WBF_WEIGHTS          = [1, 1]  # equal weight for both models

EMBED_IMGSZ          = 300
EMBED_BATCH          = 64
RERANK_SIM_THRESHOLD = 0.55
YOLO_TRUST_THRESHOLD = 0.75

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Embedder helpers ──────────────────────────────────────────────────────

def load_embedder(weights_path: Path, device: torch.device):
    import timm
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def load_reference_index(index_dir: Path):
    # Load from .json (not .npy) to stay within the 3 weight-file limit
    with open(index_dir / "reference_embeds.json", encoding="utf-8") as f:
        embeds = np.array(json.load(f), dtype=np.float32)
    with open(index_dir / "reference_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return embeds, [row["category_id"] for row in meta]


@torch.no_grad()
def embed_crops(crops: list, embedder, device: torch.device) -> np.ndarray:
    if not crops:
        return np.empty((0, 1536), dtype=np.float32)
    all_feats = []
    for i in range(0, len(crops), EMBED_BATCH):
        batch_imgs = crops[i : i + EMBED_BATCH]
        tensors = []
        for img in batch_imgs:
            img = img.convert("RGB").resize((EMBED_IMGSZ, EMBED_IMGSZ), Image.BILINEAR)
            t   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        t     = torch.stack(tensors).to(device)
        t     = (t - _MEAN.to(device)) / _STD.to(device)
        feats = embedder(t)
        feats = F.normalize(feats, dim=1)
        all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


def rerank_categories(yolo_cats, yolo_confs, crop_embeds,
                      ref_embeds, ref_cat_ids) -> list:
    if len(crop_embeds) == 0:
        return list(yolo_cats)
    sim    = crop_embeds @ ref_embeds.T
    result = []
    for i, (cat, conf) in enumerate(zip(yolo_cats, yolo_confs)):
        if conf >= YOLO_TRUST_THRESHOLD:
            result.append(cat)
            continue
        best     = int(np.argmax(sim[i]))
        best_sim = float(sim[i, best])
        result.append(ref_cat_ids[best] if best_sim >= RERANK_SIM_THRESHOLD else cat)
    return result


# ── WBF helper ────────────────────────────────────────────────────────────

def run_model(model, img_path: str, orig_w: int, orig_h: int):
    """
    Run a YOLO model and return boxes in normalised [0,1] coords,
    scores, and class labels — format required by WBF.
    """
    results = model(
        img_path,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
        augment=True,
        verbose=False,
    )
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            # Normalise to [0, 1]
            boxes.append([
                x1 / orig_w, y1 / orig_h,
                x2 / orig_w, y2 / orig_h,
            ])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes, scores, labels


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_dir   = Path(args.input)
    out_path = Path(args.output)
    here     = Path(__file__).parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ───────────────────────────────────────────────────────
    path_l = here / "best_l.onnx"
    path_x = here / "best_x.onnx"
    assert path_l.exists(), f"best_l.onnx not found"
    assert path_x.exists(), f"best_x.onnx not found"

    print("Loading YOLOv8l ...")
    model_l = YOLO(str(path_l))
    print("Loading YOLOv8x ...")
    model_x = YOLO(str(path_x))
    print("Both models loaded.")

    # ── Load WBF ─────────────────────────────────────────────────────────
    from ensemble_boxes import weighted_boxes_fusion

    # ── Load embedder ────────────────────────────────────────────────────
    index_dir    = here / "embedding_index"
    use_reranker = False
    embedder     = None
    ref_embeds   = None
    ref_cat_ids  = None

    if index_dir.exists() and (index_dir / "embedder.pt").exists() and (index_dir / "reference_embeds.json").exists():
        try:
            embedder                = load_embedder(index_dir / "embedder.pt", device)
            ref_embeds, ref_cat_ids = load_reference_index(index_dir)
            use_reranker            = True
            print(f"Embedder loaded — {ref_embeds.shape[0]} reference products")
        except Exception as e:
            print(f"WARNING: embedder failed ({e}) — detection-only")

    # ── Collect images ────────────────────────────────────────────────────
    valid_exts  = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in in_dir.iterdir()
                         if p.suffix.lower() in valid_exts)
    print(f"Processing {len(image_paths)} images ...")

    # ── Inference loop ────────────────────────────────────────────────────
    predictions = []

    with torch.no_grad():
        for img_path in image_paths:
            try:
                image_id = int(img_path.stem.split("_")[-1])
            except ValueError:
                print(f"  WARNING: cannot parse image_id from {img_path.name}")
                continue

            img_pil        = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size

            # Run both models
            boxes_l, scores_l, labels_l = run_model(model_l, str(img_path), orig_w, orig_h)
            boxes_x, scores_x, labels_x = run_model(model_x, str(img_path), orig_w, orig_h)

            if not boxes_l and not boxes_x:
                continue

            # WBF fusion — requires at least one non-empty list
            boxes_l  = boxes_l  or [[0, 0, 0.001, 0.001]]
            scores_l = scores_l or [0.0]
            labels_l = labels_l or [0]
            boxes_x  = boxes_x  or [[0, 0, 0.001, 0.001]]
            scores_x = scores_x or [0.0]
            labels_x = labels_x or [0]

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                [boxes_l, boxes_x],
                [scores_l, scores_x],
                [labels_l, labels_x],
                weights=WBF_WEIGHTS,
                iou_thr=WBF_IOU_THR,
                skip_box_thr=WBF_SKIP_BOX_THR,
            )

            if len(fused_boxes) == 0:
                continue

            # Convert normalised boxes back to pixel coords [x, y, w, h]
            boxes_px = []
            for (x1n, y1n, x2n, y2n) in fused_boxes:
                x1 = round(float(x1n) * orig_w, 1)
                y1 = round(float(y1n) * orig_h, 1)
                w  = round((float(x2n) - float(x1n)) * orig_w, 1)
                h  = round((float(y2n) - float(y1n)) * orig_h, 1)
                boxes_px.append([x1, y1, w, h])

            fused_cats  = [int(c) for c in fused_labels]
            fused_confs = [round(float(s), 4) for s in fused_scores]

            # Stage 2: embedding re-ranking
            if use_reranker:
                crops = []
                for (x1, y1, w, h) in boxes_px:
                    x1c = max(0, int(x1));  y1c = max(0, int(y1))
                    x2c = min(orig_w, int(x1 + w))
                    y2c = min(orig_h, int(y1 + h))
                    crops.append(
                        img_pil.crop((x1c, y1c, x2c, y2c))
                        if x2c > x1c and y2c > y1c
                        else img_pil.crop((0, 0, 2, 2))
                    )
                crop_embeds = embed_crops(crops, embedder, device)
                final_cats  = rerank_categories(
                    fused_cats, fused_confs, crop_embeds, ref_embeds, ref_cat_ids
                )
            else:
                final_cats = fused_cats

            for (x1, y1, w, h), cat_id, conf in zip(boxes_px, final_cats, fused_confs):
                predictions.append({
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, w, h],
                    "score":       conf,
                })

    print(f"Total predictions: {len(predictions)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
