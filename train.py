"""
train.py
Fine-tunes YOLOv8 on the NM in AI 2026 grocery dataset.

Requirements (training machine only — NOT the sandbox):
    pip install ultralytics   # latest — we export to ONNX for the sandbox

Usage:
    # Recommended — YOLOv8l, good balance of speed & accuracy
    python train.py --data /path/to/yolo_dataset/dataset.yaml --model yolov8l

    # Larger model if you have a strong GPU (fits on L4 for inference)
    python train.py --data /path/to/yolo_dataset/dataset.yaml --model yolov8x

    # Faster iteration / weaker GPU
    python train.py --data /path/to/yolo_dataset/dataset.yaml --model yolov8m

After training, your best weights will be at:
    runs/detect/nmiai_<model>/weights/best.pt

Submission zip:
    zip -r submission.zip run.py best.pt
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument(
        "--model",
        default="yolov8l",
        help="Model to fine-tune (e.g. yolov8l, yolov8x, rtdetr-x.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size (px)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument("--device", default="0", help="CUDA device(s) or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--name", default=None, help="Run name (default: nmiai_<model>)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here so the file can be read without ultralytics installed
    from ultralytics import YOLO

    data_path = Path(args.data)
    assert data_path.exists(), f"dataset.yaml not found: {data_path}"

    model_name = args.name or f"nmiai_{args.model}"

    print(f"Model:      {args.model}.pt  (pretrained COCO weights)")
    print(f"Dataset:    {data_path}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs:     {args.epochs}  (early stop patience={args.patience})")
    print(f"Run name:   {model_name}")
    print()

    # Load pretrained YOLOv8 — nc will be overridden by dataset.yaml (357 classes)
    model = YOLO(f"{args.model}.pt")

    # ------------------------------------------------------------------ #
    # Augmentation notes:
    #   - mosaic=1.0    : combines 4 images → great for small datasets
    #   - mixup=0.1     : mild blending
    #   - copy_paste=0.1: paste objects across images
    #   - degrees/shear : mild geometric
    #   - hsv_*         : colour jitter for varying store lighting
    #   - flipud=0.0    : shelves don't appear upside-down
    #   - scale=0.5     : zoom in/out — products appear at many scales
    #
    # imgsz=1280 is important: shelf images are 2000×1500 and products
    # can be small. 1280 preserves more detail than the default 640.
    # ------------------------------------------------------------------ #

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,           # -1 = auto-tune to GPU memory
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=model_name,
        exist_ok=True,
        resume=args.resume,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,                   # final LR = lr0 * lrf
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        shear=2.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,                 # shelves are always right-way up
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0002,
        translate=0.1,
        # Loss weights — slightly upweight cls since we have many classes
        cls=1.5,
        box=7.5,
        dfl=1.5,
        # Evaluation
        val=True,
        save_period=10,             # save checkpoint every N epochs
        plots=True,
    )

    # Print best weights path
    best = Path(f"runs/detect/{model_name}/weights/best.pt")
    if best.exists():
        print(f"\n✓ Training complete.")
        print(f"  Best weights: {best.resolve()}")
        print(f"\nCreate your submission zip:")
        print(f"  cp {best} best.pt")
        print(f"  zip -r submission.zip run.py best.pt")
    else:
        print("\nTraining complete. Check runs/detect/ for weights.")

    return results


if __name__ == "__main__":
    main()
