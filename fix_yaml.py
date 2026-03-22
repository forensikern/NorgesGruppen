"""
fix_yaml.py
Regenerates yolo_dataset/dataset.yaml with correct YAML syntax.
Run once, then train as normal.

Usage:
    py -3.11 fix_yaml.py
"""

import json
from pathlib import Path

import sys
dataset = sys.argv[1] if len(sys.argv) > 1 else "yolo_dataset"
ANN_PATH  = Path("NM_NGD_coco_dataset/train/annotations.json")
YAML_PATH = Path(dataset) / "dataset.yaml"

ann   = json.load(open(ANN_PATH, encoding="utf-8"))
cats  = {c["id"]: c["name"] for c in ann["categories"]}
names = [cats[i] for i in sorted(cats.keys())]

# Absolute path to yolo_dataset, forward slashes (YOLO prefers this on Windows)
dataset_path = Path(dataset).resolve().as_posix()

lines = [
    f"path: {dataset_path}",
    "train: images/train",
    "val: images/val",
    f"nc: {len(names)}",
    "names:",
]

for name in names:
    # Use double-quoted YAML strings — escape \ and " only
    safe = name.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'  - "{safe}"')

YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Written {len(names)} categories to {YAML_PATH}")
print("First 5 entries:")
for line in lines[5:10]:
    print(" ", line)
