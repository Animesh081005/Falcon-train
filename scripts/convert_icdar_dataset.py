#!/usr/bin/env python3
"""Convert ICDAR/SROIE-style quadrilateral text files to canonical JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from convert_textocr import reading_order
except ImportError:  # Imported as scripts.convert_icdar_dataset in tests/tools.
    from scripts.convert_textocr import reading_order


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
SCRIPT_NAMES = {"Arabic", "Bangla", "Chinese", "Devanagari", "Japanese", "Korean",
                "Latin", "Symbols", "None", "Mixed"}


def image_for(annotation: Path, images_root: Path) -> Path | None:
    stem = annotation.stem
    if stem.startswith("gt_"):
        stem = stem[3:]
    for extension in IMAGE_EXTENSIONS:
        candidate = images_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
        candidate = images_root / f"{stem}{extension.upper()}"
        if candidate.exists():
            return candidate
    return None


def parse_annotation(path: Path, width: int, height: int, preserve_file_order: bool) -> list[dict]:
    words = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) < 9:
            raise ValueError(f"{path}:{line_number}: expected 8 coordinates and text")
        try:
            coordinates = [float(value.strip()) for value in fields[:8]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid coordinate") from error
        remainder = fields[8:]
        if len(remainder) > 1 and remainder[0].strip() in SCRIPT_NAMES:
            remainder = remainder[1:]
        text = ",".join(remainder).strip()
        if not text or text == "###":
            continue
        xs, ys = coordinates[0::2], coordinates[1::2]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        x1, y1 = min(float(width), max(xs)), min(float(height), max(ys))
        if x1 > x0 and y1 > y0:
            words.append({"text": text, "bbox": [x0, y0, x1, y1]})
    return words if preserve_file_order else reading_order(words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="icdar")
    parser.add_argument("--preserve-file-order", action="store_true",
                        help="Use annotation line order instead of inferred raster order")
    args = parser.parse_args()

    images_root = Path(args.images_root).resolve()
    annotations_root = Path(args.annotations_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as destination:
        for annotation_path in sorted(annotations_root.rglob("*.txt")):
            image_path = image_for(annotation_path, images_root)
            if image_path is None:
                raise FileNotFoundError(f"No matching image for {annotation_path.name}")
            with Image.open(image_path) as image:
                width, height = image.size
            words = parse_annotation(annotation_path, width, height, args.preserve_file_order)
            if not words:
                continue
            row = {"image": str(image_path), "width": width, "height": height,
                   "source": args.source, "source_id": annotation_path.stem,
                   "words": words}
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
