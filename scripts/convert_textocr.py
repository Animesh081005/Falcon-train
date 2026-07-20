#!/usr/bin/env python3
"""Convert official TextOCR JSON into the canonical word-box JSONL manifest."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def reading_order(words: list[dict]) -> list[dict]:
    """Approximate raster order by grouping vertically overlapping words."""
    if not words:
        return []
    median_height = statistics.median(word["bbox"][3] - word["bbox"][1] for word in words)
    lines: list[dict] = []
    for word in sorted(words, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        x0, y0, x1, y1 = word["bbox"]
        cy = (y0 + y1) / 2
        candidates = [line for line in lines
                      if abs(cy - line["cy"]) <= 0.65 * max(median_height, y1 - y0, line["height"])]
        if candidates:
            line = min(candidates, key=lambda candidate: abs(cy - candidate["cy"]))
            line["words"].append(word)
            line["cy"] = sum((item["bbox"][1] + item["bbox"][3]) / 2
                             for item in line["words"]) / len(line["words"])
            line["height"] = max(item["bbox"][3] - item["bbox"][1]
                                 for item in line["words"])
        else:
            lines.append({"cy": cy, "height": y1 - y0, "words": [word]})
    ordered = []
    for line in sorted(lines, key=lambda item: item["cy"]):
        ordered.extend(sorted(line["words"], key=lambda item: item["bbox"][0]))
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True,
                        help="TextOCR_0.1_train.json or TextOCR_0.1_val.json")
    parser.add_argument("--images-root", required=True,
                        help="Root containing the train/ and test/ image folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=1)
    args = parser.parse_args()
    if args.min_words < 1:
        parser.error("--min-words must be positive")

    source = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    images = source["imgs"]
    annotations = source["anns"]
    image_to_annotations = source.get("imgToAnns")
    if image_to_annotations is None:
        image_to_annotations = defaultdict(list)
        for annotation_id, annotation in annotations.items():
            image_to_annotations[str(annotation["image_id"])].append(annotation_id)

    image_root = Path(args.images_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped = 0, 0
    with output.open("w", encoding="utf-8") as destination:
        for image_id, image in images.items():
            width, height = int(image["width"]), int(image["height"])
            words = []
            annotation_ids = (image_to_annotations.get(str(image_id)) or
                              image_to_annotations.get(image_id) or [])
            for annotation_id in annotation_ids:
                annotation = annotations.get(str(annotation_id), annotations.get(annotation_id))
                if annotation is None:
                    continue
                text = str(annotation.get("utf8_string", "")).strip()
                points = annotation.get("points", [])
                if not text or text == "###" or len(points) < 8 or len(points) % 2:
                    continue
                xs = [float(value) for value in points[0::2]]
                ys = [float(value) for value in points[1::2]]
                x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
                x1, y1 = min(float(width), max(xs)), min(float(height), max(ys))
                if x1 <= x0 or y1 <= y0:
                    continue
                words.append({"text": text, "bbox": [x0, y0, x1, y1]})
            if len(words) < args.min_words:
                skipped += 1
                continue
            filename = image.get("file_name", image.get("filename"))
            if not filename:
                raise ValueError(f"TextOCR image {image_id} has no filename")
            row = {"image": str(image_root / filename), "width": width, "height": height,
                   "source": "textocr", "source_id": str(image_id),
                   "words": reading_order(words)}
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    print(f"wrote {rows} rows to {output}; skipped {skipped} images with too few valid words")


if __name__ == "__main__":
    main()
