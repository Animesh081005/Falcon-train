#!/usr/bin/env python3
"""Report spatial bias and annotation counts from a word-box JSONL manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    position = min(len(values) - 1, round(fraction * (len(values) - 1)))
    return values[position]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    rows = []
    with Path(args.manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if args.max_samples and len(rows) >= args.max_samples:
                    break
    if not rows:
        raise SystemExit("Manifest is empty")
    x_centers, y_centers = [], []
    for row in rows:
        width, height = row["width"], row["height"]
        for word in row["words"]:
            x0, y0, x1, y1 = word["bbox"]
            x_centers.append(((x0 + x1) / 2) / width)
            y_centers.append(((y0 + y1) / 2) / height)
    quartiles = [sum(lo <= y < hi for y in y_centers) / len(y_centers)
                 for lo, hi in zip((0, .25, .5, .75), (.25, .5, .75, 1.01))]
    print(f"images: {len(rows):,}")
    print(f"words: {len(y_centers):,}")
    print(f"words/image: {len(y_centers) / len(rows):.2f}")
    print("normalized y center percentiles: " + ", ".join(
        f"p{round(q * 100)}={percentile(y_centers, q):.3f}" for q in (.05, .25, .5, .75, .95)))
    print("vertical quartile fractions: " + ", ".join(f"{value:.3f}" for value in quartiles))
    print("normalized x center percentiles: " + ", ".join(
        f"p{round(q * 100)}={percentile(x_centers, q):.3f}" for q in (.05, .5, .95)))


if __name__ == "__main__":
    main()
