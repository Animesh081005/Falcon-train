#!/usr/bin/env python3
"""Create a deterministic weighted training mixture from canonical manifests."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def source_spec(value: str) -> tuple[Path, float]:
    try:
        path_text, weight_text = value.rsplit(":", 1)
        weight = float(weight_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected MANIFEST:WEIGHT") from error
    if weight <= 0:
        raise argparse.ArgumentTypeError("source weight must be positive")
    return Path(path_text).resolve(), weight


def load_rows(manifest: Path) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            image = Path(row["image"])
            if not image.is_absolute():
                row["image"] = str((manifest.parent / image).resolve())
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=source_spec,
                        metavar="MANIFEST:WEIGHT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    rng = random.Random(args.seed)
    datasets = [(load_rows(path), weight, path) for path, weight in args.source]
    total_weight = sum(weight for _, weight, _ in datasets)
    exact_counts = [args.num_samples * weight / total_weight for _, weight, _ in datasets]
    counts = [int(value) for value in exact_counts]
    for index in sorted(range(len(counts)), key=lambda i: exact_counts[i] - counts[i], reverse=True):
        if sum(counts) == args.num_samples:
            break
        counts[index] += 1

    mixed = []
    for (rows, _, manifest), count in zip(datasets, counts):
        if count <= len(rows):
            selected = rng.sample(rows, count)
        else:
            selected = rows.copy()
            selected.extend(rng.choice(rows) for _ in range(count - len(rows)))
        mixed.extend(selected)
        print(f"{manifest}: {count} samples")
    rng.shuffle(mixed)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for row in mixed:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(mixed)} rows to {output}")


if __name__ == "__main__":
    main()
