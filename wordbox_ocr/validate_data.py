from __future__ import annotations

import argparse
from .dataset import WordBoxDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    ds = WordBoxDataset(args.manifest)
    words = 0
    for i in range(len(ds)):
        sample = ds[i]
        words += len(sample["words"])
    print(f"OK: {len(ds)} images, {words} annotated words")


if __name__ == "__main__":
    main()

