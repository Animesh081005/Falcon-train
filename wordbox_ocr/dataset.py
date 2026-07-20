from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .format import serialize


class WordBoxDataset(Dataset):
    def __init__(self, manifest: str, max_samples: int | None = None):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        with self.manifest.open(encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No samples in {self.manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if row.get("width", width) != width or row.get("height", height) != height:
                raise ValueError(f"Manifest dimensions disagree with {path}")
            target = serialize(row["words"], width, height)
            if not target:
                raise ValueError(f"Sample has no valid non-empty word annotations: {path}")
            return {"image": image.copy(), "target": target, "path": str(path),
                    "width": width, "height": height, "words": row["words"]}
