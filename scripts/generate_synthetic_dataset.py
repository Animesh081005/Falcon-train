#!/usr/bin/env python3
"""Generate document-like OCR pages with exact word quadrilaterals.

The output contains both the requested comma-separated annotation files and
JSONL manifests consumed directly by ``wordbox_ocr.train``. Generation is
deterministic for a given seed, sample index, vocabulary, and font set.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


FALLBACK_WORDS = """
account address amount approved balance bank billing city code company contact
customer date description discount document due email invoice item name number
order payment phone price product quantity receipt reference statement subtotal
tax total transaction unit value after align object report service status terms
warehouse delivery department signature authorized information summary notes
January February March April May June July August September October November December
""".split()

FONT_ROOTS = (
    "/usr/share/fonts", "/usr/local/share/fonts",
    str(Path.home() / ".local/share/fonts"), str(Path.home() / ".fonts"),
    # Also permit local development/testing on macOS; GPU deployment uses Linux paths above.
    "/Library/Fonts", "/System/Library/Fonts", str(Path.home() / "Library/Fonts"),
)


@dataclass(frozen=True)
class Config:
    output: str
    vocabulary: tuple[str, ...]
    fonts: tuple[str, ...]
    width: int
    height: int
    min_words: int
    max_words: int
    jpeg_quality: int
    seed: int


def discover_fonts(font_dir: str | None) -> list[str]:
    roots = [font_dir] if font_dir else list(FONT_ROOTS)
    fonts = []
    for root in roots:
        path = Path(root).expanduser()
        if path.exists():
            fonts.extend(str(p) for p in path.rglob("*")
                         if p.suffix.lower() in {".ttf", ".otf"})
    # Exclude common symbol/emoji fonts that do not cover normal Latin text.
    bad = re.compile(r"emoji|symbol|dingbat|awesome|icons?", re.I)
    candidates = sorted({p for p in fonts if not bad.search(Path(p).name)})
    usable = []
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, 24)
            box = font.getbbox("Ag09")
            if box and box[2] > box[0] and box[3] > box[1]:
                usable.append(candidate)
        except (OSError, ValueError):
            continue
    return usable


def load_vocabulary(path: str | None) -> list[str]:
    if path:
        words = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
        words = [w for w in words if w and "\n" not in w and "\r" not in w]
    else:
        words = FALLBACK_WORDS
    if len(words) < 20:
        raise ValueError("Vocabulary needs at least 20 non-empty entries")
    return words


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    box = draw.textbbox((0, 0), text, font=font, anchor="lt")
    return box[2] - box[0], box[3] - box[1]


def random_token(rng: random.Random, vocabulary: tuple[str, ...]) -> str:
    kind = rng.random()
    if kind < 0.10:
        return f"{rng.randint(1, 9999):,}"
    if kind < 0.15:
        return f"{rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if kind < 0.19:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2018, 2032)}"
    word = rng.choice(vocabulary)
    if rng.random() < 0.18:
        word = word.upper()
    elif rng.random() < 0.65:
        word = word.capitalize()
    if rng.random() < 0.08:
        word += rng.choice([":", ",", ".", "-", "#"])
    return word


def add_word(draw, annotations, text, x, y, font, fill):
    # textbbox accounts for ascender offsets; use the exact painted glyph extent.
    b = draw.textbbox((x, y), text, font=font, anchor="lt")
    x0, y0, x1, y1 = map(int, b)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Font produced an empty bounding box for {text!r}")
    draw.text((x, y), text, font=font, fill=fill, anchor="lt")
    annotations.append({"text": text, "bbox": [x0, y0, x1, y1],
                        "quad": [x0, y0, x1, y0, x1, y1, x0, y1]})
    return x1


def add_spatially_balanced_words(draw, annotations, rng, cfg, width, height,
                                 margin, target_words, font_path, base_size, fill):
    """Place sparse OCR words across the full page, in stable reading order."""
    line_count = rng.randint(3, min(5, target_words))
    counts = [target_words // line_count] * line_count
    for index in range(target_words % line_count):
        counts[index] += 1
    usable_height = height - 2 * margin
    band_height = usable_height / line_count
    for line_index, count in enumerate(counts):
        size = max(11, round(base_size * rng.uniform(0.85, 1.2)))
        line_font_path = rng.choice(cfg.fonts) if rng.random() < 0.25 else font_path
        tokens = [random_token(rng, cfg.vocabulary) for _ in range(count)]
        gap = rng.randint(max(5, size // 3), max(9, size))
        while True:
            font = ImageFont.truetype(line_font_path, size)
            widths = [text_size(draw, token, font)[0] for token in tokens]
            total_width = sum(widths) + gap * (count - 1)
            if total_width <= width - 2 * margin or size <= 10:
                break
            size -= 1
        if total_width > width - 2 * margin:
            raise RuntimeError("Sparse line cannot fit; shorten vocabulary entries")
        max_start = width - margin - total_width
        x = rng.randint(margin, max(margin, int(max_start)))
        band_top = margin + line_index * band_height
        y = round(band_top + rng.uniform(0.18, 0.72) * band_height)
        for token in tokens:
            x = add_word(draw, annotations, token, x, y, font, fill) + gap


def render_page(index: int, cfg: Config) -> dict:
    rng = random.Random(cfg.seed + index * 1_000_003)
    width = max(512, round(cfg.width * rng.uniform(0.85, 1.15)))
    height = max(640, round(cfg.height * rng.uniform(0.85, 1.15)))
    paper = rng.choice([(255, 255, 252), (248, 247, 242), (242, 244, 246), (255, 255, 255)])
    image = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(image)
    margin = rng.randint(max(20, width // 30), max(30, width // 14))
    target_words = rng.randint(cfg.min_words, cfg.max_words)
    annotations = []
    font_path = rng.choice(cfg.fonts)
    base_size = rng.randint(max(14, width // 65), max(20, width // 38))
    ink = rng.randint(10, 65)
    fill = (ink, ink, ink)
    y = margin

    if target_words <= 15:
        add_spatially_balanced_words(
            draw, annotations, rng, cfg, width, height, margin,
            target_words, font_path, base_size, fill)
    else:
        # Dense document: header/title followed by conventional body lines.
        if rng.random() < 0.75:
            title_font = ImageFont.truetype(font_path, round(base_size * rng.uniform(1.25, 1.7)))
            title_start = len(annotations)
            x = margin
            for token in [random_token(rng, cfg.vocabulary) for _ in range(rng.randint(1, 4))]:
                if len(annotations) >= target_words:
                    break
                token_width, _ = text_size(draw, token, title_font)
                if x + token_width >= width - margin:
                    break
                x = add_word(draw, annotations, token, x, y, title_font, fill) + base_size // 2
            if len(annotations) > title_start:
                y = max(a["bbox"][3] for a in annotations[title_start:]) + rng.randint(base_size, base_size * 2)

        while len(annotations) < target_words and y < height - margin - base_size * 2:
            line_size = max(11, round(base_size * rng.uniform(0.82, 1.15)))
            line_font_path = rng.choice(cfg.fonts) if rng.random() < 0.16 else font_path
            font = ImageFont.truetype(line_font_path, line_size)
            x = margin + (rng.randint(0, width // 8) if rng.random() < 0.18 else 0)
            gap = rng.randint(max(4, line_size // 4), max(8, line_size))
            words_this_line = 0
            while len(annotations) < target_words:
                token = random_token(rng, cfg.vocabulary)
                tw, _ = text_size(draw, token, font)
                if x + tw >= width - margin:
                    break
                x = add_word(draw, annotations, token, x, y, font, fill) + gap
                words_this_line += 1
                if words_this_line >= rng.randint(3, 12):
                    break
            if not words_this_line:
                break
            y = max(a["bbox"][3] for a in annotations[-words_this_line:]) + rng.randint(7, max(9, line_size))
            if rng.random() < 0.09:
                draw.line((margin, y, width - margin, y), fill=(150, 150, 150), width=1)
                y += rng.randint(4, 10)

    # Mild degradations that preserve exact boxes in the image coordinate frame.
    if rng.random() < 0.45:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.72, 1.18))
    if rng.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.65)))

    if len(annotations) != target_words:
        raise RuntimeError(
            f"Could only place {len(annotations)}/{target_words} words on sample {index}; "
            "increase page size or shorten the vocabulary"
        )

    root = Path(cfg.output)
    stem = f"syn_{index:07d}"
    image_rel = Path("images") / f"{stem}.jpg"
    annotation_rel = Path("annotations") / f"{stem}.txt"
    image.save(root / image_rel, quality=cfg.jpeg_quality, optimize=False)
    with (root / annotation_rel).open("w", encoding="utf-8", newline="\n") as f:
        for item in annotations:
            values = [*item["quad"], "Latin", item["text"]]
            f.write(",".join(map(str, values)) + "\n")
    return {"image": str(Path("..") / image_rel), "width": width, "height": height,
            "words": [{"text": a["text"], "bbox": a["bbox"]} for a in annotations]}


_WORKER_CONFIG = None


def init_worker(config):
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def worker(index):
    return index, render_page(index, _WORKER_CONFIG)


def split_for(index: int, seed: int, train_ratio: float, val_ratio: float) -> str:
    # Per-index deterministic split without holding or shuffling 100k rows.
    value = random.Random(seed ^ (index * 2_654_435_761)).random()
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "validation"
    return "test"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--num-images", type=int, default=100_000)
    p.add_argument("--vocabulary", help="UTF-8 file containing one word per line")
    p.add_argument("--font-dir", help="Only use .ttf/.otf files below this directory")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--min-words", type=int, default=9)
    p.add_argument("--max-words", type=int, default=10)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--validation-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunksize", type=int, default=32)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if args.num_images < 1 or args.min_words < 1 or args.max_words < args.min_words:
        p.error("invalid image/word counts")
    if not (0 < args.train_ratio < 1 and 0 <= args.validation_ratio < 1
            and args.train_ratio + args.validation_ratio < 1):
        p.error("ratios must leave a non-empty test fraction")
    fonts = discover_fonts(args.font_dir)
    if not fonts:
        p.error("No usable fonts found; install fonts-dejavu-core or pass --font-dir")
    vocabulary = load_vocabulary(args.vocabulary)
    root = Path(args.output).resolve()
    for name in ("images", "annotations", "manifests"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config = Config(str(root), tuple(vocabulary), tuple(fonts), args.width, args.height,
                    args.min_words, args.max_words, args.jpeg_quality, args.seed)
    manifests = {name: (root / "manifests" / f"{name}.jsonl").open("w", encoding="utf-8")
                 for name in ("train", "validation", "test")}
    counts = {name: 0 for name in manifests}
    try:
        context = mp.get_context("spawn")
        with context.Pool(args.workers, init_worker, (config,)) as pool:
            results = pool.imap(worker, range(args.num_images), chunksize=args.chunksize)
            for completed, (index, row) in enumerate(results, 1):
                split = split_for(index, args.seed, args.train_ratio, args.validation_ratio)
                manifests[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
                if completed % 1000 == 0 or completed == args.num_images:
                    print(f"generated {completed:,}/{args.num_images:,}", flush=True)
    finally:
        for handle in manifests.values():
            handle.close()
    (root / "dataset_info.json").write_text(json.dumps({
        "num_images": args.num_images, "splits": counts, "seed": args.seed,
        "num_fonts": len(fonts), "vocabulary_size": len(vocabulary),
        "layout_version": "spatially-balanced-v2",
        "annotation_format": "x1,y1,x2,y2,x3,y3,x4,y4,script,text",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
