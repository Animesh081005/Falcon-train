from __future__ import annotations

import html
import re
from dataclasses import dataclass

# Keep Falcon-OCR's pretrained OCR task token and specialize its response format
# through the instruction/targets. A novel pseudo-special token would not have a
# pretrained embedding and may collapse to unknown/subword pieces.
PROMPT = (
    "<|image|>Extract every word in reading order with its bounding box as "
    "<word>text</word><box>x0,y0,x1,y1</box>, using coordinates from 0 to 1000.\n"
    "<|OCR_PLAIN|>"
)
_ITEM = re.compile(
    r"<word>(.*?)</word>\s*<box>\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*</box>",
    re.DOTALL,
)


@dataclass(frozen=True)
class WordBox:
    text: str
    bbox: tuple[float, float, float, float]


def normalize_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid xyxy box {box} for image {width}x{height}")
    vals = (x0 / width, y0 / height, x1 / width, y1 / height)
    return tuple(min(1000, max(0, round(v * 1000))) for v in vals)


def serialize(words: list[dict], width: int, height: int) -> str:
    rows = []
    for item in words:
        text = str(item["text"]).strip()
        if not text:
            continue
        box = normalize_box(item["bbox"], width, height)
        rows.append(
            f"<word>{html.escape(text, quote=False)}</word>"
            f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>"
        )
    return "\n".join(rows)


def parse(text: str, width: int, height: int) -> list[WordBox]:
    output = []
    for match in _ITEM.finditer(text):
        coords = tuple(int(v) for v in match.groups()[1:])
        if any(v < 0 or v > 1000 for v in coords):
            continue
        x0, y0, x1, y1 = coords
        if x1 <= x0 or y1 <= y0:
            continue
        output.append(
            WordBox(
                html.unescape(match.group(1)).strip(),
                (x0 * width / 1000, y0 * height / 1000,
                 x1 * width / 1000, y1 * height / 1000),
            )
        )
    return [x for x in output if x.text]


def parse_validity(text: str) -> float:
    """1.0 only when every nonblank line is a valid, geometrically sound item."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return 0.0
    for line in lines:
        match = _ITEM.fullmatch(line.strip())
        if match is None:
            return 0.0
        x0, y0, x1, y1 = (int(value) for value in match.groups()[1:])
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            return 0.0
        if not html.unescape(match.group(1)).strip():
            return 0.0
    return 1.0
