from __future__ import annotations

import html
import re
from dataclasses import dataclass

# Keep Falcon-OCR's pretrained OCR task token and specialize its response format
# through the instruction/targets. A novel pseudo-special token would not have a
# pretrained embedding and may collapse to unknown/subword pieces.
FORMAT_V1 = "wordbox-v1-normalized-1000"
FORMAT_V2 = "wordbox-v2-cxcywh-fixed4"
CURRENT_FORMAT = FORMAT_V2

PROMPT_V1 = (
    "<|image|>Extract every word in reading order with its bounding box as "
    "<word>text</word><box>x0,y0,x1,y1</box>, using coordinates from 0 to 1000.\n"
    "<|OCR_PLAIN|>"
)
PROMPT_V2 = (
    "<|image|>Extract every word in reading order with its bounding box as "
    "<word>text</word><box>center_x,center_y,width,height</box>. Use exactly four "
    "digits per value, normalized from 0000 to 1000.\n"
    "<|OCR_PLAIN|>"
)
PROMPT = PROMPT_V2
_ITEM = re.compile(
    r"<word>(.*?)</word>\s*<box>\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*</box>",
    re.DOTALL,
)


@dataclass(frozen=True)
class WordBox:
    text: str
    bbox: tuple[float, float, float, float]


def prompt_for_format(format_version: str) -> str:
    if format_version == FORMAT_V1:
        return PROMPT_V1
    if format_version == FORMAT_V2:
        return PROMPT_V2
    raise ValueError(f"Unsupported word-box format: {format_version}")


def normalize_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid xyxy box {box} for image {width}x{height}")
    vals = (x0 / width, y0 / height, x1 / width, y1 / height)
    return tuple(min(1000, max(0, round(v * 1000))) for v in vals)


def serialize(words: list[dict], width: int, height: int,
              format_version: str = CURRENT_FORMAT) -> str:
    rows = []
    for item in words:
        text = str(item["text"]).strip()
        if not text:
            continue
        x0, y0, x1, y1 = normalize_box(item["bbox"], width, height)
        if format_version == FORMAT_V1:
            box = (x0, y0, x1, y1)
        elif format_version == FORMAT_V2:
            box = (round((x0 + x1) / 2), round((y0 + y1) / 2), x1 - x0, y1 - y0)
        else:
            raise ValueError(f"Unsupported word-box format: {format_version}")
        rows.append(
            f"<word>{html.escape(text, quote=False)}</word>"
            f"<box>{box[0]:04d},{box[1]:04d},{box[2]:04d},{box[3]:04d}</box>"
        )
    return "\n".join(rows)


def parse(text: str, width: int, height: int,
          format_version: str = CURRENT_FORMAT) -> list[WordBox]:
    output = []
    for match in _ITEM.finditer(text):
        coords = tuple(int(v) for v in match.groups()[1:])
        if any(v < 0 or v > 1000 for v in coords):
            continue
        if format_version == FORMAT_V1:
            x0, y0, x1, y1 = coords
            if x1 <= x0 or y1 <= y0:
                continue
        elif format_version == FORMAT_V2:
            cx, cy, box_w, box_h = coords
            if box_w <= 0 or box_h <= 0:
                continue
            x0, y0 = cx - box_w / 2, cy - box_h / 2
            x1, y1 = cx + box_w / 2, cy + box_h / 2
            # Integer center/size quantization can overshoot an image edge by
            # exactly half a normalized unit for odd-width edge-touching boxes.
            if not (-0.5 <= x0 < x1 <= 1000.5 and -0.5 <= y0 < y1 <= 1000.5):
                continue
            if x0 < 0:
                x1, x0 = x1 - x0, 0
            if y0 < 0:
                y1, y0 = y1 - y0, 0
            if x1 > 1000:
                x0, x1 = x0 - (x1 - 1000), 1000
            if y1 > 1000:
                y0, y1 = y0 - (y1 - 1000), 1000
        else:
            raise ValueError(f"Unsupported word-box format: {format_version}")
        output.append(
            WordBox(
                html.unescape(match.group(1)).strip(),
                (x0 * width / 1000, y0 * height / 1000,
                 x1 * width / 1000, y1 * height / 1000),
            )
        )
    return [x for x in output if x.text]


def parse_validity(text: str, format_version: str = CURRENT_FORMAT) -> float:
    """1.0 only when every nonblank line is a valid, geometrically sound item."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return 0.0
    for line in lines:
        match = _ITEM.fullmatch(line.strip())
        if match is None:
            return 0.0
        coords = tuple(int(value) for value in match.groups()[1:])
        if format_version == FORMAT_V1:
            x0, y0, x1, y1 = coords
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                return 0.0
        elif format_version == FORMAT_V2:
            cx, cy, box_w, box_h = coords
            if not (0 < box_w <= 1000 and 0 < box_h <= 1000):
                return 0.0
            if not (-0.5 <= cx - box_w / 2 and cx + box_w / 2 <= 1000.5
                    and -0.5 <= cy - box_h / 2 and cy + box_h / 2 <= 1000.5):
                return 0.0
        else:
            raise ValueError(f"Unsupported word-box format: {format_version}")
        if not html.unescape(match.group(1)).strip():
            return 0.0
    return 1.0
