from __future__ import annotations

from difflib import SequenceMatcher

from .format import WordBox


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def end_to_end_counts(pred: list[WordBox], truth: list[WordBox], threshold=0.5):
    candidates = []
    for pi, p in enumerate(pred):
        for ti, t in enumerate(truth):
            overlap = iou(p.bbox, t.bbox)
            if overlap >= threshold and p.text.casefold() == t.text.casefold():
                candidates.append((overlap, pi, ti))
    used_p, used_t = set(), set()
    for _, pi, ti in sorted(candidates, reverse=True):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi); used_t.add(ti)
    return len(used_p), len(pred) - len(used_p), len(truth) - len(used_t)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def normalized_edit_similarity(pred: list[WordBox], truth: list[WordBox]) -> float:
    return SequenceMatcher(None, " ".join(x.text for x in pred),
                           " ".join(x.text for x in truth)).ratio()

