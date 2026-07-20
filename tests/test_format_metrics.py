from wordbox_ocr.format import parse, parse_validity, serialize
from wordbox_ocr.metrics import end_to_end_counts, f1


def test_roundtrip():
    source = [{"text": "A&B", "bbox": [10, 20, 30, 40]}]
    encoded = serialize(source, 100, 100)
    decoded = parse(encoded, 100, 100)
    assert decoded[0].text == "A&B"
    assert decoded[0].bbox == (10, 20, 30, 40)
    assert parse_validity(encoded) == 1.0


def test_matching():
    truth = parse("<word>hello</word><box>0,0,500,500</box>", 100, 100)
    pred = parse("<word>hello</word><box>10,10,490,490</box>", 100, 100)
    assert end_to_end_counts(pred, truth) == (1, 0, 0)
    assert f1(1, 0, 0) == 1.0


def test_parse_validity_rejects_bad_geometry():
    assert parse_validity("<word>x</word><box>0,0,1001,10</box>") == 0.0
    assert parse_validity("<word>x</word><box>10,0,5,10</box>") == 0.0
    assert parse_validity("<word> </word><box>0,0,5,10</box>") == 0.0
