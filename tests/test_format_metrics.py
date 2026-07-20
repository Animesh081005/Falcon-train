from wordbox_ocr.format import FORMAT_V1, FORMAT_V2, parse, parse_validity, prompt_for_format, serialize
from wordbox_ocr.metrics import end_to_end_counts, f1


def test_roundtrip():
    source = [{"text": "A&B", "bbox": [10, 20, 30, 40]}]
    encoded = serialize(source, 100, 100)
    decoded = parse(encoded, 100, 100)
    assert decoded[0].text == "A&B"
    assert decoded[0].bbox == (10, 20, 30, 40)
    assert parse_validity(encoded) == 1.0


def test_matching():
    truth = parse("<word>hello</word><box>0500,0500,0500,0500</box>", 100, 100)
    pred = parse("<word>hello</word><box>0500,0500,0480,0480</box>", 100, 100)
    assert end_to_end_counts(pred, truth) == (1, 0, 0)
    assert f1(1, 0, 0) == 1.0


def test_parse_validity_rejects_bad_geometry():
    assert parse_validity("<word>x</word><box>0001,0001,1001,0010</box>") == 0.0
    assert parse_validity("<word>x</word><box>0010,0010,0000,0010</box>") == 0.0
    assert parse_validity("<word> </word><box>0010,0010,0005,0010</box>") == 0.0


def test_v1_checkpoint_compatibility():
    old = "<word>x</word><box>10,20,30,40</box>"
    assert parse(old, 100, 100, FORMAT_V1)[0].bbox == (1, 2, 3, 4)
    assert "x0,y0,x1,y1" in prompt_for_format(FORMAT_V1)
    assert "center_x,center_y,width,height" in prompt_for_format(FORMAT_V2)


def test_v2_edge_quantization_is_valid():
    encoded = serialize([{"text": "edge", "bbox": [0, 0, 1, 1]}], 1000, 1000)
    assert parse_validity(encoded) == 1.0
    assert parse(encoded, 1000, 1000)[0].bbox == (0, 0, 1, 1)
