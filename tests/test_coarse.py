"""Tests der Coarse-Gaze-Auswahllogik (Kreis-Rechteck-Schnitt)."""
from src.detection.types import Detection
from src.selection.coarse import select_with_attention_area


def make(cx, cy, size=40, label="obj"):
    return Detection.from_xyxy(
        0, label, 0.9, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2
    )


def test_none_when_empty():
    assert select_with_attention_area((0, 0), []).is_none


def test_none_when_gaze_is_none():
    assert select_with_attention_area(None, [make(0, 0)]).is_none


def test_direct_single_in_circle():
    r = select_with_attention_area((100, 100), [make(100, 100)], radius=180)
    assert r.is_direct


def test_none_when_outside_circle():
    # Box weit weg, kleiner Radius
    r = select_with_attention_area((0, 0), [make(500, 500)], radius=50)
    assert r.is_none


def test_assisted_multiple_in_circle():
    dets = [make(100, 100), make(150, 150), make(120, 80)]
    r = select_with_attention_area((120, 120), dets, radius=180)
    assert r.is_assisted
    assert len(r.candidates) == 3


def test_circle_tangent_is_inclusive():
    # Box x:[200,240], y:[90,110]; Blick (0,100). Nächster Punkt (200,100) -> Distanz 200.
    box = Detection.from_xyxy(0, "b", 0.9, 200, 90, 240, 110)
    assert select_with_attention_area((0, 100), [box], radius=200).is_direct   # 200 == radius
    assert select_with_attention_area((0, 100), [box], radius=199).is_none     # knapp daneben
