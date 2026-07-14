"""Tests der Exact-Gaze-Auswahllogik (unabhängig von YOLO/Kamera)."""
from src.detection.types import Detection
from src.selection.exact import select_object


def make(cx, cy, label="obj", size=40, conf=0.9, cid=0):
    """Hilfsfunktion: Detection mit Zentrum (cx, cy) und quadratischer Box."""
    return Detection.from_xyxy(
        cid, label, conf, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2
    )


def test_none_when_no_detections():
    assert select_object((100, 100), []).is_none


def test_none_when_gaze_is_none():
    assert select_object(None, [make(100, 100)]).is_none


def test_none_when_nearest_too_far():
    # nächstes Objekt 300 px entfernt (>= 80) -> none
    assert select_object((100, 400), [make(100, 100)]).is_none


def test_direct_single_close_object():
    r = select_object((110, 100), [make(100, 100, "A"), make(400, 100, "B")])
    assert r.is_direct
    assert r.selected.label == "A"


def test_assisted_second_within_margin():
    # A: d=0, B: d=30 (<= 0 + 40) -> zwei Kandidaten -> assisted
    r = select_object((100, 100), [make(100, 100, "A"), make(130, 100, "B")])
    assert r.is_assisted
    assert len(r.candidates) == 2


def test_direct_when_second_outside_margin():
    # A: d=0, B: d=50 (> 40) -> nur A -> direct
    r = select_object((100, 100), [make(100, 100, "A"), make(150, 100, "B")])
    assert r.is_direct
    assert r.selected.label == "A"


def test_margin_boundary_is_inclusive():
    # B genau 40 px weiter -> <= margin -> assisted
    r = select_object((100, 100), [make(100, 100, "A"), make(140, 100, "B")])
    assert r.is_assisted


def test_direct_distance_boundary():
    # Distanz genau 80 -> nicht < 80 -> none
    assert select_object((100, 100), [make(180, 100)]).is_none
    # Distanz 79 -> direct
    assert select_object((100, 100), [make(179, 100)]).is_direct
