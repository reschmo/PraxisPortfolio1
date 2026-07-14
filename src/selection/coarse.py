"""Coarse-Gaze: Blick = Bereich (Kreis).

Sicherheitsnetz für ungenaue Blicke: Bounding Boxes, die den Kreis um den
Blickpunkt schneiden, gelten als Kandidaten.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .result import SelectionResult, SelectionState
from .. import config


def _circle_intersects_bbox(cx: float, cy: float, radius: float, bbox) -> bool:
    """Kreis-Rechteck-Schnitt: nächster Punkt der Box zum Kreismittelpunkt."""
    x1, y1, x2, y2 = bbox
    nearest_x = min(max(cx, x1), x2)
    nearest_y = min(max(cy, y1), y2)
    dx, dy = cx - nearest_x, cy - nearest_y
    return dx * dx + dy * dy <= radius * radius


def _to_int_point(p):
    if p is None:
        return None
    return (int(round(p[0])), int(round(p[1])))


def select_with_attention_area(
    gaze_point: Optional[Tuple[float, float]],
    detections: Sequence,
    radius: float = config.COARSE_RADIUS,
    strategy: str = "coarse",
) -> SelectionResult:
    """1 Treffer -> direct, mehrere -> assisted, keiner -> none."""
    gp = _to_int_point(gaze_point)
    if gaze_point is None or not detections:
        return SelectionResult(SelectionState.NONE, strategy, gaze_point=gp)

    cx, cy = gaze_point
    candidates = [
        d for d in detections if _circle_intersects_bbox(cx, cy, radius, d.bbox)
    ]

    if not candidates:
        return SelectionResult(SelectionState.NONE, strategy, gaze_point=gp)
    if len(candidates) == 1:
        return SelectionResult(
            SelectionState.DIRECT,
            strategy,
            selected=candidates[0],
            candidates=candidates,
            gaze_point=gp,
        )
    return SelectionResult(
        SelectionState.ASSISTED, strategy, candidates=candidates, gaze_point=gp
    )
