"""Exact-Gaze: Blick = Punkt.

Euklidische Distanz vom Blickpunkt zum Objektzentrum entscheidet.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .result import SelectionResult, SelectionState
from .. import config


def _distance(p: Tuple[float, float], c: Tuple[float, float]) -> float:
    return math.hypot(p[0] - c[0], p[1] - c[1])


def _to_int_point(p):
    if p is None:
        return None
    return (int(round(p[0])), int(round(p[1])))


def select_object(
    gaze_point: Optional[Tuple[float, float]],
    detections: Sequence,
    direct_max_dist: float = config.EXACT_DIRECT_MAX_DIST,
    ambiguity_margin: float = config.EXACT_AMBIGUITY_MARGIN,
    strategy: str = "exact",
) -> SelectionResult:
    """Entscheidet den Auswahlzustand nach Exact-Gaze-Regeln.

    Regeln:
      * kein nächstes Objekt innerhalb `direct_max_dist`         -> none
      * ein zweites Objekt liegt <= `ambiguity_margin` weiter    -> assisted
      * sonst genau ein naher Kandidat                           -> direct
    """
    gp = _to_int_point(gaze_point)
    if gaze_point is None or not detections:
        return SelectionResult(SelectionState.NONE, strategy, gaze_point=gp)

    scored = sorted(
        ((_distance(gaze_point, d.center), d) for d in detections),
        key=lambda t: t[0],
    )
    distances = [round(dist, 2) for dist, _ in scored]
    nearest_dist, nearest = scored[0]

    if nearest_dist >= direct_max_dist:
        return SelectionResult(
            SelectionState.NONE, strategy, gaze_point=gp, distances=distances
        )

    candidates = [d for dist, d in scored if dist <= nearest_dist + ambiguity_margin]
    if len(candidates) >= 2:
        return SelectionResult(
            SelectionState.ASSISTED,
            strategy,
            candidates=candidates,
            gaze_point=gp,
            distances=distances,
        )

    return SelectionResult(
        SelectionState.DIRECT,
        strategy,
        selected=nearest,
        candidates=[nearest],
        gaze_point=gp,
        distances=distances,
    )
