"""Einheitliches, YOLO-unabhängiges Detektionsformat.

So bleibt die Auswahllogik unabhängig von der konkreten YOLO-Ausgabe:
sie braucht nur `center` und `bbox`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    """Eine erkannte Objektinstanz in einheitlichem Format."""

    class_id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]              # (cx, cy)

    @classmethod
    def from_xyxy(cls, class_id, label, confidence, x1, y1, x2, y2) -> "Detection":
        """Baut eine Detection aus Eckkoordinaten; Zentrum wird berechnet."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cls(
            int(class_id),
            str(label),
            float(confidence),
            (float(x1), float(y1), float(x2), float(y2)),
            (cx, cy),
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
