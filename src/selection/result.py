"""Gemeinsame Ergebnis- und Zustandstypen der Auswahllogik."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class SelectionState(str, Enum):
    """Die drei möglichen Zustände einer Auswahl."""

    DIRECT = "direct"      # genau ein Objekt eindeutig -> auswählen
    ASSISTED = "assisted"  # mehrere plausibel -> Kandidaten anzeigen (nicht autom. entscheiden)
    NONE = "none"          # kein ausreichend plausibles Objekt

    def __str__(self) -> str:
        return self.value


@dataclass
class SelectionResult:
    """Ergebnis eines Auswahlschritts.

    - direct   : `selected` ist gesetzt (und in `candidates` enthalten).
    - assisted : `candidates` enthält mehrere Objekte.
    - none     : weder `selected` noch `candidates`.
    """

    state: SelectionState
    strategy: str
    selected: Optional[Any] = None
    candidates: List[Any] = field(default_factory=list)
    gaze_point: Optional[Tuple[int, int]] = None
    distances: Optional[List[float]] = None  # Debug: sortierte Distanzen (Exact)

    @property
    def is_direct(self) -> bool:
        return self.state == SelectionState.DIRECT

    @property
    def is_assisted(self) -> bool:
        return self.state == SelectionState.ASSISTED

    @property
    def is_none(self) -> bool:
        return self.state == SelectionState.NONE
