"""Interface für austauschbare Blickquellen.

Kernidee des Projekts: Eine Blickquelle liefert *nur* einen Punkt
`gaze_point = (x, y)`. Ob dieser von der Maus, einem DL-Modell oder später
einem Eye-Tracker stammt, ist für den Rest der Pipeline egal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

GazePoint = Optional[Tuple[int, int]]


class GazeSource(ABC):
    """Abstrakte Blickquelle."""

    @abstractmethod
    def get_gaze_point(self) -> GazePoint:
        """Aktueller Blickpunkt (x, y) in Bildkoordinaten – oder None."""
        raise NotImplementedError

    def update(self, frame=None) -> None:
        """Optionaler Pro-Frame-Hook (z. B. DL-Inferenz auf dem aktuellen Frame).

        Für einfache Quellen (Maus) ein No-Op.
        """

    def close(self) -> None:
        """Ressourcen freigeben (Kamera, Modell etc.)."""

    def __enter__(self) -> "GazeSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
