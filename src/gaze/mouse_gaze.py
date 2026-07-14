"""Maus als Blickquelle – kontrollierte Referenz zum Testen der Auswahllogik.

Damit lässt sich die gesamte Pipeline (YOLO -> Auswahl -> Visualisierung)
prüfen, ohne dass ein Blickmodell nötig ist.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2

from .base import GazePoint, GazeSource


class MouseGaze(GazeSource):
    """Nutzt die Mausposition in einem OpenCV-Fenster als "Blickpunkt".

    Das Fenster muss vor der Instanziierung existieren
    (`cv2.namedWindow(window_name)`), da der Maus-Callback daran hängt.
    """

    def __init__(self, window_name: str, initial: Optional[Tuple[int, int]] = None) -> None:
        self.window_name = window_name
        self._point: GazePoint = initial
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        # Bei Bewegung folgt der "Blick" der Maus; Linksklick setzt ihn ebenfalls.
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self._point = (int(x), int(y))

    def get_gaze_point(self) -> GazePoint:
        return self._point
