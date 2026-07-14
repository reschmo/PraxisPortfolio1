"""Austauschbare Blickquellen.

Nur die leichten Quellen (Interface, Maus) werden hier direkt importiert.
Die DL-Blickquelle (`dl_gaze`) wird bewusst NICHT hier importiert, da sie
zusätzliche Pakete (onnxruntime, uniface) braucht -> lazy import beim Nutzen.
"""
from .base import GazeSource
from .mouse_gaze import MouseGaze

__all__ = ["GazeSource", "MouseGaze"]
