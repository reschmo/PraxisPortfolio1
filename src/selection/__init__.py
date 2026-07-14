"""Auswahllogik: nimmt Blickpunkt + Detektionen -> Zustand direct/assisted/none."""
from .result import SelectionResult, SelectionState
from .exact import select_object
from .coarse import select_with_attention_area

__all__ = [
    "SelectionResult",
    "SelectionState",
    "select_object",
    "select_with_attention_area",
]
