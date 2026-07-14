"""YOLO-Objektdetektion (Ultralytics YOLOv8) -> einheitliches Detection-Format.

Ultralytics wird bewusst *lazy* importiert, damit der Rest des Projekts
(Auswahllogik, Tests, Visualisierung) auch ohne installiertes PyTorch nutzbar
bleibt.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from .types import Detection
from .. import config


class ObjectDetector:
    """Dünner Wrapper um Ultralytics YOLO.

    Lädt das Modell einmalig und liefert pro Bild eine Liste von `Detection`.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import (benötigt PyTorch)

        self.model_path = str(model_path or config.YOLO_MODEL)
        self.imgsz = int(imgsz or config.YOLO_IMGSZ)
        self.conf = float(conf if conf is not None else config.YOLO_CONF)
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names  # dict: id -> label

    def detect(self, image) -> List[Detection]:
        """Führt Inferenz auf einem BGR-Bild (numpy-Array) aus."""
        results = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False
        )
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            label = self._label_for(class_id)
            detections.append(
                Detection.from_xyxy(class_id, label, confidence, x1, y1, x2, y2)
            )
        return detections

    def _label_for(self, class_id: int) -> str:
        names = self.class_names
        if isinstance(names, dict):
            return names.get(class_id, str(class_id))
        try:
            return names[class_id]
        except (IndexError, KeyError, TypeError):
            return str(class_id)
