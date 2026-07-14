"""
DL-Blickquelle auf Basis von yakhyo/MobileGaze (Appearance-based Gaze Estimation).

Pipeline pro Frame:
    Webcam-Frame
      -> RetinaFace (uniface)        : größtes Gesicht finden + zuschneiden
      -> GazeEstimatorONNX           : (yaw, pitch) in Radiant schätzen
      -> EMA-Glättung                : Zittern reduzieren
      -> Kalibrierung (Polynom-Fit)  : (yaw, pitch) -> (x, y) in Bildkoordinaten
    Ergebnis: gaze_point = (x, y)

Nur (x, y) verlässt dieses Modul. YOLO, Auswahllogik und Visualisierung bleiben
unverändert (austauschbare Blickquelle, siehe gaze/base.py).

Abhängigkeiten (dürfen in dieselbe venv wie YOLO; Python 3.10+):
    pip install "uniface[cpu]"        # bringt onnxruntime mit (GPU: uniface[gpu])

Standard-Backend ist uniface MobileGaze: die Modellgewichte werden beim ersten
Start automatisch geladen (Cache: ~/.uniface/models). Kein manueller Download.

Optionales Alt-Backend (GazeEstimatorONNX) für rohe yakhyo-ONNX-Gewichte ist
weiter enthalten (siehe README) und 1:1 an das Original angelehnt
(Gaze360: 90 Bins, Binbreite 4, Offset 180).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import GazePoint, GazeSource
from .. import config


# ===========================================================================
# 1) ONNX-Blickschätzer  (faithful port von yakhyo GazeEstimationONNX)
# ===========================================================================
class GazeEstimatorONNX:
    """Schätzt (yaw, pitch) in Radiant aus einem Gesichts-Crop (BGR-Bild)."""

    def __init__(self, model_path=None, session=None) -> None:
        import onnxruntime as ort

        self.session = session
        if self.session is None:
            if model_path is None:
                raise ValueError("model_path erforderlich (Pfad zur .onnx-Datei).")
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        # Dekodier-Parameter (Gaze360-Konfiguration wie im Original)
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        # ImageNet-Normalisierung
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        shape = input_cfg.shape
        try:
            # Modell-Eingabe ist NCHW -> (W, H) für cv2.resize
            self.input_size = (int(shape[3]), int(shape[2]))
        except (TypeError, ValueError):
            self.input_size = (448, 448)

        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self.input_mean) / self.input_std
        img = np.transpose(img, (2, 0, 1))                # HWC -> CHW
        return np.expand_dims(img, axis=0).astype(np.float32)  # -> BCHW

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _decode(self, yaw_logits: np.ndarray, pitch_logits: np.ndarray) -> Tuple[float, float]:
        yaw = np.sum(self._softmax(yaw_logits) * self.idx_tensor, axis=1)
        pitch = np.sum(self._softmax(pitch_logits) * self.idx_tensor, axis=1)
        yaw = yaw * self._binwidth - self._angle_offset
        pitch = pitch * self._binwidth - self._angle_offset
        return float(np.radians(yaw[0])), float(np.radians(pitch[0]))

    def estimate(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        """(yaw, pitch) in Radiant. yaw = horizontal, pitch = vertikal."""
        tensor = self._preprocess(face_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._decode(outputs[0], outputs[1])


# ===========================================================================
# 1b) Empfohlenes Backend: uniface MobileGaze (lädt Gewichte automatisch)
# ===========================================================================
class MobileGazeBackend:
    """Blickschätzer über uniface MobileGaze.

    Gibt (yaw, pitch) in Radiant zurück – gleiche Konvention wie
    GazeEstimatorONNX, aber deutlich einfacher: Die Gewichte werden beim ersten
    Aufruf automatisch geladen und zwischengespeichert.
    """

    def __init__(self, model=None) -> None:
        if model is None:
            from uniface.gaze import MobileGaze
            model = MobileGaze()
        self._gaze = model

    def estimate(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        result = self._gaze.estimate(face_bgr)  # GazeResult: .yaw, .pitch (Radiant)
        return float(result.yaw), float(result.pitch)


# ===========================================================================
# 2) Gesichtserkennung  (uniface RetinaFace) – größtes Gesicht
# ===========================================================================
def _import_retinaface():
    """Import robust gegen uniface-Versionen (neu: uniface.detection)."""
    try:
        from uniface.detection import RetinaFace
    except ImportError:  # ältere uniface-Versionen
        from uniface import RetinaFace
    return RetinaFace


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _extract_bboxes(faces) -> List[Tuple[float, float, float, float]]:
    """Robust gegen unterschiedliche uniface-Rückgabeformate.

    Unterstützt: (boxes, landmarks)-Tupel, numpy-Array Nx5 (x1,y1,x2,y2,score)
    und Iterables von Objekten mit `.bbox`.
    """
    boxes: List[Tuple[float, float, float, float]] = []
    if faces is None:
        return boxes
    if isinstance(faces, tuple) and len(faces) >= 1:  # (boxes, landmarks)
        faces = faces[0]
    if isinstance(faces, np.ndarray):
        for row in faces:
            if len(row) >= 4:
                boxes.append(tuple(float(v) for v in row[:4]))
        return boxes
    try:
        for f in faces:
            bbox = getattr(f, "bbox", f)
            boxes.append(tuple(float(v) for v in bbox[:4]))
    except TypeError:
        pass
    return boxes


class FaceDetector:
    """Dünner Wrapper um uniface RetinaFace; liefert den größten Gesichts-Crop."""

    def __init__(self, detector=None, confidence_threshold: float = 0.5) -> None:
        if detector is None:
            RetinaFace = _import_retinaface()
            detector = RetinaFace(confidence_threshold=confidence_threshold)
        self._detector = detector

    def largest_face_crop(self, frame_bgr: np.ndarray):
        faces = self._detector.detect(frame_bgr)
        boxes = _extract_bboxes(faces)
        if not boxes:
            return None, None
        x1, y1, x2, y2 = max(boxes, key=_bbox_area)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


# ===========================================================================
# 3) Kalibrierung: (yaw, pitch) -> (x, y) via Polynom-Regression (2. Grad)
# ===========================================================================
def _poly_features(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Merkmalsmatrix [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch]."""
    ones = np.ones_like(yaw)
    return np.stack([ones, yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch], axis=1)


def _head_features_from_bbox(bbox, width, height):
    """Kopf-Merkmale aus der Gesichts-Box: Position (zentriert) + Größe (~Nähe).

    hx/hy: -0.5 (links/oben) .. +0.5 (rechts/unten); hs: ~0 bei mittlerem Abstand.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hx = cx / width - 0.5
    hy = cy / height - 0.5
    hs = (x2 - x1) / width - 0.3
    return float(hx), float(hy), float(hs)


def _head_poly_features(yaw, pitch, hx, hy, hs):
    """13 Merkmale: Blick (Grad 2) + Kopfposition/-größe + wichtige Kreuzterme."""
    ones = np.ones_like(yaw)
    return np.stack([
        ones, yaw, pitch, yaw * yaw, pitch * pitch, yaw * pitch,
        hx, hy, hs, yaw * hx, pitch * hy, yaw * hs, pitch * hs,
    ], axis=1)


@dataclass
class GazeCalibration:
    """Bildet Blickwinkel (yaw, pitch) auf Bild-/Bildschirmkoordinaten ab.

    Kurze persönliche Kalibrierung: Der Nutzer schaut nacheinander auf bekannte
    Punkte; wir sammeln (yaw, pitch) -> (x, y) und passen zwei
    Least-Squares-Polynome (für x und y) an. Ein Polynom 2. Grades gleicht
    leichte Nichtlinearitäten aus, bleibt aber stabil.
    """

    coef_x: Optional[np.ndarray] = None
    coef_y: Optional[np.ndarray] = None
    n_points: int = 0  # Anzahl der beim Fit verwendeten Kalibrierpunkte

    requires_head = False  # Klassenattribut: braucht keine Kopf-Merkmale
    mode = "standard"

    @property
    def is_ready(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    def fit(
        self,
        angles: Sequence[Tuple[float, float]],
        points: Sequence[Tuple[float, float]],
    ) -> "GazeCalibration":
        angles = np.asarray(angles, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if angles.ndim != 2 or angles.shape[1] != 2 or len(angles) != len(points):
            raise ValueError("angles/points müssen jeweils Nx2 und gleich lang sein.")
        if len(angles) < 6:
            raise ValueError("Mind. 6 Kalibrierpunkte für das Polynom 2. Grades nötig.")
        feats = _poly_features(angles[:, 0], angles[:, 1])
        self.coef_x, *_ = np.linalg.lstsq(feats, points[:, 0], rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(feats, points[:, 1], rcond=None)
        self.n_points = int(len(angles))
        return self

    def transform(self, yaw: float, pitch: float) -> Tuple[int, int]:
        if not self.is_ready:
            raise RuntimeError("Kalibrierung nicht angepasst (fit zuerst aufrufen).")
        feats = _poly_features(np.array([yaw], dtype=float), np.array([pitch], dtype=float))[0]
        x = float(feats @ self.coef_x)
        y = float(feats @ self.coef_y)
        return int(round(x)), int(round(y))

    def save(self, path) -> None:
        np.savez(str(path), mode="standard", coef_x=self.coef_x, coef_y=self.coef_y,
                 n_points=self.n_points)

    @classmethod
    def load(cls, path) -> "GazeCalibration":
        data = np.load(str(path))
        n = int(data["n_points"]) if "n_points" in data.files else 0
        return cls(coef_x=data["coef_x"], coef_y=data["coef_y"], n_points=n)


@dataclass
class HeadAwareCalibration:
    """Kalibrierung, die zusätzlich Kopfposition/-größe berücksichtigt.

    Fit-Eingabe: Zeilen (yaw, pitch, hx, hy, hs) -> (x, y). Damit das etwas
    bringt, muss beim Kalibrieren der Kopf leicht bewegt werden – dann lernt das
    Modell, Kopfbewegung/Abstand zu kompensieren.
    """

    coef_x: Optional[np.ndarray] = None
    coef_y: Optional[np.ndarray] = None
    n_points: int = 0  # Anzahl der Ziel-Punkte (zur Beschriftung)

    requires_head = True
    mode = "head"

    @property
    def is_ready(self) -> bool:
        return self.coef_x is not None and self.coef_y is not None

    def fit(self, rows, points) -> "HeadAwareCalibration":
        rows = np.asarray(rows, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 5 or len(rows) != len(points):
            raise ValueError("rows müssen Nx5 (yaw,pitch,hx,hy,hs) und = len(points) sein.")
        if len(rows) < 13:
            raise ValueError("Zu wenige Proben für die kopf-aware Kalibrierung (>= 13).")
        feats = _head_poly_features(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])
        self.coef_x, *_ = np.linalg.lstsq(feats, points[:, 0], rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(feats, points[:, 1], rcond=None)
        return self

    def transform(self, yaw, pitch, hx, hy, hs):
        if not self.is_ready:
            raise RuntimeError("Kalibrierung nicht angepasst (fit zuerst aufrufen).")
        f = _head_poly_features(
            np.array([yaw], float), np.array([pitch], float),
            np.array([hx], float), np.array([hy], float), np.array([hs], float),
        )[0]
        x = float(f @ self.coef_x)
        y = float(f @ self.coef_y)
        return int(round(x)), int(round(y))

    def save(self, path) -> None:
        np.savez(str(path), mode="head", coef_x=self.coef_x, coef_y=self.coef_y,
                 n_points=self.n_points)

    @classmethod
    def load(cls, path) -> "HeadAwareCalibration":
        data = np.load(str(path))
        n = int(data["n_points"]) if "n_points" in data.files else 0
        return cls(coef_x=data["coef_x"], coef_y=data["coef_y"], n_points=n)


def load_calibration(path):
    """Lädt eine gespeicherte Kalibrierung und erkennt automatisch den Modus."""
    data = np.load(str(path))
    mode = str(data["mode"]) if "mode" in data.files else "standard"
    n = int(data["n_points"]) if "n_points" in data.files else 0
    if mode == "head":
        return HeadAwareCalibration(coef_x=data["coef_x"], coef_y=data["coef_y"], n_points=n)
    return GazeCalibration(coef_x=data["coef_x"], coef_y=data["coef_y"], n_points=n)


# ===========================================================================
# 4) Zusammenbau: DL-Blickquelle
# ===========================================================================
class DLGaze(GazeSource):
    """Blickquelle auf Basis eines vortrainierten DL-Modells (yakhyo/MobileGaze)."""

    def __init__(
        self,
        model_path=None,
        calibration: Optional[GazeCalibration] = None,
        smoothing_alpha: Optional[float] = None,
        face_detector: Optional[FaceDetector] = None,
        estimator=None,
    ) -> None:
        if estimator is not None:
            self.estimator = estimator
        elif model_path:
            self.estimator = GazeEstimatorONNX(model_path)   # optional: rohe ONNX-Gewichte
        else:
            self.estimator = MobileGazeBackend()             # Standard: uniface (auto-Download)
        self.face_detector = face_detector or FaceDetector()
        self.calibration = calibration or GazeCalibration()
        self.alpha = config.GAZE_SMOOTHING_ALPHA if smoothing_alpha is None else smoothing_alpha

        self._raw_angles: Optional[Tuple[float, float]] = None
        self._point: GazePoint = None
        self._smooth: Optional[Tuple[float, float]] = None
        self.last_face_bbox = None

    # --- Pro-Frame-Verarbeitung -------------------------------------------
    def process(self, frame_bgr):
        """Liefert ((yaw, pitch) oder None, face_bbox oder None) – OHNE Kalibrierung.

        Wird auch während der Kalibrierung genutzt, um Rohwinkel zu sammeln.
        """
        crop, bbox = self.face_detector.largest_face_crop(frame_bgr)
        self.last_face_bbox = bbox
        if crop is None or crop.size == 0:
            self._raw_angles = None
            return None, None
        yaw, pitch = self.estimator.estimate(crop)
        self._raw_angles = (yaw, pitch)
        return (yaw, pitch), bbox

    def _predict_from(self, angles, bbox, frame_shape):
        """Roher Blickpunkt aus Winkeln – berücksichtigt kopf-aware Kalibrierung."""
        if angles is None:
            return None
        if getattr(self.calibration, "requires_head", False):
            if bbox is None:
                return None
            h, w = frame_shape[:2]
            hx, hy, hs = _head_features_from_bbox(bbox, w, h)
            return self.calibration.transform(angles[0], angles[1], hx, hy, hs)
        return self.calibration.transform(angles[0], angles[1])

    def update(self, frame=None) -> None:
        if frame is None:
            return
        angles, bbox = self.process(frame)
        if angles is None:
            return
        if not self.calibration.is_ready:
            # Ohne Kalibrierung kein sinnvoller Bildpunkt (nur Rohwinkel verfügbar).
            self._point = None
            return
        point = self._predict_from(angles, bbox, frame.shape)
        if point is None:
            self._point = None
            return
        self._point = self._apply_smoothing(point[0], point[1])

    def _apply_smoothing(self, x, y) -> Tuple[int, int]:
        if self.alpha <= 0 or self._smooth is None:
            self._smooth = (float(x), float(y))
        else:
            sx, sy = self._smooth
            a = self.alpha
            self._smooth = (a * sx + (1 - a) * x, a * sy + (1 - a) * y)
        return int(round(self._smooth[0])), int(round(self._smooth[1]))

    def get_gaze_point(self) -> GazePoint:
        return self._point

    @property
    def raw_angles(self) -> Optional[Tuple[float, float]]:
        return self._raw_angles


# ===========================================================================
# 5) Interaktive Kalibrierung (braucht Kamera + Fenster; nur bei Live-Nutzung)
# ===========================================================================
def default_calibration_targets(width, height, grid=3, margin_ratio=0.12):
    """grid×grid-Raster von Zielpunkten über das Bild verteilt.

    grid=3 -> 9 Punkte, grid=4 -> 16, grid=5 -> 25.
    """
    grid = max(2, int(grid))
    mx, my = int(width * margin_ratio), int(height * margin_ratio)
    span_x = width - 1 - 2 * mx
    span_y = height - 1 - 2 * my
    xs = [int(round(mx + span_x * i / (grid - 1))) for i in range(grid)]
    ys = [int(round(my + span_y * j / (grid - 1))) for j in range(grid)]
    return [(x, y) for y in ys for x in xs]


def _draw_target(img, point, index, total, progress=0.0, done=False):
    """Kalibrier-/Messpunkt mit Fortschritts-Ring.

    progress (0..1) füllt den Ring; done=True zeigt ihn voll und grün (Bestätigung).
    Reine Anzeige – hat keinen Einfluss auf die aufgenommenen Daten.
    """
    import cv2

    x, y = int(point[0]), int(point[1])
    radius = 26
    color = (0, 200, 0) if done else (0, 0, 255)  # grün = fertig, sonst rot
    prog = max(0.0, min(1.0, progress))

    cv2.circle(img, (x, y), radius, (70, 70, 70), 2, cv2.LINE_AA)          # Hintergrund-Ring
    if prog > 0.0:
        cv2.ellipse(img, (x, y), (radius, radius), 0, -90, -90 + int(360 * prog),
                    color, 4, cv2.LINE_AA)                                 # Füll-Bogen
    cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)                     # Mittelpunkt

    cv2.putText(img, f"Kalibrierung {index}/{total} - Punkt anschauen",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run_interactive_calibration(
    dl_gaze: DLGaze,
    cap,
    window_name: str,
    targets=None,
    grid: int = 3,
    samples_per_target: int = 15,
    settle_frames: int = 8,
    flip: bool = True,
    head_aware: bool = False,
):
    """Kalibrierung; passt `dl_gaze.calibration` an.

    head_aware=False: Standard, (yaw,pitch) -> (x,y), ein gemittelter Wert je Ziel.
    head_aware=True : zusätzlich Kopfposition/-größe. Pro Ziel viele Proben,
                      dabei den Kopf leicht bewegen -> das Modell lernt, Kopf-
                      bewegung/Abstand zu kompensieren.
    """
    import cv2

    ok, frame = cap.read()
    if not ok:
        raise IOError("Keine Frames von der Kamera für die Kalibrierung.")
    if flip:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    targets = targets or default_calibration_targets(w, h, grid=grid)
    if head_aware:
        samples_per_target = max(samples_per_target, 30)

    std_angles: List[Tuple[float, float]] = []   # gemittelt je Ziel (Standard)
    std_points: List[Tuple[int, int]] = []
    head_rows = []                               # (yaw,pitch,hx,hy,hs) je Frame
    head_points = []

    move_hint = "Punkt anschauen + Kopf langsam bewegen" if head_aware else "Punkt anschauen"
    for i, (tx, ty) in enumerate(targets, start=1):
        samples = []
        settle = settle_frames
        while len(samples) < samples_per_target:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            angles, bbox = dl_gaze.process(frame)

            board = frame.copy()
            _draw_target(board, (tx, ty), i, len(targets),
                         progress=len(samples) / samples_per_target)
            cv2.putText(board, move_hint, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, board)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt("Kalibrierung durch Nutzer abgebrochen.")

            if angles is None:
                continue
            if settle > 0:  # kurze Einschwingzeit pro Punkt
                settle -= 1
                continue

            if head_aware:
                if bbox is None:
                    continue
                hx, hy, hs = _head_features_from_bbox(bbox, w, h)
                head_rows.append((angles[0], angles[1], hx, hy, hs))
                head_points.append((tx, ty))
            samples.append(angles)

        # Ring voll -> kurz grün bestätigen, dann nächster Punkt
        if samples:
            board = frame.copy()
            _draw_target(board, (tx, ty), i, len(targets), progress=1.0, done=True)
            cv2.imshow(window_name, board)
            cv2.waitKey(180)

        if samples and not head_aware:
            mean_angles = tuple(np.asarray(samples, dtype=np.float64).mean(axis=0))
            std_angles.append(mean_angles)
            std_points.append((tx, ty))

    if head_aware:
        if len(head_rows) < 13:
            raise RuntimeError("Zu wenige Proben (Gesicht sichtbar? Kopf bewegt?).")
        calib = HeadAwareCalibration().fit(head_rows, head_points)
        calib.n_points = len(targets)
        dl_gaze.calibration = calib
    else:
        if len(std_points) < 6:
            raise RuntimeError(
                "Zu wenige gültige Kalibrierpunkte (war das Gesicht durchgehend sichtbar?)."
            )
        calib = GazeCalibration().fit(std_angles, std_points)
        calib.n_points = len(std_points)
        dl_gaze.calibration = calib
    return dl_gaze.calibration


# ===========================================================================
# 6) Genauigkeits-Messung (Validierung auf UNABHÄNGIGEN Testpunkten)
# ===========================================================================
def _validation_targets(width, height):
    """Testpunkte an anderen Stellen als die Kalibrierung (3x3, größerer Rand)."""
    return default_calibration_targets(width, height, grid=3, margin_ratio=0.25)


def rate_calibration(mean_error_pct_diag: float) -> str:
    """Einfache Qualitätsbewertung aus dem mittleren Blickfehler (% der Bilddiagonale):
    <= 4 % -> 'gut', 4-6 % -> 'mittel', > 6 % -> 'schlecht'.
    """
    if mean_error_pct_diag <= 4.0:
        return "gut"
    if mean_error_pct_diag <= 6.0:
        return "mittel"
    return "schlecht"


def run_validation(
    dl_gaze: DLGaze,
    cap,
    window_name: str,
    targets=None,
    samples_per_target: int = 12,
    settle_frames: int = 8,
    flip: bool = True,
):
    """Misst die Blickgenauigkeit auf unabhängigen Testpunkten.

    Der Nutzer schaut nacheinander auf bekannte Punkte; wir vergleichen den
    geschätzten Blickpunkt mit der wahren Position. Rückgabe: dict mit
    mittlerem Fehler (px, % der Bilddiagonale) usw. – objektiver Beleg.
    """
    import cv2

    if not dl_gaze.calibration.is_ready:
        raise RuntimeError("Erst kalibrieren, dann messen.")
    ok, frame = cap.read()
    if not ok:
        raise IOError("Keine Frames von der Kamera für die Messung.")
    if flip:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    diag = (w * w + h * h) ** 0.5
    targets = targets or _validation_targets(w, h)

    per_point = []
    errors = []
    for i, (tx, ty) in enumerate(targets, start=1):
        preds = []
        settle = settle_frames
        while len(preds) < samples_per_target:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            angles, bbox = dl_gaze.process(frame)

            board = frame.copy()
            _draw_target(board, (tx, ty), i, len(targets),
                         progress=len(preds) / samples_per_target)
            cv2.putText(board, "MESSUNG - Punkt anschauen", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, board)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise KeyboardInterrupt("Messung durch Nutzer abgebrochen.")

            if angles is None:
                continue
            if settle > 0:
                settle -= 1
                continue
            point = dl_gaze._predict_from(angles, bbox, frame.shape)
            if point is None:
                continue
            preds.append(point)

        if preds:
            px = float(np.mean([p[0] for p in preds]))
            py = float(np.mean([p[1] for p in preds]))
            err = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
            per_point.append(((tx, ty), (round(px, 1), round(py, 1)), round(err, 1)))
            errors.append(err)

    if not errors:
        raise RuntimeError("Keine gültigen Messpunkte (war das Gesicht sichtbar?).")
    errors = np.asarray(errors, dtype=float)
    mean_px = float(errors.mean())
    mean_pct = float(mean_px / diag * 100.0)
    return {
        "mode": getattr(dl_gaze.calibration, "mode", "standard"),
        "n_calib_points": dl_gaze.calibration.n_points,
        "n_test_points": int(len(errors)),
        "mean_error_px": mean_px,
        "std_error_px": float(errors.std()),
        "max_error_px": float(errors.max()),      # größter Einzelfehler (Ausreißer-Check)
        "mean_error_pct_diag": mean_pct,
        "rating": rate_calibration(mean_pct),     # gut / mittel / schlecht
        "frame_size": (w, h),
        "per_point": per_point,
    }
