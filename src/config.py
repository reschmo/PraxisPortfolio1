"""
Zentrale Konfiguration: Schwellen, Modellnamen, Pfade, Farben.

Alle "magischen Zahlen" stehen an einer Stelle und sind leicht anpassbar.
Die Auswahl-Schwellen gibt es doppelt: als feste Pixelwerte (Fallback/Tests)
und normalisiert als Anteil der Bilddiagonale (auflösungsunabhängig, empfohlen).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TEST_IMAGE = ASSETS_DIR / "test.jpg"

# ---------------------------------------------------------------------------
# YOLO-Objektdetektion (Ultralytics, vortrainiert auf COCO – 80 Klassen)
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8m.pt"   # Modellname/-pfad; wird bei Bedarf autom. geladen
YOLO_IMGSZ = 960            # Inferenz-Bildgröße
YOLO_CONF = 0.35            # Konfidenzschwelle

# ---------------------------------------------------------------------------
# Auswahllogik – feste Startwerte in Pixel (Fallback/Defaults für Tests)
# ---------------------------------------------------------------------------
EXACT_DIRECT_MAX_DIST = 80.0    # nächstes Objektzentrum < 80 px -> Kandidat "direct"
EXACT_AMBIGUITY_MARGIN = 40.0   # zweites Objekt <= 40 px weiter -> "assisted"
COARSE_RADIUS = 90.0            # Radius des Aufmerksamkeitskreises (Fallback in px)

# ---------------------------------------------------------------------------
# Normalisierte Schwellen (Anteil der BILDDIAGONALE) -> auflösungsunabhängig.
# Empfohlen: Die Apps berechnen die Pixelwerte je Frame mit thresholds_for().
# So bleibt der "Suchbereich" bei jeder Auflösung optisch gleich groß.
# ---------------------------------------------------------------------------
EXACT_DIRECT_MAX_FRAC = 0.06    # ~ 80 px bei der Bilddiagonale des Testbilds
EXACT_AMBIGUITY_FRAC = 0.03     # ~ 40 px
COARSE_RADIUS_FRAC = 0.066      # ~ 90 px (halbierter Standard-Suchbereich)

# Webcam-Wunschauflösung (Live-Demo); real abhängig von der Kamera
CAM_WIDTH = 1280
CAM_HEIGHT = 720


def thresholds_for(width, height):
    """Pixelschwellen aus der Bilddiagonale ableiten (auflösungsunabhängig)."""
    diag = (float(width) ** 2 + float(height) ** 2) ** 0.5
    return {
        "direct_max": EXACT_DIRECT_MAX_FRAC * diag,
        "ambiguity": EXACT_AMBIGUITY_FRAC * diag,
        "radius": COARSE_RADIUS_FRAC * diag,
    }


# ---------------------------------------------------------------------------
# DL-Blickquelle (MobileGaze)
# Standard-Backend: uniface MobileGaze -> Gewichte werden automatisch geladen.
# GAZE_ONNX_MODEL wird nur für das optionale ONNX-Alt-Backend gebraucht.
# ---------------------------------------------------------------------------
GAZE_ONNX_MODEL = MODELS_DIR / "resnet18_gaze.onnx"  # optional (nur ONNX-Backend)
GAZE_SMOOTHING_ALPHA = 0.4      # EMA-Glättung des Blickpunkts (0 = aus … ~1 = träge)
CALIBRATION_FILE = MODELS_DIR / "calibration.npz"

# ---------------------------------------------------------------------------
# Visualisierung (Farben im BGR-Format, wie OpenCV sie erwartet)
# ---------------------------------------------------------------------------
COLOR_BOX = (0, 200, 0)          # Standard-Bounding-Box (grün)
COLOR_SELECTED = (0, 215, 255)   # ausgewähltes Objekt (gelb)
COLOR_CANDIDATE = (0, 140, 255)  # Kandidat bei "assisted" (orange)
COLOR_GAZE = (0, 0, 255)         # Blickpunkt (rot)
COLOR_GAZE_AREA = (255, 200, 0)  # Blickkreis (hellblau)
COLOR_TEXT = (255, 255, 255)     # Text
COLOR_BANNER_BG = (0, 0, 0)      # Hintergrund des Zustands-Banners

WINDOW_NAME = "Blickbasierte Objektauswahl"

# Farbe je Zustand (für Banner/Hervorhebung)
STATE_COLORS = {
    "direct": (0, 200, 0),
    "assisted": (0, 140, 255),
    "none": (60, 60, 60),
}
