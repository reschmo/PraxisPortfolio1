"""
Standbild-Demo: YOLO auf einem Bild + Blickquelle (Maus ODER Eye-Tracking) + Auswahllogik.

Kombinationen:
  Bild + Maus:          python -m src.app_image --image assets/test1.png
  Bild + Eye-Tracking:  python -m src.app_image --image assets/test1.png --gaze dl
      (nutzt die zuvor im Live-Modus gespeicherte Kalibrierung; die Webcam liefert
       den Blick, angezeigt wird das Standbild.)

Tasten: e=Exact  c=Coarse  a=Kreis  +/-=Suchbereich  q=Quit

Headless (ohne Fenster; fixer Blickpunkt, z. B. für Tests/Doku-Bilder):
  python -m src.app_image --image assets/test1.png --headless --point 250,650 \
      --strategy coarse --out out.jpg
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from . import config
from .detection.yolo import ObjectDetector
from .gaze.mouse_gaze import MouseGaze
from .selection.coarse import select_with_attention_area
from .selection.exact import select_object
from .viz.draw import draw_scene


def run_selection(strategy, gaze_point, detections, th, radius):
    if strategy == "coarse":
        return select_with_attention_area(gaze_point, detections, radius=radius)
    return select_object(gaze_point, detections,
                         direct_max_dist=th["direct_max"], ambiguity_margin=th["ambiguity"])


def parse_args():
    p = argparse.ArgumentParser(description="Standbild-Demo: blickbasierte Objektauswahl")
    p.add_argument("--image", default=str(config.DEFAULT_TEST_IMAGE), help="Pfad zum Bild")
    p.add_argument("--gaze", choices=["mouse", "dl"], default="mouse",
                   help="Blickquelle: mouse (Referenz) oder dl (Eye-Tracking: Webcam + Kalibrierung)")
    p.add_argument("--strategy", choices=["exact", "coarse"], default="exact")
    p.add_argument("--model", default=None, help="Optional: ONNX-Gewichte der DL-Blickquelle")
    p.add_argument("--source", default="0", help="Kamera-Index für --gaze dl")
    p.add_argument("--headless", action="store_true", help="Ohne Fenster rendern")
    p.add_argument("--point", default=None, help="Headless: fixer Blickpunkt 'x,y'")
    p.add_argument("--out", default="out.jpg", help="Headless: Ausgabedatei")
    return p.parse_args()


def _load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden/lesbar: {path}")
    return img


def _letterbox(img, target_w, target_h):
    """Bild auf Zielgröße einpassen (Seitenverhältnis erhalten, Rand schwarz auffüllen)."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    ox, oy = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def run_headless(args, detector=None):
    """Rendert ein annotiertes Bild für einen fixen Blickpunkt (kein Fenster nötig)."""
    img = _load_image(args.image)
    detector = detector or ObjectDetector()
    detections = detector.detect(img)
    h, w = img.shape[:2]
    th = config.thresholds_for(w, h)
    if args.point:
        gx, gy = (int(v) for v in args.point.split(","))
        gaze_point = (gx, gy)
    else:
        gaze_point = (w // 2, h // 2)
    result = run_selection(args.strategy, gaze_point, detections, th, th["radius"])
    canvas = draw_scene(img, detections, result, show_area=(args.strategy == "coarse"),
                        radius=th["radius"])
    cv2.imwrite(args.out, canvas)
    print(f"{len(detections)} Objekte | {args.strategy} | Zustand={result.state} -> {args.out}")
    return result


def _interactive_loop(display, detections, th, gaze, capture, strategy):
    """Gemeinsame Anzeige-/Tastenschleife für Bild-Demos (Maus oder DL).

    capture=None  -> Maus-Modus (kein Webcam-Read).
    capture=cv2.VideoCapture -> DL-Modus (Webcam liefert den Blick).
    """
    show_area = strategy == "coarse"
    radius_scale = 1.0
    hint = "e=Exact c=Coarse a=Kreis +/-=Bereich q=Quit"
    while True:
        if capture is not None:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gaze.update(frame)  # DL-Inferenz auf dem Webcam-Frame

        gaze_point = gaze.get_gaze_point()
        radius = th["radius"] * radius_scale
        result = run_selection(strategy, gaze_point, detections, th, radius)
        canvas = draw_scene(display, detections, result, show_area=show_area,
                            radius=radius, hint=hint)
        if strategy == "coarse":
            cv2.putText(canvas, f"Suchbereich: {int(radius)} px", (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_GAZE_AREA, 2, cv2.LINE_AA)
        if capture is not None:  # DL: Gesichtsbox als Feedback
            bbox = getattr(gaze, "last_face_bbox", None)
            if bbox:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 1)

        cv2.imshow(config.WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            strategy, show_area = "exact", False
        elif key == ord("c"):
            strategy, show_area = "coarse", True
        elif key == ord("a"):
            show_area = not show_area
        elif key in (ord("+"), ord("=")):
            radius_scale = min(3.0, radius_scale * 1.1)
        elif key == ord("-"):
            radius_scale = max(0.2, radius_scale / 1.1)

    cv2.destroyAllWindows()


def run_interactive_mouse(args):
    """Bild + Maus (Referenz)."""
    img = _load_image(args.image)
    detector = ObjectDetector()
    detections = detector.detect(img)
    h, w = img.shape[:2]
    th = config.thresholds_for(w, h)

    cv2.namedWindow(config.WINDOW_NAME)
    gaze = MouseGaze(config.WINDOW_NAME, initial=(w // 2, h // 2))
    _interactive_loop(img, detections, th, gaze, None, args.strategy)


def run_interactive_dl(args):
    """Bild + Eye-Tracking. Nutzt die im Live-Modus gespeicherte Kalibrierung."""
    from .gaze.dl_gaze import DLGaze, load_calibration

    if not config.CALIBRATION_FILE.exists():
        raise RuntimeError(
            "Keine Kalibrierung gefunden. Bitte zuerst im Live-Modus kalibrieren:\n"
            "  python -m src.app_live --source 0 --gaze dl   (dann Taste 'k')"
        )

    # Bild auf die Kalibrier-Koordinaten (Webcam-/Fenstergröße) bringen, damit der
    # kalibrierte Blickpunkt exakt auf das angezeigte Bild passt.
    w, h = config.CAM_WIDTH, config.CAM_HEIGHT
    display = _letterbox(_load_image(args.image), w, h)
    detector = ObjectDetector()
    detections = detector.detect(display)
    th = config.thresholds_for(w, h)

    gaze = DLGaze(model_path=args.model)
    gaze.calibration = load_calibration(config.CALIBRATION_FILE)
    print(f"Kalibrierung geladen ({gaze.calibration.mode}, {gaze.calibration.n_points} Punkte).")

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise IOError("Webcam nicht zu öffnen (wird für die Blickschätzung gebraucht).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, w, h)
    try:
        _interactive_loop(display, detections, th, gaze, cap, args.strategy)
    finally:
        cap.release()
        gaze.close()


def main():
    args = parse_args()
    if args.headless:
        run_headless(args)
    elif args.gaze == "dl":
        run_interactive_dl(args)
    else:
        run_interactive_mouse(args)


if __name__ == "__main__":
    main()
