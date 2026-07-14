"""
Livebild-Demo: Webcam + YOLO + Blickquelle (Maus oder DL) + Auswahllogik.

Beispiele:
    # Maus als Blickquelle (kein Blickmodell nötig)
    python -m src.app_live --source 0 --gaze mouse

    # DL-Blickquelle (MobileGaze, lädt Gewichte automatisch) + Kalibrierung
    python -m src.app_live --source 0 --gaze dl

Tasten: e=Exact  c=Coarse  a=Kreis  +/-=Suchbereich  k=Kalibrieren (nur dl)  q=Quit

Aus Performance-Gründen läuft YOLO nur jedes n-te Frame (--yolo-every),
der Blick wird jedes Frame aktualisiert.
"""
from __future__ import annotations

import argparse

import cv2

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
    p = argparse.ArgumentParser(description="Livebild-Demo: blickbasierte Objektauswahl")
    p.add_argument("--source", default="0", help="Kamera-Index (z. B. 0) oder Videopfad")
    p.add_argument("--gaze", choices=["mouse", "dl"], default="mouse")
    p.add_argument("--strategy", choices=["exact", "coarse"], default="coarse")
    p.add_argument("--model", default=None,
                   help="Optional: Pfad zu rohen ONNX-Gewichten. "
                        "Standard: uniface MobileGaze (lädt automatisch).")
    p.add_argument("--yolo-every", type=int, default=3,
                   help="YOLO nur jedes n-te Frame ausführen (Performance)")
    p.add_argument("--calib-grid", type=int, default=3,
                   help="Kalibrier-Raster NxN (3=9, 4=16, 5=25 Punkte)")
    p.add_argument("--head-aware", action="store_true",
                   help="Kopf-aware Kalibrierung (Kopfposition/-größe als Zusatz-Eingabe)")
    p.add_argument("--samples", type=int, default=15,
                   help="Samples pro Punkt bei Kalibrierung UND Messung (Standard 15)")
    return p.parse_args()


def _open_capture(source):
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        raise IOError(f"Kamera/Video konnte nicht geöffnet werden: {source}")
    if is_webcam:  # höhere Auflösung anfragen (Kamera entscheidet, was möglich ist)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    return cap, is_webcam


def _make_gaze_source(kind, window_name, model_path):
    if kind == "dl":
        from .gaze.dl_gaze import DLGaze, load_calibration  # lazy: uniface nur bei Bedarf
        source = DLGaze(model_path=model_path)
        # Falls schon einmal kalibriert wurde: gespeicherte Kalibrierung laden.
        if config.CALIBRATION_FILE.exists():
            try:
                source.calibration = load_calibration(config.CALIBRATION_FILE)
                print(f"Gespeicherte Kalibrierung geladen "
                      f"({source.calibration.mode}, {source.calibration.n_points} Punkte): "
                      f"{config.CALIBRATION_FILE}")
            except Exception as e:
                print(f"Kalibrierung konnte nicht geladen werden: {e}")
        return source
    return MouseGaze(window_name)


def _dl_status_overlay(canvas, gaze):
    """Bei DL-Blick: Gesichtsbox zeigen und – falls unkalibriert – Hinweis einblenden."""
    bbox = getattr(gaze, "last_face_bbox", None)
    if bbox:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 1)
    calib = getattr(gaze, "calibration", None)
    if calib is None or not calib.is_ready:
        cv2.putText(canvas, "DL-Blick aktiv - 'k' druecken zum Kalibrieren",
                    (16, canvas.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)


def _calibrate(gaze, cap, is_webcam, grid, head_aware, samples):
    from .gaze.dl_gaze import run_interactive_calibration
    try:
        run_interactive_calibration(gaze, cap, config.WINDOW_NAME, grid=grid,
                                    samples_per_target=samples,
                                    head_aware=head_aware, flip=is_webcam)
        gaze.calibration.save(config.CALIBRATION_FILE)
        modus = "kopf-aware" if head_aware else "standard"
        print(f"Kalibrierung ({grid}x{grid} = {grid * grid} Punkte, {modus}, "
              f"{samples} Samples/Punkt) gespeichert -> {config.CALIBRATION_FILE}")
    except Exception as e:  # Demo bei Abbruch nicht abstürzen lassen
        print(f"Kalibrierung abgebrochen: {e}")


def _validate(gaze, cap, is_webcam, samples):
    """Misst die Genauigkeit und protokolliert sie in eine CSV (für die Doku)."""
    import csv
    import time
    from .gaze.dl_gaze import run_validation
    try:
        res = run_validation(gaze, cap, config.WINDOW_NAME, samples_per_target=samples,
                             flip=is_webcam)
    except Exception as e:
        print(f"Messung abgebrochen: {e}")
        return
    print("\n=== Genauigkeits-Messung ===")
    print(f"Modus: {res['mode']} | Kalibrierpunkte: {res['n_calib_points']} | "
          f"Samples/Punkt: {samples} | Testpunkte: {res['n_test_points']}")
    print(f"Mittlerer Fehler: {res['mean_error_px']:.1f} px "
          f"(+/- {res['std_error_px']:.1f}, max {res['max_error_px']:.1f}) "
          f"= {res['mean_error_pct_diag']:.1f}% der Bilddiagonale  ->  Bewertung: {res['rating'].upper()}")
    csv_path = config.MODELS_DIR / "calibration_eval.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["zeit", "modus", "kalibrierpunkte", "samples_pro_punkt", "testpunkte",
                             "mittel_fehler_px", "std_px", "fehler_pct_diagonale", "max_fehler_px",
                             "bewertung", "frame_w", "frame_h"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), res["mode"], res["n_calib_points"],
                         samples, res["n_test_points"], round(res["mean_error_px"], 1),
                         round(res["std_error_px"], 1), round(res["mean_error_pct_diag"], 2),
                         round(res["max_error_px"], 1), res["rating"],
                         res["frame_size"][0], res["frame_size"][1]])
    print(f"Ergebnis protokolliert -> {csv_path}\n")


def main():
    args = parse_args()
    cap, is_webcam = _open_capture(args.source)

    detector = ObjectDetector()
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)  # Fenster frei skalierbar
    cv2.resizeWindow(config.WINDOW_NAME, config.CAM_WIDTH, config.CAM_HEIGHT)
    gaze = _make_gaze_source(args.gaze, config.WINDOW_NAME, args.model)

    strategy = args.strategy
    show_area = strategy == "coarse"
    detections = []
    frame_idx = 0
    radius_scale = 1.0  # per +/- live veränderbar
    hint = ("e=Exact c=Coarse a=Kreis +/-=Bereich "
            + ("k=Kalibrieren v=Messen " if args.gaze == "dl" else "") + "q=Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_webcam:
            frame = cv2.flip(frame, 1)

        if frame_idx % max(1, args.yolo_every) == 0:
            detections = detector.detect(frame)
        frame_idx += 1

        gaze.update(frame)  # DL: Inferenz auf dem Frame; Maus: No-Op
        gaze_point = gaze.get_gaze_point()
        th = config.thresholds_for(frame.shape[1], frame.shape[0])
        radius = th["radius"] * radius_scale
        result = run_selection(strategy, gaze_point, detections, th, radius)
        canvas = draw_scene(frame, detections, result, show_area=show_area,
                            radius=radius, hint=hint)
        if strategy == "coarse":
            cv2.putText(canvas, f"Suchbereich: {int(radius)} px", (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_GAZE_AREA, 2, cv2.LINE_AA)
        if args.gaze == "dl":
            _dl_status_overlay(canvas, gaze)
        cv2.imshow(config.WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            strategy, show_area = "exact", False
        elif key == ord("c"):
            strategy, show_area = "coarse", True
        elif key == ord("a"):
            show_area = not show_area
        elif key in (ord("+"), ord("=")):     # Suchbereich vergrößern
            radius_scale = min(3.0, radius_scale * 1.1)
        elif key == ord("-"):                 # Suchbereich verkleinern
            radius_scale = max(0.2, radius_scale / 1.1)
        elif key == ord("k") and args.gaze == "dl":
            _calibrate(gaze, cap, is_webcam, args.calib_grid, args.head_aware, args.samples)
        elif key == ord("v") and args.gaze == "dl":
            _validate(gaze, cap, is_webcam, args.samples)

    cap.release()
    gaze.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
