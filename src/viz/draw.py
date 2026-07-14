"""Visualisierung: Bounding Boxes, Blickpunkt/-bereich, Auswahl, Zustand, Legende."""
from __future__ import annotations

from typing import Optional

import cv2

from .. import config
from ..selection.result import SelectionResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(img, detections, color=config.COLOR_BOX, thickness=2) -> None:
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        _label(img, f"{d.label} {d.confidence:.2f}", (x1, y1), color)


def _label(img, text, org, color) -> None:
    x, y = org
    (tw, th), base = cv2.getTextSize(text, _FONT, 0.5, 1)
    y_top = max(0, y - th - base - 2)
    cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + base + 2), color, -1)
    cv2.putText(img, text, (x + 2, y_top + th + 1), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def highlight(img, detection, color, thickness=3) -> None:
    x1, y1, x2, y2 = (int(v) for v in detection.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_gaze_point(img, point, color=config.COLOR_GAZE) -> None:
    """Der eigentliche Blickpunkt – prominent, mit weißem Kontrastring."""
    if point is None:
        return
    x, y = int(point[0]), int(point[1])
    cv2.circle(img, (x, y), 11, (255, 255, 255), 2)          # weißer Ring (Kontrast)
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (x, y), 4, color, -1)


def draw_gaze_area(img, point, radius, color=config.COLOR_GAZE_AREA) -> None:
    """Coarse-Gaze-Suchbereich als transparente Fläche mit Rand + Beschriftung."""
    if point is None:
        return
    x, y, r = int(point[0]), int(point[1]), int(radius)
    overlay = img.copy()
    cv2.circle(overlay, (x, y), r, color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)        # transparente Füllung
    cv2.circle(img, (x, y), r, color, 2)                     # Rand
    ly = max(14, y - r - 6)
    cv2.putText(img, "Suchbereich", (max(2, x - r), ly), _FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_state_banner(img, result: SelectionResult) -> None:
    state = str(result.state)
    color = config.STATE_COLORS.get(state, (60, 60, 60))
    text = f"[{result.strategy}]  Zustand: {state.upper()}"
    if result.is_direct and result.selected is not None:
        text += f"  ->  {result.selected.label}"
    elif result.is_assisted:
        text += f"  ->  {len(result.candidates)} Kandidaten"
    _banner(img, text, color)


def _banner(img, text, color) -> None:
    h, w = img.shape[:2]
    bar_h = 34
    cv2.rectangle(img, (0, 0), (w, bar_h), config.COLOR_BANNER_BG, -1)
    cv2.rectangle(img, (0, 0), (10, bar_h), color, -1)
    cv2.putText(img, text, (18, 23), _FONT, 0.6, config.COLOR_TEXT, 2, cv2.LINE_AA)


def draw_legend(img) -> None:
    """Kompakte Farb-Legende unten rechts."""
    items = [
        ("Blickpunkt", config.COLOR_GAZE),
        ("Suchbereich", config.COLOR_GAZE_AREA),
        ("Auswahl", config.COLOR_SELECTED),
        ("Kandidat", config.COLOR_CANDIDATE),
        ("Objekt", config.COLOR_BOX),
    ]
    h, w = img.shape[:2]
    line_h, pad, sw = 20, 8, 14
    box_w = 150
    box_h = pad * 2 + line_h * len(items)
    x0, y0 = w - box_w - 8, h - box_h - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    for i, (name, color) in enumerate(items):
        cy = y0 + pad + i * line_h + line_h // 2
        cv2.rectangle(img, (x0 + pad, cy - sw // 2), (x0 + pad + sw, cy + sw // 2), color, -1)
        cv2.putText(img, name, (x0 + pad + sw + 8, cy + 5), _FONT, 0.45,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)


def _hint(img, text) -> None:
    h, w = img.shape[:2]
    cv2.putText(img, text, (10, h - 12), _FONT, 0.5, config.COLOR_TEXT, 1, cv2.LINE_AA)


def draw_scene(
    img,
    detections,
    result: SelectionResult,
    show_area: bool = False,
    radius: float = config.COARSE_RADIUS,
    hint: Optional[str] = None,
    show_legend: bool = True,
):
    """Alles-in-einem: Detektionen, Auswahl, Blick(bereich), Banner, Legende – auf einer Kopie."""
    canvas = img.copy()
    draw_detections(canvas, detections)

    # Auswahl hervorheben
    if result.is_assisted:
        for d in result.candidates:
            highlight(canvas, d, config.COLOR_CANDIDATE, 3)
    elif result.is_direct and result.selected is not None:
        highlight(canvas, result.selected, config.COLOR_SELECTED, 3)

    # Blick darstellen (Suchbereich nur bei Coarse-Gaze)
    if show_area:
        draw_gaze_area(canvas, result.gaze_point, radius)
    draw_gaze_point(canvas, result.gaze_point)

    draw_state_banner(canvas, result)
    if show_legend:
        draw_legend(canvas)
    if hint:
        _hint(canvas, hint)
    return canvas
