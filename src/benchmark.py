"""
Automatischer Benchmark: Exact-Gaze vs. Coarse-Gaze (mit Standard-Radius).

Idee: Für jedes von YOLO erkannte Objekt werden viele *simulierte* Blickpunkte um
das Objektzentrum gestreut (Gauß-Rauschen ~ realistische Blickungenauigkeit). Für
jeden Punkt werden beide Strategien ausgewertet und relativ zum Zielobjekt
klassifiziert:

  - Direct Hit   : direct, und das ausgewählte Objekt IST das Zielobjekt.
  - False Direct : direct, aber das FALSCHE Objekt (gefährliche Auto-Auswahl).
  - Assisted     : Kandidaten werden angeboten (keine Auto-Auswahl).
  - None         : kein Objekt.
  - Recoverable  : Ziel ist erreichbar (Direct Hit ODER Ziel unter den Kandidaten).
  - O Kandidaten : mittlere Anzahl angebotener Objekte.

Rein rechnerischer Vergleich (kein Roboterarm, keine Kamera) -> reproduzierbar.

Start:
    python -m src.benchmark                                  # nutzt Bilder in assets/
    python -m src.benchmark --images assets/test1.png --samples 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .selection.coarse import select_with_attention_area
from .selection.exact import select_object


def _new_stat():
    return {"trials": 0, "direct_hit": 0, "false_direct": 0, "assisted": 0,
            "none": 0, "assisted_has_target": 0, "cand_sum": 0}


def _record(stat, result, target):
    stat["trials"] += 1
    stat["cand_sum"] += len(result.candidates)
    if result.is_direct:
        if result.selected is target:
            stat["direct_hit"] += 1
        else:
            stat["false_direct"] += 1
    elif result.is_assisted:
        stat["assisted"] += 1
        if any(c is target for c in result.candidates):
            stat["assisted_has_target"] += 1
    else:
        stat["none"] += 1


def _merge(dst, src):
    for k in dst:
        dst[k] += src[k]


def simulate_trials(detections, image_size, samples_per_object=50,
                    noise_frac=0.035, seed=0, th=None, radius=None):
    """Streut Blickpunkte um jedes Objekt und wertet Exact & Coarse aus."""
    w, h = image_size
    diag = (w * w + h * h) ** 0.5
    th = th or config.thresholds_for(w, h)
    radius = th["radius"] if radius is None else radius
    sigma = noise_frac * diag
    rng = np.random.default_rng(seed)

    stats = {"exact": _new_stat(), "coarse": _new_stat()}
    for target in detections:
        cx, cy = target.center
        for _ in range(samples_per_object):
            gp = (cx + rng.normal(0, sigma), cy + rng.normal(0, sigma))
            r_ex = select_object(gp, detections, direct_max_dist=th["direct_max"],
                                 ambiguity_margin=th["ambiguity"])
            r_co = select_with_attention_area(gp, detections, radius=radius)
            _record(stats["exact"], r_ex, target)
            _record(stats["coarse"], r_co, target)
    return stats


def summarize(stat):
    n = max(1, stat["trials"])
    recoverable = stat["direct_hit"] + stat["assisted_has_target"]
    return {
        "trials": stat["trials"],
        "direct_hit_pct": 100.0 * stat["direct_hit"] / n,
        "false_direct_pct": 100.0 * stat["false_direct"] / n,
        "assisted": stat["assisted"],
        "none": stat["none"],
        "recoverable_pct": 100.0 * recoverable / n,
        "avg_candidates": stat["cand_sum"] / n,
    }


def _default_images():
    imgs = []
    for name in ("test0.jpg","test1.jpg","test2.jpg", "test3.jpg", "test4.jpg"):
        p = config.ASSETS_DIR / name
        if p.exists():
            imgs.append(p)
    return imgs


def run_benchmark(image_paths, samples_per_object=50, noise_frac=0.035, seed=0):
    import cv2

    from .detection.yolo import ObjectDetector

    detector = ObjectDetector()
    total = {"exact": _new_stat(), "coarse": _new_stat()}
    used = 0
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  uebersprungen (nicht lesbar): {p}")
            continue
        h, w = img.shape[:2]
        dets = detector.detect(img)
        print(f"  {Path(p).name}: {len(dets)} Objekte")
        if not dets:
            continue
        used += 1
        stats = simulate_trials(dets, (w, h), samples_per_object, noise_frac, seed)
        for strat in ("exact", "coarse"):
            _merge(total[strat], stats[strat])
    return total, used


def render_table(total):
    """Gibt die Tabelle auf der Konsole aus und liefert sie als Markdown zurück."""
    header = ["Strategie", "Versuche", "Direct Hits", "False Direct",
              "Assisted", "None", "Recoverable", "O Kandidaten"]
    rows = []
    for strat, label in (("exact", "Exact-Gaze"), ("coarse", "Coarse-Gaze")):
        s = summarize(total[strat])
        rows.append([label, str(s["trials"]), f"{s['direct_hit_pct']:.1f} %",
                     f"{s['false_direct_pct']:.1f} %", str(s["assisted"]), str(s["none"]),
                     f"{s['recoverable_pct']:.1f} %", f"{s['avg_candidates']:.3f}"])

    widths = [max(len(str(header[i])), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = " | ".join(str(header[i]).ljust(widths[i]) for i in range(len(header)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(len(header))))

    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description="Benchmark Exact- vs. Coarse-Gaze")
    p.add_argument("--images", nargs="*", default=None, help="Bildpfade (Default: assets/)")
    p.add_argument("--samples", type=int, default=50, help="simulierte Blickpunkte pro Objekt")
    p.add_argument("--noise-frac", type=float, default=0.035,
                   help="Blick-Streuung als Anteil der Bilddiagonale (~gemessene Ungenauigkeit)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    images = [Path(x) for x in args.images] if args.images else _default_images()
    if not images:
        raise SystemExit("Keine Bilder gefunden. Lege Bilder in assets/ oder nutze --images.")

    print(f"Benchmark auf {len(images)} Bild(ern), {args.samples} Punkte/Objekt, "
          f"Rauschen={args.noise_frac} x Diagonale ...")
    total, used = run_benchmark(images, args.samples, args.noise_frac, args.seed)
    print(f"\nAusgewertete Bilder: {used}\n")
    md = render_table(total)

    out = config.PROJECT_ROOT / "benchmark_results.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nMarkdown-Tabelle gespeichert -> {out}")


if __name__ == "__main__":
    main()
