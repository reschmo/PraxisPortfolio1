# Blickbasierte Objektauswahl für assistive Roboterarme

Modulare, nachvollziehbare Demo für den **Auswahlschritt**: Ein Nutzer *schaut ein
Objekt an*, eine Kamera *erkennt Objekte* (YOLO), und eine Logik entscheidet,
**welches Objekt gemeint ist – oder fragt nach**.

Praxisprojekt & Portfolio 1, FH Salzburg · Autor: Mohammad Akhlas Ahmadi ·
Betreuer: Prof. Simon Kirchgasser.

> Kein realer Roboterarm, keine Greifplanung, keine 3D-Lokalisierung – nur der
> Auswahlschritt.

---

## Aktueller Stand

| Baustein | Status |
| --- | --- |
| Auswahllogik (Exact + Coarse, Zustände direct/assisted/none) | ✅ fertig, per Unit-Tests geprüft |
| YOLO-Wrapper (einheitliches Detection-Format) | ✅ fertig |
| Maus-Blickquelle (Referenz) | ✅ fertig |
| Visualisierung (Boxen, Blick, Zustand) | ✅ fertig, auf Beispielbild geprüft |
| Standbild-App (`app_image`) | ✅ fertig (interaktiv + headless) |
| Livebild-App (`app_live`) | ✅ fertig |
| DL-Blickquelle (MobileGaze über uniface) + Kalibrierung | ✅ implementiert; Gewichte laden automatisch, braucht Webcam zum Testen |

Die Beispielbilder in `examples/` zeigen die drei Zustände (mit Platzhalter-Objekten
zur Illustration der Logik).

---

## Kernidee / Architektur (bewusst modular)

Drei klar getrennte Komponenten:

1. **Blickquelle** → liefert nur einen Punkt `gaze_point = (x, y)`
2. **Objektdetektion (YOLO)** → Liste von Objekten `{label, confidence, bbox, center}`
3. **Auswahllogik** → nimmt Blickpunkt + Objekte → Zustand `direct` / `assisted` / `none`

> Die Trennung ist zentral: Die Blickquelle ist **austauschbar** (Maus,
> Webcam-DL-Modell, später Eye-Tracker). Eine neue Quelle muss *nur*
> `gaze_point = (x, y)` liefern – der Rest bleibt gleich.

**Die drei Zustände**

- `direct` – genau ein Objekt eindeutig → auswählen
- `assisted` – mehrere plausibel → Kandidaten anzeigen (nicht automatisch entscheiden)
- `none` – kein ausreichend plausibles Objekt

**Zwei Auswahlstrategien** (beide auf denselben YOLO-Detektionen)

- **Exact-Gaze** – Blick = Punkt. Euklidische Distanz zum Objektzentrum:
  nächstes Objekt `< 80 px` → `direct`; liegt ein zweites `≤ 40 px` weiter → `assisted`.
- **Coarse-Gaze** – Blick = Bereich. Kreis (Standard `90 px`) um den Punkt;
  Boxen, die den Kreis schneiden, sind Kandidaten: 1 → `direct`, mehrere →
  `assisted`, keiner → `none`. **Sicherheitsnetz** für ungenaue Webcam-Blicke.

---

## Projektstruktur

```text
src/
  config.py              # Schwellen, Modellnamen, Pfade, Farben
  detection/
    types.py             # einheitliches Detection-Format (YOLO-unabhängig)
    yolo.py              # YOLO laden + Inferenz -> Detection
  gaze/
    base.py              # Interface: get_gaze_point() -> (x, y)
    mouse_gaze.py        # Maus als Blickquelle (Referenz)
    dl_gaze.py           # DL-Blickquelle (yakhyo/MobileGaze, ONNX) + Kalibrierung
  selection/
    result.py            # SelectionState / SelectionResult
    exact.py             # select_object(gaze, detections)
    coarse.py            # select_with_attention_area(gaze, detections, radius)
  viz/draw.py            # Zeichnen (Boxen, Blick, Zustand)
  app_image.py           # Standbild + YOLO + Blickquelle
  app_live.py            # Livebild + YOLO + Blickquelle
tests/                   # Unit-Tests der Auswahllogik + Kalibrierung
assets/test.jpg          # Beispielbild
examples/                # gerenderte Beispielausgaben (drei Zustände)
```

---

## Installation

Zwei getrennte Umgebungen vermeiden Paketkonflikte (YOLO/PyTorch vs. Gaze/ONNX).

**1) Kernumgebung (YOLO + Auswahllogik + Standbild/Live mit Maus)**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Das YOLO-Modell (`yolov8m.pt`) wird beim ersten Start automatisch geladen.

**2) DL-Blickquelle (MobileGaze über uniface)** – darf in **dieselbe** venv:

```bash
pip install "uniface[cpu]"     # NVIDIA-GPU: "uniface[gpu]"
```

Kein manueller Download: Die Blick- und Gesichtsmodelle werden beim ersten Start
automatisch geladen (Cache: `~/.uniface/models`). Python 3.10+ genügt.

> Optional/fortgeschritten: Statt uniface können rohe ONNX-Gewichte von
> [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation/releases)
> genutzt werden – dann `--model models/<datei>.onnx` angeben.

---

## Nutzung

Alle Befehle vom Projekt-Wurzelverzeichnis aus.

### Die vier Kombinationen (didaktischer Aufbau)

| # | Eingabe | Blickquelle | Befehl |
| --- | --- | --- | --- |
| 1 | Bild | Maus (Referenz) | `python -m src.app_image --image assets/test1.png` |
| 2 | Webcam | Maus | `python -m src.app_live --source 0 --gaze mouse` |
| 3 | Bild | Eye-Tracking | `python -m src.app_image --image assets/test1.png --gaze dl` |
| 4 | Webcam | Eye-Tracking | `python -m src.app_live --source 0 --gaze dl` |

Für #3 und #4 zuerst **einmal im Live-Modus kalibrieren** (`k`); die Kalibrierung
wird gespeichert und von #3 automatisch geladen. In jeder Kombination gilt:
`e`/`c` = Exact/Coarse, `+`/`-` = Suchbereich größer/kleiner, `q` = beenden.
Die Modus-Logik (direct/assisted/none) ist überall identisch.

**Standbild (interaktiv)** – Maus bewegen = Blick:

```bash
python -m src.app_image --image assets/test.jpg
```

**Standbild (headless)** – rendert ein annotiertes Bild ohne Fenster:

```bash
python -m src.app_image --image assets/test.jpg --headless \
    --gaze 250,650 --strategy coarse --out out.jpg
```

**Livebild – Maus als Blickquelle** (kein Modell nötig):

```bash
python -m src.app_live --source 0 --gaze mouse
```

**Livebild – DL-Blickquelle** (MobileGaze, lädt Gewichte automatisch):

```bash
python -m src.app_live --source 0 --gaze dl                 # 9 Punkte (3x3)
python -m src.app_live --source 0 --gaze dl --calib-grid 5  # 25 Punkte (5x5)
```

Im Fenster: `k` = kalibrieren (Punkte nacheinander ~1 s anschauen). Danach
steuern deine Augen den Blickpunkt; die Kalibrierung wird gespeichert und beim
nächsten Start automatisch geladen. `v` = **Genauigkeit messen**: du schaust auf
unabhängige Testpunkte, das Programm nennt den **mittleren Blickfehler** (in px
und % der Bilddiagonale) und schreibt ihn nach `models/calibration_eval.csv` –
so lassen sich z. B. 9 vs. 25 Punkte objektiv vergleichen.

**Kopf-aware Kalibrierung** (Experiment): mit `--head-aware` nutzt die
Kalibrierung zusätzlich die Kopfposition/-größe (aus der Gesichts-Box), um
Kopfbewegung/Abstand zu kompensieren. Dabei beim Kalibrieren den **Kopf leicht
bewegen**:

```bash
python -m src.app_live --source 0 --gaze dl --head-aware    # 9 Punkte, kopf-aware
```

Der Modus (`standard`/`head`) steht als Spalte in `calibration_eval.csv`, sodass
sich 9-Standard vs. 9-kopf-aware sauber vergleichen lässt. Der Effekt zeigt sich
vor allem, wenn man den Kopf **bewegt** (bei ruhigem Kopf sind beide ähnlich).

**Steuerung (Tastatur)**

| Taste | Wirkung |
| --- | --- |
| `e` | Exact-Gaze |
| `c` | Coarse-Gaze |
| `a` | Blickkreis ein/aus |
| `k` | Kalibrierung (nur DL-Blickquelle) |
| `v` | Genauigkeit messen + CSV (nur DL-Blickquelle) |
| `q` | Beenden |

---

## Beispiele

`examples/` enthält gerenderte Ausgaben der drei Zustände:

- `beispiel_exact_direct.jpg` – Exact-Gaze wählt ein Objekt eindeutig (`direct`).
- `beispiel_coarse_assisted.jpg` – Coarse-Gaze zeigt mehrere Kandidaten (`assisted`).
- `beispiel_exact_none.jpg` – kein Objekt nah genug (`none`).

---

## Wichtige Parameter (`src/config.py`)

| Parameter | Startwert | Bedeutung |
| --- | --- | --- |
| `YOLO_IMGSZ` | `960` | Inferenz-Bildgröße |
| `YOLO_CONF` | `0.35` | Konfidenzschwelle |
| `EXACT_DIRECT_MAX_DIST` | `80` px | Exact: `direct`, wenn nächstes Objekt näher |
| `EXACT_AMBIGUITY_MARGIN` | `40` px | Exact: zweites Objekt so nah → `assisted` |
| `COARSE_RADIUS` | `90` px | Coarse: Radius des Aufmerksamkeitskreises |
| `GAZE_SMOOTHING_ALPHA` | `0.4` | EMA-Glättung des DL-Blickpunkts |


---

## Tests

```bash
pip install pytest
pytest -q
```

Getestet werden die Auswahllogik (Exact/Coarse, alle drei Zustände, Grenzfälle)
und die Kalibrierungs-Mathematik (Polynom-Fit, Speichern/Laden) – ohne YOLO/Kamera.

---

## DL-Blickquelle: Funktionsweise

Appearance-based Deep-Learning-Blickschätzung: Ein vortrainiertes Netz schätzt die
Blickrichtung direkt aus dem Gesichtsbild (robuster gegen Kopfbewegung/Licht als der
frühere MediaPipe-Iris+KNN-Ansatz).

Pipeline pro Frame (`src/gaze/dl_gaze.py`):

```text
Frame -> RetinaFace (uniface): Gesicht finden
      -> MobileGaze (uniface): (yaw, pitch) in Radiant
      -> EMA-Glättung:         Zittern reduzieren
      -> Kalibrierung:         (yaw, pitch) -> (x, y)    [Polynom 2. Grades]
      => gaze_point = (x, y)
```



---

## Abgrenzung (nicht Teil dieser Demo)

Kein realer Roboterarm, keine Greifplanung, keine 3D-Lokalisierung, kein eigenes
Detektor-Training, keine Nutzerstudie. Fokus: der Auswahlschritt.



## Referenzen

- **yakhyo/gaze-estimation (MobileGaze)** – https://github.com/yakhyo/gaze-estimation (MIT)
- **L2CS-Net** (Abdelrahman et al. 2022) – https://arxiv.org/abs/2203.03339
- **Cheng et al. 2024**, Review appearance-based gaze estimation – https://arxiv.org/abs/2104.12668
- **Fischer-Janzen et al. 2024**, Scoping Review (gaze/assistive arms) – https://doi.org/10.3389/frobt.2024.1326670
- **Ultralytics YOLOv8** – https://github.com/ultralytics/ultralytics
- Beispielbild `assets/test.jpg`: Ultralytics-Beispiel `bus.jpg`.
