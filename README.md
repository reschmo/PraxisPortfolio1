# Blickbasierte Objektauswahl für assistive Roboterarme

Dieses Projekt untersucht den Auswahlschritt bei einer blickbasierten Steuerung für assistive Roboterarme. Der Nutzer betrachtet ein Objekt, YOLO erkennt die sichtbaren Objekte und die Auswahllogik entscheidet, welches Objekt gemeint ist. Ist die Zuordnung nicht eindeutig, werden mehrere Kandidaten angezeigt oder es erfolgt keine Auswahl.

**Praxisprojekt & Portfolio 1, FH Salzburg**  
Autor: Mohammad Akhlas Ahmadi  
Betreuer: Prof. Simon Kirchgasser

Der Schwerpunkt liegt ausschließlich auf der Objektauswahl. Ein realer Roboterarm, die Greifplanung und eine 3D-Lokalisierung sind nicht Bestandteil des Projekts.

---

## Aktueller Stand

| Baustein | Status |
| --- | --- |
| Auswahllogik mit Exact- und Coarse-Gaze sowie den Zuständen `direct`, `assisted` und `none` | fertig und mit Unit-Tests geprüft |
| YOLO-Wrapper mit einheitlichem Detection-Format | fertig |
| Maus als Referenz für die Blickquelle | fertig |
| Visualisierung von Bounding-Boxen, Blickpunkt und Auswahlzustand | fertig und mit einem Beispielbild geprüft |
| Standbild-Anwendung `app_image` | fertig, interaktiv und headless nutzbar |
| Livebild-Anwendung `app_live` | fertig |
| DL-Blickquelle mit MobileGaze über uniface und Kalibrierung | implementiert; für den Test ist eine Webcam erforderlich |

Die Bilder im Ordner `examples/` zeigen die drei möglichen Auswahlzustände. Die dort verwendeten Objekte dienen zur Veranschaulichung der Logik.

---

## Aufbau des Systems

Das System besteht aus drei getrennten Komponenten:

1. **Blickquelle**  
   Liefert einen Blickpunkt in der Form `gaze_point = (x, y)`.

2. **Objektdetektion mit YOLO**  
   Liefert eine Liste erkannter Objekte mit `label`, `confidence`, `bbox` und `center`.

3. **Auswahllogik**  
   Verknüpft den Blickpunkt mit den erkannten Objekten und gibt einen der Zustände `direct`, `assisted` oder `none` zurück.

Durch diese Trennung kann die Blickquelle ausgetauscht werden. Aktuell stehen die Maus und eine Webcam-basierte Blickschätzung zur Verfügung. Später könnte auch ein externer Eye-Tracker eingebunden werden. Die übrigen Komponenten müssen dafür nicht geändert werden, solange die Blickquelle einen Punkt `(x, y)` liefert.

### Auswahlzustände

- `direct`: Ein Objekt kann eindeutig zugeordnet werden.
- `assisted`: Mehrere Objekte sind ähnlich plausibel und werden als Kandidaten angezeigt.
- `none`: Es konnte kein ausreichend plausibles Objekt zugeordnet werden.

### Auswahlstrategien

Beide Strategien verwenden dieselben YOLO-Detektionen.

**Exact-Gaze**

Der Blick wird als einzelner Punkt behandelt. Für jedes erkannte Objekt wird die euklidische Distanz zwischen Blickpunkt und Objektzentrum berechnet.

- Liegt das nächste Objekt weniger als `80 px` entfernt, ist eine direkte Auswahl möglich.
- Liegt ein zweites Objekt höchstens `40 px` weiter entfernt, wird der Zustand `assisted` verwendet.
- Ist kein Objekt nah genug, lautet der Zustand `none`.

**Coarse-Gaze**

Der Blick wird als Bereich betrachtet. Um den Blickpunkt wird ein Kreis mit einem Standardradius von `90 px` gelegt.

- Schneidet der Kreis genau eine Bounding-Box, wird `direct` zurückgegeben.
- Schneidet er mehrere Bounding-Boxen, wird `assisted` zurückgegeben.
- Schneidet er keine Bounding-Box, wird `none` zurückgegeben.

Diese Strategie ist für ungenauere Blickschätzungen gedacht, wie sie bei einer Webcam auftreten können.

---

## Projektstruktur

```text
src/
  config.py              # Schwellenwerte, Modellnamen, Pfade und Farben
  detection/
    types.py             # einheitliches Detection-Format
    yolo.py              # YOLO laden und Inferenz ausführen
  gaze/
    base.py              # Interface: get_gaze_point() -> (x, y)
    mouse_gaze.py        # Maus als Blickquelle
    dl_gaze.py           # DL-Blickquelle mit MobileGaze und Kalibrierung
  selection/
    result.py            # SelectionState und SelectionResult
    exact.py             # Exact-Gaze-Auswahl
    coarse.py            # Coarse-Gaze-Auswahl
  viz/
    draw.py              # Darstellung von Boxen, Blickpunkt und Zustand
  app_image.py           # Anwendung für Standbilder
  app_live.py            # Anwendung für Livebilder
tests/                   # Tests für Auswahllogik und Kalibrierung
assets/test.jpg          # Beispielbild
examples/                # Beispielausgaben für die drei Zustände
```

---

## Installation

Für das Projekt kann eine virtuelle Python-Umgebung verwendet werden.

```bash
python -m venv .venv
```

Aktivierung unter Windows:

```bash
.venv\Scripts\activate
```

Aktivierung unter Linux oder macOS:

```bash
source .venv/bin/activate
```

Danach werden die Hauptabhängigkeiten installiert:

```bash
pip install -r requirements.txt
```

Das YOLO-Modell `yolov8m.pt` wird beim ersten Start automatisch geladen.

Für die DL-Blickquelle wird zusätzlich `uniface` benötigt:

```bash
pip install "uniface[cpu]"
```

Bei einer unterstützten NVIDIA-GPU kann stattdessen folgende Variante verwendet werden:

```bash
pip install "uniface[gpu]"
```

Die Modelle für Gesichts- und Blickschätzung werden beim ersten Start automatisch geladen und unter `~/.uniface/models` gespeichert. Es ist kein manueller Download erforderlich.

Alternativ können ONNX-Gewichte aus dem Projekt `yakhyo/gaze-estimation` verwendet werden:

```bash
python -m src.app_live --source 0 --gaze dl --model models/<datei>.onnx
```

---

## Nutzung

Alle Befehle werden im Wurzelverzeichnis des Projekts ausgeführt.

### Verfügbare Kombinationen

| Eingabe | Blickquelle | Befehl |
| --- | --- | --- |
| Bild | Maus | `python -m src.app_image --image assets/test1.png` |
| Webcam | Maus | `python -m src.app_live --source 0 --gaze mouse` |
| Bild | Eye-Tracking | `python -m src.app_image --image assets/test1.png --gaze dl` |
| Webcam | Eye-Tracking | `python -m src.app_live --source 0 --gaze dl` |

Für die Eye-Tracking-Varianten muss zunächst im Live-Modus kalibriert werden. Mit der Taste `k` wird die Kalibrierung gestartet. Die Kalibrierungsdaten werden gespeichert und beim nächsten Start wieder geladen.

In allen Varianten gelten dieselben Auswahlzustände und dieselbe Auswahllogik.

### Standbild mit Maussteuerung

```bash
python -m src.app_image --image assets/test.jpg
```

Der Mauszeiger dient dabei als Referenz für den Blickpunkt.

### Standbild ohne Fenster

```bash
python -m src.app_image --image assets/test.jpg --headless     --gaze 250,650 --strategy coarse --out out.jpg
```

Dieser Befehl erzeugt ein annotiertes Ausgabebild, ohne ein Fenster zu öffnen.

### Livebild mit Maus

```bash
python -m src.app_live --source 0 --gaze mouse
```

### Livebild mit DL-Blickschätzung

```bash
python -m src.app_live --source 0 --gaze dl
```

Standardmäßig wird eine Kalibrierung mit neun Punkten verwendet. Für ein Raster mit 25 Punkten:

```bash
python -m src.app_live --source 0 --gaze dl --calib-grid 5
```

Mit `k` wird die Kalibrierung gestartet. Die Punkte werden nacheinander für ungefähr eine Sekunde betrachtet. Anschließend steuert die Blickschätzung den angezeigten Blickpunkt.

Mit `v` kann die Genauigkeit an unabhängigen Testpunkten gemessen werden. Das Programm berechnet den mittleren Blickfehler in Pixeln und als Anteil der Bilddiagonale. Die Ergebnisse werden in `models/calibration_eval.csv` gespeichert. Dadurch können beispielsweise die 9-Punkt- und die 25-Punkt-Kalibrierung miteinander verglichen werden.

### Kalibrierung mit Kopfmerkmalen

Mit der Option `--head-aware` werden zusätzlich die Position und Größe des erkannten Gesichts berücksichtigt:

```bash
python -m src.app_live --source 0 --gaze dl --head-aware
```

Während der Kalibrierung sollte der Kopf leicht bewegt werden. In der Datei `calibration_eval.csv` wird der verwendete Modus in der Spalte `standard` beziehungsweise `head` gespeichert. Dadurch lassen sich beide Varianten getrennt auswerten.

### Tastatursteuerung

| Taste | Funktion |
| --- | --- |
| `e` | Exact-Gaze aktivieren |
| `c` | Coarse-Gaze aktivieren |
| `a` | Blickkreis ein- oder ausblenden |
| `k` | Kalibrierung starten |
| `v` | Genauigkeitsmessung starten und Ergebnis als CSV speichern |
| `+` / `-` | Suchbereich vergrößern oder verkleinern |
| `q` | Programm beenden |

---

## Beispiele

Der Ordner `examples/` enthält Beispielausgaben für die drei Zustände:

- `beispiel_exact_direct.jpg`: eindeutige Auswahl mit Exact-Gaze
- `beispiel_coarse_assisted.jpg`: mehrere Kandidaten mit Coarse-Gaze
- `beispiel_exact_none.jpg`: kein ausreichend nahes Objekt

---

## Wichtige Parameter

Die wichtigsten Einstellungen befinden sich in `src/config.py`.

| Parameter | Startwert | Bedeutung |
| --- | --- | --- |
| `YOLO_IMGSZ` | `960` | Bildgröße für die YOLO-Inferenz |
| `YOLO_CONF` | `0.35` | minimale Konfidenz einer Detektion |
| `EXACT_DIRECT_MAX_DIST` | `80 px` | maximale Distanz für eine direkte Auswahl |
| `EXACT_AMBIGUITY_MARGIN` | `40 px` | Abstandsmarge für eine mehrdeutige Auswahl |
| `COARSE_RADIUS` | `90 px` | Radius des Coarse-Gaze-Bereichs |
| `GAZE_SMOOTHING_ALPHA` | `0.4` | Faktor für die EMA-Glättung |

---

## Tests

Die Tests werden mit `pytest` ausgeführt:

```bash
pip install pytest
pytest -q
```

Geprüft werden:

- die Exact-Gaze-Auswahl
- die Coarse-Gaze-Auswahl
- die Zustände `direct`, `assisted` und `none`
- Grenzfälle der Auswahl
- der Polynom-Fit der Kalibrierung
- das Speichern und Laden der Kalibrierungsdaten

Für die Tests sind weder YOLO noch eine Kamera erforderlich.

---

## Funktionsweise der DL-Blickquelle

Die Blickrichtung wird mit einem vortrainierten Deep-Learning-Modell aus dem Gesichtsbild geschätzt. Das Modell liefert die Blickwinkel `yaw` und `pitch`. Diese Werte werden geglättet und durch die Kalibrierung auf Bildschirmkoordinaten abgebildet.

Ablauf pro Frame:

```text
Frame
  -> RetinaFace: Gesicht erkennen
  -> MobileGaze: yaw und pitch schätzen
  -> EMA-Glättung: Schwankungen reduzieren
  -> Kalibrierung: yaw und pitch auf x und y abbilden
  -> gaze_point = (x, y)
```

Die Zuordnung zu Bildschirmkoordinaten erfolgt mit einem Polynom zweiten Grades.

---

## Abgrenzung

Nicht Bestandteil dieses Projekts sind:

- die Steuerung eines realen Roboterarms
- die Greifplanung
- die 3D-Lokalisierung
- das Training eines eigenen Objektdetektors
- eine Nutzerstudie

Untersucht wird ausschließlich die Zuordnung eines Blicksignals zu erkannten Objekten.

---

## Referenzen

- [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) – MobileGaze, MIT-Lizenz
- [L2CS-Net](https://arxiv.org/abs/2203.03339) – Abdelrahman et al., 2022
- [Appearance-based Gaze Estimation Review](https://arxiv.org/abs/2104.12668) – Cheng et al.
- [Scoping Review zu Blicksteuerung und assistiven Roboterarmen](https://doi.org/10.3389/frobt.2024.1326670) – Fischer-Janzen et al., 2024
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Das Beispielbild `assets/test.jpg` basiert auf dem Ultralytics-Beispielbild `bus.jpg`.
