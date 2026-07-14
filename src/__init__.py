"""
Blickbasierte Objektauswahl für assistive Roboterarme.

Modularer Aufbau (siehe README):
  - detection : YOLO-Objekterkennung  -> einheitliches Detection-Format
  - gaze      : austauschbare Blickquellen (Maus, DL/yakhyo, später Eye-Tracker)
  - selection : Auswahllogik (Exact / Coarse) -> Zustand direct/assisted/none
  - viz       : Visualisierung (Boxen, Blick, Zustand)

Kernprinzip: Eine Blickquelle liefert *nur* gaze_point = (x, y).
Der Rest der Pipeline bleibt dadurch unverändert und austauschbar.
"""

__version__ = "0.1.0"
