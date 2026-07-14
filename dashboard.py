"""
Praxisprojekt Dashboard – startet die vier Demo-Modi per Klick.

Start (in der aktivierten venv, im Projekt-Wurzelverzeichnis):
    python dashboard.py

Die Oberfläche nutzt CustomTkinter (modernes Aussehen). Einmalig installieren:
    pip install customtkinter

Jeder Modus öffnet das echte OpenCV-Fenster in einem EIGENEN Prozess (unter
Windows mit eigener Konsole, in der die Ausgaben erscheinen). Es wird derselbe
Python-Interpreter benutzt, mit dem das Dashboard läuft – also automatisch die
venv, wenn du sie aktiviert hast.

Eye-Gaze-Modi brauchen eine Kalibrierung: einmal "Live-Webcam + Eye-Gaze"
starten und im Fenster 'k' drücken. Danach funktioniert auch "Bild + Eye-Gaze".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # derselbe Interpreter -> nutzt die aktive venv

# --- Modus-Definitionen -----------------------------------------------------
MODES = [
    ("bild_maus", "Bild + YOLO + Maus",
     "Statisches Bild, YOLO-Boxen, Maus als Blickpunkt (einfachste Referenz)."),
    ("webcam_maus", "Live-Webcam + YOLO + Maus",
     "Webcam-Livebild, Maus simuliert den Blick."),
    ("bild_gaze", "Bild + YOLO + Eye-Gaze",
     "Webcam beobachtet die Augen; der kalibrierte Blick wählt Objekte im Bild."),
    ("webcam_gaze", "Live-Webcam + YOLO + Eye-Gaze",
     "Livebild; der kalibrierte Blick wählt direkt wie ein Cursor."),
]


def build_command(mode_key, settings):
    """Baut die Argumentliste für 'python <args>' (reine Funktion -> testbar)."""
    img = settings.get("image", "assets/test1.png")
    cam = str(settings.get("cam", "0"))
    strat = settings.get("strategy", "coarse")
    yolo_every = str(settings.get("yolo_every", "3"))
    if mode_key == "bild_maus":
        return ["-m", "src.app_image", "--image", img, "--strategy", strat]
    if mode_key == "webcam_maus":
        return ["-m", "src.app_live", "--source", cam, "--gaze", "mouse",
                "--strategy", strat, "--yolo-every", yolo_every]
    if mode_key == "bild_gaze":
        return ["-m", "src.app_image", "--image", img, "--gaze", "dl",
                "--source", cam, "--strategy", strat]
    if mode_key == "webcam_gaze":
        return ["-m", "src.app_live", "--source", cam, "--gaze", "dl",
                "--strategy", strat, "--yolo-every", yolo_every]
    raise ValueError(f"Unbekannter Modus: {mode_key}")


def launch(mode_key, settings):
    """Startet den Modus als eigenen Prozess (öffnet das OpenCV-Fenster)."""
    args = build_command(mode_key, settings)
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    return subprocess.Popen([PY] + args, cwd=str(ROOT), creationflags=flags)


def find_cameras(max_index=5):
    """Probiert Kamera-Indizes 0..max_index-1 und gibt die verfügbaren zurück."""
    import cv2

    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
        if cap is not None:
            cap.release()
    return available


# --- GUI (nur Optik; CustomTkinter) -----------------------------------------
def run_gui():
    try:
        import customtkinter as ctk
    except ImportError:
        raise SystemExit("Bitte zuerst installieren:  pip install customtkinter")
    import tkinter as tk
    from tkinter import messagebox

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    ACCENTS = ["#2563eb", "#0ea5e9", "#16a34a", "#f59e0b"]  # je Karte eine Akzentfarbe
    HOVER = ["#1d4ed8", "#0284c7", "#15803d", "#d97706"]

    app = ctk.CTk()
    app.title("Praxisprojekt Dashboard")
    app.geometry("1000x660")
    app.minsize(920, 600)
    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure(1, weight=1)

    image_var = tk.StringVar(value="assets/test1.png")
    cam_var = tk.StringVar(value="0")
    strat_var = tk.StringVar(value="coarse")
    yolo_var = tk.StringVar(value="3")
    status_var = tk.StringVar(value="Bereit")

    # ---------- Kopfbereich ----------
    header = ctk.CTkFrame(app, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 6))
    header.grid_columnconfigure(0, weight=1)
    titlebox = ctk.CTkFrame(header, fg_color="transparent")
    titlebox.grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(titlebox, text="Praxisprojekt Dashboard",
                 font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
    ctk.CTkLabel(titlebox, text="Lokale Steuerung für die YOLO-, Webcam- und Gaze-Prototypen.",
                 text_color=("gray45", "gray70")).pack(anchor="w")
    ctk.CTkSegmentedButton(
        header, values=["Hell", "Dunkel"],
        command=lambda c: ctk.set_appearance_mode("dark" if c == "Dunkel" else "light"),
    ).grid(row=0, column=1, sticky="e")

    body = ctk.CTkFrame(app, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=22, pady=4)
    body.grid_columnconfigure(1, weight=1)
    body.grid_rowconfigure(0, weight=1)

    # ---------- Einstellungen (links) ----------
    sett = ctk.CTkFrame(body, corner_radius=14)
    sett.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
    ctk.CTkLabel(sett, text="Einstellungen",
                 font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=16, pady=(16, 10))

    def field(label, widget):
        ctk.CTkLabel(sett, text=label, text_color=("gray40", "gray70")).pack(anchor="w", padx=16)
        widget.pack(anchor="w", padx=16, pady=(2, 12), fill="x")

    field("Bildpfad (Bild-Modi)", ctk.CTkEntry(sett, textvariable=image_var, width=240))
    field("Kamera-Index (Webcam-Modi)", ctk.CTkEntry(sett, textvariable=cam_var, width=100))
    field("Strategie", ctk.CTkOptionMenu(sett, values=["coarse", "exact"], variable=strat_var, width=140))
    field("YOLO-Intervall (Live)", ctk.CTkEntry(sett, textvariable=yolo_var, width=100))

    cam_out = ctk.CTkTextbox(sett, height=64, width=240)

    def scan():
        cam_out.delete("1.0", "end")
        cam_out.insert("end", "Suche…")
        app.update_idletasks()
        try:
            found = find_cameras()
            cam_out.delete("1.0", "end")
            cam_out.insert("end", f"Verfügbar: {found}" if found else "Keine Kamera gefunden.")
        except Exception as exc:
            cam_out.delete("1.0", "end")
            cam_out.insert("end", f"Fehler: {exc}")

    ctk.CTkButton(sett, text="OpenCV-Kameras suchen", command=scan).pack(anchor="w", padx=16, pady=(4, 6), fill="x")
    cam_out.pack(anchor="w", padx=16, pady=(0, 12), fill="x")
    ctk.CTkButton(sett, text="Dashboard beenden", command=app.destroy,
                  fg_color="gray45", hover_color="gray35").pack(anchor="w", padx=16, pady=(4, 16), fill="x")

    # ---------- Modus-Karten (rechts, 2x2) ----------
    cards = ctk.CTkFrame(body, fg_color="transparent")
    cards.grid(row=0, column=1, sticky="nsew")
    cards.grid_columnconfigure(0, weight=1)
    cards.grid_columnconfigure(1, weight=1)
    cards.grid_rowconfigure(0, weight=1)
    cards.grid_rowconfigure(1, weight=1)

    def get_settings():
        return {"image": image_var.get().strip(), "cam": cam_var.get().strip(),
                "strategy": strat_var.get(), "yolo_every": yolo_var.get().strip()}

    def make_start(mode_key, title):
        def _start():
            try:
                launch(mode_key, get_settings())
                status_var.set(f"gestartet: {title}")
            except Exception as exc:
                messagebox.showerror("Fehler beim Start", str(exc))
        return _start

    for idx, (key, title, desc) in enumerate(MODES):
        r, c = divmod(idx, 2)
        card = ctk.CTkFrame(cards, corner_radius=14)
        card.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)
        ctk.CTkFrame(card, height=6, fg_color=ACCENTS[idx], corner_radius=6).grid(
            row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=1, column=0, sticky="w", padx=16)
        ctk.CTkLabel(card, text=desc, wraplength=260, justify="left",
                     text_color=("gray40", "gray70")).grid(row=2, column=0, sticky="nw", padx=16, pady=(4, 8))
        ctk.CTkButton(card, text="Start", width=120, command=make_start(key, title),
                      fg_color=ACCENTS[idx], hover_color=HOVER[idx]).grid(
            row=3, column=0, sticky="e", padx=16, pady=(0, 16))

    # ---------- Statusleiste ----------
    status = ctk.CTkFrame(app, fg_color="transparent")
    status.grid(row=2, column=0, sticky="ew", padx=24, pady=(2, 12))
    status.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(status, textvariable=status_var, text_color=("#16a34a", "#4ade80"),
                 font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
    ctk.CTkLabel(status, text="Eye-Gaze: einmal 'Live-Webcam + Eye-Gaze' starten und 'k' drücken (Kalibrierung).",
                 text_color=("gray45", "gray70")).grid(row=0, column=1, sticky="e")

    app.mainloop()


if __name__ == "__main__":
    run_gui()
