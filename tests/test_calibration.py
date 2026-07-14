"""Tests der Kalibrierung (Polynom-Fit (yaw,pitch) -> (x,y)).

Nutzt nur numpy; kein ONNX-Modell/keine Kamera nötig.
"""
import numpy as np
import pytest

from src.gaze.dl_gaze import GazeCalibration, _poly_features


def _grid_angles():
    yaws = np.linspace(-0.4, 0.4, 4)
    pitches = np.linspace(-0.3, 0.3, 4)
    return [(float(y), float(p)) for y in yaws for p in pitches]  # 16 Punkte


def test_recovers_degree2_polynomial():
    angles = _grid_angles()
    A = np.asarray(angles)
    coef_x = np.array([300, 500, 50, 20, -10, 15], dtype=float)
    coef_y = np.array([250, -40, 400, -5, 25, -8], dtype=float)
    feats = _poly_features(A[:, 0], A[:, 1])
    points = list(zip(feats @ coef_x, feats @ coef_y))

    calib = GazeCalibration().fit(angles, points)

    # An Trainingspunkten praktisch exakt
    for (y, p), (tx, ty) in zip(angles, points):
        gx, gy = calib.transform(y, p)
        assert abs(gx - tx) < 1.0 and abs(gy - ty) < 1.0

    # Neuer Punkt: Polynom 2. Grades ist exakt rekonstruierbar
    ny, npi = 0.1, -0.05
    f = _poly_features(np.array([ny]), np.array([npi]))[0]
    gx, gy = calib.transform(ny, npi)
    assert abs(gx - float(f @ coef_x)) < 1.0
    assert abs(gy - float(f @ coef_y)) < 1.0


def test_requires_minimum_points():
    with pytest.raises(ValueError):
        GazeCalibration().fit([(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_save_and_load(tmp_path):
    angles = _grid_angles()
    A = np.asarray(angles)
    feats = _poly_features(A[:, 0], A[:, 1])
    coef_x = np.arange(6, dtype=float) + 1
    coef_y = np.arange(6, dtype=float) - 2
    points = list(zip(feats @ coef_x, feats @ coef_y))

    calib = GazeCalibration().fit(angles, points)
    path = tmp_path / "cal.npz"
    calib.save(path)
    loaded = GazeCalibration.load(path)

    assert np.allclose(loaded.coef_x, calib.coef_x)
    assert np.allclose(loaded.coef_y, calib.coef_y)


def test_grid_sizes_point_counts():
    from src.gaze.dl_gaze import default_calibration_targets
    w, h = 1280, 720
    assert len(default_calibration_targets(w, h, grid=3)) == 9
    assert len(default_calibration_targets(w, h, grid=4)) == 16
    assert len(default_calibration_targets(w, h, grid=5)) == 25


def test_grid_targets_within_bounds_and_regular():
    from src.gaze.dl_gaze import default_calibration_targets
    w, h = 1280, 720
    pts = default_calibration_targets(w, h, grid=5)
    assert all(0 <= x < w and 0 <= y < h for x, y in pts)
    xs = sorted(set(x for x, _ in pts))
    ys = sorted(set(y for _, y in pts))
    assert len(xs) == 5 and len(ys) == 5  # sauberes 5x5-Raster


def test_n_points_is_persisted(tmp_path):
    from src.gaze.dl_gaze import GazeCalibration, _poly_features
    angles = [(float(y), float(p)) for y in np.linspace(-.4, .4, 4)
              for p in np.linspace(-.3, .3, 4)]
    A = np.asarray(angles)
    feats = _poly_features(A[:, 0], A[:, 1])
    points = list(zip(feats @ (np.arange(6.) + 1), feats @ (np.arange(6.) - 2)))
    calib = GazeCalibration().fit(angles, points)
    assert calib.n_points == 16
    path = tmp_path / "c.npz"
    calib.save(path)
    assert GazeCalibration.load(path).n_points == 16


def test_head_aware_recovers_linear_model():
    from src.gaze.dl_gaze import HeadAwareCalibration, _head_poly_features
    rng = np.random.default_rng(1)
    rows = rng.uniform(-0.4, 0.4, size=(60, 5))  # (yaw,pitch,hx,hy,hs)
    F = _head_poly_features(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])
    cx = rng.uniform(-100, 600, size=F.shape[1])
    cy = rng.uniform(-100, 400, size=F.shape[1])
    xs, ys = F @ cx, F @ cy
    calib = HeadAwareCalibration().fit(rows, list(zip(xs, ys)))
    assert calib.requires_head is True and calib.mode == "head"
    for k in range(5):
        yaw, pitch, hx, hy, hs = rows[k]
        gx, gy = calib.transform(yaw, pitch, hx, hy, hs)
        assert abs(gx - xs[k]) < 1.5 and abs(gy - ys[k]) < 1.5


def test_load_calibration_detects_mode(tmp_path):
    from src.gaze.dl_gaze import (GazeCalibration, HeadAwareCalibration,
                                  load_calibration, _poly_features, _head_poly_features)
    # Standard speichern -> als GazeCalibration laden
    ang = [(float(y), float(p)) for y in np.linspace(-.4, .4, 4)
           for p in np.linspace(-.3, .3, 4)]
    A = np.asarray(ang)
    f = _poly_features(A[:, 0], A[:, 1])
    std = GazeCalibration().fit(ang, list(zip(f @ (np.arange(6.) + 1), f @ (np.arange(6.) - 1))))
    sp = tmp_path / "s.npz"
    std.save(sp)
    loaded = load_calibration(sp)
    assert loaded.mode == "standard" and loaded.requires_head is False

    # Head-aware speichern -> als HeadAwareCalibration laden
    rng = np.random.default_rng(0)
    rows = rng.uniform(-.4, .4, size=(40, 5))
    F = _head_poly_features(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])
    hc = HeadAwareCalibration().fit(rows, list(zip(F @ np.ones(13), F @ (np.ones(13) * 2))))
    hp = tmp_path / "h.npz"
    hc.save(hp)
    loaded2 = load_calibration(hp)
    assert loaded2.mode == "head" and loaded2.requires_head is True


def test_rate_calibration_thresholds():
    from src.gaze.dl_gaze import rate_calibration
    assert rate_calibration(2.0) == "gut"
    assert rate_calibration(4.0) == "gut"      # Grenze inklusive
    assert rate_calibration(5.0) == "mittel"
    assert rate_calibration(6.0) == "mittel"   # Grenze inklusive
    assert rate_calibration(6.1) == "schlecht"
    assert rate_calibration(12.0) == "schlecht"
