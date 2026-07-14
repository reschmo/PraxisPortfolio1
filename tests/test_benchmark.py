"""Tests der Benchmark-Zähllogik (ohne YOLO/Kamera, mit Mock-Detektionen)."""
from src.detection.types import Detection
from src.benchmark import simulate_trials, summarize


def make(cx, cy, label, size=40):
    return Detection.from_xyxy(0, label, 0.9, cx - size / 2, cy - size / 2,
                               cx + size / 2, cy + size / 2)


def test_counts_add_up():
    dets = [make(200, 200, "cup"), make(600, 200, "scissors"), make(1000, 600, "clock")]
    stats = simulate_trials(dets, (1280, 720), samples_per_object=30, noise_frac=0.03, seed=1)
    for strat in ("exact", "coarse"):
        s = stats[strat]
        assert s["trials"] == 3 * 30
        assert s["direct_hit"] + s["false_direct"] + s["assisted"] + s["none"] == s["trials"]
        assert s["assisted_has_target"] <= s["assisted"]


def test_summarize_in_range():
    dets = [make(200, 200, "cup"), make(600, 200, "scissors")]
    stats = simulate_trials(dets, (1280, 720), samples_per_object=50, noise_frac=0.03, seed=2)
    for strat in ("exact", "coarse"):
        s = summarize(stats[strat])
        assert 0.0 <= s["direct_hit_pct"] <= 100.0
        assert 0.0 <= s["false_direct_pct"] <= 100.0
        assert 0.0 <= s["recoverable_pct"] <= 100.0
        assert s["avg_candidates"] >= 0.0


def test_coarse_recoverable_high_for_separated_objects():
    # gut getrennte Objekte + kleine Streuung -> Coarse sollte fast immer erreichbar sein
    dets = [make(150, 150, "a"), make(1100, 600, "b")]
    stats = simulate_trials(dets, (1280, 720), samples_per_object=100, noise_frac=0.02, seed=3)
    assert summarize(stats["coarse"])["recoverable_pct"] > 80.0


def test_coarse_never_false_direct_when_separated():
    # Coarse waehlt bei mehreren Kandidaten NICHT automatisch -> hier keine False Directs
    dets = [make(150, 150, "a"), make(1100, 600, "b")]
    stats = simulate_trials(dets, (1280, 720), samples_per_object=80, noise_frac=0.02, seed=4)
    assert stats["coarse"]["false_direct"] == 0
