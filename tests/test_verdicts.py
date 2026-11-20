"""Kennzahlen und Verdikte an einem handgesetzten Raster: jede Zahl im Text wird nachgerechnet; Randfälle (Nenner 0, Gleichstand, ein Zufalls-Seed)."""

import numpy as np
import pytest

import wik_constants as C
import wik_evaluation as E


def _grid(half=0.1, pop_sd=0.0, truth=0.9):
    """Synthetisches Raster: 100 Zufalls-Seeds, Schätzwert 0,5, Intervall 0,5 ± half; pop je Zufalls-Seed 0,5 + Streuung; Wahrheit 0,9."""
    n = C.N_SEEDS
    est = np.zeros((n, 3, 18, 5))
    est[..., 0] = 0.5
    est[..., 1], est[..., 2] = 0.5 - half, 0.5 + half
    est[..., 3], est[..., 4] = 0.5 - half, 0.5 + half
    pop = np.full((n, 18), 0.5) + pop_sd * np.linspace(-1, 1, n)[:, None]
    twin = np.zeros((n, 7, 3))
    twin[:, :, 0] = 0.1
    twin[:, :, 1], twin[:, :, 2] = 0.05, 0.15
    cell = {"est": est.tolist(), "pop": pop.tolist(), "twin": twin.tolist(), "r2": [0.9] * n}
    return {"meta": {"klassen": ["gbm"]}, "truth": {m: [truth] * 6 for m in C.METHODS}, "cells": {"gbm": cell, "rf": cell, "mlp": cell}}


def test_formatierung():
    assert E.fmt(1.0449, 2) == "1,04" and E.pct(0.7) == "70 %" and E.pct(0.123, 1) == "12,3 %"
    assert E.standard_error(0.9, 100) == pytest.approx(0.03)
    assert E.standard_error(0.0, 100) == 0.0


def test_abdeckung_und_ziele_handgezaehlt():
    g = _grid()
    t = E.targets(g, "gbm", "perm", 0)
    assert t["truth"] == 0.9 and t["tmean"] == pytest.approx(0.5)
    # Intervall 0,4 bis 0,6: deckt pop (0,5) und die Mitte (0,5), nicht die Wahrheit (0,9)
    assert E.coverage(g, "gbm", "perm", 0, "B", 1, "pop") == 1.0 and E.coverage(g, "gbm", "perm", 0, "B", 1, "tmean") == 1.0 and E.coverage(g, "gbm", "perm", 0, "B", 1, "truth") == 0.0
    assert E.mean_half_width(g, "gbm", "perm", 0, "B", 1) == pytest.approx(0.1)
    g2 = _grid(truth=0.55)
    assert E.coverage(g2, "gbm", "loco", 3, "N", 2, "truth") == 1.0


def test_streuung_der_modellwichtigkeit_gegen_halbe_breite():
    g = _grid(half=0.1, pop_sd=0.05)
    pop = np.full(C.N_SEEDS, 0.0) + 0.05 * np.linspace(-1, 1, C.N_SEEDS)
    assert E.ratio_v(g, "gbm", "perm", 0, "B", 1) == pytest.approx(pop.std(ddof=1) / 0.1)
    # Seeds mit pop außerhalb des Intervalls: pop = 0,5 ± 0,05 liegt im Intervall (0,4 bis 0,6): Abdeckung 1; mit half = 0,03 deckt es nur |x| <= 0,03
    g3 = _grid(half=0.03, pop_sd=0.05)
    lin = 0.05 * np.linspace(-1, 1, C.N_SEEDS)
    assert E.coverage(g3, "gbm", "perm", 0, "B", 1, "pop") == pytest.approx(float(np.mean(np.abs(lin) <= 0.03)))


def test_mittel_ueber_die_fuenf_merkmale_mit_wirkung_ohne_rauschen():
    g = _grid()
    est = np.asarray(g["cells"]["gbm"]["est"])
    est[:, 1, E.col("perm", 5), 1:] = 0.0                    # das Rauschmerkmal: Intervall [0, 0], deckt nichts
    g["cells"]["gbm"]["est"] = est.tolist()
    assert E.mean_coverage(g, "gbm", "perm", "B", 1, "pop") == 1.0                    # nur Merkmale 0 bis 4 zählen
    assert E.mean_coverage(g, "gbm", "perm", "B", 1, "pop", features=(5,)) == 0.0


def test_zwilling_zaehlung_und_gruppe():
    g = _grid()
    tw = np.asarray(g["cells"]["gbm"]["twin"])
    tw[:50, 0, 1] = -0.1          # Vorlauf-Intervall bei Permutation in 50 Zufalls-Seeds nicht über 0
    tw[:, 1, 0] = 0.2             # Zwilling größer als Vorlauf (0,2 gegen 0,1)
    g["cells"]["gbm"]["twin"] = tw.tolist()
    s = E.twin_stats(g, "gbm")
    assert s["perm"]["beide_ueber_0"] == 0.5 and s["perm"]["beide_nicht_ueber_0"] == 0.0 and s["perm"]["zwilling_groesser"] == 1.0
    assert s["gruppe_ueber_0"] == 1.0 and s["gruppe_mittel"] == pytest.approx(0.1)


def test_rangmass_an_handwerten():
    g = _grid()
    est = np.asarray(g["cells"]["gbm"]["est"])
    for j, v in enumerate((0.3, 0.2, 0.1, 0.05, 0.9, 0.0)):
        est[:, 1, E.col("perm", j), 0] = v
    g["cells"]["gbm"]["est"] = est.tolist()
    g["truth"]["perm"] = [0.3, 0.2, 0.1, 0.05, 0.9, 0.0]
    ranks, rho = E.rank_view(g, "gbm", 1, "perm", 1)
    assert ranks.tolist() == [2, 3, 4, 5, 1, 6] and rho == pytest.approx(1.0)
    g["truth"]["perm"] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.9]                          # umgekehrte Rangordnung der ersten Fünf
    assert E.rank_view(g, "gbm", 1, "perm", 1)[1] < 1.0


def test_verdicts_nennen_jede_zahl():
    g = _grid()
    kind, text = E.verdict_intervall(g, "gbm", 1, "perm", 0, "B", 3)
    assert kind == "warning" and "Boosting, 1000 Testzeilen, Permutation, Art, Zufalls-Seed 3" in text and "von 0,400 bis 0,600" in text
    assert "Wichtigkeit des Modells (0,500): deckt;" in text and "Wahrheit je Verfahren (0,900): deckt nicht" in text
    kind, text = E.verdict_abdeckung(g, "gbm", "B", 1)
    assert kind == "info" and "100 Zufalls-Seeds" in text and "Modell 100 %, Mitte 100 %, Wahrheit 0 %" in text and "0 und von Vorlauf in 0 von 100 Zufalls-Seeds" in text and "nominal 90 %" in text
    kind, text = E.verdict_testzeilen(g, "gbm", "perm", "B")
    assert "halbe Breite 0,1000 / 0,1000 / 0,1000" in text and "das 1,0-Fache" in text and "Abdeckung der Mitte über Neu-Trainings 100 % / 100 % / 100 %" in text
    kind, text = E.verdict_zwilling(g, "gbm")
    assert kind == "warning" and "beide Intervalle über 0 in 100 %" in text and "beide nicht über 0 in 0 %" in text and "Intervall über 0 in 100 %" in text


def test_verdict_intervall_gleichstand_am_rand_zaehlt_als_deckt():
    g = _grid(half=0.4, truth=0.9)       # Intervall 0,1 bis 0,9: die Wahrheit liegt genau auf dem Rand
    assert E.coverage(g, "gbm", "perm", 0, "B", 1, "truth") == 1.0
    assert E.verdict_intervall(g, "gbm", 1, "perm", 0, "B", 1)[0] == "success"
