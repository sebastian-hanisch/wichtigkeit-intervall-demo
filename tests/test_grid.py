"""Das eingefrorene Raster: Form, Herkunft, Invarianten, Beispielmodell (live gegen eingefroren) und die Neu-Rechnung (Band) über tools/wik_make_grid.py --check."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import wik_constants as C
import wik_evaluation as E
import wik_intervals as IV

ROOT = Path(__file__).resolve().parent.parent


def test_meta_nennt_quelle_stand_und_zaehleinheiten(grid):
    meta = grid["meta"]
    for key in ("stand", "quelle", "umgebung", "offset", "klassen", "zufalls_seeds", "testzeilen", "umordnungen", "neuziehungen", "felder"):
        assert key in meta, key
    assert meta["klassen"] == list(C.CLASSES) and meta["zufalls_seeds"] == C.N_SEEDS and meta["testzeilen"] == list(C.TEST_SIZES)
    assert meta["umordnungen"] == C.R_PERM and meta["neuziehungen"] == C.N_BOOT and meta["pop_zeilen"] == C.POP_ROWS


def test_form_aller_zellen(grid):
    for kl in C.CLASSES:
        c = grid["cells"][kl]
        assert np.shape(c["est"]) == (C.N_SEEDS, 3, 18, 5) and np.shape(c["pop"]) == (C.N_SEEDS, 18)
        assert np.shape(c["twin"]) == (C.N_SEEDS, 7, 3) and np.shape(c["r2"]) == (C.N_SEEDS,)
        assert np.isfinite(np.asarray(c["est"], dtype=float)).all()
    assert all(len(grid["truth"][m]) == C.D for m in C.METHODS)


def test_wahrheit_je_verfahren_im_band_um_die_bekannten_werte(grid):
    t = grid["truth"]
    assert 1.65 <= t["perm"][0] <= 1.78 and 0.065 <= t["cperm"][0] <= 0.082 and 0.032 <= t["loco"][0] <= 0.042      # Art: gewöhnlich, bedingt, LOCO (derselben Funktion μ)
    assert 1.28 <= t["perm"][4] <= 1.40 and 0.25 <= t["cperm"][4] <= 0.31 and 0.125 <= t["loco"][4] <= 0.16         # Vorlauf
    for m in C.METHODS:
        assert t[m][5] == 0.0                                                                                      # Rauschmerkmal: exakt 0
        assert t[m][3] < t[m][4]
    assert t["perm"][0] / t["cperm"][0] > 15                                                                       # es gibt keine eine Wahrheit der Wichtigkeit


def test_intervalle_sind_geordnet_und_die_normalnaeherung_ist_symmetrisch(grid):
    for kl in C.CLASSES:
        e = E.est_array(grid, kl)
        assert (e[..., 1] <= e[..., 2]).all() and (e[..., 3] <= e[..., 4]).all()
        assert np.abs((e[..., 3] + e[..., 4]) / 2 - e[..., 0]).max() < 2e-6          # N: Mittel ± 1,6449 Standardfehler (Rundung auf sechs Stellen)
        inside = ((e[..., 1] <= e[..., 0]) & (e[..., 0] <= e[..., 2])).mean()
        assert inside > 0.97, (kl, inside)                                            # B enthält den Schätzwert (Perzentile des Bootstraps um das Mittel; als Band)


def test_mehr_testzeilen_machen_das_intervall_enger_um_etwa_die_wurzel(grid):
    for kl in C.CLASSES:
        for method in C.METHODS:
            hw = [np.mean([E.mean_half_width(grid, kl, method, f, "B", mi) for f in C.EFFECT_FEATURES]) for mi in range(3)]
            assert hw[0] > hw[1] > hw[2]
            assert 0.50 < hw[1] / hw[0] < 0.65 and 0.50 < hw[2] / hw[1] < 0.65        # 1/√3,3 = 0,55 je Schritt (Band)


def test_gewoehnliche_permutation_deckt_die_wahre_wichtigkeit_von_art_und_vorlauf_nie_in_den_gezaehlten_zellen(grid):
    zellen = 0
    for kl in C.CLASSES:
        for mi in range(3):
            for f in (0, 4):
                k = round(E.coverage(grid, kl, "perm", f, "B", mi, "truth") * C.N_SEEDS)
                assert k == 0, (kl, mi, f, k)                                          # 0 von 100 Zufalls-Seeds heißt nicht, dass die Wahrscheinlichkeit 0 ist
            zellen += 1
    assert zellen == 9


def test_rauschmerkmal_intervall_ueber_null_selten_und_loco_mittel_negativ(grid):
    for kl in C.CLASSES:
        lo, _ = E.interval(grid, kl, 1, "perm", 5, "B")
        assert float(np.mean(lo > 0)) <= 0.15                                           # einseitig nominal 5 %, Standardfehler 2 bis 3 Prozentpunkte
        assert float(E.est_array(grid, kl)[:, 1, E.col("loco", 5), 0].mean()) < 0       # Weglassen eines Rauschmerkmals verbessert den Testfehler


def test_beispielmodell_live_gleich_eingefroren(grid, example):
    for kl in C.CLASSES:
        d = np.asarray(example["diff"][kl], dtype=float)
        assert d.shape == (3000, 18)
        for mi, m in enumerate(C.TEST_SIZES):
            idx = IV.bootstrap_indices(m, C.N_BOOT, C.SEED_BOOT + mi)
            for c in (0, 4, 12, 16):
                live = IV.interval_summary(d[:m, c], idx)
                frozen = E.est_array(grid, kl)[0, mi, c]
                assert np.allclose(live, frozen, atol=2e-6), (kl, mi, c)               # derselbe Bootstrap live und im Raster (Rundung auf sechs Stellen)


def test_neu_rechnen_als_band_ueber_das_pruefskript():
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("wik_make_grid", ROOT / "tools" / "wik_make_grid.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wik_make_grid"] = mod
    spec.loader.exec_module(mod)
    assert mod.check(str(ROOT / "data" / "wik_grid.json")) == 0     # Form, Wahrheit neu gerechnet (Band 5 %), Boosting Zufalls-Seed 0 im Band 2e-3
