"""Generator (Kopie aus verweilzeit-demo) und Zufallsstrom: feste Werte, Handrechnung, Träger der Verteilung."""

import math

import numpy as np

import wik_constants as C
import wik_generator as G
from wik_rng import SplitMix64


def test_kunden_effekte_sind_die_der_quelle():
    ref = np.random.default_rng(12345).normal(0.0, 0.30, 12)  # so erzeugt verweilzeit-demo (vwz_generator.kunden_effekte)
    assert np.allclose(C.UK, ref, rtol=0, atol=1e-15)


def test_splitmix64_erste_ausgabe_ist_die_bekannte():
    # Referenzwert des SplitMix64 mit Zustand 0 (Vigna): 0xE220A8397B1DCDAF
    assert SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF


def test_mu_von_hand_gerechnet():
    uk = C.UK
    # Import, Kunde 0, Zoll 1, Wochenende 0: log(3,5 e^0,7) + UK[0] * 1,0
    x = np.array([[0, 0, 1, 0, 0.0, 0.0]])
    assert math.isclose(G.mu(x)[0], math.log(3.5) + 0.7 + uk[0], rel_tol=1e-12)
    # Export, Kunde 3, Vorlauf 2: log(0,6 + 0,85 * 2) + UK[3] * 0,3
    x = np.array([[1, 3, 0, 0, 2.0, 0.0]])
    assert math.isclose(G.mu(x)[0], math.log(2.3) + uk[3] * 0.3, rel_tol=1e-12)
    # Umschlag, Kunde 5, Vorlauf 4: log(0,3 + 0,8 * 4) + UK[5] * 0,3
    x = np.array([[2, 5, 0, 0, 4.0, 0.0]])
    assert math.isclose(G.mu(x)[0], math.log(3.5) + uk[5] * 0.3, rel_tol=1e-12)
    # Leer, Kunde 1, Wochenende 1: log(7 e^0,3) + UK[1]
    x = np.array([[3, 1, 0, 1, 0.0, 0.0]])
    assert math.isclose(G.mu(x)[0], math.log(7.0) + 0.3 + uk[1], rel_tol=1e-12)


def test_vorlauf_wirkt_nur_bei_export_und_umschlag():
    for art in (0, 3):
        a = G.mu(np.array([[art, 2, 0, 0, 0.0, 0.0]]))
        b = G.mu(np.array([[art, 2, 0, 0, 7.0, 0.0]]))  # unmöglich, aber die Formel liest den Vorlauf bei Import und Leer nicht
        assert a == b
    assert G.mu(np.array([[1, 2, 0, 0, 5.0, 0.0]])) > G.mu(np.array([[1, 2, 0, 0, 1.0, 0.0]]))


def test_ziehung_ist_deterministisch_und_gueltig():
    a = G.draw_rows(500, 7)
    b = G.draw_rows(500, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, G.draw_rows(500, 8))
    assert G.is_valid(a).all()


def test_die_ersten_zeilen_haengen_nicht_von_n_ab():
    assert np.array_equal(G.draw_rows(50, 3), G.draw_rows(200, 3)[:50])


def test_art_anteile_im_band_mit_standardfehler():
    X = G.draw_rows(4000, 11)
    for a, p in enumerate(C.ART_P):
        share = float((X[:, 0] == a).mean())
        se = math.sqrt(p * (1 - p) / 4000)
        assert abs(share - p) < 4 * se, (a, share, p)


def test_traeger_regeln():
    X = np.array([[0, 0, 0, 0, 0.0, 0], [0, 0, 0, 0, 1.0, 0], [1, 0, 0, 0, 0.5, 0], [1, 0, 0, 0, 5.0, 0], [2, 0, 0, 0, 0.4, 0], [2, 0, 0, 0, 6.0, 0], [3, 0, 1, 0, 0.0, 0]])
    assert G.is_valid(X).tolist() == [True, False, False, True, False, True, False]


def test_geklemmte_funktion_ist_auf_allen_moeglichen_zeilen_gleich():
    X = G.draw_rows(3000, 21)
    assert np.abs(G.mu_clamped(X) - G.mu(X)).max() == 0.0
    out = np.array([[1, 0, 0, 0, 0.0, 0.0]])   # unmögliche Zeile (Export mit Vorlauf 0): die Fortsetzung unterscheidet sich
    assert G.mu_clamped(out)[0] != G.mu(out)[0]


def test_trainingsdaten_rauschen_im_band():
    X, y = G.make_history(3000, 5)
    res = y - G.mu(X)
    for a, s in enumerate(C.S_BASE):
        r = res[X[:, 0] == a]
        assert abs(r.mean()) < 4 * s / math.sqrt(len(r)), a
        assert abs(r.std() / s - 1) < 0.15, a


def test_zwilling_ist_vorlauf_plus_rauschen_und_wird_von_mu_nicht_gelesen():
    X = G.draw_rows(3000, 17)
    d = X[:, 6] - X[:, 4]
    assert abs(d.mean()) < 4 * 0.05 / math.sqrt(3000) and abs(d.std() / 0.05 - 1) < 0.1
    Y = X.copy()
    Y[:, 6] += 5.0
    assert np.array_equal(G.mu(X), G.mu(Y))


def test_ohne_rauschspalten_ist_das_gewicht_null_und_der_zwilling_gleich_dem_vorlauf():
    X = G.draw_rows(200, 3, noise_columns=False)
    assert (X[:, 5] == 0).all() and (X[:, 6] == X[:, 4]).all() and G.is_valid(X).all()
