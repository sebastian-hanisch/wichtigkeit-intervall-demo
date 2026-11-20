"""Orakel und Handbeispiele der drei Verfahren (andere Rechenwege als der Code): Aufzählung aller Umordnungen, sklearn.inspection.permutation_importance, Normalgleichung,
geschlossene Form im Gauß-Zwilling, bedingte Erwartung der wahren Funktion. Aus nicht gelesenen Quellen kommt kein Test."""

import itertools
from fractions import Fraction as Fr

import numpy as np
import pytest

import wik_constants as C
import wik_generator as G
import wik_importance as IMP
import wik_value as V
from wik_rng import SplitMix64


# ---------------------------------------------------------------------------------------------------------------- Handbeispiel (4 Zeilen, exakt)
def test_permutation_importance_von_hand_vier_zeilen():
    x = [Fr(v) for v in (0, 1, 2, 3)]                       # f = x0, y = x0: Fehler 0
    perms = list(itertools.permutations(range(4)))
    werte = [sum((x[p[i]] - x[i]) ** 2 for i in range(4)) / 4 for p in perms]
    mittel = sum(werte) / len(werte)
    var = sum((v - Fr(3, 2)) ** 2 for v in x) / 4           # Varianz mit Nenner n: 5/4
    assert var == Fr(5, 4) and mittel == Fr(5, 2) == 2 * var   # Mittel über alle 4! = 24 Umordnungen = 2 · Varianz
    assert len(werte) == 24 and min(werte) == 0 and max(werte) == 5
    assert (sum(w < mittel for w in werte), sum(w == mittel for w in werte), sum(w > mittel for w in werte)) == (11, 2, 11)


def test_perm_diffs_gegen_aufzaehlung_fuer_f_gleich_x0():
    """Unser Code (zufällige Umordnungen) gegen die exakte Aufzählung: Mittel über viele Umordnungen nähert sich 5/2 (4 Zeilen) an."""
    X = np.zeros((4, 7))
    X[:, 0] = [0, 1, 2, 3]
    f = lambda Z: Z[:, 0]   # noqa: E731
    vals = [IMP.perm_diffs(f, X, X[:, 0], 0, SplitMix64(1000 + k), r=1).mean() for k in range(4000)]
    assert abs(np.mean(vals) - 2.5) < 4 * np.std(vals) / np.sqrt(len(vals))


def test_merkmal_das_f_nicht_liest_hat_wichtigkeit_exakt_null():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 7))
    f = lambda Z: Z[:, 0] * 2.0   # noqa: E731
    d = IMP.perm_diffs(f, X, X[:, 0], 3, SplitMix64(5), r=5)
    assert np.all(d == 0.0)


def test_permutation_gegen_sklearn_permutation_importance_band():
    sk = pytest.importorskip("sklearn.inspection")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    X7, y = G.make_history(1500, 21)
    X = X7[:, :6]
    model = make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1])], remainder="passthrough"), LinearRegression()).fit(X, y)
    f = model.predict
    ref = sk.permutation_importance(model, X, y, scoring="neg_mean_squared_error", n_repeats=60, random_state=0)
    for j in (0, 4):
        mine = np.mean([IMP.perm_diffs(f, X, y, j, SplitMix64(300 + k), r=1).mean() for k in range(60)])
        assert abs(mine - ref.importances_mean[j]) < 4 * (ref.importances_std[j] / np.sqrt(60)) + 0.01 * ref.importances_mean[j] + 1e-3, (j, mine, ref.importances_mean[j])


# ---------------------------------------------------------------------------------------------------------------- bedingte Permutation
def test_jede_schicht_behaelt_ihre_werte_exakt():
    rng = np.random.default_rng(2)
    strata = rng.integers(0, 4, size=300)
    values = rng.normal(size=300)
    idx = IMP.permute_within(300, strata, SplitMix64(7))
    assert sorted(idx.tolist()) == list(range(300))                              # eine echte Umordnung
    for s in np.unique(strata):
        rows = strata == s
        assert sorted(values[idx][rows]) == sorted(values[rows])                  # Multimenge der Schicht bleibt (exakt)
        assert (strata[idx] == strata).all()                                      # Zeilen bleiben in ihrer Schicht


def test_eine_schicht_ist_eine_gewoehnliche_umordnung_und_einzelzeilen_bleiben():
    idx = IMP.permute_within(50, np.zeros(50, dtype=int), SplitMix64(3))
    assert sorted(idx.tolist()) == list(range(50))
    one = IMP.permute_within(5, np.arange(5), SplitMix64(3))
    assert one.tolist() == [0, 1, 2, 3, 4]                                        # Schichten aus einer Zeile ändern nichts


def test_schichten_der_bedingten_permutation():
    X = np.zeros((4, 7))
    X[:, 0] = [0, 1, 2, 3]
    X[:, 4] = [0, 3.0, 0, 2.0]
    assert IMP.strata_for(X, 1).tolist() == [0, 1, 2, 3]            # andere Merkmale: Schicht = Art
    assert IMP.strata_for(X, 0).tolist() == [0, 1, 0, 1]            # Art selbst: Vorlauf > 0 gegen = 0


# ---------------------------------------------------------------------------------------------------------------- LOCO und Zwilling
def test_loco_im_linearen_modell_gegen_normalgleichung():
    rng = np.random.default_rng(4)
    n = 4000
    A = rng.normal(size=(n, 3))
    y = A @ np.array([1.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=n)
    full = np.linalg.solve(A.T @ A, A.T @ y)
    wo = np.linalg.solve(A[:, [0, 2]].T @ A[:, [0, 2]], A[:, [0, 2]].T @ y)
    d = IMP.loco_diffs(A @ full, A[:, [0, 2]] @ wo[[0, 1]], y)
    # Verlustanstieg ohne Merkmal 1: Koeffizient 0,5 unabhängig von den anderen -> 0,25 · Var(x1) = 0,25 (Normalgleichung, nicht der Code der Demo)
    assert abs(d.mean() - 0.25) < 0.03


def test_zwilling_gauss_geschlossene_form_exakt_und_simuliert():
    rho = Fr(9, 10)
    assert Fr(1, 2) == (2 * 1) / Fr(4)                                            # Permutation je Zwilling: E[(A' - A)^2] / 4 = 2 / 4
    assert (1 - rho ** 2) / 2 == Fr(19, 200)                                      # bedingte Permutation (A' gegeben B)
    assert (1 - rho ** 2) / 4 == Fr(19, 400)                                      # LOCO einzeln
    assert (1 + rho) / 2 == Fr(19, 20)                                            # LOCO Gruppe = Var(f), das Zehnfache der Summe der beiden Einzelwerte
    assert (1 + rho) / 2 == 10 * (2 * (1 - rho ** 2) / 4)                          # das Zehnfache der Summe der beiden LOCO-Einzelwerte
    rng = np.random.default_rng(6)
    n = 400_000
    r = 0.9
    A = rng.normal(size=n)
    B = r * A + np.sqrt(1 - r * r) * rng.normal(size=n)
    f = (A + B) / 2
    perm = np.mean((((rng.permutation(A) + B) / 2) - f) ** 2)                     # A durch eine Umordnung ersetzt
    assert abs(perm - 0.5) < 0.01
    beta = (f @ B) / (B @ B)                                                      # LOCO einzeln: bester linearer Vorhersager von f aus B (kleinste Quadrate)
    assert abs(np.mean((f - beta * B) ** 2) - 19 / 400) < 0.002 and abs(beta - (1 + r) / 2) < 0.01
    assert abs(np.var(f) - 19 / 20) < 0.01                                        # LOCO beide weglassen: Varianz von f


# ---------------------------------------------------------------------------------------------------------------- Wahrheit je Verfahren
def test_loco_wahrheit_an_additiver_funktion_in_geschlossener_form():
    g = lambda Z: np.asarray(C.UK)[Z[:, 1].astype(int)] + 5.0 * Z[:, 3]            # Kunde und Wochenende unabhängig, additiv
    X = G.draw_rows(500, 11, noise_columns=False)
    uk = np.asarray(C.UK)
    exakt_kunde = float(np.mean((uk - uk.mean()) ** 2))                           # Var von UK[Kunde] bei Gleichverteilung über 12 Kunden
    p = C.P_WOCHENENDE
    exakt_we = 25 * p * (1 - p)
    assert abs(V.loco_truth(g, X, 1) - exakt_kunde) < 0.15 * exakt_kunde            # Zufallsanteil der 500 Zeilen (Band)
    assert abs(V.loco_truth(g, X, 3) - exakt_we) < 0.15 * exakt_we
    assert V.loco_truth(g, X, 0) < 1e-12 and V.loco_truth(g, X, 4) < 1e-12          # Art und Vorlauf nicht gelesen: 0
    assert V.loco_truth(G.mu, X, 5) == 0.0                                          # Rauschmerkmal


def test_loco_wahrheit_von_mu_hat_die_reihung_der_wirkung():
    X = G.draw_rows(400, 12, noise_columns=False)
    t = [V.loco_truth(G.mu, X, j) for j in range(5)]
    assert t[4] == max(t) and all(v > 0 for v in t)                               # Vorlauf wichtigster, jedes Merkmal mit Wirkung > 0


def test_bedingte_erwartung_von_mu_effizienz():
    X = G.draw_rows(20, 13, noise_columns=False)
    full = V.cond_expect(G.mu, X, np.ones(5, dtype=bool))
    assert np.abs(full - G.mu(X)).max() < 1e-12                                   # alle Merkmale bekannt: E[mu | X] = mu(X)
    marg = V.cond_expect(G.mu, X, np.zeros(5, dtype=bool))
    assert np.ptp(marg) < 1e-12                                                   # nichts bekannt: überall derselbe Grundwert
