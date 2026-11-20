"""Intervalle und Abdeckung: Handwerte durch Aufzählung (exakt), Orakel gegen numpy und scipy, Eigenschaften, Randfälle, Abdeckung des Intervall-Codes auf bekannter Verteilung (Band)."""

import itertools
from fractions import Fraction as Fr
from math import comb

import numpy as np
import pytest

import wik_formula as FM
import wik_intervals as IV

POP = (0, 0, 1, 1, 2, 6)          # Population der Verlustdifferenzen, Mittel 5/3


def test_normalnaeherung_deckt_852_von_1296_saetzen():
    """Testsatz = 4 Ziehungen mit Zurücklegen aus 6 Verlusten: 6⁴ = 1296 gleich wahrscheinliche Sätze; Mittel der Population 5/3."""
    mean = Fr(sum(POP), len(POP))
    assert mean == Fr(5, 3)
    covered = 0
    for s in itertools.product(POP, repeat=4):
        lo, hi = IV.interval_n(np.array(s, dtype=float)) or (np.nan, np.nan)
        covered += lo <= float(mean) <= hi
    assert covered == 852 and Fr(covered, 1296) == Fr(71, 108)


def test_bootstrap_perzentil_exakt_durch_aufzaehlung_zwei_wege():
    """Exakte Bootstrap-Verteilung über alle 4⁴ = 256 Neuziehungen, Perzentile als kleinster Wert mit kumulativer Wahrscheinlichkeit ≥ 0,05 bzw. ≥ 0,95: deckt in 768 von 1296 Sätzen.
    Zweiter Weg: Zählvektoren mit Multinomialgewicht."""
    mean = Fr(5, 3)
    covered = covered2 = 0
    fakt = [1, 1, 2, 6, 24]
    for s in itertools.product(POP, repeat=4):
        draws = [Fr(sum(r), 4) for r in itertools.product(s, repeat=4)]               # 256 gleich wahrscheinliche Neuziehungen
        dist1 = {}
        for v in draws:
            dist1[v] = dist1.get(v, 0) + Fr(1, 256)
        dist2 = {}
        for counts in itertools.product(range(5), repeat=4):
            if sum(counts) != 4:
                continue
            w = Fr(24, 1)
            for c in counts:
                w /= fakt[c]
            v = Fr(sum(c * x for c, x in zip(counts, s)), 4)
            dist2[v] = dist2.get(v, 0) + w / 256
        assert dist1 == dist2                                                         # beide Wege geben dieselbe exakte Verteilung
        vals = sorted(dist1)
        cum = {v: sum(dist1[u] for u in vals if u <= v) for v in vals}
        lo = next(v for v in vals if cum[v] >= Fr(5, 100))
        hi = next(v for v in vals if cum[v] >= Fr(95, 100))
        covered += lo <= mean <= hi
        covered2 += lo <= mean <= hi
    assert covered == covered2 == 768 and Fr(768, 1296) == Fr(16, 27)


def test_bernoulli_wald_abdeckung_exakt():
    for p, ziel in ((Fr(1, 2), Fr(7, 8)), (Fr(1, 5), Fr(352, 625))):
        tot = Fr(0)
        for k in range(5):
            d = np.array([1.0] * k + [0.0] * (4 - k))
            lo, hi = IV.interval_n(d)
            if lo <= float(p) <= hi:
                tot += comb(4, k) * p ** k * (1 - p) ** (4 - k)
        assert tot == ziel


def test_bootstrap_perzentil_gegen_numpy_inverted_cdf():
    d = np.random.default_rng(1).normal(size=40)
    idx = IV.bootstrap_indices(40, 400, 5)
    means = IV.bootstrap_means(d, idx)
    lo, hi = IV.interval_b(d, idx)
    assert lo == np.percentile(means, 5) and hi == np.percentile(means, 95)
    assert abs(np.percentile(means, 5, method="inverted_cdf") - lo) < 0.02 and abs(np.percentile(means, 95, method="inverted_cdf") - hi) < 0.02


def test_normalintervall_gegen_scipy():
    stats = pytest.importorskip("scipy.stats")
    d = np.random.default_rng(2).normal(size=300)
    lo, hi = IV.interval_n(d)
    se = d.std(ddof=1) / np.sqrt(300)
    z = stats.norm.ppf(0.95)
    assert abs((hi - lo) / 2 - z * se) < 1e-6 and abs(z - IV.Z90) < 1e-9
    assert abs(stats.t.ppf(0.95, 299) - 1.650) < 0.001          # t(299) = 1,650: Normalnäherung weicht bei 300 Zeilen um 0,005 ab


def test_abdeckung_des_intervall_codes_auf_bekannter_verteilung_band():
    """Verluste χ² mit 1 Freiheitsgrad (schief, Mittel exakt 1): Normalnäherung deckt nominal im Band 88 bis 92 % (2000 Testsätze, Standardfehler 0,7 Prozentpunkte)."""
    rng = np.random.default_rng(3)
    for m in (300, 1000):
        hits = [IV.covers(*IV.interval_n(rng.chisquare(1, size=m)), 1.0) for _ in range(2000)]
        assert 0.88 <= np.mean(hits) <= 0.92, (m, np.mean(hits))
    hb = [IV.covers(*IV.interval_b(d, IV.bootstrap_indices(100, 200, 9000 + k)), 1.0) for k, d in enumerate(rng.chisquare(1, size=(100, 100)))]
    assert 0.78 <= np.mean(hb) <= 0.98                                           # Bootstrap, 100 Sätze zu 100 Zeilen: Band wegen Standardfehler 3 Prozentpunkte


def test_eigenschaften_reihenfolge_verdoppeln_und_zufalls_seed():
    d = np.random.default_rng(4).normal(size=120)
    lo, hi = IV.interval_n(d)
    lo2, hi2 = IV.interval_n(d[::-1])
    assert (lo, hi) == pytest.approx((lo2, hi2))                                  # Reihenfolge der Testzeilen ändert nichts
    dd = np.concatenate([d, d])
    lo3, hi3 = IV.interval_n(dd)
    assert (lo3 + hi3) / 2 == pytest.approx((lo + hi) / 2)                        # Mitte bleibt
    assert (hi3 - lo3) / (hi - lo) == pytest.approx(1 / np.sqrt(2), rel=0.03)     # Breite etwa durch √2 (Standardfehler ∝ 1/√m; Band wegen m − 1)
    assert np.array_equal(IV.bootstrap_indices(30, 20, 7), IV.bootstrap_indices(30, 20, 7))
    assert not np.array_equal(IV.bootstrap_indices(30, 20, 7), IV.bootstrap_indices(30, 20, 8))


def test_randfaelle():
    assert IV.interval_n(np.array([3.0])) is None                                 # eine Zeile: Standardfehler nicht definiert
    lo, hi = IV.interval_b(np.array([3.0]), IV.bootstrap_indices(1, 10, 1))
    assert (lo, hi) == (3.0, 3.0)
    z = np.zeros(50)
    assert IV.interval_n(z) == (0.0, 0.0) and IV.interval_b(z, IV.bootstrap_indices(50, 20, 2)) == (0.0, 0.0)
    assert IV.interval_n(np.array([1.0, 3.0])) == pytest.approx((2 - IV.Z90 * 1.0, 2 + IV.Z90 * 1.0))     # zwei Zeilen: s = √2, se = 1
    assert IV.bootstrap_indices(5, 1, 3).shape == (1, 5)                          # B = 1
    assert IV.half_width(1.0, 3.0) == 1.0 and IV.covers(1.0, 3.0, 3.0) and not IV.covers(1.0, 3.0, 3.1)


def test_abdeckungsformel_handwerte_und_simulation():
    for r, wert in ((0, 0.9000), (0.5, 0.8588), (1, 0.7552), (2, 0.5380), (3, 0.3970)):
        assert FM.coverage_of_center(r) == pytest.approx(wert, abs=6e-5), r
    rng = np.random.default_rng(5)
    for r in (0.5, 1.0, 2.0):
        e = rng.normal(size=400_000)
        pop = rng.normal(size=400_000) * r                                         # tau = r · se, se = 1
        sim = np.mean(np.abs(pop + e) <= FM.Z)                                    # Intervall um pop + e deckt die Mitte 0 genau dann, wenn |pop + e| <= z · se
        assert abs(sim - FM.coverage_of_center(r)) < 0.004, r
    assert FM.coverage_from_ratio(1.25) == pytest.approx(0.5280, abs=5e-4)       # V = 1,25 gibt r = z · V = 2,06
    assert FM.coverage_from_ratio(0.0) == pytest.approx(0.9, abs=1e-9)
