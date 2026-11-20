"""Claims: jede Zahl der README wird aus dem eingefrorenen Raster (mit denselben Auswertungsfunktionen wie in der App) gerechnet und muss im Wortlaut der README stehen.

Die Tabelle „Zahl → Test“ der README nennt, welcher Test welche Zahl prüft; Zahlen, die nur Messung dieser README sind, stehen dort als „Messung, nicht im Test“."""

import re
from pathlib import Path

import numpy as np

import wik_constants as C
import wik_evaluation as E
import wik_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
P, F = E.pct, E.fmt
ML = {"perm": "Permutation", "cperm": "Bedingte Permutation", "loco": "LOCO"}


def test_wahrheit(grid):
    t = grid["truth"]
    for m, text in (("perm", "Permutation"), ("cperm", "bedingte Permutation"), ("loco", "LOCO")):
        assert " / ".join(F(x, 3) for x in t[m]) in README, text
    assert f"1,713, bei bedingter Permutation {F(t['cperm'][0], 3)} und bei LOCO {F(t['loco'][0], 3)}" in README
    art = {kl: F(E.targets(grid, kl, "perm", 0)["tmean"], 2) for kl in C.CLASSES}
    assert f"Boosting {art['gbm']}, Random Forest {art['rf']}, Netz {art['mlp']}" in README


def test_tabelle_abdeckung(grid):
    for kl in C.CLASSES:
        for m in C.METHODS:
            cs = [P(E.mean_coverage(grid, kl, m, "B", 1, t)) for t in C.TARGETS]
            assert f"| {C.CLASS_LABELS[kl]} | {ML[m]} | {cs[0]} | {cs[1]} | {cs[2]} |" in README, (kl, m)


def test_spannen_und_rangfolge(grid):
    for kind, mi, text in (("B", 0, "86 bis 89 %"), ("B", 1, "87 bis 91 %"), ("B", 2, "86 bis 92 %")):
        vals = [E.mean_coverage(grid, kl, m, kind, mi, "pop") for kl in C.CLASSES for m in C.METHODS]
        assert f"{P(min(vals)).replace(' %', '')} bis {P(max(vals))}" == text
        assert f"{text} (je neun Zellen" in README or f"zu {text}" in README
    nvals = [E.mean_coverage(grid, kl, m, "N", mi, "pop") for kl in C.CLASSES for m in C.METHODS for mi in range(3)]
    assert f"Normalnäherung {P(min(nvals)).replace(' %', '')} bis {P(max(nvals))}" in README
    allv = [E.mean_coverage(grid, kl, m, k, mi, "pop") for k in C.INTERVAL_KINDS for kl in C.CLASSES for m in C.METHODS for mi in range(3)]
    assert f"{P(min(allv)).replace(' %', '')} bis {P(max(allv))} (Mittel über fünf Merkmale mit Wirkung, 54 Zellen" in README
    n = sum(1 for kl in C.CLASSES for mi in range(3) if (lambda a: a[0] < a[1] < a[2])([E.mean_coverage(grid, kl, m, "B", mi, "truth") for m in C.METHODS]))
    assert n == 7 and "in **7 von 9** Zellen" in README
    mlp = [[P(E.mean_coverage(grid, "mlp", m, "B", mi, "truth")) for m in C.METHODS] for mi in range(3)]
    assert mlp[0][1:] == ["84 %", "83 %"] and mlp[1][1:] == ["78 %", "78 %"] and mlp[2][1] == mlp[2][2] == "60 %"          # die Ausnahmen der README
    for kl in C.CLASSES:
        rc = ", ".join(F(E.mean_rank_correlation(grid, kl, 1, m)) for m in C.METHODS)
        assert rc.replace(", ", " / ") in README, kl
    rho = [E.mean_rank_correlation(grid, kl, 1, m) for kl in C.CLASSES for m in C.METHODS]
    assert F(min(rho)) == "0,88" and F(max(rho)) == "0,99"


def test_tabelle_testzeilen(grid):
    cells = E.cells_135(grid, "B")
    assert len(cells) == 135
    d = np.abs(np.array([[c[1], c[2]] for c in cells]) @ np.array([1.0, -1.0]))
    assert f"im Mittel auf {F(100 * d.mean(), 1)} Prozentpunkte (größter Abstand {F(100 * d.max(), 1)}" in README
    assert f"in {P(float((d <= 0.10).mean()))} der Zellen höchstens 10, in {P(float((d <= 0.05).mean()))} höchstens 5" in README
    for kl in C.CLASSES:
        for m in C.METHODS:
            hw = [np.mean([E.mean_half_width(grid, kl, m, f, "B", mi) for f in C.EFFECT_FEATURES]) for mi in range(3)]
            vv = [np.mean([E.ratio_v(grid, kl, m, f, "B", mi) for f in C.EFFECT_FEATURES]) for mi in range(3)]
            tm = [E.mean_coverage(grid, kl, m, "B", mi, "tmean") for mi in range(3)]
            pp = [E.mean_coverage(grid, kl, m, "B", mi, "pop") for mi in range(3)]
            row = (f"| {C.CLASS_LABELS[kl]}, {ML[m]} | {F(hw[0], 4)} / {F(hw[1], 4)} / {F(hw[2], 4)} | {F(vv[0])} / {F(vv[1])} / {F(vv[2])} | "
                   f"{P(tm[0])} / {P(tm[1])} / {P(tm[2])} | {P(pp[0])} / {P(pp[1])} / {P(pp[2])} |")
            assert row in README, (kl, m)
    hw = [np.mean([E.mean_half_width(grid, "gbm", "perm", f, "B", mi) for f in C.EFFECT_FEATURES]) for mi in range(3)]
    assert f"das {F(hw[0] / hw[2], 1)}-Fache von 300 auf 3000 Zeilen" in README


def test_tabelle_zwilling(grid):
    for kl in C.CLASSES:
        s = E.twin_stats(grid, kl)
        row = (f"| {C.CLASS_LABELS[kl]} | {P(s['perm']['beide_ueber_0'])} | {P(s['cperm']['beide_ueber_0'])} | {P(s['loco']['beide_nicht_ueber_0'])} | "
               f"{P(s['loco']['beide_ueber_0'])} | {P(s['gruppe_ueber_0'])} |")
        assert row in README, kl
        assert F(s["gruppe_mittel"]) == "0,15"
    big = {kl: P(E.twin_stats(grid, kl)["perm"]["zwilling_groesser"]) for kl in C.CLASSES}
    assert f"in {big['gbm']} (Boosting), {big['rf']} (Random Forest) und {big['mlp']} (Netz)" in README


def test_grenzen_abschnitt_steht_in_der_readme(grid):
    block = re.search(r"<!-- GRENZEN -->\n(.*?)\n<!-- /GRENZEN -->", README, re.S).group(1)
    assert block.strip() == T.grenzen_markdown(grid).strip()


def test_readme_hat_keine_platzhalter_und_keine_verbotenen_woerter():
    assert "PLATZHALTER" not in README and "TODO nach dem Deploy" in README      # Live-Link-Platzhalter ist Teil der späteren Einbindung
    for bad in ("Saatwert", "Schwanz", " trägt ", " tragen "):
        assert bad not in README, bad
