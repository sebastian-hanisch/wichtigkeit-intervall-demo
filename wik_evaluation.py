"""Kennzahlen aus dem eingefrorenen Raster und Verdikt-Logik (reine Funktionen: `verdict_*` gibt (Art, Text) zurück, die App ruft st.<Art>(Text)).

Zähleinheit der Abdeckung: Zufalls-Seeds (Neu-Trainings mit eigener Historie und eigenem Testsatz). Ein Intervall hat ein Ziel: die Wichtigkeit dieses Modells („pop“),
die Wichtigkeit eines Modells aus 1000 Zeilen im Mittel über Neu-Trainings („tmean“) oder die Wahrheit je Verfahren („truth“)."""

import json
from pathlib import Path

import numpy as np

import wik_constants as C
from wik_formula import coverage_from_ratio

DATA = Path(__file__).resolve().parent / "data"


def load_grid(path=DATA / "wik_grid.json") -> dict:
    """Raster lesen; die Zahlenfelder der Zellen werden einmal in numpy-Felder umgewandelt (sonst rechnet jede Kennzahl die Listen neu um)."""
    with open(path, encoding="utf-8") as fh:
        grid = json.load(fh)
    for cell in grid["cells"].values():
        for key in ("est", "pop", "twin", "r2"):
            cell[key] = np.asarray(cell[key], dtype=float)
    return grid


def load_example(path=DATA / "wik_example.json") -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------------------------------------ Formatierung
def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def col(method: str, feature: int) -> int:
    return C.METHODS.index(method) * C.D + feature


def standard_error(p: float, n: int) -> float:
    """Standardfehler eines Anteils p aus n Zufalls-Seeds (Prozentpunkte als Anteil)."""
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


# ------------------------------------------------------------------------------------------------------------------------------ Zugriff
def est_array(grid, kl):
    return np.asarray(grid["cells"][kl]["est"], dtype=float)         # (Zufalls-Seeds, Testzeilen, 18, 5)


def pop_array(grid, kl):
    return np.asarray(grid["cells"][kl]["pop"], dtype=float)         # (Zufalls-Seeds, 18)


def interval(grid, kl, mi: int, method: str, feature: int, kind: str):
    """(unten, oben) je Zufalls-Seed für die Testzeilen-Stufe mi (Index in TEST_SIZES), Verfahren, Merkmal, Intervallart B oder N."""
    e = est_array(grid, kl)[:, mi, col(method, feature)]
    return (e[:, 1], e[:, 2]) if kind == "B" else (e[:, 3], e[:, 4])


def estimate(grid, kl, mi, method, feature):
    return est_array(grid, kl)[:, mi, col(method, feature), 0]


def targets(grid, kl, method, feature) -> dict:
    """Ziele: pop (je Zufalls-Seed das Modell auf der Bevölkerung), tmean (Mittel von pop über die Zufalls-Seeds), truth (Wahrheit je Verfahren)."""
    pop = pop_array(grid, kl)[:, col(method, feature)]
    return {"pop": pop, "tmean": float(pop.mean()), "truth": float(grid["truth"][method][feature])}


def coverage(grid, kl, method, feature, kind, mi, target):
    """Anteil der Zufalls-Seeds, deren Intervall das Ziel enthält."""
    lo, hi = interval(grid, kl, mi, method, feature, kind)
    t = targets(grid, kl, method, feature)[target]
    return float(np.mean((lo <= t) & (t <= hi)))


def mean_coverage(grid, kl, method, kind, mi, target, features=C.EFFECT_FEATURES):
    """Mittel der Abdeckung über die fünf Merkmale mit Wirkung (nicht über das Rauschmerkmal)."""
    return float(np.mean([coverage(grid, kl, method, f, kind, mi, target) for f in features]))


def mean_half_width(grid, kl, method, feature, kind, mi):
    lo, hi = interval(grid, kl, mi, method, feature, kind)
    return float(np.mean(0.5 * (hi - lo)))


def ratio_v(grid, kl, method, feature, kind, mi):
    """V = Streuung der Modellwichtigkeit über Neu-Trainings (SD von pop) geteilt durch die mittlere halbe Breite."""
    sd = float(pop_array(grid, kl)[:, col(method, feature)].std(ddof=1))
    return sd / mean_half_width(grid, kl, method, feature, kind, mi)


def cells_135(grid, kind="B"):
    """Die 135 Zellen (3 Klassen x 3 Verfahren x 5 Merkmale x 3 Testzeilen-Stufen): (V, gemessene Abdeckung der Mitte, Abdeckung nach Formel)."""
    out = []
    for kl in grid["meta"]["klassen"]:
        for method in C.METHODS:
            for f in C.EFFECT_FEATURES:
                for mi in range(len(C.TEST_SIZES)):
                    v = ratio_v(grid, kl, method, f, kind, mi)
                    out.append((v, coverage(grid, kl, method, f, kind, mi, "tmean"), coverage_from_ratio(v)))
    return out


def _ranks(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    for v in np.unique(a):
        idx = a == v
        ranks[idx] = ranks[idx].mean()
    return ranks


def _rank_stats(est_vec, truth):
    ranks = 1 + (est_vec[None, :] > est_vec[:, None]).sum(axis=1)
    rho = float(np.corrcoef(_ranks(est_vec), _ranks(truth))[0, 1])
    return ranks, rho


def rank_view(grid, kl, mi, method, seed: int):
    """Rangmaß eines Zufalls-Seeds: Rang des Merkmals unter den sechs (1 = wichtigstes) und Rangkorrelation mit dem Rang der Wahrheit je Verfahren."""
    est = est_array(grid, kl)[seed - 1, mi, [col(method, f) for f in range(C.D)], 0]
    return _rank_stats(est, np.asarray(grid["truth"][method], dtype=float))


def mean_rank_correlation(grid, kl, mi, method):
    est = est_array(grid, kl)[:, mi, :, 0][:, [col(method, f) for f in range(C.D)]]
    truth = np.asarray(grid["truth"][method], dtype=float)
    return float(np.mean([_rank_stats(row, truth)[1] for row in est]))


# ------------------------------------------------------------------------------------------------------------------------------ Zwilling
def twin_stats(grid, kl) -> dict:
    """Zwillings-Experiment (1000 Testzeilen, Intervall B): Anteil der Zufalls-Seeds mit beiden Intervallen über 0, mit beiden nicht über 0, Zwilling größer als Vorlauf; Gruppe über 0."""
    tw = np.asarray(grid["cells"][kl]["twin"], dtype=float)          # (Zufalls-Seeds, 7, 3): Schätzwert, B unten, B oben
    out = {}
    for name, (iv, itw) in {"perm": (0, 1), "cperm": (2, 3), "loco": (4, 5)}.items():
        a, b = tw[:, iv, 1] > 0, tw[:, itw, 1] > 0
        out[name] = {"beide_ueber_0": float(np.mean(a & b)), "beide_nicht_ueber_0": float(np.mean(~a & ~b)), "zwilling_groesser": float(np.mean(tw[:, itw, 0] > tw[:, iv, 0]))}
    out["gruppe_ueber_0"] = float(np.mean(tw[:, 6, 1] > 0))
    out["gruppe_mittel"] = float(tw[:, 6, 0].mean())
    return out


# ------------------------------------------------------------------------------------------------------------------------------ Verdikte
def _covers_text(lo, hi, value):
    return "deckt" if lo <= value <= hi else "deckt nicht"


def verdict_intervall(grid, kl, mi, method, feature, kind, seed):
    """Experiment 1: ein Neu-Training, ein Intervall, drei Ziele."""
    lo, hi = (a[seed - 1] for a in interval(grid, kl, mi, method, feature, kind))
    est = float(estimate(grid, kl, mi, method, feature)[seed - 1])
    t = targets(grid, kl, method, feature)
    pop, mean, truth = float(t["pop"][seed - 1]), t["tmean"], t["truth"]
    name = f"{C.CLASS_LABELS[kl]}, {C.TEST_SIZES[mi]} Testzeilen, {C.METHOD_LABELS[method].split(' (')[0]}, {C.FEATURE_LABELS[feature]}, Zufalls-Seed {seed}"
    parts = [f"Wichtigkeit des Modells ({fmt(pop, 3)}): {_covers_text(lo, hi, pop)}", f"Mitte über Neu-Trainings ({fmt(mean, 3)}): {_covers_text(lo, hi, mean)}",
             f"Wahrheit je Verfahren ({fmt(truth, 3)}): {_covers_text(lo, hi, truth)}"]
    kinds = "success" if lo <= truth <= hi else "warning"
    return kinds, f"{name}: Schätzwert {fmt(est, 3)}, Intervall {C.INTERVAL_LABELS[kind].split(' (')[0]} von {fmt(lo, 3)} bis {fmt(hi, 3)}. Gegen die drei Ziele: " + "; ".join(parts) + "."


def verdict_abdeckung(grid, kl, kind, mi):
    """Experiment 2: Abdeckung nach Ziel und Verfahren, Mittel über die fünf Merkmale mit Wirkung."""
    n = C.N_SEEDS
    lines = []
    for method in C.METHODS:
        cs = {tg: mean_coverage(grid, kl, method, kind, mi, tg) for tg in C.TARGETS}
        lines.append(f"{C.METHOD_LABELS[method].split(' (')[0]}: Modell {pct(cs['pop'])}, Mitte {pct(cs['tmean'])}, Wahrheit {pct(cs['truth'])}")
    se = standard_error(C.NOMINAL, n)
    art = coverage(grid, kl, "perm", 0, kind, mi, "truth")
    vor = coverage(grid, kl, "perm", 4, kind, mi, "truth")
    k_art, k_vor = round(art * n), round(vor * n)
    return "info", (f"{C.CLASS_LABELS[kl]}, {C.TEST_SIZES[mi]} Testzeilen, Intervall {C.INTERVAL_LABELS[kind].split(' (')[0]}, {n} Zufalls-Seeds, Mittel über die fünf Merkmale mit Wirkung (nominal 90 %, Standardfehler bei 90 % etwa {fmt(100 * se, 1)} Prozentpunkte): "
                    + "; ".join(lines) + f". Gewöhnliche Permutation deckt die wahre Wichtigkeit von Art in {k_art} und von Vorlauf in {k_vor} von {n} Zufalls-Seeds.")


def verdict_testzeilen(grid, kl, method, kind):
    """Experiment 3: mehr Testzeilen machen das Intervall enger, nicht verlässlicher für die Frage nach einem Neu-Training (Mittel über fünf Merkmale)."""
    hw = [float(np.mean([mean_half_width(grid, kl, method, f, kind, mi) for f in C.EFFECT_FEATURES])) for mi in range(3)]
    tm = [mean_coverage(grid, kl, method, kind, mi, "tmean") for mi in range(3)]
    pp = [mean_coverage(grid, kl, method, kind, mi, "pop") for mi in range(3)]
    vv = [float(np.mean([ratio_v(grid, kl, method, f, kind, mi) for f in C.EFFECT_FEATURES])) for mi in range(3)]
    t = C.TEST_SIZES
    return "info", (f"{C.CLASS_LABELS[kl]}, {C.METHOD_LABELS[method].split(' (')[0]}, {C.N_SEEDS} Zufalls-Seeds, Mittel über die fünf Merkmale mit Wirkung: halbe Breite {fmt(hw[0], 4)} / {fmt(hw[1], 4)} / {fmt(hw[2], 4)} bei {t[0]} / {t[1]} / {t[2]} Testzeilen "
                    f"(das {fmt(hw[0] / hw[2], 1)}-Fache von {t[2]} zu {t[0]} Testzeilen), Verhältnis V = SD der Modellwichtigkeit / halbe Breite {fmt(vv[0])} / {fmt(vv[1])} / {fmt(vv[2])}; "
                    f"Abdeckung der Mitte über Neu-Trainings {pct(tm[0])} / {pct(tm[1])} / {pct(tm[2])}, der Wichtigkeit des Modells {pct(pp[0])} / {pct(pp[1])} / {pct(pp[2])}.")


def verdict_zwilling(grid, kl):
    s = twin_stats(grid, kl)
    n = C.N_SEEDS
    return "warning", (f"{C.CLASS_LABELS[kl]}, {n} Zufalls-Seeds, 1000 Testzeilen, Intervall B: Permutation nennt Vorlauf und Zwilling beide wichtig (beide Intervalle über 0 in {pct(s['perm']['beide_ueber_0'])}), "
                       f"LOCO einzeln nennt beide unwichtig (beide nicht über 0 in {pct(s['loco']['beide_nicht_ueber_0'])}); LOCO ohne beide zusammen: Intervall über 0 in {pct(s['gruppe_ueber_0'])}.")
