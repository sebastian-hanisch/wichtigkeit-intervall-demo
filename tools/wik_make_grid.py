"""Rechnet das Raster der Demo und friert es als data/wik_grid.json (und data/wik_example.json) ein.

Aufruf (Test-venv mit scikit-learn, OMP_NUM_THREADS=1; Rechenzeit je nach Last 20 bis 60 Minuten):
    python tools/wik_make_grid.py                     # Raster nach data/
    python tools/wik_make_grid.py --offset 1 --out X  # Gegenprobe mit zweitem Zufalls-Seed-Bereich (alle Zufallsströme um 1 verschoben, andere Zufalls-Seeds der Zellen)
    python tools/wik_make_grid.py --check             # prüft die eingefrorene Datei: Form, Wahrheit je Verfahren und eine Zelle als Band

Je Klasse (Boosting, Random Forest, Netz) und Zufalls-Seed (= Neu-Training mit 1000 Trainingszeilen und EIGENEN 3000 Testzeilen, 7 Spalten mit dem Zwilling des Vorlaufs):
  * Verlustdifferenzen je Testzeile für 3 Verfahren x 6 Merkmale (Permutation und bedingte Permutation mit 3 Umordnungen; LOCO mit Neu-Training ohne das Merkmal),
  * Schätzwert und Intervalle B (Bootstrap-Perzentile, 300 Neuziehungen) und N (Normalnäherung) für die Testzeilen 300, 1000, 3000 (Anfänge des Testsatzes),
  * dasselbe Verfahren auf einer festen Bevölkerung von 30 000 Zeilen (Ziel „Wichtigkeit des Modells“),
  * das Zwillings-Experiment (n = 1000, 1000 Testzeilen).
Wahrheit je Verfahren für die bekannte Funktion mu: Permutation und bedingte Permutation aus 300 000 Zeilen, LOCO exakt aus der bedingten Erwartung (20 000 Zeilen)."""

import argparse
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import wik_constants as C  # noqa: E402
import wik_generator as G  # noqa: E402
import wik_importance as IMP  # noqa: E402
import wik_intervals as IV  # noqa: E402
import wik_value as V  # noqa: E402
from wik_rng import SplitMix64  # noqa: E402

DATA = ROOT / "data"
OFFSET = 0


def r6(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def make_model(kl, seed, n_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if kl == "gbm":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed)
    if kl == "rf":
        return RandomForestRegressor(n_estimators=60, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=1)
    ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]), ("sk", StandardScaler(), list(range(2, n_cols)))])
    return make_pipeline(ct, MLPRegressor(hidden_layer_sizes=(16,), alpha=3.0, max_iter=500, solver="lbfgs", random_state=seed))


def _drop(kl, X, cols):
    """Merkmale weglassen: Boosting und Wald wirklich (Spalten entfernen), das Netz durch eine Konstante ersetzen (gleiche Architektur)."""
    cols = list(cols)
    if kl == "mlp":
        Z = np.array(X, dtype=float)
        Z[:, cols] = 0.0
        return Z
    return np.delete(X, cols, axis=1)


def fit_predict(kl, Xtr, ytr, seed, drop=()):
    """Modell trainieren (ohne die Spalten `drop`) und eine Vorhersagefunktion auf Zeilen mit allen Spalten zurückgeben."""
    ncols = Xtr.shape[1]
    Z = _drop(kl, Xtr, drop) if drop else Xtr
    m = make_model(kl, seed, Z.shape[1]).fit(Z, ytr)
    return (lambda X: m.predict(_drop(kl, X, drop))) if drop else (lambda X: m.predict(X[:, :ncols]))


# ------------------------------------------------------------------------------------------------------------------------------ Wahrheit
def truth_block():
    t = {}
    X = G.draw_rows(C.TRUTH_ROWS, C.SEED_TRUTH + OFFSET, noise_columns=False)
    base = G.mu(X)
    for name, cond in (("perm", False), ("cperm", True)):
        vals = []
        for j in range(C.D):
            if j in C.COL_NOISE_FEATURES:
                vals.append(0.0)
                continue
            acc = 0.0
            for r in range(C.R_PERM):
                rng = SplitMix64(C.SEED_TRUTH + OFFSET + 1000 * (1 + cond) + 10 * j + r)
                idx = IMP.permute_within(len(X), IMP.strata_for(X, j), rng) if cond else IMP.shuffled_index(len(X), rng)
                Xp = X.copy()
                Xp[:, j] = X[idx, j]
                acc += float(np.mean((G.mu(Xp) - base) ** 2))
            vals.append(acc / C.R_PERM)
        t[name] = r6(vals)
    Xl = G.draw_rows(C.LOCO_TRUTH_ROWS, C.SEED_TRUTH + OFFSET + 5000, noise_columns=False)
    t["loco"] = r6([V.loco_truth(G.mu, Xl, j) for j in range(C.D)])
    t["rows"] = {"perm": C.TRUTH_ROWS, "cperm": C.TRUTH_ROWS, "loco": C.LOCO_TRUTH_ROWS}
    return t


@lru_cache(maxsize=1)
def _pop():
    return G.make_history(C.POP_ROWS, C.SEED_POP + OFFSET)


@lru_cache(maxsize=2)
def _data(sd):
    return G.make_history(C.N_TRAIN, C.SEED_TRAIN + OFFSET + sd), G.make_history(max(C.TEST_SIZES), C.SEED_TEST + OFFSET + sd)


def _diff_columns(preds, predict_wo, X, y, seed_base, rows=None):
    """Verlustdifferenzen je Zeile als Feld (Zeilen, 18): Permutation (6), bedingte Permutation (6), LOCO (6)."""
    cols = []
    for ci, cond in enumerate((False, True)):
        for j in range(C.D):
            rng = SplitMix64(seed_base + 1000 * ci + j)
            cols.append(IMP.perm_diffs(preds, X, y, j, rng, C.R_PERM, conditional=cond))
    full = preds(X)
    for j in range(C.D):
        cols.append(IMP.loco_diffs(full, predict_wo[j](X), y))
    return np.column_stack(cols)


def job(args):
    kl, sd, offset = args
    global OFFSET
    OFFSET = offset
    (Xtr, ytr), (Xte, yte) = _data(sd)
    Xpop, ypop = _pop()
    model_seed = 1000 * (offset + 1) + sd
    f = fit_predict(kl, Xtr[:, :C.D], ytr, model_seed)                          # 6 Spalten
    wo = [fit_predict(kl, Xtr[:, :C.D], ytr, model_seed, drop=(j,)) for j in range(C.D)]
    D = _diff_columns(f, wo, Xte[:, :C.D], yte, C.SEED_PERM + OFFSET + 100 * sd)
    Dpop = _diff_columns(f, wo, Xpop[:, :C.D], ypop, C.SEED_PERM + OFFSET + 100 * sd + 50)
    pop = Dpop.mean(axis=0)
    est = np.zeros((len(C.TEST_SIZES), 18, 5))
    for mi, m in enumerate(C.TEST_SIZES):
        idx = IV.bootstrap_indices(m, C.N_BOOT, C.SEED_BOOT + OFFSET + 10 * sd + mi)
        for c in range(18):
            est[mi, c] = IV.interval_summary(D[:m, c], idx)
    pred_te = f(Xte[:, :C.D])
    pr = f(Xpop[:5000, :C.D])
    tr = G.mu(Xpop[:5000])
    r2 = 1.0 - float(np.sum((pr - tr) ** 2) / np.sum((tr - tr.mean()) ** 2))
    # Zwilling (7 Spalten, 1000 Testzeilen)
    ft = fit_predict(kl, Xtr, ytr, model_seed)
    wt = {1: fit_predict(kl, Xtr, ytr, model_seed, drop=(4,)), 2: fit_predict(kl, Xtr, ytr, model_seed, drop=(6,)), 3: fit_predict(kl, Xtr, ytr, model_seed, drop=(4, 6))}
    m = 1000
    X1, y1 = Xte[:m], yte[:m]
    full = ft(X1)
    d = {}
    for ci, cond in enumerate((False, True)):
        for j in (4, 6):
            d[(ci, j)] = IMP.perm_diffs(ft, X1, y1, j, SplitMix64(C.SEED_PERM + OFFSET + 100 * sd + 70 + 10 * ci + j), C.R_PERM, conditional=bool(cond))
    d["l4"], d["l6"], d["lg"] = (IMP.loco_diffs(full, wt[1](X1), y1), IMP.loco_diffs(full, wt[2](X1), y1), IMP.loco_diffs(full, wt[3](X1), y1))
    idx = IV.bootstrap_indices(m, C.N_BOOT, C.SEED_BOOT + OFFSET + 10 * sd + 7)
    order = [d[(0, 4)], d[(0, 6)], d[(1, 4)], d[(1, 6)], d["l4"], d["l6"], d["lg"]]
    twin = np.array([[float(np.mean(x)), *IV.interval_b(x, idx)] for x in order])
    return {"kl": kl, "sd": sd, "est": est, "pop": pop, "r2": r2, "twin": twin, "diff": D if sd == 0 else None}


# ------------------------------------------------------------------------------------------------------------------------------ Lauf
def build(workers, classes, offset):
    global OFFSET
    OFFSET = offset
    t0 = time.time()
    print("Wahrheit je Verfahren ...", flush=True)
    truth = truth_block()
    print(f"  fertig nach {time.time() - t0:.0f} s", flush=True)
    jobs = [(kl, sd, offset) for sd in range(C.N_SEEDS) for kl in classes]
    raw = {kl: [] for kl in classes}
    example = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=len(classes))):
            raw[r["kl"]].append(r)
            if r["diff"] is not None:
                example[r["kl"]] = r6(r["diff"])
            if i % 30 == 0:
                print(f"  {i}/{len(jobs)} nach {time.time() - t0:.0f} s", flush=True)
    import sklearn
    cells = {}
    for kl, rs in raw.items():
        rs = sorted(rs, key=lambda r: r["sd"])
        cells[kl] = {"est": r6([r["est"] for r in rs]), "pop": r6([r["pop"] for r in rs]), "r2": r6([r["r2"] for r in rs]), "twin": r6([r["twin"] for r in rs])}
    meta = {
        "stand": time.strftime("%Y-%m-%d"),
        "quelle": "tools/wik_make_grid.py (Zufallsströme SplitMix64, wik_rng); Boosting (200 Bäume, Tiefe 3, Lernrate 0,1), Random Forest (60 Bäume, min. Blatt 3, max_features 0,5), "
                  "Netz (eine versteckte Schicht mit 16 Einheiten, L2 alpha 3, L-BFGS)",
        "umgebung": f"Python {platform.python_version()}, numpy {np.__version__}, scikit-learn {sklearn.__version__}, {platform.system()}",
        "offset": offset, "klassen": list(classes), "n_train": C.N_TRAIN, "testzeilen": list(C.TEST_SIZES), "zufalls_seeds": C.N_SEEDS, "umordnungen": C.R_PERM,
        "neuziehungen": C.N_BOOT, "pop_zeilen": C.POP_ROWS,
        "felder": {"cells[kl].est": "(Zufalls-Seeds 100, Testzeilen 3, Spalten 18 = Verfahren (Permutation, bedingt, LOCO) mal Merkmal (6), 5 = Schätzwert, B unten, B oben, N unten, N oben)",
                   "cells[kl].pop": "(Zufalls-Seeds, 18): dasselbe Verfahren desselben Modells auf der Bevölkerung (Ziel 'Wichtigkeit des Modells')",
                   "cells[kl].twin": "(Zufalls-Seeds, 7, 3): Zwilling, 1000 Testzeilen; 7 = Permutation Vorlauf, Permutation Zwilling, bedingt Vorlauf, bedingt Zwilling, LOCO Vorlauf, LOCO Zwilling, LOCO Gruppe; 3 = Schätzwert, B unten, B oben",
                   "cells[kl].r2": "R² gegen mu auf 5000 Bevölkerungszeilen", "truth": "Wahrheit je Verfahren, 6 Merkmale"},
        "rechenzeit_s": round(time.time() - t0),
    }
    return {"meta": meta, "truth": truth, "cells": cells}, example


def check(path):
    """Form, Wahrheit je Verfahren (neu gerechnet) und eine Zelle (Boosting, Zufalls-Seed 0, Bevölkerungswert und Intervall) als Band."""
    global OFFSET
    grid = json.load(open(path, encoding="utf-8"))
    OFFSET = int(grid["meta"]["offset"])
    ok = True

    def bad(msg):
        nonlocal ok
        ok = False
        print("FEHLER:", msg)

    for kl in grid["meta"]["klassen"]:
        c = grid["cells"][kl]
        if np.shape(c["est"]) != (C.N_SEEDS, 3, 18, 5) or np.shape(c["pop"]) != (C.N_SEEDS, 18) or np.shape(c["twin"]) != (C.N_SEEDS, 7, 3):
            bad(f"Form {kl}")
    t = truth_block_small()
    if np.abs(np.array(t) - np.array(grid["truth"]["perm"])).max() > 0.05 * max(grid["truth"]["perm"]):
        bad(f"Wahrheit Permutation: {t} gegen {grid['truth']['perm']}")
    if "gbm" in grid["meta"]["klassen"]:
        r = job(("gbm", 0, OFFSET))
        old = np.array(grid["cells"]["gbm"]["est"][0])
        if np.abs(r["est"][:, :, 0] - old[:, :, 0]).max() > 2e-3:
            bad(f"Schätzwerte Boosting Zufalls-Seed 0: größte Abweichung {np.abs(r['est'][:, :, 0] - old[:, :, 0]).max():.2e}")
    print("wik_grid.json: OK" if ok else "wik_grid.json: ABWEICHUNG")
    return 0 if ok else 1


def truth_block_small():
    """Wahrheit der Permutation mit 60 000 statt 300 000 Zeilen (nur zur Kontrolle der Größenordnung, Band 5 %)."""
    X = G.draw_rows(60_000, C.SEED_TRUTH + OFFSET + 777, noise_columns=False)
    base = G.mu(X)
    out = []
    for j in range(C.D):
        if j in C.COL_NOISE_FEATURES:
            out.append(0.0)
            continue
        acc = 0.0
        for r in range(C.R_PERM):
            idx = IMP.shuffled_index(len(X), SplitMix64(9000 + 10 * j + r))
            Xp = X.copy()
            Xp[:, j] = X[idx, j]
            acc += float(np.mean((G.mu(Xp) - base) ** 2))
        out.append(acc / C.R_PERM)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(DATA / "wik_grid.json"))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--classes", default=",".join(C.CLASSES))
    a = ap.parse_args()
    if a.check:
        return check(a.out)
    grid, example = build(a.workers, tuple(a.classes.split(",")), a.offset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(grid, fh, ensure_ascii=False, separators=(",", ":"))
    ex_path = Path(a.out).with_name("wik_example.json" if a.offset == 0 else "wik_example_off.json")
    with open(ex_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"quelle": "Verlustdifferenzen je Testzeile des Zufalls-Seeds 0 (3000 Zeilen x 18 Spalten: Permutation, bedingte Permutation, LOCO mal sechs Merkmale)", "stand": grid["meta"]["stand"]}, "diff": example}, fh, separators=(",", ":"))
    print(f"geschrieben: {a.out} und {ex_path.name}, Rechenzeit {grid['meta']['rechenzeit_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
