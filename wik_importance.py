"""Wichtigkeit der Merkmale: drei Verfahren, je Testzeile gerechnet (Zunahme des quadratischen Fehlers gegen y).

Permutation: eine Spalte unter allen Testzeilen umordnen (R Umordnungen, gemittelt). Bedingte Permutation: innerhalb von Schichten umordnen (Schicht = Art; für die Art selbst:
Vorlauf größer als 0 gegen gleich 0). LOCO (leave-one-covariate-out): ein Merkmal weglassen, NEU trainieren; Zunahme des Testfehlers gegen das Modell mit allen Merkmalen.
Das ist nicht das Leave-one-out der Trainingspunkte. „Wichtigkeit“ meint hier immer die Wichtigkeit der Merkmale (global), nicht den Beitrag einer Zeile."""

import numpy as np

import wik_constants as C
from wik_rng import SplitMix64

R_DEFAULT = 3


def sq_loss(pred, y):
    return (np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)) ** 2


def shuffled_index(m: int, rng: SplitMix64):
    idx = list(range(m))
    rng.shuffle(idx)
    return np.asarray(idx, dtype=int)


def permute_within(m: int, strata, rng: SplitMix64):
    """Index, der die Zeilen innerhalb jeder Schicht umordnet (jede Schicht behält ihre Werte als Multimenge)."""
    strata = np.asarray(strata)
    out = np.arange(m)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        out[rows] = rows[np.asarray(idx, dtype=int)]
    return out


def strata_for(X, col: int):
    """Schichten der bedingten Permutation: für die Art Vorlauf > 0 gegen = 0, sonst die Art."""
    X = np.asarray(X)
    return (X[:, C.COL_VORLAUF] > 0).astype(int) if col == C.COL_ART else X[:, C.COL_ART].astype(int)


def perm_diffs(predict, X, y, col: int, rng: SplitMix64, r: int = R_DEFAULT, conditional: bool = False):
    """Je Testzeile die mittlere Zunahme des quadratischen Fehlers über r Umordnungen der Spalte `col` (Feld (Zeilen,))."""
    X = np.asarray(X, dtype=float)
    base = sq_loss(predict(X), y)
    strata = strata_for(X, col) if conditional else None
    acc = np.zeros(len(X))
    for _ in range(r):
        idx = permute_within(len(X), strata, rng) if conditional else shuffled_index(len(X), rng)
        Xp = X.copy()
        Xp[:, col] = X[idx, col]
        acc += sq_loss(predict(Xp), y) - base
    return acc / r


def loco_diffs(pred_full, pred_without, y):
    """Je Testzeile: Fehler des Modells ohne das Merkmal minus Fehler des vollen Modells (beide aus Vorhersagen auf denselben Testzeilen)."""
    return sq_loss(pred_without, y) - sq_loss(pred_full, y)


def importance_matrix_columns():
    """Reihenfolge der 18 Spalten der Verlustdifferenzen: Verfahren (Permutation, bedingt, LOCO) mal Merkmal (sechs)."""
    return [(m, j) for m in C.METHODS for j in range(C.D)]
