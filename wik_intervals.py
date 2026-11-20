"""Intervalle für ein Mittel von Verlustdifferenzen aus den Testzeilen (90 %): Bootstrap-Perzentile (B) und Normalnäherung (N).

Beide berücksichtigen nur die Auswahl der Testzeilen, nicht das Training und nicht die Wahl der Umordnung."""

import numpy as np

import wik_constants as C
from wik_rng import SplitMix64

Z90 = 1.6448536269514722        # zweiseitig 90 % (Normalverteilung)


def bootstrap_indices(m: int, n_boot: int, seed: int):
    """n_boot Neuziehungen der m Testzeilen mit Zurücklegen (SplitMix64); Feld (n_boot, m)."""
    rng = SplitMix64(seed)
    return np.array([[rng.below(m) for _ in range(m)] for _ in range(n_boot)], dtype=np.int32)


def bootstrap_means(d, idx):
    """Mittel der Neuziehungen für eine Spalte d (Feld (m,)) bei Indexfeld idx (n_boot, m)."""
    return np.asarray(d, dtype=float)[idx].mean(axis=1)


def interval_b(d, idx):
    """Bootstrap-Perzentil-Intervall (5. bis 95. Perzentil der Neuziehungsmittel)."""
    means = bootstrap_means(d, idx)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def interval_n(d):
    """Normalnäherung: Mittel ± 1,6449 Standardfehler (Standardabweichung mit m - 1). Bei einer Zeile ist der Standardfehler nicht definiert: None."""
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return None
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    mu = float(d.mean())
    return mu - Z90 * se, mu + Z90 * se


def half_width(lo: float, hi: float) -> float:
    return 0.5 * (hi - lo)


def covers(lo: float, hi: float, target: float) -> bool:
    return lo <= target <= hi


def interval_summary(d, idx):
    """(Schätzwert, B unten, B oben, N unten, N oben) für eine Spalte."""
    b = interval_b(d, idx)
    n = interval_n(d)
    return (float(np.mean(d)), b[0], b[1], n[0], n[1])
