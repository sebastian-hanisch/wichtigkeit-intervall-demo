"""Abdeckung der Mitte über Neu-Trainings in geschlossener Form.

Streut die Wichtigkeit des Modells über Neu-Trainings mit der Standardabweichung tau und schätzt das Intervall sie mit dem Standardfehler se
(halbe Breite z * se), dann deckt das Intervall die Mitte über Neu-Trainings mit Wahrscheinlichkeit 2 Phi(z / sqrt(1 + r^2)) - 1, r = tau / se.
Das Verhältnis V = tau / halbe Breite der Tabellen ist nicht dieses r: r = z * V."""

from math import erf, sqrt

Z = 1.6448536269514722


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def coverage_of_center(r: float, z: float = Z) -> float:
    """Abdeckung der Mitte über Neu-Trainings bei r = tau / se."""
    return 2.0 * phi(z / sqrt(1.0 + r * r)) - 1.0


def coverage_from_ratio(v: float, z: float = Z) -> float:
    """Dasselbe aus dem Verhältnis V = SD(pop) / halbe Breite: r = z * V."""
    return coverage_of_center(z * v, z)
