"""Erfundener Verweilzeit-Generator (Formeln aus verweilzeit-demo/vwz_generator.py KOPIERT, nicht importiert), gezogen mit SplitMix64 (wik_rng).

Merkmale spaltenweise: 0 Art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 Kunde (0 bis 11), 2 Zoll, 3 Wochenende, 4 Vorlauf (Tage), 5 Gewicht (reines Rauschen),
6 Zwilling = Vorlauf + N(0; 0,05²) (nur im Zwillings-Experiment benutzt; die wahre Funktion liest ihn nicht).
Zielgröße ln(Verweilzeit) = mu(x) + s(art) * z; mu(x) ist analytisch bekannt und dient als Wahrheit (bekannte Funktion mu).
Es gibt keine reale Datenquelle."""

import numpy as np

import wik_constants as C
from wik_rng import SplitMix64


def mu(X):
    """Wahre Mitte der log-Verweilzeit (vektorisiert). Der Vorlauf wirkt nur bei Export und Umschlag: die strukturelle Wechselwirkung Art × Vorlauf."""
    X = np.asarray(X, dtype=float)
    art = X[:, 0].astype(int)
    u = np.asarray(C.UK)[X[:, 1].astype(int)] * np.asarray(C.U_SCALE)[art]
    zoll, we, vor = X[:, 2], X[:, 3], X[:, 4]
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vor, 0.3 + 0.8 * vor],
        default=7.0 * np.exp(0.30 * we),
    )
    return np.log(med) + u


def mu_clamped(X):
    """Dieselbe Funktion auf allen möglichen Zeilen (Vorlauf in den Träger der Art geklemmt), außerhalb der Stütze anders fortgesetzt."""
    X = np.array(X, dtype=float)
    a = X[:, 0].astype(int)
    lo_e, hi_e = C.VORLAUF_EXPORT
    lo_u, hi_u = C.VORLAUF_UMSCHLAG
    X[:, 4] = np.where(a == 1, np.clip(X[:, 4], lo_e, hi_e), np.where(a == 2, np.clip(X[:, 4], lo_u, hi_u), X[:, 4]))
    return mu(X)


def noise_sd(art):
    return np.asarray(C.S_BASE)[np.asarray(art, dtype=int)]


def draw_features(rng: SplitMix64, n: int, noise_columns: bool = True):
    """n natürliche Zeilen (n, 7). Je Zeile werden alle Zufallszahlen gezogen, damit die ersten k Zeilen nicht von n abhängen."""
    X = np.zeros((n, C.D_TWIN))
    for i in range(n):
        art = rng.choice(C.ART_P)
        kunde = rng.below(C.K_KUNDEN)
        u_zoll = rng.random()
        tday = rng.random() * 7000.0
        u_v = rng.random()
        gew = rng.normal() if noise_columns else 0.0
        tw = rng.normal() if noise_columns else 0.0
        X[i, 0] = art
        X[i, 1] = kunde
        X[i, 2] = 1.0 if (u_zoll < C.P_ZOLL and art == 0) else 0.0
        X[i, 3] = 1.0 if (int(tday) % 7) >= 5 else 0.0
        if art == 1:
            X[i, 4] = C.VORLAUF_EXPORT[0] + (C.VORLAUF_EXPORT[1] - C.VORLAUF_EXPORT[0]) * u_v
        elif art == 2:
            X[i, 4] = C.VORLAUF_UMSCHLAG[0] + (C.VORLAUF_UMSCHLAG[1] - C.VORLAUF_UMSCHLAG[0]) * u_v
        X[i, 5] = gew
        X[i, 6] = X[i, 4] + C.TWIN_SD * tw
    return X


def make_history(n: int, seed: int):
    """Daten (X mit 7 Spalten, y = ln Verweilzeit) aus dem Zufallsstrom `seed`."""
    rng = SplitMix64(seed)
    X = draw_features(rng, n)
    z = np.array([rng.normal() for _ in range(n)])
    y = mu(X) + noise_sd(X[:, 0]) * z
    return X, y


def draw_rows(n: int, seed: int, noise_columns: bool = True):
    """Nur Merkmale (ohne Zielgröße). noise_columns=False lässt das Gewicht auf 0 und den Zwilling gleich dem Vorlauf (spart die langsamen Normalzahlen; mu liest beide nicht)."""
    return draw_features(SplitMix64(seed), n, noise_columns)


def is_valid(X):  # Spalte 6 (Zwilling) wird nicht geprüft
    """True, wenn die Kombination im Generator vorkommen kann (Träger der gemeinsamen Verteilung)."""
    X = np.asarray(X, dtype=float)
    art = X[:, 0].astype(int)
    v = X[:, 4]
    ok = np.ones(len(X), dtype=bool)
    ok &= ~(np.isin(art, (0, 3)) & (v != 0))
    ok &= ~((art == 1) & ((v < C.VORLAUF_EXPORT[0]) | (v > C.VORLAUF_EXPORT[1])))
    ok &= ~((art == 2) & ((v < C.VORLAUF_UMSCHLAG[0]) | (v > C.VORLAUF_UMSCHLAG[1])))
    ok &= ~((X[:, 2] == 1) & (art != 0))
    return ok
