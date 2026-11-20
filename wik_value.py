"""Bedingte Erwartung der wahren Funktion unter der gemeinsamen Verteilung des Generators (exakte Aufzählung, Mittelpunktregel mit Q Punkten für den Vorlauf).

Gebraucht für die LOCO-Wahrheit: Verlustanstieg des besten Vorhersagers ohne das Merkmal j, E[(mu - E[mu | X ohne j])^2], exakt aus dem Generator (Kopie der
Rechnung aus wechselwirkung-lage-demo, hier unabhängig gehalten). Spieler: Art, Kunde, Zoll, Wochenende, Vorlauf; das Rauschmerkmal Gewicht ist kein Spieler (mu liest es nicht)."""

import numpy as np

import wik_constants as C

Q_VORLAUF = 20


def art_posterior(X, obs):
    n = len(X)
    pa = np.asarray(C.ART_P, dtype=float)
    if obs[0]:
        W = np.zeros((n, 4))
        W[np.arange(n), X[:, 0].astype(int)] = 1.0
        return W
    W = np.tile(pa, (n, 1))
    if obs[2]:
        z = X[:, 2]
        L = np.ones((n, 4))
        L[:, 0] = np.where(z == 1, C.P_ZOLL, 1 - C.P_ZOLL)
        L[:, 1:] = np.where(z[:, None] == 1, 0.0, 1.0)
        W = W * L
    if obs[4]:
        v = X[:, 4]
        L = np.zeros((n, 4))
        L[:, 0] = v == 0
        L[:, 3] = v == 0
        lo_e, hi_e = C.VORLAUF_EXPORT
        lo_u, hi_u = C.VORLAUF_UMSCHLAG
        L[:, 1] = np.where((v >= lo_e) & (v <= hi_e), 1 / (hi_e - lo_e), 0)
        L[:, 2] = np.where((v >= lo_u) & (v <= hi_u), 1 / (hi_u - lo_u), 0)
        W = W * L
    s = W.sum(axis=1, keepdims=True)
    return W / np.where(s > 0, s, 1.0)


def cond_expect(f, X, obs, q: int = Q_VORLAUF):
    """E[f(X) | X_S = x_S] für jede Zeile von X; obs = Maske über (Art, Kunde, Zoll, Wochenende, Vorlauf)."""
    X = np.asarray(X, dtype=float)
    obs = np.asarray(obs, dtype=bool)
    ne = len(X)
    W = art_posterior(X, np.r_[obs, False])
    tot = np.zeros(ne)
    for a in range(4):
        if W[:, a].max() <= 0:
            continue
        opt = [(None, np.array([1.0])) if obs[1] else (np.arange(float(C.K_KUNDEN)), np.full(C.K_KUNDEN, 1 / C.K_KUNDEN))]
        if obs[2]:
            opt.append((None, np.array([1.0])))
        else:
            opt.append((np.array([0.0, 1.0]), np.array([1 - C.P_ZOLL, C.P_ZOLL])) if a == 0 else (np.array([0.0]), np.array([1.0])))
        opt.append((None, np.array([1.0])) if obs[3] else (np.array([0.0, 1.0]), np.array([1 - C.P_WOCHENENDE, C.P_WOCHENENDE])))
        if obs[4]:
            opt.append((None, np.array([1.0])))
        elif a == 1:
            lo, hi = C.VORLAUF_EXPORT
            opt.append(((np.arange(q) + 0.5) / q * (hi - lo) + lo, np.full(q, 1 / q)))
        elif a == 2:
            lo, hi = C.VORLAUF_UMSCHLAG
            opt.append(((np.arange(q) + 0.5) / q * (hi - lo) + lo, np.full(q, 1 / q)))
        else:
            opt.append((np.array([0.0]), np.array([1.0])))
        idx = np.indices([len(w) for _, w in opt]).reshape(4, -1)
        wc = np.prod([opt[j][1][idx[j]] for j in range(4)], axis=0)
        nc = idx.shape[1]
        rows = np.zeros((nc, ne, X.shape[1]))
        rows[:, :, 0] = a
        for j, (vals, _) in enumerate(opt):
            rows[:, :, j + 1] = X[None, :, j + 1] if vals is None else vals[idx[j]][:, None]
        pred = f(rows.reshape(-1, X.shape[1])).reshape(nc, ne)
        tot += W[:, a] * (wc[:, None] * pred).sum(axis=0)
    return tot


def loco_truth(f, X, j: int, chunk: int = 250):
    """Wahrheit der LOCO-Wichtigkeit des Merkmals j (0 bis 4) für die Funktion f: Mittel von (f - E[f | X ohne j])² über die Zeilen X; Gewicht (5) ist 0."""
    if j >= 5:
        return 0.0
    obs = np.ones(5, dtype=bool)
    obs[j] = False
    acc = 0.0
    for s in range(0, len(X), chunk):
        sl = X[s:s + chunk]
        acc += float(((f(sl) - cond_expect(f, sl, obs)) ** 2).sum())
    return acc / len(X)
