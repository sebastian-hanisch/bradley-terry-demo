"""Die Vergleichsverfahren der vorigen Stücke (numpy/Standardbibliothek): Siegquote, Massey (Kleinste Quadrate auf der Punktedifferenz, hier aus den Ausgängen:
jeder Sieg zählt einen Punkt) und Colley, dazu Rangordnung mit Toleranz, Kendall-τ und der Anteil falsch geordneter Staffel-Paare.

Notation: G[i, j] = Zahl der Partien von i gegen j (symmetrisch), W[i, j] = Siege von i gegen j."""

import numpy as np

import turnier_generator as T


def laplacian(G) -> np.ndarray:
    """Laplace-Matrix des Spielgraphen: Diagonale = Zahl der Partien, außerhalb = minus Zahl der Partien zwischen den beiden."""
    A = np.asarray(G, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def massey_rating(W, G) -> np.ndarray:
    """Massey-Wertung per Minimum-Norm-Lösung von M r = p mit p_i = Siege minus Niederlagen (jeder Sieg ein Punkt Vorsprung)."""
    W = np.asarray(W, dtype=float)
    p = (W - W.T).sum(axis=1)
    r, *_ = np.linalg.lstsq(laplacian(G), p, rcond=None)
    return r


def colley_rating(W, G) -> np.ndarray:
    """Colley-Wertung: (2I + M) r = 1 + (Siege - Niederlagen)/2; Mittel aller Wertungen 1/2."""
    W = np.asarray(W, dtype=float)
    G = np.asarray(G, dtype=float)
    wins = W.sum(axis=1)
    losses = G.sum(axis=1) - wins
    return np.linalg.solve(2.0 * np.eye(len(G)) + laplacian(G), 1.0 + (wins - losses) / 2.0)


def win_share(W, G) -> np.ndarray:
    """Siegquote je Teilnehmer (Siege je Partie; ohne Partien 0)."""
    wins = np.asarray(W, dtype=float).sum(axis=1)
    games = np.asarray(G, dtype=float).sum(axis=1)
    return np.divide(wins, games, out=np.zeros_like(wins), where=games > 0)


def rank_with_tolerance(score, rng: T.SplitMix64, tol: float = 1e-9) -> tuple:
    """Platz je Teilnehmer (0 = bester) nach absteigender Wertung. Wertungen, die sich um weniger als `tol` unterscheiden, sind gleich und werden
    durch das Los getrennt: Sonst entschiede Gleitkomma-Rauschen (1e-16) über Gleichstände, die in vollständigen Plänen exakt vorkommen."""
    score = np.asarray(score, dtype=float)
    key = np.round(score / tol).astype(np.int64)
    lots = rng.uniform(len(score))
    order = sorted(range(len(score)), key=lambda i: (-key[i], lots[i]))
    pos = [0] * len(score)
    for place, i in enumerate(order):
        pos[i] = place
    return tuple(pos)


def kendall_tau(rank, values) -> float:
    """τ zwischen den Plätzen (kleiner = besser) und einer Größe `values` (größer = besser); Paare mit gleichem Wert zählen nicht."""
    r = np.asarray(rank, dtype=float)
    v = np.asarray(values, dtype=float)
    iu = np.triu_indices(len(r), 1)
    sv = np.sign(v[iu[0]] - v[iu[1]])
    sr = np.sign(r[iu[1]] - r[iu[0]])
    counted = sv != 0
    if not counted.any():
        return float("nan")
    return float((sv[counted] * sr[counted]).mean())


def crossing_wrong(rank, n_weak: int) -> float:
    """Anteil der Paare (stark, schwach) aus zwei Staffeln, in denen der Schwache in der Tabelle vor dem Starken steht (Indexhälften)."""
    r = np.asarray(rank)
    weak, strong = r[:n_weak], r[n_weak:]
    if len(weak) == 0 or len(strong) == 0:
        return float("nan")
    return float((weak[:, None] < strong[None, :]).mean())
