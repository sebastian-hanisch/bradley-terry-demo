"""Vorhersage außerhalb der Stichprobe: frische Partien aus derselben Matrix P wie das Turnier (gleiche Stärken, gleiche Stile), Log-Verlust je Partie und
die einfachen Gegenmodelle. Zufall nur über SplitMix64 (abgeleiteter Seed), der gemeinsame Generator bleibt unverändert."""

import numpy as np

import turnier_generator as T

CLIP = 1e-6          # Vorhersagen werden für den Log-Verlust in [CLIP, 1 − CLIP] gehalten
Q_CLIP = 0.02        # geglättete Siegquote in log5 liegt in [0,02; 0,98]


def held_out_games(P, per_pair: int, rng: T.SplitMix64) -> np.ndarray:
    """W_test[i, j] = Siege von i in per_pair frischen Partien gegen j, jede Partie mit Wahrscheinlichkeit P[i, j] (i < j gezogen, j > i ergänzt)."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    u = rng.uniform(m * per_pair).reshape(m, per_pair)
    wins = (u < P[iu][:, None]).sum(axis=1)
    W = np.zeros((n, n), dtype=int)
    W[iu] = wins
    W[iu[1], iu[0]] = per_pair - wins
    return W


def log_loss(P_hat, W_test) -> float:
    """Mittlerer negativer Log-Likelihood je Partie: −Σ W_ij log P̂_ij / Σ W_ij (Einheit nat). Ohne Partien: nan."""
    P_hat = np.asarray(P_hat, dtype=float)
    W_test = np.asarray(W_test, dtype=float)
    mask = (W_test > 0) & ~np.eye(len(W_test), dtype=bool)
    total = W_test[mask].sum()
    if total == 0:
        return float("nan")
    return float(-(W_test[mask] * np.log(np.clip(P_hat[mask], CLIP, 1.0 - CLIP))).sum() / total)


def coin_probabilities(n: int) -> np.ndarray:
    """Das Gegenmodell ohne Wissen: jede Partie 50 zu 50."""
    return np.full((n, n), 0.5)


def log5_probabilities(W, G) -> np.ndarray:
    """log5 aus den geglätteten Siegquoten q_i = (w_i + 1/2)/(g_i + 1): P_ij = q_i(1 − q_j) / (q_i(1 − q_j) + q_j(1 − q_i)). Das ist Bradley-Terry mit
    θ = logit(q), ohne Rücksicht auf die Gegner."""
    W = np.asarray(W, dtype=float)
    G = np.asarray(G, dtype=float)
    q = np.clip((W.sum(axis=1) + 0.5) / (G.sum(axis=1) + 1.0), Q_CLIP, 1.0 - Q_CLIP)
    a = q[:, None] * (1.0 - q[None, :])
    return a / (a + a.T)


def saturated_probabilities(W, G) -> np.ndarray:
    """Das Modell ohne Annahme: je Paar eine eigene Wahrscheinlichkeit (w + 1/2)/(g + 1); ohne Partien 1/2."""
    W = np.asarray(W, dtype=float)
    G = np.asarray(G, dtype=float)
    return (W + 0.5) / (G + 1.0)
