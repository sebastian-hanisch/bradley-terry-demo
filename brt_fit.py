"""Bradley-Terry-Modell: Siegwahrscheinlichkeit P(i schlägt j) = σ(θ_i − θ_j). Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben
(DEMO-PLAYBOOK §1b); nur numpy.

Notation: W[i, j] = Siege von i gegen j, G = W + Wᵀ = Zahl der Partien je Paar. Die Log-Likelihood ist Σ_{i≠j} W_ij log σ(θ_i − θ_j), penalisiert mit
−(λ/2) Σ θ_i² (Ridge: eine Vorannahme „die Stärken streuen um 0 mit Varianz 1/λ“). Die Wahrscheinlichkeit hängt nur an Differenzen, deshalb werden die
Stärken auf Mittelwert 0 gesetzt."""

from dataclasses import dataclass

import numpy as np


def logistic(x) -> np.ndarray:
    """σ(x) = 1/(1 + e^(−x)), numerisch stabil (kein Überlauf für große |x|)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def log_logistic(x) -> np.ndarray:
    """log σ(x), stabil."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def _prepared(W) -> np.ndarray:
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    return W


def win_probabilities(theta) -> np.ndarray:
    """P[i, j] = σ(θ_i − θ_j); die Diagonale ist 1/2 und wird nirgends gebraucht."""
    th = np.asarray(theta, dtype=float)
    return logistic(th[:, None] - th[None, :])


def log_likelihood(theta, W, lam: float = 0.0) -> float:
    """Penalisierte Log-Likelihood: Σ_{i≠j} W_ij log σ(θ_i − θ_j) − (λ/2) Σ θ²."""
    th = np.asarray(theta, dtype=float)
    W = _prepared(W)
    ll = float((W * log_logistic(th[:, None] - th[None, :])).sum())
    return ll - 0.5 * lam * float(th @ th)


def gradient(theta, W, lam: float = 0.0) -> np.ndarray:
    """∂/∂θ_i = Siege_i − Σ_j G_ij σ(θ_i − θ_j) − λ θ_i."""
    th = np.asarray(theta, dtype=float)
    W = _prepared(W)
    G = W + W.T
    P = win_probabilities(th)
    return W.sum(axis=1) - (G * P).sum(axis=1) - lam * th


def fisher_information(theta, W) -> np.ndarray:
    """Beobachtete Information der Likelihood (ohne Penalty): diag(V 1) − V mit V_ij = G_ij σ_ij (1 − σ_ij). Positiv semidefinit, Kern = Einsvektor."""
    th = np.asarray(theta, dtype=float)
    W = _prepared(W)
    G = W + W.T
    P = win_probabilities(th)
    V = G * P * (1.0 - P)
    np.fill_diagonal(V, 0.0)
    return np.diag(V.sum(axis=1)) - V


def hessian(theta, W, lam: float = 0.0) -> np.ndarray:
    """Hesse-Matrix der penalisierten Log-Likelihood: −(Information + λ I)."""
    return -(fisher_information(theta, W) + lam * np.eye(len(theta)))


@dataclass(frozen=True)
class Fit:
    theta: np.ndarray          # Stärken, Mittelwert 0
    iterations: int            # Newton-Schritte
    converged: bool
    information: np.ndarray    # Information + λ I an der Lösung (positiv definit)


def fit_penalized(W, lam: float, max_iter: int = 200, tol: float = 1e-10) -> Fit:
    """Penalisierte Maximum Likelihood per Newton-Verfahren mit Schrittverkürzung (die Zielfunktion ist für λ > 0 streng konkav). Existiert der
    ungeschützte Schätzer nicht, so liefert die Penalty trotzdem eine endliche Lösung; ihr Gewicht λ ist dann das, was die Ordnung bestimmt."""
    if lam <= 0:
        raise ValueError("lam muss positiv sein; ohne Penalty existiert der Schätzer nicht immer (siehe brt_existence.ford_condition)")
    W = _prepared(W)
    n = len(W)
    theta = np.zeros(n)
    obj = log_likelihood(theta, W, lam)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = gradient(theta, W, lam)
        step = np.linalg.solve(fisher_information(theta, W) + lam * np.eye(n), g)
        # Der exakte Schritt hat Mittelwert 0 (die Gradientensumme ist −λ Σθ = 0). Rundungsrauschen in der Richtung „alle gleich“ würde durch 1/λ verstärkt,
        # die Schrittzahl wäre dann plattformabhängig; die Projektion ändert die Lösung nicht.
        step = step - step.mean()
        t = 1.0
        while True:
            cand = theta + t * step
            new = log_likelihood(cand, W, lam)
            if new >= obj - 1e-12 * (1.0 + abs(obj)) or t < 1e-12:
                break
            t *= 0.5
        moved = float(np.abs(cand - theta).max())
        theta, obj = cand, new
        if moved < tol:
            converged = True
            break
    theta = theta - theta.mean()
    return Fit(theta, it, converged, fisher_information(theta, W) + lam * np.eye(n))


def fit_mm(W, max_iter: int = 20000, tol: float = 1e-10) -> tuple:
    """Zermelo-Iteration (MM-Algorithmus, Hunter 2004) ohne Penalty: π_i ← Siege_i / Σ_j G_ij/(π_i + π_j). Nur wo der Schätzer existiert; sonst (None, max_iter).
    Rückgabe (θ = log π mit Mittelwert 0, Iterationen)."""
    W = _prepared(W)
    G = W + W.T
    w = W.sum(axis=1)
    n = len(W)
    if (w == 0).any():
        return None, max_iter
    pi = np.ones(n)
    for it in range(1, max_iter + 1):
        denom = (G / (pi[:, None] + pi[None, :])).sum(axis=1)
        new = w / denom
        new = new / np.exp(np.mean(np.log(new)))
        if float(np.abs(np.log(new) - np.log(pi)).max()) < tol:
            return np.log(new), it
        pi = new
    return None, max_iter


def standard_errors(information) -> np.ndarray:
    """Standardfehler von θ_i − Mittel(θ): Wurzel der Diagonale von C Σ C mit Σ = information⁻¹ und C = I − 11ᵀ/n. Nur die identifizierte Richtung zählt;
    ohne die Projektion steckte 1/(nλ) aus der Nullrichtung der Likelihood im Ergebnis."""
    info = np.asarray(information, dtype=float)
    n = len(info)
    C = np.eye(n) - np.ones((n, n)) / n
    cov = C @ np.linalg.inv(info) @ C
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def oracle_lambda(n: int, delta: float) -> float:
    """λ, die zur Streuung der Generator-Stärken passt: 1/Varianz mit Var(s) = δ²(n² − 1)/12 (nur in der Simulation bekannt)."""
    return 12.0 / (delta ** 2 * (n ** 2 - 1))
