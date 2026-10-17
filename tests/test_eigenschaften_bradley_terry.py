"""Eigenschaftstests bradley-terry-demo (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für den (penalisierten) Bradley-Terry-Schätzer gelten müssen, ohne dass das
richtige Ergebnis bekannt ist. Instanzen: 30 Turniere aus dem Generator (feste Seeds, n = 2 bis 8, alle drei Spielpläne, 1 bis 4 Partien je Paar) plus Randfälle
(zwei Teilnehmer, Zyklus, Ungeschlagener, zerfallender Plan, keine Partie). Die Aussagen folgen aus der Definition (Log-Likelihood Σ W_ij log σ(θ_i − θ_j) − (λ/2) Σ θ²).

1. Umnummerieren der Teilnehmer permutiert die Schätzung (λ > 0, auch wo der ungeschützte Schätzer fehlt; und die Zermelo-Iteration wo er existiert).
2. Siege und Niederlagen vertauschen (W -> Wᵀ) negiert die Schätzung; Existenz, Ungeschlagene und Sieglose tauschen die Rollen.
3. Zentriert: Mittelwert 0 (Newton und Zermelo).
4. Score-Gleichung: Siege_i - erwartete Siege_i = λ θ_i (λ > 0) bzw. = 0 (ungeschützt, Zermelo); die Lösung maximiert die penalisierte Log-Likelihood (keine Störung verbessert).
5. Größeres λ schrumpft den Abstand zum Prior 0 monoton (‖θ‖₂ fällt: aus der Optimalität beider Lösungen), und λ → ∞ liefert den Prior.
6. Verdoppeln aller Partien: der ungeschützte Schätzer bleibt gleich (Likelihood-Gleichung ist skaleninvariant), die Standardfehler teilen sich durch √2 (λ → 0);
   an derselben Stelle ist die Information mit 2W größer, also das Intervall enger (jedes λ).
7. Wahrscheinlichkeiten und Log-Verlust: P_ij + P_ji = 1; der Münz-Log-Verlust ist ln 2 für jede Partienmatrix; der Log-Verlust ist unter Umnummerieren invariant.
"""

import math

import numpy as np
import pytest

import brt_existence as E
import brt_fit as F
import brt_heldout as H
import turnier_generator as T

N_INSTANCES = 30


def instance(seed):
    n = 2 + seed % 7
    t = T.build_turnier(n, 0.5, 1 + seed % 4, T.PLANS[seed % 3], 1, 0.7, 0.0 if seed % 5 else 0.2, "ausgewogen", 7000 + seed)
    return np.asarray(t.wins, dtype=float)


def instances():
    return [instance(s) for s in range(N_INSTANCES)]


def extra_cases():
    cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    trans = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
    pair = np.array([[0, 3], [1, 0]], dtype=float)
    islands = np.zeros((4, 4))
    islands[0, 1] = 2
    islands[2, 3] = 1
    return [cyc, trans, pair, islands, np.zeros((3, 3)), np.zeros((2, 2))]


def expected_wins(theta, W):
    G = W + W.T
    return (G * F.win_probabilities(theta)).sum(axis=1)          # die Diagonale von G ist 0


# --- 1. Umnummerieren ---------------------------------------------------------------------------------------------------------------------

def test_relabelling_the_participants_permutes_the_estimate():
    rng = T.SplitMix64(5)
    for W in instances() + extra_cases():
        perm = rng.permutation(len(W))
        W2 = W[np.ix_(perm, perm)]
        for lam in (0.05, 1.0):
            assert np.allclose(F.fit_penalized(W2, lam).theta, F.fit_penalized(W, lam).theta[perm], atol=1e-7)
        if E.ford_condition(W) and len(W) > 1:
            assert np.allclose(F.fit_mm(W2)[0], F.fit_mm(W)[0][perm], atol=1e-6)


# --- 2. Sieger und Verlierer vertauschen ----------------------------------------------------------------------------------------------------

def test_swapping_winners_and_losers_negates_the_estimate():
    for W in instances() + extra_cases():
        for lam in (0.05, 1.0):
            assert np.allclose(F.fit_penalized(W.T, lam).theta, -F.fit_penalized(W, lam).theta, atol=1e-7)
        if E.ford_condition(W) and len(W) > 1:
            assert np.allclose(F.fit_mm(W.T)[0], -F.fit_mm(W)[0], atol=1e-6)


def test_swapping_winners_and_losers_swaps_the_roles_in_the_existence_report():
    for W in instances() + extra_cases():
        assert E.ford_condition(W) == E.ford_condition(W.T)
        assert E.undefeated(W) == E.winless(W.T) and E.winless(W) == E.undefeated(W.T)
        assert sorted(E.win_graph_components(W)) == sorted(E.win_graph_components(W.T))


# --- 3. Zentrierung ------------------------------------------------------------------------------------------------------------------------

def test_the_estimate_is_centred():
    for W in instances() + extra_cases():
        for lam in (0.01, 0.3, 3.0):
            fit = F.fit_penalized(W, lam)
            assert fit.theta.mean() == pytest.approx(0.0, abs=1e-10)
        if E.ford_condition(W) and len(W) > 1:
            assert F.fit_mm(W)[0].mean() == pytest.approx(0.0, abs=1e-9)


# --- 4. Score-Gleichung ----------------------------------------------------------------------------------------------------------------------

def test_score_equation_wins_minus_expected_wins_equals_lambda_theta():
    checked = 0
    for W in instances() + extra_cases():
        if not W.any():
            continue
        for lam in (0.02, 0.5, 4.0):
            fit = F.fit_penalized(W, lam)
            assert fit.converged
            assert np.allclose(W.sum(axis=1) - expected_wins(fit.theta, W), lam * fit.theta, atol=1e-6)
        checked += 1
    assert checked >= 30


def test_unpenalised_estimate_reproduces_the_wins_of_every_participant():
    checked = 0
    for W in instances() + extra_cases()[:3]:
        if len(W) < 2 or not E.ford_condition(W):
            continue
        theta, _ = F.fit_mm(W)
        assert np.allclose(W.sum(axis=1), expected_wins(theta, W), atol=1e-5)
        checked += 1
    assert checked >= 8


def test_the_solution_maximises_the_penalised_log_likelihood():
    rng = T.SplitMix64(9)
    for W in instances():
        if not W.any():
            continue
        lam = 0.3
        fit = F.fit_penalized(W, lam)
        best = F.log_likelihood(fit.theta, W, lam)
        for _ in range(5):
            noise = (rng.uniform(len(W)) - 0.5) * 0.4
            assert F.log_likelihood(fit.theta + noise, W, lam) <= best + 1e-9


# --- 5. Schrumpfung --------------------------------------------------------------------------------------------------------------------------

def test_a_larger_lambda_shrinks_the_estimate_towards_the_prior_monotonically():
    grid = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
    checked = 0
    for W in instances() + extra_cases()[:4]:
        if not W.any():
            continue
        norms = [float(np.linalg.norm(F.fit_penalized(W, lam).theta)) for lam in grid]
        assert all(b <= a + 1e-9 for a, b in zip(norms, norms[1:]))
        checked += 1
    assert checked >= 30


def test_an_infinite_prior_returns_the_prior():
    for W in instances():
        if not W.any():
            continue
        theta = F.fit_penalized(W, 1e6).theta
        assert np.abs(theta).max() < 1e-4
    # kein Spiel: jedes λ liefert den Prior
    assert np.allclose(F.fit_penalized(np.zeros((3, 3)), 0.5).theta, 0.0)


# --- 6. Verdoppeln aller Partien -------------------------------------------------------------------------------------------------------------

def test_doubling_all_games_keeps_the_unpenalised_estimate_and_divides_the_standard_errors_by_root_two():
    checked = 0
    for W in instances():
        if len(W) < 3 or not E.ford_condition(W):
            continue
        t1, _ = F.fit_mm(W)
        t2, _ = F.fit_mm(2 * W)
        assert np.allclose(t1, t2, atol=1e-6)
        f1, f2 = F.fit_penalized(W, 1e-7, max_iter=500), F.fit_penalized(2 * W, 1e-7, max_iter=500)
        se1, se2 = F.standard_errors(f1.information), F.standard_errors(f2.information)
        assert np.allclose(se2 * math.sqrt(2.0), se1, rtol=1e-3)
        checked += 1
    assert checked >= 5


def test_with_more_games_at_the_same_point_the_interval_gets_narrower():
    checked = 0
    for W in instances():
        if not W.any() or len(W) < 3:
            continue
        for lam in (0.1, 1.0):
            theta = F.fit_penalized(W, lam).theta
            info1 = F.fisher_information(theta, W) + lam * np.eye(len(W))
            info2 = F.fisher_information(theta, 2 * W) + lam * np.eye(len(W))
            assert (F.standard_errors(info2) <= F.standard_errors(info1) + 1e-12).all()
        checked += 1
    assert checked >= 20


# --- 7. Wahrscheinlichkeiten und Log-Verlust -------------------------------------------------------------------------------------------------

def test_win_probabilities_add_up_to_one_and_follow_the_order_of_the_strengths():
    rng = T.SplitMix64(13)
    for k in range(N_INSTANCES):
        n = 2 + k % 7
        theta = (rng.uniform(n) - 0.5) * 6
        P = F.win_probabilities(theta)
        off = ~np.eye(n, dtype=bool)
        assert np.allclose((P + P.T)[off], 1.0, atol=1e-12)
        i, j = np.nonzero(off)
        assert ((theta[i] > theta[j]) == (P[i, j] > 0.5)).all()


def test_the_coin_log_loss_is_ln_two_and_log_loss_is_invariant_under_relabelling():
    rng = T.SplitMix64(17)
    for W in instances():
        if not W.any():
            continue
        n = len(W)
        assert H.log_loss(H.coin_probabilities(n), W) == pytest.approx(math.log(2.0), abs=1e-12)
        theta = F.fit_penalized(W, 0.5).theta
        P = F.win_probabilities(theta)
        perm = rng.permutation(n)
        assert H.log_loss(P[np.ix_(perm, perm)], W[np.ix_(perm, perm)]) == pytest.approx(H.log_loss(P, W), abs=1e-12)
