"""Einheitentests des Bradley-Terry-Schätzers gegen die von Hand gerechneten Mini-Instanzen (conftest.py)."""

import math
import random

import numpy as np
import pytest

import brt_fit as F


def test_logistic_is_stable_and_symmetric():
    assert F.logistic(0.0) == 0.5
    assert F.logistic(1000.0) == 1.0 and F.logistic(-1000.0) == 0.0
    assert F.logistic(2.0) + F.logistic(-2.0) == pytest.approx(1.0)
    assert F.log_logistic(-1000.0) == pytest.approx(-1000.0)          # kein Überlauf, kein -inf
    assert F.log_logistic(0.0) == pytest.approx(-math.log(2))


def test_pair_hand_values(pair):
    fit = F.fit_penalized(pair, 1e-9, max_iter=500)
    assert fit.converged
    assert fit.theta == pytest.approx([0.5 * math.log(3), -0.5 * math.log(3)], abs=1e-6)
    assert F.win_probabilities(fit.theta)[0, 1] == pytest.approx(0.75, abs=1e-6)


def test_pair_with_a_prior_has_a_smaller_spread_and_zero_gradient(pair):
    free = F.fit_penalized(pair, 1e-9, max_iter=500).theta[0]
    ridge = F.fit_penalized(pair, 0.5)
    assert 0 < ridge.theta[0] < free
    assert F.gradient(ridge.theta, pair, 0.5) == pytest.approx([0.0, 0.0], abs=1e-8)
    # Bedingung für A: Siege 3 − Partien 4 · σ(2t) − λ t = 0
    t = ridge.theta[0]
    assert 3 - 4 * F.logistic(2 * t) - 0.5 * t == pytest.approx(0.0, abs=1e-8)


def test_cycle_gives_equal_strengths_for_any_prior(cycle3):
    for lam in (1e-6, 0.1, 1.0, 5.0):
        fit = F.fit_penalized(cycle3, lam)
        assert fit.theta == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)
        P = F.win_probabilities(fit.theta)
        assert P[0, 1] == pytest.approx(0.5) and P[1, 2] == pytest.approx(0.5)


def test_transitive_needs_the_prior_and_matches_the_scalar_equation(transitive3):
    fit = F.fit_penalized(transitive3, 1.0)
    assert fit.converged
    a, b, c = fit.theta
    assert b == pytest.approx(0.0, abs=1e-9) and a == pytest.approx(-c)
    assert 2 - F.logistic(a) - F.logistic(2 * a) - 1.0 * a == pytest.approx(0.0, abs=1e-8)
    assert a == pytest.approx(0.5910618, abs=1e-5)
    # ohne (fast) keine Vorannahme läuft die Stärke des Ungeschlagenen weg: je kleiner λ, desto größer
    spreads = [F.fit_penalized(transitive3, lam, max_iter=400).theta[0] for lam in (1.0, 0.01, 1e-4)]
    assert spreads[0] < spreads[1] < spreads[2]


def test_fit_is_centered_and_rejects_a_non_positive_prior(pair, transitive3):
    assert F.fit_penalized(transitive3, 0.3).theta.sum() == pytest.approx(0.0, abs=1e-12)
    for lam in (0.0, -1.0, -1e-9):
        with pytest.raises(ValueError, match="lam muss positiv sein"):       # die eigene Prüfung, nicht erst die Singularität der Matrix (LinAlgError ist ein ValueError)
            F.fit_penalized(pair, lam)


def test_gradient_and_hessian_match_finite_differences():
    rnd = random.Random(5)
    for _ in range(20):
        n = rnd.randint(3, 6)
        W = np.array([[0 if i == j else rnd.randint(0, 4) for j in range(n)] for i in range(n)])
        theta = np.array([rnd.uniform(-1.5, 1.5) for _ in range(n)])
        lam = rnd.choice([0.0, 0.3, 2.0])
        eps = 1e-6
        num_g = np.array([(F.log_likelihood(theta + eps * e, W, lam) - F.log_likelihood(theta - eps * e, W, lam)) / (2 * eps) for e in np.eye(n)])
        assert F.gradient(theta, W, lam) == pytest.approx(num_g, abs=1e-5)
        num_h = np.array([(F.gradient(theta + eps * e, W, lam) - F.gradient(theta - eps * e, W, lam)) / (2 * eps) for e in np.eye(n)])
        assert F.hessian(theta, W, lam) == pytest.approx(num_h, abs=1e-5)


def test_fisher_information_has_the_ones_vector_in_its_kernel():
    W = np.array([[0, 2, 1], [1, 0, 3], [2, 0, 0]])
    info = F.fisher_information(np.array([0.3, -0.2, 0.1]), W)
    assert info @ np.ones(3) == pytest.approx(np.zeros(3), abs=1e-12)
    assert np.allclose(info, info.T)
    assert min(np.linalg.eigvalsh(info)) > -1e-12


def test_standard_errors_hand_value_for_the_pair(pair):
    fit = F.fit_penalized(pair, 1e-9, max_iter=500)
    se = F.standard_errors(fit.information)
    assert se == pytest.approx([0.5 * math.sqrt(4 / 3), 0.5 * math.sqrt(4 / 3)], abs=1e-4)


def test_information_is_the_fisher_information_plus_the_prior_and_gives_the_closed_form_for_the_pair(pair):
    lam = 0.5
    fit = F.fit_penalized(pair, lam)
    assert fit.information == pytest.approx(F.fisher_information(fit.theta, pair) + lam * np.eye(2))
    # Paar: Information entlang (1, −1)/√2 ist 2V mit V = g σ(1 − σ) an θ̂, dazu λ; θ_A − Mittel = (θ_A − θ_B)/2, also se = √(½ / (2V + λ))
    s = F.logistic(fit.theta[0] - fit.theta[1])
    V = 4 * s * (1 - s)
    assert F.standard_errors(fit.information) == pytest.approx([math.sqrt(0.5 / (2 * V + lam))] * 2, abs=1e-9)


def test_standard_errors_ignore_the_unidentified_direction(pair):
    # ohne Projektion stünde 1/(nλ) in der Diagonalen; mit Projektion hängt der Wert bei kleinem λ nicht an λ
    se_small = F.standard_errors(F.fit_penalized(pair, 1e-6, max_iter=500).information)
    se_smaller = F.standard_errors(F.fit_penalized(pair, 1e-9, max_iter=500).information)
    assert se_small == pytest.approx(se_smaller, rel=1e-3)
    assert se_small.max() < 1.0


def test_complete_plan_orders_like_the_wins_for_every_prior():
    rnd = random.Random(11)
    checked = 0
    for _ in range(60):
        n = rnd.randint(3, 7)
        W = np.zeros((n, n), dtype=int)
        k = rnd.randint(1, 4)
        for i in range(n):
            for j in range(i + 1, n):
                w = sum(rnd.random() < 0.5 for _ in range(k))
                W[i, j], W[j, i] = w, k - w
        wins = W.sum(axis=1)
        if len(set(wins)) != n:
            continue
        for lam in (0.01, 1.0, 5.0):
            theta = F.fit_penalized(W, lam).theta
            assert list(np.argsort(-theta)) == list(np.argsort(-wins)), (lam, wins)
        checked += 1
    assert checked > 10


def test_mm_agrees_with_newton_where_the_estimator_exists_and_stops_where_it_does_not(cycle3, transitive3):
    W = np.array([[0, 3, 2, 4], [1, 0, 3, 2], [2, 1, 0, 3], [0, 2, 1, 0]])
    mm, it = F.fit_mm(W)
    assert mm is not None and it < 20000
    assert mm == pytest.approx(F.fit_penalized(W, 1e-9, max_iter=500).theta, abs=1e-4)
    with np.errstate(all="raise"):                       # ohne den frühen Abbruch liefe log(0) durch die Iteration (NaN, 20000 Schritte, gleicher Rückgabewert)
        assert F.fit_mm(transitive3) == (None, 20000)
    assert F.fit_mm(cycle3)[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)


def test_oracle_lambda_formula():
    assert F.oracle_lambda(8, 0.3) == pytest.approx(12 / (0.09 * 63))
    assert F.oracle_lambda(12, 0.6) == pytest.approx(12 / (0.36 * 143))
