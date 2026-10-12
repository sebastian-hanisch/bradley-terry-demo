"""Orakel-Tests bradley-terry-demo: Schätzer, Existenz und Standardfehler gegen ANDERE Rechenwege (DEMO-PLAYBOOK §12).

1. Newton-Lösung gegen scipy.optimize.minimize auf der negativen penalisierten Log-Likelihood, aufgebaut aus der PARTIENLISTE (jede Partie einzeln).
2. Newton (kaum penalisiert) gegen die Zermelo/MM-Iteration, wo der Schätzer existiert.
3. Existenz: Ford-Bedingung gegen networkx.is_strongly_connected; wo sie verletzt ist, wächst die Stärke ohne Grenze, wenn die Vorannahme verschwindet;
   wo sie gilt, konvergiert sie.
4. Vollständig und ausgeglichen: Bradley-Terry-Ordnung gleich Siegtabelle (aus dem Generator, mit der Likelihood-Gleichung hergeleitet).
5. Standardfehler gegen die Pseudoinverse der Information (λ → 0) und gegen die Handrechnung des Paars.
6. Frische Partien: Mittel gegen die Wahrscheinlichkeitsmatrix; Log-Verlust der Wahrheit nähert sich der Entropie (Erwartungswert).
"""

import itertools
import random

import networkx as nx
import numpy as np
import pytest

import brt_existence as E
import brt_fit as F
import brt_heldout as H
import turnier_generator as T

optimize = pytest.importorskip("scipy.optimize")


def random_plan(n, rnd, p=0.7, k_max=4):
    """Zufälliger Spielplan mit Ausgängen als Siegematrix und als Partienliste (Sieger, Verlierer)."""
    W = np.zeros((n, n), dtype=int)
    games = []
    for i, j in itertools.combinations(range(n), 2):
        if rnd.random() < p:
            for _ in range(rnd.randint(1, k_max)):
                if rnd.random() < 0.5:
                    W[i, j] += 1
                    games.append((i, j))
                else:
                    W[j, i] += 1
                    games.append((j, i))
    return W, games


def neg_penalized_loglik(theta, games, lam):
    d = np.array([theta[w] - theta[l] for w, l in games])
    return float(np.logaddexp(0.0, -d).sum() + 0.5 * lam * float(theta @ theta))


# --- 1. Newton gegen scipy auf der Partienliste ---------------------------------------------------------------------------------------------

def test_newton_solution_matches_scipy_minimize_on_the_game_list():
    rnd = random.Random(2026)
    checked = 0
    for _ in range(40):
        n = rnd.randint(3, 6)
        W, games = random_plan(n, rnd)
        if not games:
            continue
        for lam in (0.1, 1.0):
            ours = F.fit_penalized(W, lam).theta
            res = optimize.minimize(neg_penalized_loglik, np.zeros(n), args=(games, lam), method="BFGS", options={"gtol": 1e-10})
            ref = res.x - res.x.mean()
            assert ours == pytest.approx(ref, abs=1e-5)
        checked += 1
    assert checked > 25


# --- 2. Newton gegen Zermelo/MM ------------------------------------------------------------------------------------------------------------

def test_newton_matches_the_mm_iteration_where_the_estimator_exists():
    rnd = random.Random(7)
    compared = 0
    for _ in range(80):
        n = rnd.randint(3, 6)
        W, _ = random_plan(n, rnd, p=0.9, k_max=6)
        if not E.ford_condition(W):
            continue
        mm, _ = F.fit_mm(W)
        assert mm is not None
        assert F.fit_penalized(W, 1e-9, max_iter=500).theta == pytest.approx(mm, abs=1e-4)
        compared += 1
    assert compared > 20


# --- 3. Existenz -----------------------------------------------------------------------------------------------------------------------------

def test_existence_matches_networkx_and_decides_between_divergence_and_convergence():
    rnd = random.Random(3)
    seen = {True: 0, False: 0}
    for _ in range(80):
        n = rnd.randint(3, 5)
        W, _ = random_plan(n, rnd, p=0.8, k_max=2)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, j) for i in range(n) for j in range(n) if W[i, j] > 0)
        exists = E.ford_condition(W)
        assert exists == nx.is_strongly_connected(g)
        theta = {lam: F.fit_penalized(W, lam, max_iter=2000).theta for lam in (1e-3, 1e-5, 1e-7)}
        span = {lam: float(th.max() - th.min()) for lam, th in theta.items()}
        if exists:
            assert span[1e-7] - span[1e-5] < 1e-3                      # konvergiert: die Vorannahme verschwindet
        elif span[1e-7] > 0:
            assert span[1e-3] < span[1e-5] < span[1e-7]              # läuft weg, je kleiner λ
            assert span[1e-7] > 2 * span[1e-3] or span[1e-7] > 8
        seen[exists] += 1
    assert seen[True] > 5 and seen[False] > 5


def test_undefeated_and_winless_match_the_definition_on_the_game_list():
    rnd = random.Random(21)
    for _ in range(120):
        n = rnd.randint(2, 7)
        W, games = random_plan(n, rnd, p=0.5, k_max=2)         # viele Spieler mit 0 oder genau 1 Partie
        played, lost, won = [0] * n, [0] * n, [0] * n
        for w, l in games:
            played[w] += 1
            played[l] += 1
            won[w] += 1
            lost[l] += 1
        assert E.undefeated(W) == [i for i in range(n) if played[i] > 0 and lost[i] == 0]
        assert E.winless(W) == [i for i in range(n) if played[i] > 0 and won[i] == 0]


def test_a_non_positive_prior_is_rejected_at_the_boundary():
    W = np.array([[0, 2], [1, 0]])
    with pytest.raises(ValueError, match="lam muss positiv sein"):
        F.fit_penalized(W, 0.0)
    assert F.fit_penalized(W, 1e-12, max_iter=50).theta.shape == (2,)       # knapp über null wird gerechnet, nicht abgelehnt


# --- 4. Vollständig und ausgeglichen ----------------------------------------------------------------------------------------------------

def test_complete_balanced_plan_orders_like_the_win_table_for_every_prior():
    seen = 0
    for seed in range(100000, 100120):
        t = T.build_turnier(8, 0.3, 4, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", seed)
        wins = t.wins.sum(axis=1)
        if len(set(wins)) != 8:
            continue
        for lam in (0.05, 1.0):
            theta = F.fit_penalized(t.wins, lam).theta
            assert list(np.argsort(-theta)) == list(np.argsort(-wins))
        seen += 1
    assert seen > 30


# --- 5. Standardfehler ----------------------------------------------------------------------------------------------------------------------

def test_standard_errors_match_the_pseudoinverse_of_the_information_when_the_prior_vanishes():
    rnd = random.Random(12)
    checked = 0
    for _ in range(30):
        n = rnd.randint(3, 6)
        W, _ = random_plan(n, rnd, p=1.0, k_max=5)
        if not E.ford_condition(W):
            continue
        fit = F.fit_penalized(W, 1e-8, max_iter=500)
        info_free = F.fisher_information(fit.theta, W)
        ref = np.sqrt(np.clip(np.diag(np.linalg.pinv(info_free)), 0, None))
        assert F.standard_errors(fit.information) == pytest.approx(ref, rel=2e-3)
        checked += 1
    assert checked > 10


def test_standard_error_of_the_pair_is_the_closed_form():
    W = np.array([[0, 30], [10, 0]])            # 30 von 40: p = 0,75, g = 40
    fit = F.fit_penalized(W, 1e-9, max_iter=500)
    expected = 0.5 / np.sqrt(40 * 0.75 * 0.25)
    assert F.standard_errors(fit.information) == pytest.approx([expected, expected], abs=1e-6)


# --- 6. Frische Partien ----------------------------------------------------------------------------------------------------------------

def test_the_truth_has_the_lowest_expected_log_loss_among_probability_matrices():
    t = T.build_turnier(6, 0.4, 1, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 5)
    W_test = H.held_out_games(t.P, 4000, T.SplitMix64(9))
    truth = H.log_loss(t.P, W_test)
    # untere Schranke: mittlere Entropie der Paare (Erwartungswert des Log-Verlusts der Wahrheit)
    iu = np.triu_indices(6, 1)
    p = t.P[iu]
    entropy = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())
    assert truth == pytest.approx(entropy, abs=0.01)
    # echte Gegenwahrscheinlichkeiten (P_ij + P_ji = 1): zur Münze hin gedämpft und überschärft; die Wahrheit ist der Minimierer des erwarteten Verlusts
    for wrong in (H.coin_probabilities(6), 0.5 + 0.5 * (t.P - 0.5), np.clip(0.5 + 1.5 * (t.P - 0.5), 0.02, 0.98)):
        assert H.log_loss(wrong, W_test) > truth
