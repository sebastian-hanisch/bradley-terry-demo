"""Einheitentests der Vergleichsverfahren (Massey, Colley, Rangordnung mit Toleranz, Kendall-τ, Staffel-Paare) gegen Handwerte."""

import math

import numpy as np
import pytest

import brt_baselines as B
import turnier_generator as T


def test_laplacian_of_the_transitive_instance(transitive3):
    G = transitive3 + transitive3.T
    assert B.laplacian(G).tolist() == [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]


def test_massey_and_colley_hand_values(transitive3):
    G = transitive3 + transitive3.T
    r = B.massey_rating(transitive3, G)
    assert r == pytest.approx([2 / 3, 0.0, -2 / 3])                 # M = 3I − J, p = (2, 0, −2)
    c = B.colley_rating(transitive3, G)
    assert c == pytest.approx([0.7, 0.5, 0.3])                      # C = 5I − J, b = (2, 1, 0), Summe n/2
    assert c.sum() == pytest.approx(1.5)


def test_massey_handles_a_split_plan_with_the_minimum_norm_solution(islands):
    G = islands + islands.T
    r = B.massey_rating(islands, G)
    assert np.isfinite(r).all()
    assert r[0] > r[1] and r[2] > r[3]
    assert r[0] + r[1] == pytest.approx(0.0) and r[2] + r[3] == pytest.approx(0.0)


def test_win_share_without_games_is_zero_not_nan():
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 1
    q = B.win_share(W, W + W.T)
    assert q.tolist() == [1.0, 0.0, 0.0]


def test_rank_with_tolerance_ties_noise_but_not_real_differences():
    for s in range(30):
        assert B.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))[2] == 2
    assert {B.rank_with_tolerance([1.0, 1.0 + 1e-13, 0.5], T.SplitMix64(s))[0] for s in range(40)} == {0, 1}
    assert {B.rank_with_tolerance([1.0, 1.0 + 1e-6, 0.5], T.SplitMix64(s))[1] for s in range(40)} == {0}


def test_kendall_tau_and_crossing_wrong_hand_instances():
    assert B.kendall_tau((0, 2, 1), (3, 1, 2)) == 1.0
    assert B.kendall_tau((0, 1, 2), (1, 2, 3)) == -1.0
    assert math.isnan(B.kendall_tau((0, 1), (5.0, 5.0)))
    assert B.crossing_wrong((0, 3, 1, 2), 2) == pytest.approx(0.5)
    assert B.crossing_wrong((3, 2, 1, 0), 2) == 0.0
    assert math.isnan(B.crossing_wrong((0, 1, 2), 0)) and math.isnan(B.crossing_wrong((0, 1, 2), 3))
