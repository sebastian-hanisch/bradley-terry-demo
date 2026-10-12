"""Frische Partien, Log-Verlust und die Gegenmodelle gegen von Hand gerechnete Werte."""

import math

import numpy as np
import pytest

import brt_heldout as H
import turnier_generator as T


def test_held_out_games_are_complete_antisymmetric_and_seed_stable():
    P = np.array([[0.5, 0.7, 0.2], [0.3, 0.5, 0.9], [0.8, 0.1, 0.5]])
    W = H.held_out_games(P, 25, T.SplitMix64(7))
    assert (W + W.T == 25 * (1 - np.eye(3, dtype=int))).all()
    assert np.array_equal(W, H.held_out_games(P, 25, T.SplitMix64(7)))
    assert not np.array_equal(W, H.held_out_games(P, 25, T.SplitMix64(8)))


def test_held_out_games_follow_the_probability_matrix():
    P = np.array([[0.5, 0.2, 0.9], [0.8, 0.5, 0.4], [0.1, 0.6, 0.5]])
    W = H.held_out_games(P, 6000, T.SplitMix64(3))
    for i in range(3):
        for j in range(3):
            if i < j:
                assert W[i, j] / 6000 == pytest.approx(P[i, j], abs=0.03)


def test_certain_outcomes():
    P = np.array([[0.5, 1.0], [0.0, 0.5]])
    W = H.held_out_games(P, 10, T.SplitMix64(1))
    assert W[0, 1] == 10 and W[1, 0] == 0


def test_log_loss_hand_value():
    P_hat = np.array([[0.5, 0.75], [0.25, 0.5]])
    W_test = np.array([[0, 3], [1, 0]])
    assert H.log_loss(P_hat, W_test) == pytest.approx(-(3 * math.log(0.75) + 1 * math.log(0.25)) / 4)
    assert H.log_loss(P_hat, W_test) == pytest.approx(0.562335, abs=1e-6)
    assert H.log_loss(H.coin_probabilities(2), W_test) == pytest.approx(math.log(2))


def test_log_loss_without_games_is_nan_without_a_numpy_warning_and_never_infinite():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # 0/0 ergäbe ebenfalls nan, aber mit RuntimeWarning
        assert math.isnan(H.log_loss(np.full((2, 2), 0.5), np.zeros((2, 2))))
    certain_wrong = np.array([[0.5, 0.0], [1.0, 0.5]])
    loss = H.log_loss(certain_wrong, np.array([[0, 1], [0, 0]]))
    assert math.isfinite(loss) and loss > 10


def test_log5_and_saturated_hand_values():
    W = np.array([[0, 3], [1, 0]])
    G = W + W.T
    # q_A = (3 + 0,5)/(4 + 1) = 0,7; q_B = 0,3; P_AB = 0,7 · 0,7 / (0,7 · 0,7 + 0,3 · 0,3) = 0,49/0,58
    P5 = H.log5_probabilities(W, G)
    assert P5[0, 1] == pytest.approx(0.49 / 0.58)
    assert P5[0, 1] + P5[1, 0] == pytest.approx(1.0)
    S = H.saturated_probabilities(W, G)
    assert S[0, 1] == pytest.approx(3.5 / 5) and S[1, 0] == pytest.approx(1.5 / 5)
    assert H.saturated_probabilities(np.zeros((2, 2)), np.zeros((2, 2)))[0, 1] == 0.5      # ohne Partien: ½
