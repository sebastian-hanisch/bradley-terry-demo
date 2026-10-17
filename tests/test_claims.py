"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import brt_constants as C
import brt_evaluation as ev


@pytest.fixture(scope="module")
def existence():
    return {(r["delta"], r["k"]): r for r in ev.sweep_existence()}


@pytest.fixture(scope="module")
def plans():
    return {r["label"]: r for r in ev.sweep_existence_plans()}


@pytest.fixture(scope="module")
def extreme():
    return ev.existence_extreme()


@pytest.fixture(scope="module")
def staffeln():
    return {r["x"]: r for r in ev.sweep_staffeln()}


@pytest.fixture(scope="module")
def predict():
    return {(r["delta"], r["k"]): r for r in ev.sweep_predict()}


@pytest.fixture(scope="module")
def se():
    return {(r["k"], r["lam"]): r for r in ev.sweep_se()}


@pytest.fixture(scope="module")
def cycles():
    return {(r["c"], r["k"]): r for r in ev.sweep_cycles()}


@pytest.fixture(scope="module")
def iterations():
    return {(r["delta"], r["k"]): r for r in ev.sweep_iterations()}


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_with_one_game_per_pair_the_estimator_often_does_not_exist(existence):
    assert existence[(0.3, 1)]["exists"] == pytest.approx(0.66, abs=0.006)
    assert existence[(0.3, 1)]["extreme"] == pytest.approx(0.33, abs=0.006)
    assert existence[(0.3, 2)]["exists"] == pytest.approx(0.957, abs=0.006)
    assert existence[(0.3, 4)]["exists"] == 1.0


def test_a_larger_strength_gap_makes_it_worse(existence):
    assert existence[(0.1, 1)]["exists"] == pytest.approx(0.863, abs=0.006)
    assert existence[(0.6, 1)]["exists"] == pytest.approx(0.213, abs=0.006)
    assert existence[(1.0, 1)]["exists"] == pytest.approx(0.02, abs=0.006)
    assert existence[(1.0, 4)]["exists"] == pytest.approx(0.617, abs=0.006)
    for d in (0.1, 0.3, 0.6, 1.0):
        assert existence[(d, 1)]["exists"] <= existence[(d, 2)]["exists"] <= existence[(d, 4)]["exists"]


def test_by_plan_and_the_extreme_case(plans, extreme):
    assert plans["Vollständig"]["exists"] == 1.0
    assert plans["Zwei Staffeln, eine Kreuzpartie"]["exists"] == pytest.approx(0.983, abs=0.006)
    assert plans["Zufällige Lücken, p = 0,5"]["exists"] == pytest.approx(0.81, abs=0.006)
    assert extreme["exists"] == 0.0 and extreme["extreme"] == pytest.approx(0.937, abs=0.006)


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_the_prior_decides_the_order_across_the_leagues(staffeln):
    r = staffeln[1]
    assert r["Siege"] == pytest.approx(0.368, abs=0.006) and r["Massey"] == pytest.approx(0.155, abs=0.006) and r["Colley"] == pytest.approx(0.244, abs=0.006)
    assert r[ev.bt_key(0.01)] == pytest.approx(0.150, abs=0.006)
    assert r[ev.bt_key(0.1)] == pytest.approx(0.184, abs=0.006)
    assert r[ev.bt_key(1.0)] == pytest.approx(0.287, abs=0.006)
    assert r[ev.bt_key(0.01)] < r[ev.bt_key(0.1)] < r["Colley"] < r[ev.bt_key(1.0)] < r["Siege"]
    r2 = staffeln[2]
    assert r2["Siege"] == pytest.approx(0.266, abs=0.006) and r2["Massey"] == pytest.approx(0.119, abs=0.006)
    assert r2["Colley"] == pytest.approx(0.158, abs=0.006)
    assert r2[ev.bt_key(0.01)] == pytest.approx(0.11, abs=0.006) and r2[ev.bt_key(0.1)] == pytest.approx(0.125, abs=0.006)
    assert r2[ev.bt_key(1.0)] == pytest.approx(0.183, abs=0.006)


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_with_one_game_and_almost_no_prior_the_prediction_is_worse_than_a_coin(predict):
    r = predict[(0.3, 1)]
    assert r["Münze"] == pytest.approx(0.693, abs=0.002)
    assert r[ev.bt_key(0.01)] == pytest.approx(0.843, abs=0.006) and r[ev.bt_key(0.01)] > r["Münze"]
    assert r[ev.bt_key(2.0)] == pytest.approx(0.644, abs=0.006) and r[ev.bt_key(2.0)] < r["Münze"]
    assert r["Wahrheit"] == pytest.approx(0.587, abs=0.006)


def test_the_best_prior_is_the_inverse_variance_of_the_true_strengths(predict):
    assert predict[(0.3, 1)]["oracle_lam"] == pytest.approx(2.116, abs=0.002) and predict[(0.6, 1)]["oracle_lam"] == pytest.approx(0.529, abs=0.002)
    for k in (1, 2, 4, 10):
        assert predict[(0.3, k)]["best_lam"] == 2.0 and predict[(0.6, k)]["best_lam"] == 0.5
        for d in (0.3, 0.6):
            r = predict[(d, k)]
            best = min(r[ev.bt_key(l)] for l in C.PREDICT_LAMS)
            assert r["oracle"] <= best + 0.005


def test_bradley_terry_beats_the_saturated_model_for_few_games(predict):
    for d in (0.3, 0.6):
        for k in (1, 2, 4, 10):
            assert predict[(d, k)]["oracle"] < predict[(d, k)]["gesättigt"]
    assert predict[(0.3, 4)]["oracle"] < predict[(0.3, 4)]["log5"]


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_the_ninety_percent_interval_covers_about_ninety_percent(se):
    for r in se.values():
        assert 0.88 <= r["coverage"] <= 0.93, r
    assert se[(1, 1.0)]["se"] == pytest.approx(0.573, abs=0.006) and se[(40, 1.0)]["se"] == pytest.approx(0.117, abs=0.006)
    assert se[(1, 1.0)]["err"] > se[(4, 1.0)]["err"] > se[(40, 1.0)]["err"]


# --- Experiment 5 ------------------------------------------------------------------------------------------------------------------------

def test_the_unmodelled_cycles_cost_little_and_the_saturated_model_pays_off_late(cycles):
    r = cycles[(0.3, 20)]
    assert r["bt_excess"] == pytest.approx(0.010, abs=0.002) and r["sat_excess"] == pytest.approx(0.023, abs=0.002)
    assert cycles[(0.3, 600)]["bt_excess"] == pytest.approx(0.004, abs=0.002) and cycles[(0.3, 600)]["sat_excess"] == pytest.approx(0.001, abs=0.001)
    for c in (0.3, 0.6):
        assert cycles[(c, 60)]["bt_excess"] < cycles[(c, 60)]["sat_excess"]
        assert cycles[(c, 200)]["bt_excess"] > cycles[(c, 200)]["sat_excess"]
    assert cycles[(0.15, 200)]["bt_excess"] < cycles[(0.15, 200)]["sat_excess"]
    assert cycles[(0.15, 600)]["bt_excess"] > cycles[(0.15, 600)]["sat_excess"]
    for key in cycles:
        assert cycles[key]["bt_excess"] < 0.06


# --- Rechenaufwand (README) --------------------------------------------------------------------------------------------------------------

def test_newton_needs_far_fewer_iterations_than_the_mm_iteration(iterations):
    ratios = [r["mm"] / r["newton"] for r in iterations.values()]
    assert min(ratios) == pytest.approx(9.4, abs=0.8) and max(ratios) == pytest.approx(150, abs=12)     # README: "das 9- bis 150-Fache"; App: "Neun- bis Hundertfünfzigfache"
    assert iterations[(0.1, 4)]["newton"] == pytest.approx(5.1, abs=0.4) and iterations[(0.1, 4)]["mm"] == pytest.approx(53, abs=3)
    assert iterations[(1.0, 4)]["newton"] == pytest.approx(9.5, abs=0.5) and iterations[(1.0, 4)]["mm"] == pytest.approx(1419, abs=60)
    assert min(r["newton"] for r in iterations.values()) == pytest.approx(5.0, abs=0.4) and min(r["mm"] for r in iterations.values()) == pytest.approx(47, abs=3)
    assert max(r["newton"] for r in iterations.values()) < 10.5
    assert max(r["mm"] for r in iterations.values()) > 1000
    assert iterations[(1.0, 4)]["runs"] < 70                        # bei großem Abstand fehlt der Schätzer oft: nur 57 von 100


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(existence, plans, extreme, staffeln, predict, se, cycles):
    level, text = ev.verdict_existence(list(existence.values()), list(plans.values()), extreme)
    assert level == "warning" and "66 %" in text and "100 %" in text
    level, text = ev.verdict_staffeln(list(staffeln.values()))
    assert level == "warning" and ev.pct(staffeln[1]["Siege"]) in text and ev.pct(staffeln[1][ev.bt_key(1.0)]) in text
    level, text = ev.verdict_predict(list(predict.values()))
    assert ev.de(predict[(0.3, 1)][ev.bt_key(0.01)], 3) in text and "schlechter als der Münzwurf" in text
    level, text = ev.verdict_se(list(se.values()))
    assert level == "success" and ev.de(se[(1, 1.0)]["se"]) in text
    level, text = ev.verdict_cycles(list(cycles.values()))
    assert ev.de(cycles[(0.3, 20)]["bt_excess"], 3) in text and ev.de(cycles[(0.3, 600)]["sat_excess"], 3) in text
