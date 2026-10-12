"""Analyse, Szenario und Verdikte; Randfälle der Regler."""

import itertools
import math

import numpy as np
import pytest

import brt_constants as C
import brt_evaluation as ev
import turnier_generator as T
from brt_scenario import Settings, make_turnier

CASES = [Settings(n=n, k=k, plan=plan, x=x, p=0.5, c=c, stil="konter", lam=lam, seed=seed)
         for n, k, plan, x, c, lam, seed in itertools.product((4, 8, 12), (1, 3), ("vollstaendig", "staffeln", "luecken"), (0, 1), (0.0, 0.3), (0.01, 1.0), (3, 4))]


def test_analysis_invariants_over_many_settings():
    for s in CASES:
        a = ev.analyse(s)
        n = s.n
        for m in C.METHODS:
            assert sorted(a.rank[m]) == list(range(n)), (s, m)
            assert 1 <= a.best_place[m] <= n
        assert a.fit.theta.sum() == pytest.approx(0.0, abs=1e-9)
        assert a.scores["Colley"].mean() == pytest.approx(0.5, abs=1e-9)
        assert (a.se >= 0).all() and (a.lo <= a.fit.theta).all() and (a.fit.theta <= a.hi).all()
        assert a.ford == (len(a.components) == 1)
        assert sorted(v for comp in a.components for v in comp) == list(range(n))
        assert (s.plan == "staffeln") == (not math.isnan(a.crossing["Siege"]))
        assert a.heldout.shape == (n, n) and (a.heldout + a.heldout.T == C.HELDOUT_PER_PAIR * (1 - np.eye(n, dtype=int))).all()
        assert a.log_loss["Wahrheit"] <= a.log_loss["Münze"] + 1e-9 or s.c > 0     # nur Rauschen kann die Wahrheit über die Münze heben
        assert np.isnan(a.p_next).sum() == 1 and a.oracle_lam > 0


def test_complete_plan_gives_the_win_table_order_without_ties_for_any_prior():
    seen = 0
    for s in range(100000, 100150):
        a = ev.analyse(Settings(n=8, k=4, plan="vollstaendig", lam=0.5, seed=s))
        if len(set(a.scores["Siege"])) == 8:
            assert a.rank["Bradley-Terry"] == a.rank["Siege"]
            seen += 1
    assert seen > 30


def test_without_any_crossing_game_no_method_beats_a_coin_flip_across_the_leagues():
    wrongs = [ev.analyse(Settings(n=12, plan="staffeln", x=0, lam=0.1, seed=s)).crossing for s in range(100000, 100120)]
    for m in C.METHODS:
        assert abs(np.mean([w[m] for w in wrongs]) - 0.5) < 0.05, m


def test_analysis_is_cached_and_deterministic():
    s = Settings(n=8, seed=11)
    assert ev.analyse(s) is ev.analyse(s)
    assert np.array_equal(make_turnier(s).wins, make_turnier(Settings(n=8, seed=11)).wins)
    assert np.array_equal(ev.analyse(s).heldout, ev.analyse(Settings(n=8, seed=11)).heldout)


def test_a_tournament_with_an_undefeated_player_is_flagged_and_named():
    s = Settings(n=8, k=1, plan="vollstaendig", seed=C.DEFAULT_SEED)
    a = ev.analyse(s)
    assert not a.ford and a.undefeated == (a.turnier.names.index("B"),) and a.winless == ()
    level, text = ev.verdict_analysis(a)
    assert level == "warning" and "existiert hier nicht" in text and "ungeschlagen: B" in text


def test_verdict_cases():
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=12, plan="staffeln", x=0, seed=3)))
    assert level == "warning" and "zerfällt" in text and "existiert hier nicht" in text
    a = ev.analyse(Settings(n=8, k=10, plan="vollstaendig", lam=1.0, seed=3))
    level, text = ev.verdict_analysis(a)
    assert level == "info" and "ordnet Bradley-Terry wie die Siegtabelle" in text
    candidates = (ev.analyse(Settings(n=8, k=1, lam=0.01, seed=s)) for s in range(100000, 100100))
    bad = next(c for c in candidates if c.ford and c.log_loss["Bradley-Terry"] > c.log_loss["Münze"])
    level, text = ev.verdict_analysis(bad)
    assert level == "warning" and "schlechter als ein Münzwurf" in text


def test_sweeps_have_consistent_shapes():
    seeds = C.SWEEP_SEEDS[:20]
    grid = ev.sweep_existence(deltas=(0.3,), ks=(1, 4), seeds=seeds)
    assert [(r["delta"], r["k"]) for r in grid] == [(0.3, 1), (0.3, 4)] and grid[1]["exists"] == 1.0
    assert [r["label"] for r in ev.sweep_existence_plans(seeds=seeds)][0] == "Vollständig"
    rows = ev.sweep_staffeln(xs=(0, 1), lams=(0.1,), seeds=seeds)
    assert [r["x"] for r in rows] == [0, 1] and set(rows[0]) == {"x", "Siege", "Massey", "Colley", "BT 0.1"}
    rows = ev.sweep_predict(deltas=(0.3,), ks=(2,), lams=(0.5, 2.0), seeds=seeds)
    assert rows[0]["best_lam"] in (0.5, 2.0) and rows[0]["Wahrheit"] < rows[0]["Münze"]
    rows = ev.sweep_se(ks=(4,), lams=(1.0,), seeds=seeds)
    assert 0.5 < rows[0]["coverage"] <= 1.0
    rows = ev.sweep_cycles(cs=(0.3,), ks=(4, 20), seeds=seeds)
    assert rows[0]["bt_excess"] > 0 and rows[0]["sat_excess"] > 0
    rows = ev.sweep_iterations(deltas=(0.3,), ks=(4,), seeds=seeds)
    assert rows[0]["newton"] < rows[0]["mm"]


def test_decimal_comma_helpers():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 3) == "3,142" and ev.pct1(0.0526) == "5,3" and ev.pct(0.405) in ("40", "41")
    assert ev.bt_key(1.0) == "BT 1" and ev.bt_key(0.01) == "BT 0.01"
