"""Gewinngraph, Ford-Bedingung, Ungeschlagene und Sieglose gegen die Mini-Instanzen und gegen networkx."""

import random

import networkx as nx
import numpy as np

import brt_existence as E


def test_transitive_chain_has_one_component_per_player_in_dominance_order(transitive3):
    assert E.win_graph_components(transitive3) == [(0,), (1,), (2,)]
    assert not E.ford_condition(transitive3)
    assert E.undefeated(transitive3) == [0] and E.winless(transitive3) == [2]


def test_cycle_is_one_component(cycle3):
    assert E.win_graph_components(cycle3) == [(0, 1, 2)]
    assert E.ford_condition(cycle3)
    assert E.undefeated(cycle3) == [] and E.winless(cycle3) == []


def test_islands(islands):
    comps = E.win_graph_components(islands)
    assert sorted(comps) == [(0,), (1,), (2,), (3,)]
    assert not E.ford_condition(islands)
    assert E.undefeated(islands) == [0, 2] and E.winless(islands) == [1, 3]


def test_a_player_without_games_is_neither_undefeated_nor_winless():
    W = np.zeros((3, 3), dtype=int)
    W[0, 1] = 1
    assert E.undefeated(W) == [0] and E.winless(W) == [1]
    assert not E.ford_condition(W)                     # Spieler 2 hat nie gespielt: Plan nicht stark zusammenhängend


def test_components_match_networkx_and_come_in_topological_order():
    rnd = random.Random(2026)
    for _ in range(120):
        n = rnd.randint(2, 9)
        W = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and rnd.random() < 0.3:
                    W[i, j] = rnd.randint(1, 3)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, j) for i in range(n) for j in range(n) if W[i, j] > 0)
        ours = E.win_graph_components(W)
        assert sorted(ours) == sorted(tuple(sorted(c)) for c in nx.strongly_connected_components(g))
        assert E.ford_condition(W) == nx.is_strongly_connected(g)
        position = {v: idx for idx, comp in enumerate(ours) for v in comp}
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    assert position[i] <= position[j]             # nie ein Pfeil zurück in eine frühere Komponente
