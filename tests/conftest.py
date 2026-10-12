"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def pair():
    """2 Teilnehmer A=0, B=1: A schlägt B in 3 von 4 Partien. Handrechnung (λ → 0): θ_A − θ_B = ln 3, zentriert θ_A = ½ ln 3 = 0,5493, P = 0,75;
    Var(θ_A − θ_B) = 1/(g p (1 − p)) = 1/(4 · 0,1875) = 1,3333, SE(θ_A − Mittel) = ½ √1,3333 = 0,5774."""
    return np.array([[0, 3], [1, 0]])


@pytest.fixture
def cycle3():
    """Zyklus A schlägt B, B schlägt C, C schlägt A (je eine Partie). Handrechnung: θ = 0 ist stationär (Siege 1 − Erwartung 2 · ½ = 0 in jeder Zeile),
    alle P = ½; der Schätzer existiert."""
    return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


@pytest.fixture
def transitive3():
    """A schlägt B, B schlägt C, A schlägt C (je eine Partie). A ist ungeschlagen, C sieglos: der Schätzer existiert nicht. Mit λ ist θ_B = 0 und
    θ_A = −θ_C = t mit 2 − σ(t) − σ(2t) = λ t; für λ = 1 ist t ≈ 0,591."""
    return np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])


@pytest.fixture
def islands():
    """4 Teilnehmer, zwei getrennte Paare: 0 schlägt 1 (zweimal), 2 schlägt 3 (einmal). Vier Komponenten des Gewinngraphen (kein Pfeil zurück),
    ungeschlagen 0 und 2, ohne Sieg 1 und 3."""
    W = np.zeros((4, 4), dtype=int)
    W[0, 1] = 2
    W[2, 3] = 1
    return W
