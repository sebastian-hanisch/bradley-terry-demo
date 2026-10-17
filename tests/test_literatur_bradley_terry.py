"""Literatur-Referenzwerte bradley-terry-demo (ORACLE-PLAYBOOK §5.1): der ungeschützte Bradley-Terry-Schätzer (λ → 0) gegen Zahlen aus eingesehenen Quellen.
Alle Quellen am 2026-10-17 eingesehen (WebFetch). Die Werte stammen aus den Quellen, nicht aus dem Code der Demo. Beide Datensätze haben einen existierenden
Schätzer (Ford-Bedingung erfüllt); die Quellen setzen eine Mannschaft auf 0 (Bezugsgröße), die Demo zentriert auf Mittelwert 0: verglichen werden Differenzen.

1. Stigler-Zitationsdaten (Bradley-Terry-Beispiel mit vier Zeitschriften), BradleyTerry2-Vignette "Bradley-Terry Models in R" (Turner, Firth), Abschnitt 2.1,
   https://cran.r-project.org/web/packages/BradleyTerry2/vignettes/BradleyTerry.html (Kopfzeile: "BradleyTerry2 version 1.1.4, 2026-09-29").
   Gedruckte Daten (citations.sf, "win1 win2"): Biometrika/Comm Statist 730/33, Biometrika/JASA 498/320, Biometrika/JRSS-B 221/284, Comm Statist/JASA 68/813,
   Comm Statist/JRSS-B 17/276, JASA/JRSS-B 142/325. Gedruckter Fit citeModel (Bezug Biometrika = 0): Comm Statist -2.9491, JASA -0.4796, JRSS-B 0.2690;
   mit Bezug JASA: Biometrika 0.4796, Comm Statist -2.4695, JRSS-B 0.7485.
2. Baseball 1987, American League East, sieben Mannschaften (Agresti 2002, Categorical Data Analysis, S. 437-438; je Paar 13 Partien, kein Remis).
   a) Gedruckte Koeffizienten baseballModel1 (BTm, nur Mannschaftsfähigkeit, Bezug Baltimore = 0) in derselben Vignette, Abschnitt 3.3: Boston 1.1077,
      Cleveland 0.6839, Detroit 1.4364, Milwaukee 1.5814, New York 1.2476, Toronto 1.2945, mit Standardfehlern (Bezug Baltimore) 0.3339, 0.3319, 0.3396,
      0.3433, 0.3359, 0.3367. Die vollständigen Daten sind in der Vignette NICHT gedruckt (nur head); die Siegmatrix unten stammt aus
      https://homepage.divms.uiowa.edu/~luke/xls/glim/glim/node8.html (Bradley-Terry-Beispiel zu GLIM-Notizen), dort aus der Ergebnisliste rekonstruiert
      (die gedruckte Tabelle ist dort ein Bild). Prüfung der Matrix: jedes Paar summiert sich zu 13 Partien; die ungeschützten Schätzungen reproduzieren
      sowohl die BradleyTerry2-Koeffizienten (vier Stellen) als auch die dort gedruckten Schätzungen (Bezug Milwaukee: Detroit -0.144948, Toronto -0.286871,
      New York -0.333738, Boston -0.473658, Cleveland -0.897502, Baltimore -1.58134; die GLIM-Zahlen weichen um bis zu 2e-5 vom exakten Maximum ab, ihre
      Standardfehler um 1e-4, deshalb Toleranz 3e-5 und keine Standardfehler daraus). Ein Zahlendreher in der Matrix würde beides verfehlen.

Nicht eingesehen (kein Test, siehe Bericht "offen"): Bradley/Terry (1952), Zermelo (1929), Ford (1957), Hunter (2004) im Original; Agresti (2002) selbst
(nur über 2.). Die Ridge-Schätzung (λ > 0) und die Standardfehler der Demo (Projektion auf θ - Mittel) haben keinen Literaturwert; dazu gibt es nur das
Orakel gegen die Pseudoinverse.
"""

import numpy as np
import pytest

import brt_existence as E
import brt_fit as F

# --- Stigler/Citations: Biometrika=0, Comm Statist=1, JASA=2, JRSS-B=3 ---------------------------------------------------------------------

CITATIONS = [(0, 1, 730, 33), (0, 2, 498, 320), (0, 3, 221, 284), (1, 2, 68, 813), (1, 3, 17, 276), (2, 3, 142, 325)]


def citations_wins():
    W = np.zeros((4, 4))
    for a, b, wa, wb in CITATIONS:
        W[a, b], W[b, a] = wa, wb
    return W


def test_citations_example_matches_the_printed_abilities_for_both_reference_journals():
    W = citations_wins()
    assert E.ford_condition(W)                                   # der Schätzer existiert (sonst wäre der Vergleich sinnlos)
    for theta in (F.fit_mm(W)[0], F.fit_penalized(W, 1e-9, max_iter=500).theta):
        d = theta - theta[0]                                     # Bezug Biometrika
        assert d[1:] == pytest.approx([-2.9491, -0.4796, 0.2690], abs=6e-5)
        d = theta - theta[2]                                     # Bezug JASA
        assert d[[0, 1, 3]] == pytest.approx([0.4796, -2.4695, 0.7485], abs=6e-5)


# --- Baseball 1987, Agresti (2002) ----------------------------------------------------------------------------------------------------------

TEAMS = ["Milwaukee", "Detroit", "Toronto", "New York", "Boston", "Cleveland", "Baltimore"]
BASEBALL = np.array([                       # BASEBALL[i, j] = Siege von i gegen j
    [0, 7, 9, 7, 7, 9, 11],
    [6, 0, 7, 5, 11, 9, 9],
    [4, 6, 0, 7, 7, 8, 12],
    [6, 8, 6, 0, 6, 7, 10],
    [6, 2, 6, 7, 0, 7, 12],
    [4, 4, 5, 6, 6, 0, 6],
    [2, 4, 1, 3, 1, 7, 0],
], dtype=float)


def test_baseball_matrix_has_thirteen_games_per_pair():
    G = BASEBALL + BASEBALL.T
    off = ~np.eye(7, dtype=bool)
    assert (G[off] == 13).all()


def test_baseball_abilities_match_the_printed_coefficients():
    assert E.ford_condition(BASEBALL)
    theta, _ = F.fit_mm(BASEBALL)
    vs_baltimore = theta - theta[6]
    printed = {"Boston": 1.1077, "Cleveland": 0.6839, "Detroit": 1.4364, "Milwaukee": 1.5814, "New York": 1.2476, "Toronto": 1.2945}
    for team, value in printed.items():
        assert vs_baltimore[TEAMS.index(team)] == pytest.approx(value, abs=6e-5)
    vs_milwaukee = theta - theta[0]
    printed2 = {"Detroit": -0.144948, "Toronto": -0.286871, "New York": -0.333738, "Boston": -0.473658, "Cleveland": -0.897502, "Baltimore": -1.58134}
    for team, value in printed2.items():
        assert vs_milwaukee[TEAMS.index(team)] == pytest.approx(value, abs=3e-5)      # GLIM-Ausgabe läuft weniger genau (Baltimore weicht um 2e-5 ab)
    # das Newton-Verfahren der Demo (λ sehr klein) landet am selben Punkt
    newton = F.fit_penalized(BASEBALL, 1e-9, max_iter=500).theta
    assert newton - newton[6] == pytest.approx(vs_baltimore, abs=1e-5)


def test_baseball_standard_errors_of_the_contrasts_match_bradleyterry2():
    theta, _ = F.fit_mm(BASEBALL)
    cov = np.linalg.pinv(F.fisher_information(theta, BASEBALL))           # Kovarianz der identifizierten Richtungen (Information hat den Kern 1)
    se_vs_baltimore = [np.sqrt(cov[i, i] + cov[6, 6] - 2 * cov[i, 6]) for i in range(6)]
    printed = {"Boston": 0.3339, "Cleveland": 0.3319, "Detroit": 0.3396, "Milwaukee": 0.3433, "New York": 0.3359, "Toronto": 0.3367}
    for team, value in printed.items():
        assert se_vs_baltimore[TEAMS.index(team)] == pytest.approx(value, abs=6e-5)
