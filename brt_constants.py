"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 16, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
K_MIN, K_MAX, DEFAULT_K = 1, 20, 1
X_MIN, X_MAX, DEFAULT_X = 0, 8, 1                 # Kreuzpartien je Spieler der starken Staffel (nur Plan "staffeln")
P_MIN, P_MAX, P_STEP, DEFAULT_P = 0.1, 1.0, 0.05, 0.5   # Spielwahrscheinlichkeit je Paar (nur Plan "luecken")
CYCLE_MIN, CYCLE_MAX, CYCLE_STEP, DEFAULT_CYCLE = 0.0, 0.5, 0.05, 0.0
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.01, 5.0, 0.01, 1.0     # Gewicht der Vorannahme (Ridge auf die Stärken)
SEED_MAX, DEFAULT_SEED = 999999, 40         # Seed 40 (n = 8, 1 Partie je Paar): B bleibt ungeschlagen, der Schätzer existiert nicht

PLAN_LABELS = {"vollstaendig": "Vollständig: jeder gegen jeden", "staffeln": "Zwei Staffeln", "luecken": "Zufällige Lücken"}
STIL_LABELS = {
    "ausgewogen": "Ausgewogen (reihum)",
    "konter": "Konter-Stil dominiert (der Stärkste wird gekontert)",
    "beute": "Beute-Stil dominiert (der Stärkste kontert)",
}
METHODS = ("Siege", "Siegquote", "Massey", "Colley", "Bradley-Terry")
TOL = 1e-9                                  # Wertungen, die sich um weniger unterscheiden, sind gleich (Gleitkomma-Rauschen)
Z90 = 1.6448536269514722                    # zweiseitiges 90-%-Intervall der Normalverteilung
HELDOUT_PER_PAIR = 100                      # frische Partien je Paar für den Log-Verlust (Anzeige und Experimente)

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100300))
SWEEP_SEEDS_CYCLES = SWEEP_SEEDS[:100]      # Experiment 5 (Hunderte Partien je Paar): 100 feste Turniere
EXP_N = 8                   # Experimente 1, 3, 4, 5: feste Instanz mit 8 Teilnehmern
EXP_N_STAFFELN = 12         # Experiment 2: zwei Staffeln mit je 6
EXP_N_BIG = 16
EXP_DELTA = 0.3
EXISTENCE_KS = (1, 2, 4)
EXISTENCE_DELTAS = (0.1, 0.3, 0.6, 1.0)
EXISTENCE_PLAN_K = 4        # Experiment 1, Zeilen nach Spielplan (n = 12)
X_SWEEP = (1, 2, 3)
STAFFEL_LAMS = (0.01, 0.1, 1.0)
PREDICT_DELTAS = (0.3, 0.6)
PREDICT_KS = (1, 2, 4, 10)
PREDICT_LAMS = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
SE_KS = (1, 2, 4, 10, 40)
SE_LAMS = (0.1, 1.0)
CYCLE_CS = (0.15, 0.3, 0.6)
CYCLE_KS = (4, 20, 60, 200, 600)

COLOR_SIEGE = "rgba(150,150,150,0.9)"
COLOR_QUOTE = "#8e6bb0"
COLOR_MASSEY = "#c0392b"
COLOR_COLLEY = "#d98c1f"
COLOR_BT = "#2F6B65"
COLOR_MAIN = "#2F6B65"
METHOD_COLORS = {"Siege": COLOR_SIEGE, "Siegquote": COLOR_QUOTE, "Massey": COLOR_MASSEY, "Colley": COLOR_COLLEY, "Bradley-Terry": COLOR_BT}

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "k": DEFAULT_K, "plan": "vollstaendig", "x": DEFAULT_X, "p": DEFAULT_P, "c": 0.0, "stil": "ausgewogen",
         "lam": DEFAULT_LAM, "seed": DEFAULT_SEED}
PRESETS = {
    "Eine Partie je Paar": dict(_BASE),
    "Zwei Staffeln, eine Kreuzpartie": {**_BASE, "n": 12, "plan": "staffeln", "lam": 0.1},
    "Falsche Vorannahme": {**_BASE, "lam": LAM_MIN},
    "Viele Partien": {**_BASE, "k": 20},
}
PRESET_HELP = {
    "Eine Partie je Paar": "8 Teilnehmer, jeder gegen jeden genau einmal: Der Schätzer existiert oft nicht, weil jemand ungeschlagen bleibt; die Vorannahme λ hält die Rechnung am Laufen.",
    "Zwei Staffeln, eine Kreuzpartie": "12 Teilnehmer in einer starken und einer schwachen Staffel, jeder Starke spielt nur einmal gegen einen Schwachen: Hier entscheidet λ, ob die Ordnung wie bei der Siegtabelle oder wie bei Massey ausfällt.",
    "Falsche Vorannahme": "Dieselbe Eine-Partie-Runde mit fast keiner Vorannahme (λ = 0,01): Die Stärken laufen auseinander, und die Vorhersage ist schlechter als ein Münzwurf.",
    "Viele Partien": "Jedes Paar spielt 20-mal: Die Ordnung ist die der Siegtabelle, die Intervalle sind eng, und die Vorannahme ist fast bedeutungslos.",
}
