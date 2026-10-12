"""Kennzahlen, Verdikt-Logik und Experimente (über die festen Instanzen SWEEP_SEEDS)."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import brt_baselines as B
import brt_constants as C
import brt_existence as E
import brt_fit as F
import brt_heldout as H
import turnier_generator as T
from brt_scenario import Settings, make_turnier

_LOT_SALT = 0xC0FFEE
_HELDOUT_SALT = 0xBEEF
LL_LABELS = ("Münze", "log5", "Bradley-Terry", "gesättigt", "Wahrheit")


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    turnier: T.Turnier
    fit: F.Fit
    se: np.ndarray             # Standardfehler von θ_i − Mittel
    lo: np.ndarray             # untere Grenze des 90-%-Intervalls
    hi: np.ndarray             # obere Grenze
    scores: dict               # Verfahren -> Wertung je Teilnehmer
    rank: dict                 # Verfahren -> Platz je Teilnehmer (0 = bester)
    tau_s: dict                # Verfahren -> τ gegen die Generator-Stärke
    tau_e: dict                # Verfahren -> τ gegen die erwartete Siegquote
    crossing: dict             # Verfahren -> falsch geordnete Staffel-Paare (nan außer Plan "zwei Staffeln")
    components: tuple          # starke Zusammenhangskomponenten des Gewinngraphen in Dominanzordnung
    plan_parts: int            # Zusammenhangskomponenten des Spielplans (wer gegen wen gespielt hat)
    ford: bool                 # Schätzer ohne Penalty existiert
    undefeated: tuple
    winless: tuple
    heldout: np.ndarray        # frische Partien aus P
    log_loss: dict             # Modell -> Log-Verlust je frischer Partie
    p_next: np.ndarray         # P(i schlägt den im Bradley-Terry-Rang Nächsten); nan für den Letzten
    oracle_lam: float
    best_place: dict           # Verfahren -> Platz des Stärksten (1 = Spitze)


def _scores(t: T.Turnier, theta: np.ndarray) -> dict:
    return {
        "Siege": t.wins.sum(axis=1).astype(float),
        "Siegquote": B.win_share(t.wins, t.games),
        "Massey": B.massey_rating(t.wins, t.games),
        "Colley": B.colley_rating(t.wins, t.games),
        "Bradley-Terry": theta,
    }


@functools.lru_cache(maxsize=16384)
def analyse(settings: Settings) -> Analysis:
    """Gleiche Reglerwerte, gleiche Analyse; der Zwischenspeicher lässt die Experimente Turniere teilen (Settings ist unveränderlich)."""
    t = make_turnier(settings)
    n = settings.n
    fit = F.fit_penalized(t.wins, settings.lam)
    se = F.standard_errors(fit.information)
    scores = _scores(t, fit.theta)
    rank = {m: B.rank_with_tolerance(scores[m], T.SplitMix64(settings.seed ^ _LOT_SALT), C.TOL) for m in C.METHODS}
    comps = E.win_graph_components(t.wins)
    heldout = H.held_out_games(t.P, C.HELDOUT_PER_PAIR, T.SplitMix64(settings.seed ^ _HELDOUT_SALT))
    bt_p = F.win_probabilities(fit.theta)
    ll = {
        "Münze": H.log_loss(H.coin_probabilities(n), heldout),
        "log5": H.log_loss(H.log5_probabilities(t.wins, t.games), heldout),
        "Bradley-Terry": H.log_loss(bt_p, heldout),
        "gesättigt": H.log_loss(H.saturated_probabilities(t.wins, t.games), heldout),
        "Wahrheit": H.log_loss(t.P, heldout),
    }
    order = sorted(range(n), key=lambda i: rank["Bradley-Terry"][i])        # Platz 1 zuerst
    p_next = np.full(n, float("nan"))
    for a, b in zip(order[:-1], order[1:]):
        p_next[a] = bt_p[a, b]
    n_weak = n // 2
    return Analysis(
        settings=settings, turnier=t, fit=fit, se=se, lo=fit.theta - C.Z90 * se, hi=fit.theta + C.Z90 * se,
        scores=scores, rank=rank,
        tau_s={m: B.kendall_tau(rank[m], t.strengths) for m in C.METHODS},
        tau_e={m: B.kendall_tau(rank[m], t.expected) for m in C.METHODS},
        crossing={m: (B.crossing_wrong(rank[m], n_weak) if settings.plan == "staffeln" else float("nan")) for m in C.METHODS},
        components=tuple(comps), plan_parts=T.components(t.games), ford=len(comps) == 1,
        undefeated=tuple(E.undefeated(t.wins)), winless=tuple(E.winless(t.wins)),
        heldout=heldout, log_loss=ll, p_next=p_next, oracle_lam=F.oracle_lambda(n, settings.delta),
        best_place={m: rank[m][n - 1] + 1 for m in C.METHODS},
    )


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


def bt_key(lam: float) -> str:
    return f"BT {lam:g}"


# --- Experiment 1: wann existiert der Schätzer ---------------------------------------------------------------------------------------------

def _existence(n, delta, k, plan, x, p, seeds) -> dict:
    ok, extreme = [], []
    for s in seeds:
        W = T.build_turnier(n, delta, k, plan, x, p, 0.0, "ausgewogen", s).wins
        ok.append(float(E.ford_condition(W)))
        extreme.append(float(bool(E.undefeated(W)) or bool(E.winless(W))))
    return {"exists": _mean(ok), "extreme": _mean(extreme)}


def sweep_existence(deltas=C.EXISTENCE_DELTAS, ks=C.EXISTENCE_KS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Vollständiger Plan, ausgeglichene Stile: Anteil der Turniere, in denen der Schätzer existiert, nach Stärkeabstand und Partien je Paar."""
    return [{"delta": d, "k": k, **_existence(n, d, k, "vollstaendig", 1, 0.5, seeds)} for d in deltas for k in ks]


def sweep_existence_plans(n=12, delta=C.EXP_DELTA, k=C.EXISTENCE_PLAN_K, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Derselbe Anteil nach Spielplan: vollständig, zwei Staffeln mit einer Kreuzpartie, zufällige Lücken mit p = 0,5."""
    return [{"label": label, **_existence(n, delta, k, plan, x, p, seeds)}
            for label, plan, x, p in (("Vollständig", "vollstaendig", 1, 0.5), ("Zwei Staffeln, eine Kreuzpartie", "staffeln", 1, 0.5),
                                      ("Zufällige Lücken, p = 0,5", "luecken", 1, 0.5))]


def existence_extreme(n=C.EXP_N_BIG, delta=1.0, k=1, seeds=C.SWEEP_SEEDS) -> dict:
    """Der Extremfall: 16 Teilnehmer, großer Stärkeabstand, eine Partie je Paar."""
    return {"n": n, "delta": delta, "k": k, **_existence(n, delta, k, "vollstaendig", 1, 0.5, seeds)}


# --- Experiment 2: die Vorannahme ist das Ranking -------------------------------------------------------------------------------------------

def sweep_staffeln(xs=C.X_SWEEP, lams=C.STAFFEL_LAMS, n=C.EXP_N_STAFFELN, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Zwei Staffeln, eine Partie je Paar, x Kreuzpartien je Starkem: Anteil falsch geordneter Paare über die Staffeln nach Verfahren und λ."""
    rows = []
    n_weak = n // 2
    for x in xs:
        acc = {m: [] for m in ("Siege", "Massey", "Colley", *(bt_key(l) for l in lams))}
        for s in seeds:
            t = T.build_turnier(n, delta, 1, "staffeln", x, 0.5, 0.0, "ausgewogen", s)
            scores = {"Siege": t.wins.sum(axis=1).astype(float), "Massey": B.massey_rating(t.wins, t.games), "Colley": B.colley_rating(t.wins, t.games)}
            for l in lams:
                scores[bt_key(l)] = F.fit_penalized(t.wins, l).theta
            for m, v in scores.items():
                acc[m].append(B.crossing_wrong(B.rank_with_tolerance(v, T.SplitMix64(s ^ _LOT_SALT), C.TOL), n_weak))
        rows.append({"x": x, **{m: _mean(v) for m, v in acc.items()}})
    return rows


# --- Experiment 3: Vorhersage statt Rang, und das richtige λ -------------------------------------------------------------------------------

def sweep_predict(deltas=C.PREDICT_DELTAS, ks=C.PREDICT_KS, lams=C.PREDICT_LAMS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Log-Verlust auf 100 frischen Partien je Paar: Münze, log5, gesättigt, Bradley-Terry je λ (und beim Orakel-λ), Wahrheit."""
    rows = []
    for d in deltas:
        oracle = F.oracle_lambda(n, d)
        for k in ks:
            acc = {"Münze": [], "log5": [], "gesättigt": [], "Wahrheit": [], "oracle": [], **{bt_key(l): [] for l in lams}}
            for s in seeds:
                t = T.build_turnier(n, d, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", s)
                W_test = H.held_out_games(t.P, C.HELDOUT_PER_PAIR, T.SplitMix64(s ^ _HELDOUT_SALT))
                acc["Münze"].append(H.log_loss(H.coin_probabilities(n), W_test))
                acc["log5"].append(H.log_loss(H.log5_probabilities(t.wins, t.games), W_test))
                acc["gesättigt"].append(H.log_loss(H.saturated_probabilities(t.wins, t.games), W_test))
                acc["Wahrheit"].append(H.log_loss(t.P, W_test))
                for l in lams:
                    acc[bt_key(l)].append(H.log_loss(F.win_probabilities(F.fit_penalized(t.wins, l).theta), W_test))
                acc["oracle"].append(H.log_loss(F.win_probabilities(F.fit_penalized(t.wins, oracle).theta), W_test))
            row = {"delta": d, "k": k, "oracle_lam": oracle, **{m: _mean(v) for m, v in acc.items()}}
            row["best_lam"] = min(lams, key=lambda l: row[bt_key(l)])
            rows.append(row)
    return rows


# --- Experiment 4: stimmen die Standardfehler -----------------------------------------------------------------------------------------------

def sweep_se(ks=C.SE_KS, lams=C.SE_LAMS, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Abdeckung des 90-%-Intervalls für θ_i − Mittel gegen die Generator-Stärke (vollständig, ausgeglichen)."""
    truth = T.make_strengths(n, delta)
    truth = truth - truth.mean()
    rows = []
    for k in ks:
        for lam in lams:
            cover, se_m, err_m = [], [], []
            for s in seeds:
                t = T.build_turnier(n, delta, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", s)
                fit = F.fit_penalized(t.wins, lam)
                se = F.standard_errors(fit.information)
                err = fit.theta - truth
                cover.append(float(np.mean(np.abs(err) <= C.Z90 * se)))
                se_m.append(float(se.mean()))
                err_m.append(float(np.abs(err).mean()))
            rows.append({"k": k, "lam": lam, "coverage": _mean(cover), "se": _mean(se_m), "err": _mean(err_m)})
    return rows


# --- Experiment 5: Zyklen, wann lohnt das Modell ohne Annahme --------------------------------------------------------------------------------

def sweep_cycles(cs=C.CYCLE_CS, ks=C.CYCLE_KS, n=C.EXP_N, delta=C.EXP_DELTA, seeds=C.SWEEP_SEEDS_CYCLES) -> list[dict]:
    """Konter-Stile: Mehrverlust gegenüber der Wahrheit für Bradley-Terry (beim Orakel-λ) und das gesättigte Modell, nach Partien je Paar."""
    lam = F.oracle_lambda(n, delta)
    rows = []
    for c in cs:
        for k in ks:
            bt, sat, tru = [], [], []
            for s in seeds:
                t = T.build_turnier(n, delta, k, "vollstaendig", 1, 0.5, c, "konter", s)
                W_test = H.held_out_games(t.P, C.HELDOUT_PER_PAIR, T.SplitMix64(s ^ _HELDOUT_SALT))
                bt.append(H.log_loss(F.win_probabilities(F.fit_penalized(t.wins, lam).theta), W_test))
                sat.append(H.log_loss(H.saturated_probabilities(t.wins, t.games), W_test))
                tru.append(H.log_loss(t.P, W_test))
            rows.append({"c": c, "k": k, "bt": _mean(bt), "sat": _mean(sat), "truth": _mean(tru),
                         "bt_excess": _mean(bt) - _mean(tru), "sat_excess": _mean(sat) - _mean(tru)})
    return rows


# --- Rechenaufwand (README und Tests, nicht in der App) ----------------------------------------------------------------------------------------

def sweep_iterations(deltas=(0.1, 0.3, 0.6, 1.0), ks=(4, 10), n=12, seeds=C.SWEEP_SEEDS[:100]) -> list[dict]:
    """Iterationen bis zur Konvergenz (Toleranz 1e-10), nur Turniere, in denen der Schätzer existiert: Newton (kaum penalisiert) gegen Zermelo/MM."""
    rows = []
    for d in deltas:
        for k in ks:
            nt, mm, runs = [], [], 0
            for s in seeds:
                t = T.build_turnier(n, d, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", s)
                if not E.ford_condition(t.wins):
                    continue
                theta_mm, it_mm = F.fit_mm(t.wins)
                if theta_mm is None:
                    continue
                runs += 1
                nt.append(float(F.fit_penalized(t.wins, 1e-8).iterations))
                mm.append(float(it_mm))
            rows.append({"delta": d, "k": k, "runs": runs, "newton": _mean(nt), "mm": _mean(mm)})
    return rows


# --- Verdikte (live aus den Messwerten) --------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Was liefert Bradley-Terry hier, und existiert der Schätzer?"""
    s, t = a.settings, a.turnier
    names = t.names
    if not a.ford:
        who = []
        if a.plan_parts > 1:
            who.append(f"der Spielplan zerfällt in {a.plan_parts} Teile, die nie gegeneinander gespielt haben")
        if a.undefeated:
            who.append("ungeschlagen: " + ", ".join(names[i] for i in a.undefeated))
        if a.winless:
            who.append("ohne Sieg: " + ", ".join(names[i] for i in a.winless))
        reason = "; ".join(who) if who else f"der Gewinngraph hat {len(a.components)} Komponenten, ohne dass jemand ungeschlagen oder sieglos ist"
        return "warning", (f"Der Schätzer ohne Vorannahme **existiert hier nicht** ({reason}): Die Stärke liefe ins Unendliche. Die gezeigte Ordnung kommt von "
                           f"der Vorannahme λ = {de(s.lam)}; ein anderes λ gäbe eine andere Ordnung, solange die Daten nichts entscheiden.")
    tau = a.tau_s
    text = (f"τ gegen die Generator-Stärke: Siegtabelle {de(tau['Siege'])}, Colley {de(tau['Colley'])}, Bradley-Terry {de(tau['Bradley-Terry'])}. "
            f"Log-Verlust auf frischen Partien: Bradley-Terry {de(a.log_loss['Bradley-Terry'], 3)}, Münze {de(a.log_loss['Münze'], 3)}, "
            f"Wahrheit {de(a.log_loss['Wahrheit'], 3)}.")
    if s.plan == "vollstaendig":
        text += " Im vollständigen Plan ordnet Bradley-Terry wie die Siegtabelle; der Gewinn liegt in den Wahrscheinlichkeiten und Intervallen, nicht im Rang."
    if a.log_loss["Bradley-Terry"] > a.log_loss["Münze"]:
        return "warning", text + " **Die Vorhersage ist schlechter als ein Münzwurf**: Mit dieser Vorannahme sind die Stärken zu weit gespreizt."
    return "info", text


def verdict_existence(grid, plans, extreme) -> tuple[str, str]:
    g = {(r["delta"], r["k"]): r for r in grid}
    base = {k: g[(0.3, k)] for k in (1, 2, 4)}
    plan = {r["label"]: r for r in plans}
    return "warning", (f"Bei 8 Teilnehmern (Stärkeabstand 0,3) existiert der Schätzer bei einer Partie je Paar in **{pct(base[1]['exists'])} %** der Turniere, bei zwei in "
                       f"{pct(base[2]['exists'])} %, bei vier in {pct(base[4]['exists'])} %. Je größer der Abstand, desto öfter bleibt jemand ungeschlagen: bei Abstand "
                       f"1,0 sind es nur {pct(g[(1.0, 1)]['exists'])} %, bei 16 Teilnehmern {pct(extreme['exists'])} %. Nach Spielplan (12 Teilnehmer, 4 Partien): "
                       f"vollständig {pct(plan['Vollständig']['exists'])} %, zwei Staffeln {pct(plan['Zwei Staffeln, eine Kreuzpartie']['exists'])} %, "
                       f"Lücken {pct(plan['Zufällige Lücken, p = 0,5']['exists'])} %.")


def verdict_staffeln(rows, lams=C.STAFFEL_LAMS) -> tuple[str, str]:
    r = next(r for r in rows if r["x"] == 1)
    lo, mid, hi = (r[bt_key(l)] for l in lams)
    return "warning", (f"Bei einer Kreuzpartie je Starkem sind {pct(r['Siege'])} % der Paare über die Staffeln falsch geordnet (Siegtabelle), {pct(r['Massey'])} % "
                       f"(Massey), {pct(r['Colley'])} % (Colley). Bradley-Terry liegt dazwischen und **hängt an der Vorannahme**: {pct(lo)} % bei λ = {de(lams[0])}, "
                       f"{pct(mid)} % bei λ = {de(lams[1])}, {pct(hi)} % bei λ = {de(lams[2])}. Die Daten allein entscheiden hier nicht; λ ist das Ranking.")


def verdict_predict(rows) -> tuple[str, str]:
    r1 = next(r for r in rows if r["delta"] == 0.3 and r["k"] == 1)
    r6 = next(r for r in rows if r["delta"] == 0.6 and r["k"] == 1)
    return "warning", (f"Mit einer Partie je Paar und fast ohne Vorannahme (λ = 0,01) ist der Log-Verlust {de(r1[bt_key(0.01)], 3)}, **schlechter als der Münzwurf "
                       f"({de(r1['Münze'], 3)})**; mit λ = 2 sind es {de(r1[bt_key(2.0)], 3)} (Wahrheit {de(r1['Wahrheit'], 3)}). Das beste λ des Rasters ist "
                       f"{de(r1['best_lam'])} bei Abstand 0,3 (Orakel {de(r1['oracle_lam'])}) und {de(r6['best_lam'])} bei Abstand 0,6 (Orakel {de(r6['oracle_lam'])}): "
                       "die Vorannahme gehört zur Streuung der wahren Stärken. Das Orakel-λ gibt es nur in der Simulation.")


def verdict_se(rows) -> tuple[str, str]:
    lo = min(r["coverage"] for r in rows)
    hi = max(r["coverage"] for r in rows)
    k1 = next(r for r in rows if r["k"] == 1 and r["lam"] == 1.0)
    k40 = next(r for r in rows if r["k"] == 40 and r["lam"] == 1.0)
    return "success", (f"Das 90-%-Intervall deckt die wahre Stärke in {pct1(lo)} bis {pct1(hi)} % der Fälle ab, bei allen Partienzahlen und beiden Vorannahmen. "
                       f"Der Standardfehler schrumpft von {de(k1['se'])} (1 Partie je Paar) auf {de(k40['se'])} (40 Partien). Das gilt für vollständige, "
                       "ausgeglichene Pläne ohne Zyklen; für Staffeln und Lücken ist es hier nicht gemessen.")


def verdict_cycles(rows) -> tuple[str, str]:
    r = {(x["c"], x["k"]): x for x in rows}
    return "info", (f"Bei Konter-Stilen (c = 0,3) kostet das unmodellierte Zyklenmodell Bradley-Terry {de(r[(0.3, 20)]['bt_excess'], 3)} nat je Partie bei 20 Partien "
                    f"je Paar und {de(r[(0.3, 600)]['bt_excess'], 3)} bei 600; das gesättigte Modell kostet {de(r[(0.3, 20)]['sat_excess'], 3)} bzw. "
                    f"{de(r[(0.3, 600)]['sat_excess'], 3)}. Erst bei sehr vielen Partien je Paar zahlt sich das Modell ohne Annahme aus; bei wenigen ist die "
                    "Annahme billiger. Die Rangfrage ist damit nicht beantwortet: Wer Zyklen hat, hat keine eindeutige Ordnung.")
