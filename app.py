"""Bradley-Terry: Siegwahrscheinlichkeit aus der Stärke - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Ranking-Linie: Ein Modell, in dem jedem Teilnehmer eine Stärke gehört und jede Partie mit der Wahrscheinlichkeit σ(Stärkedifferenz) entschieden
wird. Fünf Experimente messen, wo es nützt (Wahrscheinlichkeiten, Intervalle) und wo nicht (Rang, fehlender Schätzer). Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import brt_constants as C
import brt_evaluation as ev
from brt_fit import oracle_lambda
from brt_pdf_export import build_pdf
from brt_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from brt_scenario import Settings
from brt_visualization import (
    build_cycles,
    build_existence,
    build_matrices,
    build_places,
    build_predict,
    build_se,
    build_staffeln,
    build_strengths,
)

st.set_page_config(page_title="Bradley-Terry – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _existence():
    return ev.sweep_existence(), ev.sweep_existence_plans(), ev.existence_extreme()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _staffeln():
    return ev.sweep_staffeln()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _predict():
    return ev.sweep_predict()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Turniere …")
def _se():
    return ev.sweep_se()


@st.cache_data(show_spinner="Experiment rechnet über 100 feste Turniere mit bis zu 600 Partien je Paar …")
def _cycles():
    return ev.sweep_cycles()


def verdict(level, text):
    SHOW[level](text)


st.title("📈 Bradley-Terry: Siegwahrscheinlichkeit aus der Stärke")
st.markdown(
    """
Die Siegtabelle zählt, Massey und Colley lösen ein lineares System. Diese Demo ist das **fünfte Stück der Ranking-Linie** und fragt anders: Was wäre, wenn
jeder Teilnehmer eine **Stärke** hätte und jede Partie mit der Wahrscheinlichkeit **σ(Stärke des einen minus Stärke des anderen)** entschieden würde? Das
**Bradley-Terry-Modell** schätzt diese Stärken aus den Ausgängen. Es ordnet **nicht besser** als die Verfahren davor; es liefert **Wahrscheinlichkeiten und
Intervalle**, und es hat einen Preis: **Der Schätzer existiert oft nicht**, und wo er fehlt, ist die Vorannahme das Ranking.

Weiter unten: fünf **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So schätzt man Stärken aus Paarvergleichen", expanded=True):
    st.markdown(
        """
1. **Das Modell:** P(i schlägt j) = σ(θ_i − θ_j) mit σ(x) = 1/(1 + e^(−x)). Nur Differenzen zählen; die Stärken werden auf Mittelwert 0 gesetzt.
   Ausgänge sind Sieg oder Niederlage, keine Punkte: Kantersiege können nichts verzerren, aber auch nichts nützen.
2. **Die Schätzung:** Gesucht ist das θ, das die beobachteten Ausgänge am wahrscheinlichsten macht (Maximum Likelihood). Die Funktion ist konkav; das
   **Newton-Verfahren** löst sie in wenigen Dutzend Schritten, die ältere Zermelo-Iteration braucht je nach Stärkeabstand das Drei- bis Dreißigfache.
3. **Der Haken:** Hat jemand nie verloren, wäre seine Stärke unendlich; die Likelihood hat dann kein Maximum. Genau dann existiert der Schätzer, wenn
   der Graph „i → j, wenn i mindestens einmal gegen j gewann“ **stark zusammenhängend** ist (Ford-Bedingung).
4. **Die Vorannahme λ:** Ein Zuschlag −(λ/2) · Σθ² hält die Stärken endlich: „die Stärken streuen um 0 mit Varianz 1/λ“. Wo die Daten entscheiden,
   ist λ fast ohne Wirkung; wo nicht, bestimmt λ die Ordnung.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Teilnehmer**")
    n = st.slider("Teilnehmer n", *bounds("n_slider"), key="n_slider", help="Wie viele Teilnehmer im Turnier antreten.")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Teilnehmer (logistische Skala). Je größer, desto öfter bleibt jemand ungeschlagen.")
    st.markdown("**Spielplan**")
    plan = st.selectbox("Spielplan", list(C.PLAN_LABELS), format_func=C.PLAN_LABELS.get, key="plan_select")
    k = st.slider("Partien je Paar", *bounds("k_slider"), key="k_slider", help="Wie oft ein Paar gegeneinander spielt, wenn es überhaupt spielt.")
    if plan == "staffeln":
        seed_widget("x_slider")
        x = st.slider("Kreuzpartien je Spieler der starken Staffel", *bounds("x_slider"), key="x_slider",
                      help="Gegen wie viele zufällige Spieler der schwachen Staffel jeder Starke zusätzlich antritt. 0 = die Staffeln begegnen sich nie.")
        remember("x_slider", x)
    if plan == "luecken":
        seed_widget("p_slider")
        p = st.slider("Wahrscheinlichkeit, dass ein Paar spielt", *bounds("p_slider"), step=C.P_STEP, key="p_slider",
                      help="Jedes Paar spielt mit dieser Wahrscheinlichkeit (dann k-mal), sonst gar nicht.")
        remember("p_slider", p)
    st.markdown("**Zyklen**")
    c = st.slider("Zyklenstärke c", *bounds("c_slider"), step=C.CYCLE_STEP, key="c_slider",
                  help="Wie stark drei Spielstile einander kontern. 0 = keine Zyklen (dann stimmt das Modell).")
    if c > 0:
        seed_widget("stil_select")
        stil = st.selectbox("Stilverteilung", list(C.STIL_LABELS), format_func=C.STIL_LABELS.get, key="stil_select",
                            help="Wie die drei Spielstile verteilt sind. Der Stärkste hat immer Stil 0.")
        remember("stil_select", stil)
    st.markdown("**Vorannahme**")
    lam = st.slider("Gewicht der Vorannahme λ", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider", format="%.2f",
                    help="Ridge auf die Stärken: sie streuen um 0 mit Varianz 1/λ. Klein = fast keine Annahme, groß = alle nahe 0 (dann ordnet die Siegtabelle).")
    st.caption(f"Orakel-λ dieses Turniers (1/Varianz der wahren Stärken, nur in der Simulation bekannt): {ev.de(oracle_lambda(int(n), float(delta)))}")
    seed = st.number_input("Zufalls-Seed des Turniers", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Turnier würfeln", width="stretch", on_click=randomize_seed)

x_value = int(current_value("x_slider"))
p_value = round(float(current_value("p_slider")), 2)
stil_value = str(current_value("stil_select"))
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "k_slider": int(k), "plan_select": plan, "x_slider": x_value,
                   "p_slider": p_value, "c_slider": round(float(c), 2), "stil_select": stil_value, "lam_slider": round(float(lam), 2),
                   "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(k), plan, x_value, p_value, round(float(c), 2), stil_value, round(float(lam), 2), int(seed))
a = _analysis(settings)
t = a.turnier

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Schätzung")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Schätzer existiert", "ja" if a.ford else "nein",
          delta=f"{len(a.components)} Teil(e) des Gewinngraphen", delta_color="off",
          help="Ford-Bedingung: Der Gewinngraph „i → j, wenn i gegen j mindestens einmal gewann“ muss stark zusammenhängend sein.")
m2.metric("τ gegen die Stärke", ev.de(a.tau_s["Bradley-Terry"]),
          delta=f"Siegtabelle {ev.de(a.tau_s['Siege'])}, Colley {ev.de(a.tau_s['Colley'])}", delta_color="off",
          help="Kendall-τ der Bradley-Terry-Ordnung gegen die Generator-Stärke.")
m3.metric("Log-Verlust, frische Partien", ev.de(a.log_loss["Bradley-Terry"], 3),
          delta=f"Münze {ev.de(a.log_loss['Münze'], 3)}, Wahrheit {ev.de(a.log_loss['Wahrheit'], 3)}", delta_color="off",
          help="Mittlerer Verlust je Partie (nat) auf 100 frischen Partien je Paar aus derselben Siegwahrscheinlichkeitsmatrix. Kleiner ist besser.")
if plan == "staffeln":
    m4.metric("Falsch geordnete Staffel-Paare", "–" if math.isnan(a.crossing["Bradley-Terry"]) else f"{ev.pct(a.crossing['Bradley-Terry'])} %",
              delta=f"Siegtabelle {ev.pct(a.crossing['Siege'])} %, Massey {ev.pct(a.crossing['Massey'])} %, Colley {ev.pct(a.crossing['Colley'])} %",
              delta_color="off")
else:
    m4.metric("Newton-Schritte", a.fit.iterations, delta=f"λ = {ev.de(settings.lam)}, Orakel-λ {ev.de(a.oracle_lam)}", delta_color="off")

if not a.ford:
    parts = " → ".join("{" + ", ".join(t.names[i] for i in comp) + "}" for comp in a.components)
    st.warning(f"**Gewinngraph in Dominanzordnung:** {parts}. Eine Komponente steht vor der, die sie schlägt; der Schätzer ohne Vorannahme existiert nur bei "
               "einer einzigen Komponente.")

order = list(range(t.n - 1, -1, -1))
truth = t.strengths - t.strengths.mean()
df = pd.DataFrame({
    "Teilnehmer": [t.names[i] for i in order],
    "Platz nach Stärke": list(range(1, t.n + 1)),
    "Siege": [int(a.scores["Siege"][i]) for i in order],
    "Partien": [int(t.games.sum(axis=1)[i]) for i in order],
    "θ": [round(float(a.fit.theta[i]), 2) for i in order],
    "90-%-Intervall": [f"{a.lo[i]:.2f} bis {a.hi[i]:.2f}".replace(".", ",") for i in order],
    "P(schlägt den Nächsten)": ["–" if math.isnan(a.p_next[i]) else ev.de(float(a.p_next[i])) for i in order],
    "Platz Siegtabelle": [a.rank["Siege"][i] + 1 for i in order],
    "Platz Bradley-Terry": [a.rank["Bradley-Terry"][i] + 1 for i in order],
    "wahre Stärke": [round(float(truth[i]), 2) for i in order],
})
st.dataframe(df, hide_index=True, width="stretch")
st.plotly_chart(build_strengths(a), width="stretch", key="strengths_chart")
st.caption("Geschätzte Stärke mit 90-%-Intervall (Normalnäherung unter der Vorannahme) und die wahre Generator-Stärke; sie ist hier bekannt, weil das Turnier "
           "simuliert ist. Die Schätzung liegt wegen der Vorannahme näher an 0 als die wahre Stärke.")
st.plotly_chart(build_places(a), width="stretch", key="places_chart")
st.caption("Je Teilnehmer der Platz nach jedem Verfahren, in der wahren Reihenfolge; auf der gestrichelten Linie stimmt der Platz.")

if t.n <= 8:
    st.markdown("**Beobachtet und vorhergesagt** (nur für n ≤ 8)")
    st.plotly_chart(build_matrices(a), width="stretch", key="matrix_chart")
st.download_button("📄 PDF der aktuellen Schätzung", data=build_pdf(a), file_name="bradley_terry.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was liefert Bradley-Terry, und wann fehlt der Schätzer?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wann existiert der Schätzer?")
st.caption("8 Teilnehmer (feste Instanz, unabhängig vom Regler), jeder gegen jeden, ausgeglichene Stile; Anteil der 300 festen Turniere, in denen der "
           "Gewinngraph stark zusammenhängt, der Schätzer also existiert.")
grid, plans, extreme = _existence()
st.plotly_chart(build_existence(grid), width="stretch", key="exp1_chart")
verdict(*ev.verdict_existence(grid, plans, extreme))
st.dataframe(pd.DataFrame([{"Spielplan (12 Teilnehmer, 4 Partien je Paar)": r["label"], "Schätzer existiert": f"{ev.pct1(r['exists'])} %",
                            "mit Ungeschlagenem oder Sieglosem": f"{ev.pct1(r['extreme'])} %"} for r in plans]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Die Vorannahme ist das Ranking")
st.caption("12 Teilnehmer (feste Instanz), eine starke und eine schwache Staffel, innerhalb jede gegen jeden, dazu x Kreuzpartien je Starkem; Mittel über "
           "300 feste Turniere. Bradley-Terry mit drei Vorannahmen.")
rows2 = _staffeln()
st.plotly_chart(build_staffeln(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_staffeln(rows2))

st.subheader("🔬 Experiment 3: Vorhersage statt Rang, und das richtige λ")
st.caption("8 Teilnehmer (feste Instanz), vollständig; gemessen an 100 frischen Partien je Paar aus derselben Siegwahrscheinlichkeitsmatrix. Log-Verlust je "
           "Partie in nat, kleiner ist besser; Mittel über 300 feste Turniere.")
rows3 = _predict()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Stärkeabstand 0,3**")
    st.plotly_chart(build_predict(rows3, 0.3), width="stretch", key="exp3_chart_a")
with c2:
    st.markdown("**Stärkeabstand 0,6**")
    st.plotly_chart(build_predict(rows3, 0.6), width="stretch", key="exp3_chart_b")
verdict(*ev.verdict_predict(rows3))
st.dataframe(pd.DataFrame([{"Abstand": ev.de(r["delta"]), "Partien je Paar": r["k"], "Münze": round(r["Münze"], 3), "log5": round(r["log5"], 3),
                            "gesättigt": round(r["gesättigt"], 3), "BT, λ = 1": round(r[ev.bt_key(1.0)], 3),
                            "BT, Orakel-λ": round(r["oracle"], 3), "Orakel-λ": round(r["oracle_lam"], 2), "Wahrheit": round(r["Wahrheit"], 3)}
                           for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Stimmen die Standardfehler?")
st.caption("8 Teilnehmer (feste Instanz), vollständig, ausgeglichene Stile; Anteil der Teilnehmer, deren wahre Stärke im 90-%-Intervall liegt, gemittelt über 300 "
           "feste Turniere.")
rows4 = _se()
st.plotly_chart(build_se(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_se(rows4))

st.subheader("🔬 Experiment 5: Zyklen, wann lohnt das Modell ohne Annahme?")
st.caption("8 Teilnehmer (feste Instanz), vollständig, Konter-Stil dominiert, Zyklenstärke c = 0,3 im Diagramm; Bradley-Terry mit dem Orakel-λ gegen das "
           "gesättigte Modell (je Paar eine eigene Wahrscheinlichkeit); 100 feste Turniere.")
rows5 = _cycles()
st.plotly_chart(build_cycles(rows5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_cycles(rows5))
st.dataframe(pd.DataFrame([{"Zyklenstärke c": ev.de(r["c"]), "Partien je Paar": r["k"], "Mehrverlust Bradley-Terry": round(r["bt_excess"], 4),
                            "Mehrverlust gesättigt": round(r["sat_excess"], 4), "Wahrheit": round(r["truth"], 3)} for r in rows5]),
             hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Schätzer existiert** (Gewinngraph stark zusammenhängend) | Ein Ungeschlagener oder ein zerfallender Plan macht die Stärke unendlich bzw. unbestimmt; nur die Vorannahme λ liefert dann eine Ordnung (Experimente 1 und 2). | Rangintervalle und Shrinkage |
| **Die Vorannahme λ passt zur Streuung** | Zu klein: die Stärken laufen auseinander und die Vorhersage ist schlechter als ein Münzwurf; zu groß: alles nahe 0, die Ordnung ist die der Siegtabelle (Experiment 3). Das Orakel-λ kennt nur die Simulation. | Rangintervalle und Shrinkage |
| **Die Leistung hängt nur von der Stärke ab** (keine Zyklen) | Zyklen sind für das Modell unsichtbar; der Verlust ist klein, aber die Rangfrage ist dann nicht mehr eindeutig (Experiment 5). | Nicht-Transitivität |
| **Ein Spiel ist Sieg oder Niederlage** | Remis, Tore und Heimvorteil haben ein eigenes Modell; hier gäbe es nur 0 oder 1. | Tore, Remis, Heimvorteil |
| **Die Stärke ist konstant, die Partien sind unabhängig** | Verändert sich die Stärke über die Zeit, folgt ihr nur ein fortlaufendes Verfahren. | Elo |
| **Die Standardfehler gelten** | Sie sind eine Normalnäherung unter der Vorannahme; gemessen nur für vollständige, ausgeglichene Pläne ohne Zyklen (Experiment 4). | – |
| **Alles ist simuliert** | Die Stärke ist eine Konstruktion des Generators; das Orakel-λ und die frischen Partien gibt es nur hier. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $P(i \text{ schlägt } j) = \sigma(\theta_i - \theta_j)$ mit $\sigma(x) = 1/(1 + e^{-x})$. $W_{ij}$ sind die Siege von $i$ gegen $j$,
$G = W + W^{\top}$ die Zahl der Partien je Paar. Die Stärken sind nur bis auf eine Konstante bestimmt; gesetzt wird $\sum_i \theta_i = 0$.

**Log-Likelihood.** $\ell(\theta) = \sum_{i \ne j} W_{ij} \log \sigma(\theta_i - \theta_j)$. Gradient: $\partial \ell / \partial \theta_i = w_i - \sum_j G_{ij}\,
\sigma(\theta_i - \theta_j)$ mit den Siegen $w_i$. Die Hesse-Matrix ist $-(\mathrm{diag}(V\mathbf{1}) - V)$ mit $V_{ij} = G_{ij}\,\sigma_{ij}(1 - \sigma_{ij})$,
negativ semidefinit, ihr Kern ist der Einsvektor.

**Existenz (Zermelo 1929, Ford 1957).** Der Maximierer ist endlich genau dann, wenn der Gewinngraph ($i \to j$, wenn $W_{ij} > 0$) stark zusammenhängend
ist; äquivalent: in jeder Zerlegung der Teilnehmer in zwei nichtleere Gruppen hat jemand aus der ersten gegen jemanden aus der zweiten gewonnen.

**Vorannahme.** $\ell_\lambda(\theta) = \ell(\theta) - \tfrac{\lambda}{2}\sum_i \theta_i^2$, streng konkav für $\lambda > 0$ (Normalverteilung mit Varianz $1/\lambda$ als
Vorverteilung). Newton: $\theta \leftarrow \theta + (F + \lambda I)^{-1}\,\nabla \ell_\lambda$ mit der Information $F = \mathrm{diag}(V\mathbf{1}) - V$, mit
Schrittverkürzung. Bei $\lambda \to \infty$ ist die Ordnung die der Siege; im vollständigen, ausgeglichenen Plan ist $\hat\theta$ für jedes $\lambda$ streng
monoton in den Siegen.

**Orakel-$\lambda$.** Die Generator-Stärken haben die Varianz $\delta^2 (n^2 - 1)/12$; $\lambda^\ast = 12/(\delta^2 (n^2 - 1))$ ist die dazu passende
Vorannahme (nur in der Simulation bekannt).

**Standardfehler.** $\hat\Sigma = (F + \lambda I)^{-1}$ an $\hat\theta$, projiziert auf die identifizierte Richtung: $C \hat\Sigma C$ mit
$C = I - \mathbf{1}\mathbf{1}^{\top}/n$; das 90-%-Intervall ist $\hat\theta_i \pm 1{,}645\,\mathrm{se}_i$ (Normalnäherung).

**Zermelo-Iteration (MM).** $\pi_i \leftarrow w_i \big/ \sum_j G_{ij}/(\pi_i + \pi_j)$ mit $\pi = e^{\theta}$; Newman (2023) beschleunigt sie.

**Vorhersage.** Log-Verlust $-\sum W^{\text{neu}}_{ij} \log \hat P_{ij} / \sum W^{\text{neu}}_{ij}$ auf frischen Partien. Vergleichsmodelle: Münze ($1/2$), log5
($q_i = (w_i + \tfrac12)/(g_i + 1)$, $P_{ij} = q_i(1 - q_j)/(q_i(1 - q_j) + q_j(1 - q_i))$), gesättigt ($(W_{ij} + \tfrac12)/(G_{ij} + 1)$), Wahrheit.

**Rangordnung.** Wertungen, die sich um weniger als $10^{-9}$ unterscheiden, gelten als gleich (Gleitkomma-Rauschen); Gleichstand entscheidet das Los.

**Literatur.**
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345
  (das logistische Paarvergleichsmodell).
- Zermelo, E. (1929): Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. *Mathematische Zeitschrift* 29,
  436–460; Ford, L. R., Jr. (1957): Solution of a ranking problem from binary comparisons. *American Mathematical Monthly* 64, 28–33 (Existenzbedingung
  des Schätzers; der Wortlaut der Bedingung nach einer Sekundärquelle, das Original wurde nicht eingesehen).
- Hunter, D. R. (2004): MM algorithms for generalized Bradley-Terry models. *The Annals of Statistics* 32 (1), 384–406, doi:10.1214/aos/1079120141
  (die Zermelo-Iteration als MM-Algorithmus, Konvergenzbedingungen).
- Newman, M. E. J. (2023): Efficient computation of rankings from pairwise comparisons. *Journal of Machine Learning Research* 24 (238), 1–25
  (schnellere Iteration; hier nicht umgesetzt).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
