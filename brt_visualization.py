"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import brt_constants as C
import brt_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_strengths(a):
    """Geschätzte Stärke mit 90-%-Intervall je Teilnehmer, in der wahren Reihenfolge (Stärkster links); dazu die wahre Generator-Stärke."""
    t = a.turnier
    n = t.n
    order = list(range(n - 1, -1, -1))
    names = [t.names[i] for i in order]
    truth = t.strengths - t.strengths.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names, y=[float(truth[i]) for i in order], mode="markers", name="wahre Stärke (Generator)",
                             marker=dict(size=11, symbol="diamond", color="#e67e22", line=dict(width=1, color="#ffffff")),
                             hovertemplate="%{x}: wahre Stärke %{y:.2f}<extra></extra>"))
    th = a.fit.theta
    fig.add_trace(go.Scatter(
        x=names, y=[float(th[i]) for i in order], mode="markers", name="Bradley-Terry mit 90-%-Intervall", marker=dict(size=9, color=C.COLOR_BT),
        error_y=dict(type="data", symmetric=False, array=[float(a.hi[i] - th[i]) for i in order], arrayminus=[float(th[i] - a.lo[i]) for i in order],
                     color=C.COLOR_BT, thickness=2, width=6),
        hovertemplate="%{x}: θ = %{y:.2f}<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Teilnehmer (wahre Reihenfolge, Stärkster links)")
    fig.update_yaxes(title_text="Stärke θ (logistische Skala, Mittel 0)")
    return _base(fig, height=400)


def build_places(a):
    """Platz je Verfahren für jeden Teilnehmer, in der wahren Reihenfolge (Stärkster links): auf der Diagonale stimmt das Verfahren."""
    t = a.turnier
    n = t.n
    order = list(range(n - 1, -1, -1))
    names = [t.names[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names, y=list(range(1, n + 1)), mode="lines", line=dict(color="rgba(150,150,150,0.5)", dash="dash"), name="wahre Ordnung",
                             hoverinfo="skip"))
    symbols = {"Siege": "circle-open", "Colley": "square-open", "Bradley-Terry": "circle"}
    for m in ("Siege", "Colley", "Bradley-Terry"):
        fig.add_trace(go.Scatter(x=names, y=[a.rank[m][i] + 1 for i in order], mode="markers", name=m,
                                 marker=dict(size=11 if m == "Bradley-Terry" else 8, color=C.METHOD_COLORS[m], symbol=symbols[m]),
                                 hovertemplate="%{x}: Platz %{y} (" + m + ")<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Teilnehmer (wahre Reihenfolge, Stärkster links)")
    fig.update_yaxes(title_text="Platz", autorange="reversed", dtick=1)
    return _base(fig, height=360)


def build_matrices(a):
    """Beobachteter Sieganteil je Paar und vorhergesagte Siegwahrscheinlichkeit nebeneinander (nur für kleine n), in der wahren Reihenfolge."""
    from brt_fit import win_probabilities
    t = a.turnier
    n = t.n
    order = list(range(n - 1, -1, -1))
    names = [t.names[i] for i in order]
    P_hat = win_probabilities(a.fit.theta)[np.ix_(order, order)]
    G = t.games[np.ix_(order, order)].astype(float)
    obs = np.divide(t.wins[np.ix_(order, order)], G, out=np.full((n, n), np.nan), where=G > 0)
    for m in (P_hat, obs):
        np.fill_diagonal(m, np.nan)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("beobachtet: Siege / Partien", "vorhergesagt: σ(θ_i − θ_j)"), horizontal_spacing=0.12)
    for col, mat in ((1, obs), (2, P_hat)):
        text = [["" if np.isnan(v) else f"{v:.2f}".replace(".", ",") for v in row] for row in mat]
        fig.add_trace(go.Heatmap(z=mat, x=names, y=names, text=text, texttemplate="%{text}", colorscale="RdBu", zmin=0, zmax=1, zmid=0.5, showscale=False,
                                 hovertemplate="%{y} gegen %{x}: %{text}<extra></extra>", xgap=1, ygap=1), row=1, col=col)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top")
    return _base(fig, height=max(300, 36 * n + 120))


def build_existence(grid):
    """Anteil der Turniere mit existierendem Schätzer, nach Partien je Paar, eine Linie je Stärkeabstand."""
    ks = sorted({r["k"] for r in grid})
    fig = go.Figure()
    for d in sorted({r["delta"] for r in grid}):
        rows = sorted((r for r in grid if r["delta"] == d), key=lambda r: r["k"])
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in rows], y=[r["exists"] for r in rows], mode="lines+markers", name=f"Abstand {ev.de(d)}",
                                 hovertemplate=f"Abstand {ev.de(d)}, %{{x}} Partien je Paar: %{{y:.0%}}<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category", categoryorder="array", categoryarray=[str(k) for k in ks])
    fig.update_yaxes(title_text="Turniere mit existierendem Schätzer", tickformat=".0%", range=[0, 1.05])
    return _base(fig)


def build_staffeln(rows, lams=C.STAFFEL_LAMS):
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    series = [("Siege", "Siegtabelle", C.COLOR_SIEGE), ("Massey", "Massey", C.COLOR_MASSEY), ("Colley", "Colley", C.COLOR_COLLEY)]
    shades = ("#9cc5bf", "#5f9d95", "#1f4a45")
    series += [(ev.bt_key(l), f"Bradley-Terry, λ = {ev.de(l)}", shades[i]) for i, l in enumerate(lams)]
    for key, label, color in series:
        fig.add_trace(go.Bar(x=xs, y=[r[key] for r in rows], name=label, marker_color=color,
                             hovertemplate=label + ", %{x} Kreuzpartien: %{y:.0%} falsch<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#c0392b", annotation_text="Münzwurf", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Kreuzpartien je Spieler der starken Staffel", type="category")
    fig.update_yaxes(title_text="falsch geordnete Paare über die Staffeln", tickformat=".0%", range=[0, 0.6])
    return _base(fig, height=400)


def build_predict(rows, delta, lams=C.PREDICT_LAMS):
    """Log-Verlust gegen λ, eine Linie je Partienzahl; Münze und Wahrheit als Bezugslinien (Wahrheit für die kleinste Partienzahl dieselbe)."""
    sel = sorted((r for r in rows if r["delta"] == delta), key=lambda r: r["k"])
    xs = [ev.de(l) for l in lams]
    fig = go.Figure()
    for r in sel:
        fig.add_trace(go.Scatter(x=xs, y=[r[ev.bt_key(l)] for l in lams], mode="lines+markers", name=f"{r['k']} Partien je Paar",
                                 hovertemplate=f"{r['k']} Partien, λ = %{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=sel[0]["Münze"], line_dash="dash", line_color="#c0392b", annotation_text="Münzwurf", annotation_position="top right")
    fig.add_hline(y=sel[0]["Wahrheit"], line_dash="dot", line_color="#222222", annotation_text="Wahrheit", annotation_position="bottom right")
    fig.update_xaxes(title_text="Vorannahme λ", type="category")
    fig.update_yaxes(title_text="Log-Verlust je frischer Partie (nat)")
    return _base(fig, height=400)


def build_se(rows):
    fig = go.Figure()
    for lam in sorted({r["lam"] for r in rows}):
        sel = sorted((r for r in rows if r["lam"] == lam), key=lambda r: r["k"])
        fig.add_trace(go.Scatter(x=[str(r["k"]) for r in sel], y=[r["coverage"] for r in sel], mode="lines+markers", name=f"λ = {ev.de(lam)}",
                                 hovertemplate=f"λ = {ev.de(lam)}, %{{x}} Partien: %{{y:.1%}}<extra></extra>"))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#c0392b", annotation_text="nominal 90 %", annotation_position="top right")
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="Abdeckung des 90-%-Intervalls", tickformat=".0%", range=[0.8, 1.0])
    return _base(fig)


def build_cycles(rows, c=0.3):
    sel = sorted((r for r in rows if r["c"] == c), key=lambda r: r["k"])
    xs = [str(r["k"]) for r in sel]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["bt_excess"] for r in sel], mode="lines+markers", name="Bradley-Terry (eine Stärke je Teilnehmer)",
                             line=dict(color=C.COLOR_BT, width=3), hovertemplate="%{x} Partien: %{y:.4f} nat<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sat_excess"] for r in sel], mode="lines+markers", name="gesättigt (eine Wahrscheinlichkeit je Paar)",
                             line=dict(color="#8e6bb0", width=3), hovertemplate="%{x} Partien: %{y:.4f} nat<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="Mehrverlust (nat je Partie)", type="log")
    return _base(fig, height=380)
