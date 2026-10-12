"""PDF-Export der aktuellen Konfiguration und der Schätzung (fpdf2, Kernschrift: nur Zeichen aus Latin-1, daher „Tau“ statt τ und „lambda“ statt λ)."""

from fpdf import FPDF

import brt_constants as C
import brt_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("λ", "lambda"), ("θ", "theta"), ("σ", "sigma"), ("≥", ">="), ("≤", "<="), ("→", "->"),
                     ("**", ""), ("€", "EUR"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, t = a.settings, a.turnier
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Bradley-Terry: Siegwahrscheinlichkeit aus der Stärke"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    extra = f", Kreuzpartien: {s.x}" if s.plan == "staffeln" else (f", Spielwahrscheinlichkeit: {ev.de(s.p)}" if s.plan == "luecken" else "")
    pdf.multi_cell(0, 5.5, _latin(f"Teilnehmer: {s.n}, Partien je Paar: {s.k}, Spielplan: {C.PLAN_LABELS[s.plan]}{extra}, Zyklenstärke: {ev.de(s.c)}, "
                                  f"Vorannahme λ: {ev.de(s.lam)}, Seed: {s.seed}."))
    pdf.ln(2)
    level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text))
    pdf.ln(3)
    headers = ("Teilnehmer", "Siege", "Partien", "theta", "90-%-Intervall", "wahre Staerke")
    widths = (28, 18, 20, 24, 50, 30)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in sorted(range(t.n), key=lambda j: a.rank["Bradley-Terry"][j]):
        row = (t.names[i], str(int(a.scores["Siege"][i])), str(int(t.games.sum(axis=1)[i])), ev.de(a.fit.theta[i]),
               f"{ev.de(a.lo[i])} bis {ev.de(a.hi[i])}", ev.de(t.strengths[i] - t.strengths.mean()))
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Generator der Ranking-Linie); die Stärke ist eine Konstruktion der Simulation. Die Intervalle sind eine Normalnäherung unter der "
        "Vorannahme (Ridge) und gelten für vollständige, ausgeglichene Pläne ohne Zyklen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."))
    return bytes(pdf.output())
