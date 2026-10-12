"""PDF-Export: erzeugt für jede Planart und mit fehlendem Schätzer gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import brt_evaluation as ev
import brt_pdf_export as X
from brt_scenario import Settings


def test_pdf_is_built_for_every_plan_and_for_a_missing_estimator():
    for s in (Settings(), Settings(n=12, plan="staffeln", x=0, lam=0.1, seed=3), Settings(n=6, plan="luecken", p=0.3, k=2, seed=5),
              Settings(n=16, plan="vollstaendig", k=20, c=0.3, stil="konter", lam=0.05, seed=7)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≥ 0,5; λ = 1 → θ − σ **fett** €")
    assert out == "Tau >= 0,5; lambda = 1 -> theta - sigma fett EUR"
    assert out.encode("latin-1")
