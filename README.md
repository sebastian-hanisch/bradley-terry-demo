# Bradley-Terry: Siegwahrscheinlichkeit aus der Stärke – Streamlit-Demo

Fünftes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `siegtabelle-demo`, `rangdistanz-demo`, `rangaggregation-demo` und `massey-colley-demo`: Die
Siegtabelle zählt, Massey und Colley lösen ein lineares System. Hier bekommt jeder Teilnehmer eine **Stärke**, und jede Partie wird mit der Wahrscheinlichkeit
**σ(Stärke des einen minus Stärke des anderen)** entschieden: das **Bradley-Terry-Modell**. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel
(Generator Version 1, wortgleich); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Was liefert ein Modell mit Stärken gegenüber der Tabelle und den linearen Systemen, und was kostet es? **Es ordnet nicht besser:** im vollständigen Plan ist die
Ordnung die der Siegtabelle, bei Lücken liegt sie bei Colley und Massey. Es liefert **Wahrscheinlichkeiten und Intervalle**, und die Intervalle stimmen. Der
Preis: **der Schätzer existiert oft nicht**, und wo er fehlt, **ist die Vorannahme das Ranking**.

## Modell

- **Bradley-Terry:** P(i schlägt j) = σ(θ_i − θ_j) mit σ(x) = 1/(1 + e^(−x)). Nur Differenzen zählen; die Stärken werden auf Mittelwert 0 gesetzt. Ausgänge sind Sieg
  oder Niederlage, keine Punkte.
- **Schätzung:** Maximum Likelihood auf der konkaven Log-Likelihood, mit einem Zuschlag −(λ/2) Σ θ² (Ridge: „die Stärken streuen um 0 mit Varianz 1/λ“). Gelöst
  per **Newton-Verfahren** mit Schrittverkürzung; die ältere **Zermelo-Iteration** (MM-Algorithmus) steht als Orakel und für die Zählung der Schritte im Test.
- **Existenz:** Der Schätzer ohne Zuschlag ist genau dann endlich, wenn der Gewinngraph („i → j, wenn i mindestens einmal gegen j gewann“) stark
  zusammenhängend ist (Ford-Bedingung). Die App zeigt die Komponenten in Dominanzordnung sowie Ungeschlagene und Sieglose.
- **Intervalle:** Normalnäherung aus der Information plus Zuschlag, projiziert auf die identifizierte Richtung (θ_i minus Mittel); 90-%-Intervall θ ± 1,645 · se.
- **Vorhersage:** Log-Verlust (nat je Partie) auf **100 frischen Partien je Paar**, gezogen aus derselben Siegwahrscheinlichkeitsmatrix wie das Turnier (gleiche
  Stärken, gleiche Stile; abgeleiteter Seed, der gemeinsame Generator bleibt unverändert). Gegenmodelle: Münze, log5, gesättigt (je Paar eine eigene
  Wahrscheinlichkeit), Wahrheit.
- **Orakel-λ:** die Generator-Stärken haben die Varianz δ²(n² − 1)/12; 1/Varianz ist die dazu passende Vorannahme. Das gibt es nur in der Simulation.
- **Rangordnung:** Wertungen, die sich um weniger als 10⁻⁹ unterscheiden, gelten als gleich; Gleichstand entscheidet das Los.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 300 feste Turniere (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanzen unabhängig vom Regler
  (8 Teilnehmer, in Experiment 2 zwölf). Experiment 5 (bis zu 600 Partien je Paar) über die ersten 100 dieser Turniere.
- **Maße:** Anteil der Turniere mit existierendem Schätzer, falsch geordnete Paare über die Staffeln, Kendall-τ gegen die Generator-Stärke, Log-Verlust auf
  frischen Partien, Abdeckung des 90-%-Intervalls, Iterationen.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit `siegtabelle-demo`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 300 feste Turniere, 8 Teilnehmer, Stärkeabstand 0,3, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Existiert der Schätzer? (vollständig, 8 Teilnehmer) | **1 Partie je Paar: 66 %**, 2 Partien: 95,7 %, 4 Partien: 100 %. Bei 1 Partie bleibt in 33 % der Turniere jemand ungeschlagen oder sieglos. | `test_with_one_game_per_pair_the_estimator_often_does_not_exist` |
| … nach Stärkeabstand | Eine Partie je Paar: Abstand 0,1: 86 %, 0,6: **21 %**, 1,0: **2 %**; bei vier Partien und Abstand 1,0 noch 62 %. | `test_a_larger_strength_gap_makes_it_worse` |
| … nach Spielplan (12 Teilnehmer, 4 Partien) | Vollständig 100 %, zwei Staffeln mit einer Kreuzpartie 98 %, zufällige Lücken (p = 0,5) 81 %. 16 Teilnehmer, Abstand 1,0, 1 Partie: **0 %**. | `test_by_plan_and_the_extreme_case` |
| Zwei Staffeln, eine Kreuzpartie je Starkem (12 Teilnehmer, 1 Partie) | Falsch geordnete Paare: Siegtabelle **37 %**, Massey 15,5 %, Colley 24 %. **Bradley-Terry hängt an λ: 15 % (λ = 0,01), 18 % (0,1), 29 % (1).** Mit 2 Kreuzpartien: 27 / 12 / 16 % und 11 / 12,5 / 18 %. | `test_the_prior_decides_the_order_across_the_leagues` |
| Vorhersage bei 1 Partie je Paar (Log-Verlust, Münze 0,693) | λ = 0,01: **0,843, schlechter als die Münze**; λ = 0,1: 0,741; λ = 1: 0,650; λ = 2: **0,644**; Wahrheit 0,587. | `test_with_one_game_and_almost_no_prior_the_prediction_is_worse_than_a_coin` |
| Bestes λ | Bei Abstand 0,3 ist λ = 2 das beste des Rasters (Orakel 2,12), bei Abstand 0,6 λ = 0,5 (Orakel 0,53), für 1, 2, 4 und 10 Partien je Paar; das Orakel-λ liegt höchstens 0,005 über dem besten Rasterwert. | `test_the_best_prior_is_the_inverse_variance_of_the_true_strengths` |
| Gegen das Modell ohne Annahme | Mit dem Orakel-λ schlägt Bradley-Terry das gesättigte Modell (je Paar eine Wahrscheinlichkeit) bei 1 bis 10 Partien je Paar in allen Zeilen, und log5 bei 4 Partien (Abstand 0,3). | `test_bradley_terry_beats_the_saturated_model_for_few_games` |
| Standardfehler | Das 90-%-Intervall deckt die wahre Stärke in **89 bis 93 %** der Fälle ab (1 bis 40 Partien, λ = 0,1 und 1). Der Standardfehler fällt von 0,57 (1 Partie) auf 0,12 (40 Partien). | `test_the_ninety_percent_interval_covers_about_ninety_percent` |
| Zyklen (Konter-Stil, c = 0,3) | Bradley-Terry verliert gegenüber der Wahrheit 0,010 nat je Partie bei 20 Partien je Paar und 0,004 bei 600; das gesättigte Modell 0,023 und 0,001. **Die Kreuzung liegt zwischen 60 und 200 Partien je Paar** (c = 0,3 und 0,6), zwischen 200 und 600 bei c = 0,15. | `test_the_unmodelled_cycles_cost_little_and_the_saturated_model_pays_off_late` |
| Rechenaufwand (12 Teilnehmer, nur Turniere mit Schätzer) | Newton **5 bis 9,5 Schritte**, Zermelo-Iteration **47 bis 1419**: das 9- bis 150-Fache, wachsend mit dem Stärkeabstand (Abstand 0,1: 5 gegen 47 bis 53; Abstand 1,0: 9,5 gegen 1235 bis 1419). | `test_newton_needs_far_fewer_iterations_than_the_mm_iteration` |

## Befunde und Korrekturen gegenüber dem Plan

- **Standardfehler, Messartefakt der Vorab-Messreihe:** Die Kovarianz muss auf θ_i minus Mittel projiziert werden; sonst steckt 1/(nλ) aus der Nullrichtung der
  Likelihood im Standardfehler (in der Vorab-Messreihe 3,5 statt 1,1) und die Intervalle decken scheinbar 100 % ab. Die Demo projiziert, der Orakel-Test prüft gegen die Pseudoinverse.
- **Frische Partien aus derselben Matrix:** Ein anderer Seed für die Testpartien erzeugte andere Stile und ließ das gesättigte Modell schlecht aussehen. Die
  Testpartien werden aus P des Trainingsturniers gezogen.
- **Schrittzahl war Rauschen, von der Linux-CI gefunden:** Die erste Fassung zählte für Newton 13 bis 71 Schritte (Vorab-Messreihe 17 bis 59) und im App-Text
  stand „hundert- bis tausendmal so viele“ für Zermelo; der Claims-Test maß das 2,6- bis 29-Fache und der Text wurde danach auf „Drei- bis Dreißigfache“
  gesetzt. Auf Linux kam derselbe Test auf 18 statt 20 Schritte. Ursache: Bei λ = 10⁻⁸ verstärkt die Lösung des Newton-Systems Rundungsrauschen in der
  Richtung „alle Stärken gleich“ mit 1/λ; der Mittelwert der Stärken driftete auf 2·10⁻⁷, und die Zahl der Schritte hing am Rauschen (Maximum 116). Der
  exakte Schritt hat Mittelwert 0; mit zentriertem Schritt sind es 5 bis 9,5 Schritte, plattformstabil. Das Verhältnis zu Zermelo ist damit das 9- bis
  150-Fache; der App-Text lautet „Zehn- bis Hundertfünfzigfache“. Die Orakel-Vergleiche der Lösung (scipy, MM-Iteration) waren auch vorher grün; der Mittelwert wird am Ende ohnehin auf 0 gesetzt.
- **Verdikt unvollständig, vom Test gefunden:** Bei zwei Staffeln ohne Kreuzpartie nannte das Verdikt nur den Ungeschlagenen, nicht den zerfallenden Plan. Es
  nennt jetzt beides.
- **Zahlen wichen im Rauschen ab:** Existenz bei 1 Partie 66 % statt 61 % der Vorab-Messreihe (andere feste Turniere). Das Zyklen-Experiment rechnet
  Bradley-Terry mit dem Orakel-λ statt mit λ = 0,1: ein fairerer Vergleich, kleinere Zahlen bei wenigen Partien.
- **Seed:** Der Standard-Seed 40 zeigt den Fall, in dem der Schätzer fehlt (B bleibt ungeschlagen); die Presets zeigen in den Mittelwerten über 80 Turniere, was
  ihre Beschreibung verspricht.

## Abgrenzung

- [Newton-Verfahren](https://sebastianhanisch-newton-verfahren-demo.streamlit.app/) (nichtlineare Optimierung) erklärt Newton für allgemeine Funktionen; hier
  löst es die konkave Log-Likelihood. Querverweis, kein Nachbau.
- **Siegtabelle** (Wurzel) und **Massey und Colley** liefern Spielpläne und Vergleichsverfahren; dieses Stück erklärt, warum die Ordnung im vollständigen Plan
  dieselbe ist, und was das Modell zusätzlich liefert.
- **Rangintervalle und Shrinkage** schätzen λ (empirisches Bayes) und bilden Rangintervalle; hier ist λ gesetzt und die Intervalle gelten für θ.

## Ehrliche Grenzen

- **Kein Gewinn an Rangtreue** gegenüber Colley und der Siegtabelle; der Wert liegt in Wahrscheinlichkeit und Unsicherheit.
- **λ ist eine Annahme:** bei Staffeln und wenigen Partien bestimmt sie das Ergebnis; das Orakel-λ gibt es nur in der Simulation.
- **Stark stochastische Transitivität** steckt im Modell: Zyklen sind unsichtbar; der Verlust ist klein, aber die Rangfrage ist dann nicht mehr eindeutig.
- **Unentschieden, Tore, Heimvorteil** fehlen: nur Sieg oder Niederlage.
- **Standardfehler sind eine Normalnäherung** unter der Vorannahme; gemessen nur für vollständige, ausgeglichene Pläne ohne Zyklen, nicht für Staffeln und Lücken.
- **Alles simuliert**, ein Generator (Version 1), n ≤ 16, gleichabständige Stärken, keine Zeitabhängigkeit.

## Quellen

- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39 (3/4), 324–345.
- Zermelo, E. (1929): Die Berechnung der Turnier-Ergebnisse als ein Maximumproblem der Wahrscheinlichkeitsrechnung. *Mathematische Zeitschrift* 29, 436–460.
- Ford, L. R., Jr. (1957): Solution of a ranking problem from binary comparisons. *American Mathematical Monthly* 64, 28–33 (Existenzbedingung des Schätzers;
  der Wortlaut der Bedingung nach einer Sekundärquelle, das Original wurde nicht eingesehen).
- Hunter, D. R. (2004): MM algorithms for generalized Bradley-Terry models. *The Annals of Statistics* 32 (1), 384–406, doi:10.1214/aos/1079120141.
- Newman, M. E. J. (2023): Efficient computation of rankings from pairwise comparisons. *Journal of Machine Learning Research* 24 (238), 1–25 (hier nicht
  umgesetzt).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472.

## Tests

94 Tests, rund 75 Sekunden: Einheiten gegen die von Hand gerechneten Mini-Instanzen (`test_fit.py`, `test_existence.py`, `test_heldout.py`,
`test_baselines.py`), Generator (`test_generator.py`, wortgleich mit der Wurzel), **Orakel** (`test_oracle_bradley_terry.py`: Newton gegen `scipy.optimize` auf
der Partienliste, Newton gegen die Zermelo-Iteration, Existenz gegen `networkx` samt Divergenz bei verschwindender Vorannahme, Ordnung gegen die Siegtabelle,
Standardfehler gegen die Pseudoinverse und die geschlossene Form, frische Partien gegen die Entropie), Presets, ausblendbare Regler
(`test_hidden_widget_state.py`), Analyse und Verdikte (`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, 22 von 30 geplanten Mutanten, dann Zeitbudget): Nur gegen die Orakel-Tests wurden 10 getötet, 12
überlebten (Rate 45 %). Die 12 Überlebenden wurden einzeln gelesen und gegen die Einheitentests geprüft: 6 von den bestehenden Einheits- und Generator-Tests
getötet, 2 äquivalent (`>=` statt `>` in der Schrittverkürzung, `<=` statt `<` bei einer Gleitkomma-Zufallszahl), 4 echte Lücken (Vorzeichen des Zuschlags in der
Information, Meldung der λ-Prüfung, früher Abbruch der Zermelo-Iteration, Warnung bei 0/0 im Log-Verlust): mit Tests geschlossen und danach einzeln erneut
getötet, nicht erneut mit dem Werkzeug gemessen. Das Orakel selbst wurde um die Definition von Ungeschlagenen und Sieglosen gegen die Partienliste ergänzt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `brt_fit.py` | Log-Likelihood, Gradient, Information, Newton mit Zuschlag, Zermelo/MM, Standardfehler, Orakel-λ |
| `brt_existence.py` | Gewinngraph, Komponenten in Dominanzordnung, Ford-Bedingung, Ungeschlagene und Sieglose |
| `brt_heldout.py` | frische Partien, Log-Verlust, Münze, log5, gesättigtes Modell |
| `brt_baselines.py` | Siegquote, Massey, Colley, Rangordnung mit Toleranz, Kendall-τ, Staffel-Paare |
| `brt_scenario.py` | Reglerwerte (`Settings`) und das Turnier |
| `brt_evaluation.py` | Analyse, fünf Experimente über die festen Turniere, Verdikte |
| `brt_presets.py`, `brt_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `brt_visualization.py`, `brt_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Schätzung von λ** (empirisches Bayes, Kreuzvalidierung): gehört zu „Rangintervalle und Shrinkage“; hier ist λ ein Regler mit dem Orakel-Wert als Marke.
- **Zyklenmodelle** (Davidson, Rao-Kupper, Hodge-Zerlegung): „Nicht-Transitivität“; hier nur das gesättigte Modell als Referenz.
- **Remis, Tore, Heimvorteil:** „Tore, Remis, Heimvorteil“; hier nur Sieg oder Niederlage.
- **Fortlaufende Schätzung** (Elo und Nachfolger): eigene Stücke; hier ein Schätzer über alle Partien.
- **Schnellere Iteration** (Newman 2023): nicht umgesetzt; Newton genügt für n ≤ 16.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
