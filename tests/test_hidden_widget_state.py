"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider} | {s.key for s in at.selectbox}


def test_widgets_are_only_drawn_where_they_act():
    at = fresh()
    assert at.selectbox(key="plan_select").value == "vollstaendig"
    keys = widget_keys(at)
    assert not {"x_slider", "p_slider", "stil_select"} & keys
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    assert "x_slider" in widget_keys(at) and "p_slider" not in widget_keys(at)
    at.selectbox(key="plan_select").select("luecken").run(timeout=TIMEOUT)
    assert "p_slider" in widget_keys(at) and "x_slider" not in widget_keys(at)
    at.slider(key="c_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert "stil_select" in widget_keys(at)
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    at.slider(key="x_slider").set_value(4).run(timeout=TIMEOUT)
    at.selectbox(key="plan_select").select("vollstaendig").run(timeout=TIMEOUT)
    assert "x_slider" not in at.session_state
    at.selectbox(key="plan_select").select("staffeln").run(timeout=TIMEOUT)
    assert at.slider(key="x_slider").value == 4
    at.slider(key="c_slider").set_value(0.2).run(timeout=TIMEOUT)
    at.selectbox(key="stil_select").select("konter").run(timeout=TIMEOUT)
    at.slider(key="c_slider").set_value(0.0).run(timeout=TIMEOUT)
    assert "stil_select" not in at.session_state
    at.slider(key="c_slider").set_value(0.2).run(timeout=TIMEOUT)
    assert at.selectbox(key="stil_select").value == "konter"
    assert not at.exception


def test_preset_sets_values_and_hidden_widgets_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Zwei Staffeln, eine Kreuzpartie"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="plan_select").value == "staffeln"
    assert at.slider(key="n_slider").value == 12 and abs(at.slider(key="lam_slider").value - 0.1) < 1e-9
    assert at.slider(key="x_slider").value == 1
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Viele Partien"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="plan_select").value == "vollstaendig" and at.slider(key="k_slider").value == 20
    assert "x_slider" not in at.session_state
    assert not at.exception
