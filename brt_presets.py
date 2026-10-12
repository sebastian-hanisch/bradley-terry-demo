"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Ausblendbare Regler (Kreuzpartien x nur bei Staffeln, Spielwahrscheinlichkeit p nur bei Lücken,
Stilverteilung nur bei c > 0) folgen dem KEPT-Muster (DEMO-PLAYBOOK §3): der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben,
sondern in KEPT abgelegt und unmittelbar vor dem Zeichnen per `seed_widget` eingesetzt."""

import random
from dataclasses import dataclass

import streamlit as st

import brt_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "plan_select": SettingSpec("plan", str, "vollstaendig", choices=tuple(C.PLAN_LABELS)),
    "x_slider": SettingSpec("x", int, C.DEFAULT_X, C.X_MIN, C.X_MAX),
    "p_slider": SettingSpec("p", float, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "c_slider": SettingSpec("c", float, C.DEFAULT_CYCLE, C.CYCLE_MIN, C.CYCLE_MAX),
    "stil_select": SettingSpec("stil", str, "ausgewogen", choices=tuple(C.STIL_LABELS)),
    "lam_slider": SettingSpec("lam", float, C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "delta": "delta_slider", "k": "k_slider", "plan": "plan_select", "x": "x_slider", "p": "p_slider", "c": "c_slider",
               "stil": "stil_select", "lam": "lam_slider", "seed": "seed_input"}
KEPT_KEYS = ("x_slider", "p_slider", "stil_select")


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
