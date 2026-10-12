"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt, was seine Beschreibung verspricht."""

import dataclasses

import brt_constants as C
import brt_evaluation as ev
import brt_presets as P
from brt_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX, name
        assert C.K_MIN <= p["k"] <= C.K_MAX and C.X_MIN <= p["x"] <= C.X_MAX and C.P_MIN <= p["p"] <= C.P_MAX, name
        assert p["plan"] in C.PLAN_LABELS and p["stil"] in C.STIL_LABELS, name
        assert C.CYCLE_MIN <= p["c"] <= C.CYCLE_MAX and C.LAM_MIN <= p["lam"] <= C.LAM_MAX, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_first_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Eine Partie je Paar"]) == Settings()


def test_setting_specs_cover_every_preset_key_and_settings_fields_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_widgets():
    assert set(P.KEPT_KEYS) == {"x_slider", "p_slider", "stil_select"}


def _mean(preset, attr, seeds=range(100000, 100080)):
    vals = [attr(ev.analyse(Settings(**{**C.PRESETS[preset], "seed": s}))) for s in seeds]
    return sum(vals) / len(vals)


def test_each_preset_tells_its_story():
    one = "Eine Partie je Paar"
    assert _mean(one, lambda a: float(not a.ford)) > 0.2                      # der Schätzer fehlt in einem guten Teil der Turniere
    staffeln = "Zwei Staffeln, eine Kreuzpartie"
    assert _mean(staffeln, lambda a: a.crossing["Bradley-Terry"]) < _mean(staffeln, lambda a: a.crossing["Siege"]) - 0.1
    wrong = "Falsche Vorannahme"
    assert _mean(wrong, lambda a: a.log_loss["Bradley-Terry"]) > _mean(wrong, lambda a: a.log_loss["Münze"])
    many = "Viele Partien"
    assert _mean(many, lambda a: float(a.ford)) == 1.0
    assert _mean(many, lambda a: a.log_loss["Bradley-Terry"]) < _mean(many, lambda a: a.log_loss["Wahrheit"]) + 0.02
    assert _mean(many, lambda a: float(a.rank["Bradley-Terry"] == a.rank["Siege"])) > 0.8
