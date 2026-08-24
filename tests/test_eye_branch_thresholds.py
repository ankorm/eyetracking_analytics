"""Глазная ветка берёт пороги из модели, а не из зашитых констант.

Прежде `mark_eye_rest_from_angles` и
`mark_eye_data_with_full_angle_from_angles` по умолчанию использовали 15 и
5 град/с и 1 град -- константы вырожденной ветки старой `eye_params`, которые
не зависели от частоты записи. Это противоречило главному свойству методики:
пороги выводятся из частоты. Здесь проверяется, что пороги сходятся -- обе
функции берут их из одной модели и на высоких частотах совпадают с порогами
разметки взора.
"""
from __future__ import annotations

import pytest

from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk
from eyetracking_analytics import models as mdl

from . import synthetic

_EYE_FUNCS = (
    gm.mark_eye_rest_from_angles,
    gm.mark_eye_data_with_full_angle_from_angles,
)
_KW = dict(col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", col_time="T_s", deg=True)


def _params(func, fs_hz: float) -> dict:
    df = synthetic.gaze_2d(fs_hz=fs_hz, duration_s=4.0, n_steps=6, seed=int(fs_hz))
    _, events, _ = func(df, **_KW)
    return events.attrs["params_used"]


@pytest.mark.parametrize("func", _EYE_FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("fs", [90.0, 250.0, 500.0])
def test_eye_thresholds_come_from_the_model(func, fs) -> None:
    """По умолчанию все три порога -- модельные при N_SAMPLES_SACCADE."""
    model = mdl.min_saccade_minima_by_sampling(fs, n_samples=mdl.N_SAMPLES_SACCADE)
    p = _params(func, fs)

    assert p["порог пика скорости"]["источник"] == "модель"
    assert p["порог скорости покоя"]["источник"] == "модель"
    assert p["минимальная амплитуда"]["источник"] == "модель"

    assert p["порог пика скорости"]["значение"] == pytest.approx(model.v_peak_deg_s)
    assert p["порог скорости покоя"]["значение"] == pytest.approx(model.v_fixation_deg_s)
    assert p["минимальная амплитуда"]["значение"] == pytest.approx(model.a_min_deg)


@pytest.mark.parametrize("fs", [90.0, 250.0])
def test_both_eye_functions_agree_on_thresholds(fs) -> None:
    """Ветки различаются только урезанием интервала, но не порогами."""
    a = _params(gm.mark_eye_rest_from_angles, fs)
    b = _params(gm.mark_eye_data_with_full_angle_from_angles, fs)
    for key in ("порог пика скорости", "порог скорости покоя", "минимальная амплитуда"):
        assert a[key]["значение"] == b[key]["значение"], key


def test_eye_and_gaze_thresholds_converge_at_high_rate() -> None:
    """Выше 222 Гц оба физиологических предела связывают, и `n_samples` перестаёт влиять.

    Ниже этой частоты пороги саккады и перевода взора расходятся -- намеренно:
    требования к переводу взора ниже, потому что модель по нему не восстанавливается.
    """
    saccade = mdl.min_saccade_minima_by_sampling(250.0, n_samples=mdl.N_SAMPLES_SACCADE)
    gaze = mdl.min_saccade_minima_by_sampling(250.0, n_samples=mdl.N_SAMPLES_GAZE_SHIFT)

    assert saccade.a_min_deg == pytest.approx(gaze.a_min_deg)
    assert saccade.v_peak_deg_s == pytest.approx(gaze.v_peak_deg_s)
    assert saccade.v_fixation_deg_s == pytest.approx(gaze.v_fixation_deg_s)

    # На 90 Гц -- расходятся, и это тоже надо видеть.
    low_s = mdl.min_saccade_minima_by_sampling(90.0, n_samples=mdl.N_SAMPLES_SACCADE)
    low_g = mdl.min_saccade_minima_by_sampling(90.0, n_samples=mdl.N_SAMPLES_GAZE_SHIFT)
    assert low_s.a_min_deg > low_g.a_min_deg
    assert low_s.v_peak_deg_s > low_g.v_peak_deg_s


@pytest.mark.parametrize("func", _EYE_FUNCS, ids=lambda f: f.__name__)
def test_explicit_thresholds_still_win(func) -> None:
    """Заданное пользователем важнее модельного -- контракт всей библиотеки."""
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=4.0, n_steps=6, seed=90)
    _, events, _ = func(df, **_KW, min_peak_velocity_deg_s=15.0,
                        fixation_velocity_limit_deg_s=5.0, min_amplitude_deg=1.0)
    p = events.attrs["params_used"]

    assert p["порог пика скорости"]["источник"] == "пользователь"
    assert p["порог пика скорости"]["значение"] == pytest.approx(15.0)
    assert p["минимальная амплитуда"]["значение"] == pytest.approx(1.0)


def test_gaze_branch_is_unaffected() -> None:
    """Разметка взора продолжает брать N_SAMPLES_GAZE_SHIFT."""
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=4.0, n_steps=6, seed=90)
    res = mk.mark_gaze_events_direction(
        df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", geometry="angles")
    model = mdl.min_saccade_minima_by_sampling(90.0, n_samples=mdl.N_SAMPLES_GAZE_SHIFT)
    p = res.params_used
    assert p["порог пика скорости"]["значение"] == pytest.approx(model.v_peak_deg_s)
