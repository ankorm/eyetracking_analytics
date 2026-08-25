"""Журнал вычислений: какие параметры применены и откуда взяты.

Требование: по результату всегда можно восстановить условия расчёта, а библиотека
ничего не печатает в stdout.
"""
from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout

import numpy as np
import pytest

from eyetracking_analytics import eyetracking_funs as ef
from eyetracking_analytics import for_raw_data as frd
from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import report as rp

from . import synthetic

_SIGMOID_KW = dict(
    col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s",
    min_peak_velocity_deg_s=30.0, min_amplitude_deg=0.5, max_axis_dt_s=0.012, dt_window_s=0.05,
    r2_min_keep=0.5,
)


def _record(fs_hz: float = 250.0, **kw):
    return synthetic.gaze_2d(fs_hz=fs_hz, duration_s=10.0, n_steps=15, seed=606, **kw)


def _arrays(df):
    return (df["T_s"].to_numpy(dtype=float),
            df["gaze_yaw"].to_numpy(dtype=float),
            df["gaze_pitch"].to_numpy(dtype=float))


def test_library_does_not_print_to_stdout() -> None:
    """Ни одна функция не должна писать в stdout, даже при verbose."""
    df = _record()
    t, x, y = _arrays(df)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        ef.ksv(t, x, y)
        frd.angular_velocity_at_rate(t, x, y)
        gm.mark_eye_data_with_sigmoid(df, verbose=True, **_SIGMOID_KW)
    assert buffer.getvalue() == ""


def test_verbose_messages_go_to_logger(caplog: pytest.LogCaptureFixture) -> None:
    df = _record()
    with caplog.at_level(logging.INFO, logger=rp.LOGGER_NAME):
        gm.mark_eye_data_with_sigmoid(df, verbose=True, **_SIGMOID_KW)
    assert any("sigmoid" in r.message for r in caplog.records)


# ─────────────────────────────────────────────── params_used в результатах
def _assert_params_documented(params_used: dict) -> None:
    assert params_used, "журнал пуст"
    for name, item in params_used.items():
        assert set(item) == {"значение", "единицы", "источник", "пояснение"}, name
        assert item["источник"], f"у параметра {name} не указан источник"


def test_ksv_result_carries_params_used() -> None:
    t, x, y = _arrays(_record())
    r = ef.ksv(t, x, y)
    _assert_params_documented(r.params_used)
    assert "порог скорости фиксации" in r.params_used
    assert r.params_used["порог скорости фиксации"]["источник"] == rp.FROM_MODEL


def test_velocity_result_carries_params_used() -> None:
    t, x, y = _arrays(_record())
    r = frd.angular_velocity_at_rate(t, x, y)
    _assert_params_documented(r.params_used)
    assert r.params_used["частота записи"]["источник"] == rp.FROM_DATA


def test_markup_results_carry_params_used() -> None:
    df = _record()
    _, saccades, fixations = gm.mark_eye_data_with_sigmoid(df, **_SIGMOID_KW)
    for frame in (saccades, fixations):
        _assert_params_documented(frame.attrs["params_used"])

    _, s2, f2 = gm.mark_eye_data_with_full_angle_from_angles(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s")
    for frame in (s2, f2):
        _assert_params_documented(frame.attrs["params_used"])
        assert "схема расчёта" in frame.attrs["params_used"]


def test_source_marks_user_values_as_user() -> None:
    t, x, y = _arrays(_record())
    r = ef.ksv(t, x, y, v0_deg_s=25.0)
    assert r.params_used["порог скорости фиксации"]["источник"] == rp.FROM_USER


# ─────────────────────────────────────────────────────── предупреждения
def test_warns_when_user_threshold_far_from_model(caplog: pytest.LogCaptureFixture) -> None:
    t, x, y = _arrays(_record())
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        r = ef.ksv(t, x, y, v0_deg_s=0.2)
    assert r.v0_deg_s_used == 0.2
    assert any("отличие в" in rec.message for rec in caplog.records)


def test_no_warning_when_threshold_close_to_model(caplog: pytest.LogCaptureFixture) -> None:
    t, x, y = _arrays(_record())
    model_v0 = ef.ksv(t, x, y).v0_deg_s_model
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        ef.ksv(t, x, y, v0_deg_s=model_v0 * 1.2)
    assert not [rec for rec in caplog.records if "отличие в" in rec.message]


def test_warns_about_dropped_samples(caplog: pytest.LogCaptureFixture) -> None:
    """Молчаливая потеря наблюдений — отдельный дефект; теперь она видна."""
    df = _record()
    df.loc[10:19, "gaze_yaw"] = np.nan
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        gm.mark_eye_data_with_sigmoid(df, **_SIGMOID_KW)
    assert any("отброшено неконечных отсчётов" in rec.message for rec in caplog.records)


def test_warns_about_duplicated_timestamps(caplog: pytest.LogCaptureFixture) -> None:
    df = _record()
    t = df["T_s"].to_numpy(dtype=float).copy()
    t[100] = t[99]
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        frd.angular_velocity_at_rate(t, df["gaze_yaw"].to_numpy(dtype=float),
                               df["gaze_pitch"].to_numpy(dtype=float))
    assert any("дубли" in rec.message for rec in caplog.records)


# ────────────────────────────────────────────────── формат и восстановление
def test_describe_restores_report_text() -> None:
    t, x, y = _arrays(_record())
    r = ef.ksv(t, x, y)
    text = rp.describe(r.params_used, title="коэффициент стабилизации взора")
    assert "коэффициент стабилизации взора" in text
    assert "порог скорости фиксации" in text
    assert rp.FROM_MODEL in text


def test_param_formats_units_and_source() -> None:
    p = rp.Param(19.2137, "град/с", rp.FROM_MODEL, "пиковая скорость")
    text = rp.format_report("проверка", {"порог": p})
    assert "19.2137 град/с" in text
    assert "модель: пиковая скорость" in text
    assert rp.Param(float("nan")).format_value() == "нет данных"
    assert rp.Param(True).format_value() == "да"


def test_enable_report_logging_is_removable() -> None:
    logger = logging.getLogger(rp.LOGGER_NAME)
    before = len(logger.handlers)
    handler = rp.enable_report_logging(stream=io.StringIO())
    assert len(logger.handlers) == before + 1
    logger.removeHandler(handler)
    assert len(logger.handlers) == before


def test_smoothing_window_reported_in_both_units() -> None:
    """M-3: окно в отсчётах даёт разное время на разных частотах — это должно быть видно."""
    windows = {}
    for fs in (90.0, 250.0):
        df = synthetic.gaze_2d(fs_hz=fs, duration_s=10.0, n_steps=10, seed=17)
        _, saccades, _ = gm.mark_eye_data_with_sigmoid(df, **_SIGMOID_KW)
        windows[fs] = saccades.attrs["params_used"]["гауссово сглаживание"]["значение"]
    assert "16.7 мс" in windows[90.0]
    assert "6.0 мс" in windows[250.0]
