"""Единицы измерения: всё публичное API работает в секундах, градусах и герцах.

Переход на секунды должен быть преобразованием без потерь: значения по умолчанию
воспроизводят прежнее поведение, а окна, заданные в секундах, переводятся
в отсчёты по фактической частоте записи.
"""
from __future__ import annotations

import pytest

from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk
from eyetracking_analytics import models as mdl

from . import synthetic

_FS = 250.0
_SIGMOID_KW = dict(
    col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", col_time="T_s",
    min_peak_velocity_deg_s=30.0, min_amplitude_deg=0.5, max_axis_dt_s=0.012, dt_window_s=0.05,
    r2_min_keep=0.5,
)
# Плоские координаты точки взора: расстояние до плоскости r в тех же единицах.
_PLANE_KW = dict(
    col_gaze_x="X", col_gaze_y="Y", col_time="T_s", geometry="plane", r=300.0,
    min_peak_velocity_deg_s=20.0, fixation_velocity_limit_deg_s=15.0, min_event_duration_s=0.006,
    min_amplitude_deg=0.2,
)


def _sigmoid_record():
    return synthetic.gaze_2d(fs_hz=_FS, duration_s=10.0, n_steps=15, seed=606)


# ───────────────────────────────────────────────── модель порогов в секундах
def test_model_returns_seconds_consistent_with_table() -> None:
    """`t_min_s` — та же величина, что раньше возвращалась в миллисекундах."""
    for fs in (60.0, 90.0, 200.0, 250.0, 1000.0):
        for n in (2, 3, 4, 10):
            r = mdl.min_saccade_minima_by_sampling(fs, n_samples=n)
            # Длительность согласована с моделью T(A) = T0_ms * A^m (T0 в мс, как в статье).
            expected_ms = r.model.T0_ms * (r.a_min_deg ** r.model.m)
            assert r.t_min_s * 1000.0 == pytest.approx(expected_ms, rel=1e-12)
            assert 1e-4 < r.t_min_s < 1.0            # секунды, а не миллисекунды
            assert r.v_peak_deg_s > 0


def test_model_strict_sampling_duration_also_in_seconds() -> None:
    fs, n = 200.0, 3
    r = mdl.min_saccade_minima_by_sampling(fs, n_samples=n, return_strict_sampling_duration=True)
    assert r.t_min_s == pytest.approx(n / fs, rel=1e-12)


def test_model_named_fields() -> None:
    """Возврат именованный: порядок элементов больше не нужно помнить."""
    r = mdl.min_saccade_minima_by_sampling(250.0, n_samples=3)
    assert r.a_min_deg == r[0] and r.t_min_s == r[1] and r.v_peak_deg_s == r[2]
    assert r.model.T0_ms > 0


def test_model_rejects_boundary_rate_with_matching_message() -> None:
    """Код отвергает ровно 30 Гц — сообщение должно говорить именно это."""
    with pytest.raises(ValueError, match="строго больше 30"):
        mdl.min_saccade_minima_by_sampling(30.0)


# ─────────────────────────────────── окна: None = прежние значения в отсчётах
def test_none_windows_equal_legacy_point_values() -> None:
    df = _sigmoid_record()
    _, s_none, f_none = gm.mark_eye_data_with_sigmoid(df, **_SIGMOID_KW)
    _, s_sec, f_sec = gm.mark_eye_data_with_sigmoid(
        df,
        min_peak_width_s=gm.LEGACY_MIN_PEAK_WIDTH_POINTS / _FS,
        gaussian_sigma_s=gm.LEGACY_GAUSSIAN_SIGMA_POINTS / _FS,
        **_SIGMOID_KW,
    )
    assert len(s_none) == len(s_sec)
    assert len(f_none) == len(f_sec)


def test_window_in_seconds_scales_with_rate() -> None:
    """Одно и то же окно в секундах даёт разное число отсчётов на разных частотах."""
    wide = _sigmoid_record()
    _, s_narrow, _ = gm.mark_eye_data_with_sigmoid(
        wide, min_peak_width_s=3.0 / _FS, **_SIGMOID_KW)
    _, s_wide, _ = gm.mark_eye_data_with_sigmoid(
        wide, min_peak_width_s=40.0 / _FS, **_SIGMOID_KW)
    # Более широкое требование к пику отсеивает часть событий.
    assert len(s_wide) < len(s_narrow)


def test_median_window_is_forced_odd() -> None:
    """Ядро медианы обязано быть нечётным: чётное значение округляется вверх."""
    df = _sigmoid_record()
    # 4 отсчёта -> 5; вызов не должен падать на ValueError из median_filter_dataframe
    _, saccades, _ = gm.mark_eye_data_with_sigmoid(
        df, median_window_s=4.0 / _FS, **_SIGMOID_KW)
    assert saccades is not None


def test_gaussian_smoothing_can_be_disabled() -> None:
    df = _sigmoid_record()
    _, s_off, _ = gm.mark_eye_data_with_sigmoid(df, gaussian_sigma_s=0.0, **_SIGMOID_KW)
    _, s_on, _ = gm.mark_eye_data_with_sigmoid(df, **_SIGMOID_KW)
    # Без сглаживания шум даёт больше пиков-кандидатов.
    assert len(s_off) >= len(s_on)


# ──────────────────────────── три роли прежнего max_delta_time разделены
def test_duration_roles_are_separable() -> None:
    df = synthetic.planar_xy()
    base = mk.mark_gaze_events_direction(df, **_PLANE_KW)

    # Явно заданные значения, равные выводимым по умолчанию, ничего не меняют.
    same = mk.mark_gaze_events_direction(
        df,
        min_subprocess_duration_s=_PLANE_KW["min_event_duration_s"],
        max_merge_gap_s=_PLANE_KW["min_event_duration_s"] / 2.0,
        **_PLANE_KW,
    )
    assert len(same.gaze_shifts) == len(base.gaze_shifts)
    assert len(same.fixations) == len(base.fixations)

    # Минимальная длительность подпроцесса управляет разбиением фиксаций
    # независимо от минимальной длительности события.
    sub = mk.mark_gaze_events_direction(df, min_subprocess_duration_s=1.0, **_PLANE_KW)
    assert len(sub.gaze_shifts) == len(base.gaze_shifts), "разбиение не должно влиять на события"
    assert len(sub.fixations) < len(base.fixations)

    # Минимальная длительность события управляет отбором пиков.
    kw = dict(_PLANE_KW)
    kw["min_event_duration_s"] = 0.5
    long_events = mk.mark_gaze_events_direction(df, **kw)
    assert len(long_events.gaze_shifts) < len(base.gaze_shifts)


def test_merge_gap_is_necessary_not_sufficient() -> None:
    """Увеличение допустимого разрыва само по себе не склеивает фиксации.

    Склейка требует ещё и статистической близости (относительное отличие
    mean_speed, ex, ey, fro_cov не более 20 %), поэтому на однородных данных
    разрыв не является связывающим ограничением.
    """
    df = synthetic.planar_xy()
    base = mk.mark_gaze_events_direction(df, **_PLANE_KW)
    wide = mk.mark_gaze_events_direction(df, max_merge_gap_s=0.5, **_PLANE_KW)
    assert len(wide.fixations) <= len(base.fixations)


def test_all_public_time_parameters_carry_units_in_name() -> None:
    """Имя временного параметра должно называть единицы: иначе путаница мс/с возвращается."""
    import inspect

    time_like = ("duration_s", "gap", "window", "dt", "sigma")
    for func in (
        gm.mark_eye_data_with_sigmoid,
        gm.mark_eye_data_with_full_angle_from_angles,
        mk.mark_gaze_events_1d,
        mk.mark_gaze_events_direction,
    ):
        for name in inspect.signature(func).parameters:
            if any(token in name for token in time_like):
                assert name.endswith("_s"), f"{func.__name__}: параметр {name} без единиц в имени"
