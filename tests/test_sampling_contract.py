"""Контракт частоты дискретизации.

Решения автора методики, которые здесь закреплены:

* отсчёты времени обязательны и заданы в секундах;
* `fs_hz=None` — расчёт на частоте записи, без пересчёта;
* заданная частота — децимация: берутся реальные отсчёты, без осреднения;
* частота выше частоты записи отвергается, допуск — два СКО длины шага;
* порог модели берётся при `min(запрошенная, частота записи)`.
"""
from __future__ import annotations

import numpy as np
import pytest

from eyetracking_analytics import eyetracking_funs as ef
from eyetracking_analytics import for_raw_data as frd
from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk
from eyetracking_analytics import sampling as smp

from . import synthetic


# ───────────────────────────────────────────────────────── оценка частоты
def test_uniform_grid_gives_exact_rate() -> None:
    t = np.arange(1000) / 200.0
    info = smp.estimate_sampling_rate(t)
    assert info.fs_hz == pytest.approx(200.0, rel=1e-12)
    assert info.n_samples == 1000
    assert info.n_duplicated_timestamps == 0
    assert info.n_gaps == 0
    assert info.jitter_rel == pytest.approx(0.0, abs=1e-12)


def test_jitter_is_measured() -> None:
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, jitter=0.05, seed=11)
    info = smp.estimate_sampling_rate(df["T_s"].to_numpy(dtype=float))
    assert info.fs_hz == pytest.approx(200.0, rel=0.02)
    assert 0.02 < info.jitter_rel < 0.10
    # Дрожание даёт запас по запрашиваемой частоте, но небольшой.
    assert info.fs_max_requestable_hz > info.fs_hz
    assert info.fs_max_requestable_hz < 1.5 * info.fs_hz


def test_duplicated_timestamps_are_counted_not_crashed() -> None:
    t = np.arange(100) / 100.0
    t[50] = t[49]                       # дубль метки времени
    info = smp.estimate_sampling_rate(t)
    assert info.n_duplicated_timestamps == 1
    assert np.isfinite(info.fs_hz)


def test_gaps_are_counted() -> None:
    t = np.concatenate([np.arange(50) / 100.0, 0.5 + 1.0 + np.arange(50) / 100.0])
    info = smp.estimate_sampling_rate(t)
    assert info.n_gaps == 1
    assert info.max_gap_s > 1.0


def test_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="минимум 2"):
        smp.estimate_sampling_rate(np.array([0.0]))


def test_rejects_non_finite_time() -> None:
    with pytest.raises(ValueError, match="NaN"):
        smp.estimate_sampling_rate(np.array([0.0, 0.1, np.nan]))


# ─────────────────────────────────────────────── единицы времени: секунды
def test_milliseconds_axis_is_rejected() -> None:
    """Раньше ось в миллисекундах давала 0 саккад без единого предупреждения."""
    t_ms = np.arange(4000) * 5.0                       # 200 Гц, но в миллисекундах
    x = np.zeros(4000)
    info = smp.estimate_sampling_rate(t_ms)
    assert info.looks_like_milliseconds
    with pytest.raises(ValueError, match="миллисекунд"):
        smp.require_seconds(info)
    with pytest.raises(ValueError, match="миллисекунд"):
        frd.angular_velocity_at_rate(t_ms, x, x)


def test_milliseconds_axis_rejected_in_markup() -> None:
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=20.0, n_steps=10, seed=7)
    df_ms = df.copy()
    df_ms["T_s"] = df_ms["T_s"] * 1000.0
    for call in (
        lambda d: gm.mark_eye_data_with_full_angle_from_angles(
            d, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", col_time="T_s"),
        lambda d: mk.mark_gaze_events_direction(
            d, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", geometry="angles", col_time="T_s"),
        lambda d: mk.mark_gaze_events_1d(d, col_gaze_angle="gaze_yaw", col_time="T_s"),
        lambda d: gm.mark_eye_data_with_sigmoid(
            d, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", col_time="T_s"),
    ):
        with pytest.raises(ValueError, match="миллисекунд"):
            call(df_ms)


def test_slow_but_legitimate_recording_is_not_rejected() -> None:
    """Условие узкое: реальная медленная запись не должна ложно отвергаться."""
    t = np.arange(600) / 2.0            # 2 Гц, 300 с
    info = smp.estimate_sampling_rate(t)
    assert not info.looks_like_milliseconds
    smp.require_seconds(info)


# ────────────────────────────────────────────────── запрос частоты
def test_rate_above_recording_is_rejected() -> None:
    t = np.arange(2000) / 200.0
    x = np.zeros(2000)
    info = smp.estimate_sampling_rate(t)
    with pytest.raises(ValueError, match="недопустимое значение"):
        frd.angular_velocity_at_rate(t, x, x, fs_hz=info.fs_hz * 2)


def test_requesting_recording_rate_is_a_no_op() -> None:
    """Приведение к собственной частоте не должно менять результат."""
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=10, seed=21)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    native = frd.angular_velocity_at_rate(t, x, y)
    same = frd.angular_velocity_at_rate(t, x, y, fs_hz=native.fs_hz_recording)
    assert not native.resampled and not same.resampled
    assert np.array_equal(native.v_abs, same.v_abs, equal_nan=True)


def test_decimation_takes_real_samples() -> None:
    """Децимация выбирает измерения, а не интерполирует между ними."""
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=10, seed=33)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)

    t_dec, (x_dec,), idx = smp.decimate_to_rate(t, 50.0, x)
    assert np.array_equal(t_dec, t[idx])
    assert np.array_equal(x_dec, x[idx])
    assert np.isin(x_dec, x).all()
    info_dec = smp.estimate_sampling_rate(t_dec)
    assert info_dec.fs_hz == pytest.approx(50.0, rel=0.05)


def test_decimation_lowers_measured_peak_velocity() -> None:
    """Физическое следствие децимации: пиковые скорости снижаются."""
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=20, seed=44)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    native = frd.angular_velocity_at_rate(t, x, y)
    dec = frd.angular_velocity_at_rate(t, x, y, fs_hz=50.0)
    assert dec.resampled
    assert np.nanmax(dec.v_abs) < np.nanmax(native.v_abs)


# ──────────────────────────────────────────────────── единицы скорости
def test_velocity_components_are_per_second() -> None:
    """B-2: раньше v_x/v_y возвращались в «единицах за отсчёт», а v_abs — в «/с»."""
    fs = 100.0
    t = np.arange(1000) / fs
    x = 1.0 * t                       # ровно 1 ед./с
    y = np.zeros_like(t)
    res = frd.angular_velocity_at_rate(t, x, y)
    assert np.allclose(res.v_x, 1.0, rtol=1e-9)
    assert np.allclose(res.v_y, 0.0, atol=1e-12)
    assert np.allclose(res.v_abs, 1.0, rtol=1e-9)
    assert res.t_s.size == res.v_abs.size == t.size - 1


# ──────────────────────────────────────────────────────────── KSV
def test_ksv_does_not_depend_on_requested_rate_when_equal() -> None:
    """B-1: раньше ksv масштабировался как fs_факт/fs_запрош."""
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=10, seed=55)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    a = ef.ksv(t, x, y, v0_deg_s=10.0)
    b = ef.ksv(t, x, y, v0_deg_s=10.0, fs_hz=a.fs_hz_recording)
    assert a.ksv == pytest.approx(b.ksv, rel=1e-12)
    assert 0.0 <= a.ksv <= 1.0


def test_ksv_reports_its_conditions() -> None:
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=10, seed=66)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    from_model = ef.ksv(t, x, y)
    assert from_model.v0_source == "модель"
    assert from_model.v0_deg_s_used == pytest.approx(from_model.v0_deg_s_model)

    from_user = ef.ksv(t, x, y, v0_deg_s=123.0)
    assert from_user.v0_source == "пользователь"
    assert from_user.v0_deg_s_used == 123.0
    # Модельное значение считается всё равно — чтобы расхождение было видно.
    assert np.isfinite(from_user.v0_deg_s_model)
    assert from_user.t_stab_s <= from_user.t_total_s


def test_ksv_model_threshold_follows_effective_rate() -> None:
    """Порог модели зависит от частоты — это методика, а не дефект."""
    df = synthetic.gaze_2d(fs_hz=200.0, duration_s=10.0, n_steps=10, seed=77)
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    native = ef.ksv(t, x, y)
    decimated = ef.ksv(t, x, y, fs_hz=100.0)
    assert decimated.fs_hz_used == pytest.approx(100.0, rel=0.05)
    # Чем ниже частота, тем крупнее минимально детектируемая саккада и выше порог.
    assert decimated.v0_deg_s_used > native.v0_deg_s_used
