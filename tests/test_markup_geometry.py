"""Пункт M-1: эксцентриситет вместо полярного угла и полная угловая скорость.

Прежняя планарная схема считала `alpha = atan2(dy, dx)` — направление смещения,
а не его величину. Здесь закреплены три свойства правильной схемы:

* радиальное движение детектируется (прежняя схема его не видела);
* тангенциальное движение детектируется (буквальное восстановление
  унаследованной схемы «производная эксцентриситета» его бы не увидело);
* прохождение траектории вблизи нейтральной точки не порождает ложного события.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyetracking_analytics import constants as const
from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk

from . import synthetic

_FS = 250.0
_KW = dict(min_peak_velocity_deg_s=30.0, fixation_velocity_limit_deg_s=15.0,
           min_event_duration_s=0.012, min_amplitude_deg=0.5)


def _directions(theta_deg: np.ndarray, phi_deg: np.ndarray) -> pd.DataFrame:
    """Направления с заданными эксцентриситетом и азимутом -> yaw/pitch и плоскость."""
    t = np.arange(theta_deg.size, dtype=float) / _FS
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    ux = np.cos(th)
    uy = np.sin(th) * np.cos(ph)
    uz = np.sin(th) * np.sin(ph)
    pitch = np.degrees(np.arcsin(np.clip(uz, -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(uy, ux))
    r = 12.0
    return pd.DataFrame({
        "T_s": t, "yaw": yaw, "pitch": pitch,
        "X": r * uy / ux, "Y": r * uz / ux,
    })


def _steps(times_s: np.ndarray, amps: np.ndarray, n: int, k: float = 150.0) -> np.ndarray:
    t = np.arange(n, dtype=float) / _FS
    out = np.zeros(n)
    for ts, amp in zip(times_s, amps):
        out += amp / (1.0 + np.exp(np.clip(-k * (t - ts), -700, 700)))
    return out


# ────────────────────────────────────────── слепых зон быть не должно
def test_radial_movement_is_detected() -> None:
    """Азимут постоянен, меняется только эксцентриситет."""
    n = int(_FS * 10)
    theta = _steps(np.linspace(1, 9, 8), np.full(8, 6.0), n)
    df = _directions(theta, np.zeros(n))
    res = mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y",
                                        geometry="plane", r=12.0, **_KW)
    assert np.nanmax(res.speed_deg_s) > 100.0
    assert len(res.gaze_shifts) >= 6, "радиальные движения должны детектироваться"


def test_tangential_movement_is_detected() -> None:
    """Эксцентриситет постоянен, меняется только азимут."""
    n = int(_FS * 10)
    phi = _steps(np.linspace(1, 9, 8), np.full(8, 40.0), n)
    df = _directions(np.full(n, 12.0), phi)
    res = mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y",
                                        geometry="plane", r=12.0, **_KW)
    assert np.nanmax(res.speed_deg_s) > 100.0
    assert len(res.gaze_shifts) >= 6, "тангенциальные движения должны детектироваться"
    # Эксцентриситет при этом почти не меняется — событие видно только по
    # полной угловой скорости, а не по производной эксцентриситета.
    assert np.ptp(res.eccentricity_deg) < 0.5


def test_no_false_event_when_passing_near_neutral() -> None:
    """Медленное монотонное движение через нейтральную точку.

    В прежней схеме давало скачок угла с 0.4° до 179.6° и пик 4043 град/с.
    """
    n = int(_FS * 4)
    df = pd.DataFrame({
        "T_s": np.arange(n, dtype=float) / _FS,
        "X": np.linspace(-3.0, 3.0, n),
        "Y": np.full(n, 0.02),
    })
    res = mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y",
                                        geometry="plane", r=12.0, **_KW)
    assert np.nanmax(res.speed_deg_s) < 50.0
    assert res.eccentricity_deg.min() >= 0.0
    assert len(res.gaze_shifts) == 0


# ──────────────────────────── плоская и сферическая схемы — одна методика
def test_plane_matches_angles_on_same_directions() -> None:
    df = synthetic.gaze_2d(fs_hz=_FS, duration_s=20.0, n_steps=30)
    yaw = np.radians(df["gaze_yaw"].to_numpy(dtype=float))
    pitch = np.radians(df["gaze_pitch"].to_numpy(dtype=float))
    ux = np.cos(pitch) * np.cos(yaw)
    uy = np.cos(pitch) * np.sin(yaw)
    uz = np.sin(pitch)
    r = 12.0
    plane = pd.DataFrame({"T_s": df["T_s"], "X": r * uy / ux, "Y": r * uz / ux})

    by_angles = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles", **_KW)
    by_plane = mk.mark_gaze_events_direction(
        plane, col_horizontal="X", col_vertical="Y", geometry="plane", r=r, **_KW)

    assert np.allclose(by_plane.eccentricity_deg, by_angles.eccentricity_deg, atol=1e-8)
    assert np.allclose(by_plane.speed_deg_s, by_angles.speed_deg_s, atol=1e-8)
    assert len(by_plane.gaze_shifts) == len(by_angles.gaze_shifts)


def test_angles_reproduces_legacy_spherical_scheme() -> None:
    """Сферическая схема была методически верной — её результат сохраняется."""
    df = synthetic.gaze_2d(fs_hz=_FS, duration_s=20.0, n_steps=30)
    # При анализе данных движения глаза свой порог: интервалы там -- покой, а не фиксация.
    eye_kw = {("rest_velocity_limit_deg_s" if k == "fixation_velocity_limit_deg_s" else k): v
              for k, v in _KW.items()}
    legacy_alpha, _, legacy_fix = gm.mark_eye_data_with_full_angle_from_angles(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s", deg=True, **eye_kw)
    # Унаследованная функция не фильтрует ни по длительности, ни по скорости на
    # фиксации, поэтому оба критерия здесь выключены: сравнивается геометрия.
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        drop_short_fixations=False, trim_fixations_by_velocity=False, **_KW)
    assert np.array_equal(res.eccentricity_deg, legacy_alpha)
    assert len(res.fixations) == len(legacy_fix)


# ──────────────────────────────────────────────── обязательность r
def test_plane_requires_distance() -> None:
    df = synthetic.planar_xy()
    with pytest.raises(ValueError, match="нужен параметр r"):
        mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y", geometry="plane", **_KW)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="нужен параметр r"):
            mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y",
                                          geometry="plane", r=bad, **_KW)


def test_vector_geometry_requires_third_column() -> None:
    df = synthetic.planar_xy()
    with pytest.raises(ValueError, match="col_forward"):
        mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y", geometry="vector", **_KW)


def test_unknown_geometry_rejected() -> None:
    df = synthetic.planar_xy()
    with pytest.raises(ValueError, match="geometry"):
        mk.mark_gaze_events_direction(df, col_horizontal="X", col_vertical="Y", geometry="polar", **_KW)


def test_vector_geometry_uses_median_direction_as_neutral() -> None:
    n = int(_FS * 10)
    theta = _steps(np.linspace(1, 9, 8), np.full(8, 5.0), n)
    df = _directions(theta, np.full(n, 30.0))
    th = np.radians(theta)
    ph = np.radians(np.full(n, 30.0))
    frame = pd.DataFrame({
        "T_s": df["T_s"],
        "vx": np.cos(th), "vy": np.sin(th) * np.cos(ph), "vz": np.sin(th) * np.sin(ph),
    })
    res = mk.mark_gaze_events_direction(
        frame, col_horizontal="vx", col_vertical="vy", col_forward="vz", geometry="vector", **_KW)
    assert np.isclose(np.linalg.norm(res.neutral_direction), 1.0, atol=1e-9)
    # Эксцентриситет отсчитывается от медианного направления: движение 0..40 град
    # относительно медианы даёт 20..0..20, то есть минимум около нуля.
    assert res.eccentricity_deg.min() == pytest.approx(0.0, abs=0.5)
    assert res.eccentricity_deg.max() == pytest.approx(20.0, abs=0.5)


# ────────────────────────────── схема результата: A и azimuth разделены
def test_gaze_shift_table_separates_eccentricity_and_azimuth() -> None:
    """Прежде A_s и alpha_s содержали одно и то же значение (пункт I-1)."""
    df = synthetic.gaze_2d(fs_hz=_FS, duration_s=20.0, n_steps=30)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles", **_KW)
    assert {"eccentricity_start_deg", "eccentricity_end_deg",
            "azimuth_start_deg", "azimuth_end_deg", "amplitude_deg"} <= set(res.gaze_shifts.columns)
    if not res.gaze_shifts.empty:
        # Прежде обе величины писались в две пары колонок одним значением.
        assert not np.allclose(res.gaze_shifts["eccentricity_start_deg"],
                               res.gaze_shifts["azimuth_start_deg"])
        # Амплитуда, по которой шёл отбор, теперь сохраняется.
        assert (res.gaze_shifts["amplitude_deg"] >= _KW["min_amplitude_deg"]).all()


def test_gaze_shifts_are_deduplicated() -> None:
    """B-3: два пика внутри одного движения давали две строки с теми же границами."""
    df = synthetic.gaze_2d(fs_hz=_FS, duration_s=20.0, n_steps=30)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        min_peak_velocity_deg_s=5.0, fixation_velocity_limit_deg_s=1.0, min_event_duration_s=0.012, min_amplitude_deg=0.1)
    bounds = list(zip(res.gaze_shifts["i_start"], res.gaze_shifts["i_end"]))
    assert len(bounds) == len(set(bounds)), "границы событий не должны повторяться"


# ─────────────────────────────── изъятая планарная схема объясняет себя
def test_withdrawn_planar_functions_explain_replacement() -> None:
    df = synthetic.planar_xy()
    with pytest.raises(NotImplementedError) as info:
        gm.mark_eye_data_with_full_angle(df, col_horizontal="X", col_vertical="Y", col_time="T_s")
    text = str(info.value)
    assert "mark_gaze_events_direction" in text
    assert "mark_gaze_events_1d" in text
    # Алиас-опечатка «angel» удалён вместе с переходом на устанавливаемый пакет.
    assert not hasattr(gm, "mark_eye_data_with_full_angel")


# ──────────────────────── C-18: порог фиксации выводится из модели
def test_fixation_threshold_derived_from_model() -> None:
    """`v_fix = A_min / max(T_min, 50 мс)` — вместо ручного делителя."""
    from eyetracking_analytics import models as mdl

    for fs in (60.0, 70.0, 90.0, 100.0, 150.0):
        m = mdl.min_saccade_minima_by_sampling(fs, n_samples=const.N_SAMPLES_GAZE_SHIFT)
        assert m.t_fixation_s == pytest.approx(max(m.t_min_s, 0.05))
        assert m.v_fixation_by_model_deg_s == pytest.approx(m.a_min_deg / m.t_fixation_s)
        # В этом диапазоне модель строже физиологического предела и он не срабатывает.
        assert not m.v_fixation_at_floor
        assert m.v_fixation_deg_s == pytest.approx(m.v_fixation_by_model_deg_s)
        assert m.v_fixation_deg_s < m.v_peak_deg_s


def test_fixation_threshold_never_below_physiological_limit() -> None:
    """Чем выше частота, тем точнее оценка, но строже идеальной фиксации порог быть не может."""
    from eyetracking_analytics import models as mdl

    previous = None
    for fs in (60.0, 90.0, 150.0):
        m = mdl.min_saccade_minima_by_sampling(fs, n_samples=const.N_SAMPLES_GAZE_SHIFT)
        # Пока предел не сработал, модельная оценка убывает с ростом частоты.
        if previous is not None:
            assert m.v_fixation_by_model_deg_s < previous
        previous = m.v_fixation_by_model_deg_s
    for fs in (200.0, 250.0, 500.0, 900.0, 1250.0):
        m = mdl.min_saccade_minima_by_sampling(fs, n_samples=const.N_SAMPLES_GAZE_SHIFT)
        # Итоговый порог не опускается ниже физиологического предела.
        assert m.v_fixation_deg_s >= const.IDEAL_FIXATION_VELOCITY_DEG_S
        assert m.v_fixation_at_floor == (
            m.v_fixation_by_model_deg_s < const.IDEAL_FIXATION_VELOCITY_DEG_S)


def test_amplitude_floor_keeps_thresholds_physiological() -> None:
    """C-20: предел применяется к амплитуде, и пороги скорости выводятся из неё.

    Без предела модель при высоких частотах давала пиковую скорость саккады
    0.32 град/с при 1250 Гц — физически невозможное значение.
    """
    from eyetracking_analytics import models as mdl

    floor = const.IDEAL_FIXATION_VELOCITY_DEG_S * const.MIN_FIXATION_DURATION_S
    assert floor == pytest.approx(0.25)

    for fs in (250.0, 500.0, 900.0, 1250.0):
        m = mdl.min_saccade_minima_by_sampling(fs, n_samples=const.N_SAMPLES_GAZE_SHIFT)
        assert m.a_min_at_floor
        assert m.a_min_deg == pytest.approx(floor)
        # Пиковая скорость и длительность выходят на правдоподобные значения.
        assert 20.0 < m.v_peak_deg_s < 40.0
        assert 0.015 < m.t_min_s < 0.025
        assert m.v_fixation_deg_s == pytest.approx(const.IDEAL_FIXATION_VELOCITY_DEG_S)
        # Саккада такой длительности всё ещё разрешается записью.
        assert m.t_min_s * fs >= const.N_SAMPLES_GAZE_SHIFT

    # Инвариант «порог фиксации ниже порога саккады» держится во всём диапазоне.
    for fs in range(31, 2001, 37):
        for n in (2, 3, 4, 10):
            m = mdl.min_saccade_minima_by_sampling(float(fs), n_samples=n)
            assert m.v_fixation_deg_s < m.v_peak_deg_s, (fs, n)


def test_high_rate_markup_works_with_model_thresholds() -> None:
    df = synthetic.gaze_2d(fs_hz=500.0, duration_s=4.0, n_steps=8, seed=500)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    assert res.fs_hz == pytest.approx(500.0, rel=0.01)
    assert res.params_used["порог пика скорости"]["источник"] == "модель"


def test_fixation_threshold_minimum_duration_is_configurable() -> None:
    """`min_fixation_duration_s` действует по-разному в зависимости от того,
    связывает ли предел амплитуды.

    Пока предел не сработал, параметр снижает порог скорости. Когда сработал,
    он поднимает предел амплитуды: `a_floor = v_идеал * min_fixation_duration_s`,
    и порог скорости остаётся равным скорости идеальной фиксации.
    """
    from eyetracking_analytics import models as mdl

    # 90 Гц: предел амплитуды не связывает, параметр влияет на порог скорости.
    base_90 = mdl.min_saccade_minima_by_sampling(90.0, n_samples=const.N_SAMPLES_GAZE_SHIFT)
    longer_90 = mdl.min_saccade_minima_by_sampling(
        90.0, n_samples=const.N_SAMPLES_GAZE_SHIFT, min_fixation_duration_s=0.2)
    assert not base_90.a_min_at_floor and not longer_90.a_min_at_floor
    assert longer_90.t_fixation_s == pytest.approx(0.2)
    assert longer_90.v_fixation_deg_s < base_90.v_fixation_deg_s

    # 250 Гц: предел связывает, параметр поднимает минимальную амплитуду.
    base_250 = mdl.min_saccade_minima_by_sampling(250.0, n_samples=const.N_SAMPLES_GAZE_SHIFT)
    longer_250 = mdl.min_saccade_minima_by_sampling(
        250.0, n_samples=const.N_SAMPLES_GAZE_SHIFT, min_fixation_duration_s=0.2)
    assert base_250.a_min_at_floor and longer_250.a_min_at_floor
    assert longer_250.a_min_deg == pytest.approx(const.IDEAL_FIXATION_VELOCITY_DEG_S * 0.2)
    assert longer_250.a_min_deg > base_250.a_min_deg
    assert longer_250.v_fixation_deg_s == pytest.approx(const.IDEAL_FIXATION_VELOCITY_DEG_S)

    with pytest.raises(ValueError, match="min_fixation_duration_s"):
        mdl.min_saccade_minima_by_sampling(250.0, min_fixation_duration_s=0.0)


def test_markup_thresholds_default_to_model() -> None:
    """Прежние дефолты 15 и 5 были замороженными константами вырожденной ветки."""
    from eyetracking_analytics import models as mdl

    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    m = mdl.min_saccade_minima_by_sampling(90.0, n_samples=const.N_SAMPLES_GAZE_SHIFT)

    used = res.params_used
    assert used["порог пика скорости"]["значение"] == pytest.approx(m.v_peak_deg_s)
    assert used["порог пика скорости"]["источник"] == "модель"
    assert used["порог скорости на фиксации"]["значение"] == pytest.approx(m.v_fixation_deg_s)
    assert used["минимальная амплитуда"]["значение"] == pytest.approx(m.a_min_deg)


def test_user_thresholds_win_and_are_marked_as_user() -> None:
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        fixation_velocity_limit_deg_s=5.0)
    used = res.params_used
    assert used["порог скорости на фиксации"]["значение"] == 5.0
    assert used["порог скорости на фиксации"]["источник"] == "пользователь"


def test_fixation_threshold_above_saccade_threshold_rejected() -> None:
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=10.0, n_steps=10, seed=902)
    with pytest.raises(ValueError, match="строго ниже"):
        mk.mark_gaze_events_direction(
            df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
            min_peak_velocity_deg_s=10.0, fixation_velocity_limit_deg_s=20.0)


# ─────────────────── критерий валидности фиксации по разбросу
def test_spread_threshold_is_higher_at_low_rates() -> None:
    """Допустимый разброс = max(A_min, 5 град): на низких частотах он выше."""
    from eyetracking_analytics import models as mdl

    thresholds = {
        fs: mdl.min_saccade_minima_by_sampling(fs, n_samples=const.N_SAMPLES_GAZE_SHIFT).spread_max_deg
        for fs in (45.0, 60.0, 70.0, 90.0, 250.0, 900.0)
    }
    # Убывает с ростом частоты и выходит на разброс идеальной фовеальной фиксации.
    assert thresholds[45.0] > thresholds[60.0] > thresholds[70.0]
    assert thresholds[45.0] > 20.0
    for fs in (90.0, 250.0, 900.0):
        assert thresholds[fs] == pytest.approx(const.IDEAL_FIXATION_SPREAD_DEG)


def test_spread_criterion_filters_when_enabled() -> None:
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    kept = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        spread_limit_deg=0.5, drop_invalid_spread=True, drop_short_fixations=False,
        trim_fixations_by_velocity=False)
    all_rows = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        spread_limit_deg=0.5, drop_short_fixations=False,
        trim_fixations_by_velocity=False)
    assert len(kept.fixations) < len(all_rows.fixations)
    assert kept.fixations["spread_valid"].all()
    assert (kept.fixations["spread_deg"] <= 0.5).all()


def test_spread_criterion_in_scalar_markup() -> None:
    """В одномерной схеме разброс — СКО одной координаты."""
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    res = mk.mark_gaze_events_1d(
        pd.DataFrame({"T_s": df["T_s"], "angle": df["gaze_yaw"]}), col_gaze_angle="angle")
    assert {"spread_deg", "spread_limit_deg", "spread_valid"} <= set(res.fixations.columns)


# ─────────────────────── минимальная длительность фиксации
def test_short_fixations_are_dropped_by_frequency_dependent_threshold() -> None:
    """C-4 (решение принято): интервал короче минимальной длительности не фиксация.

    Порог зависит от частоты: `max(t_min, 50 мс)` из модели.
    """
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    kept = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    all_rows = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        drop_short_fixations=False,
        trim_fixations_by_velocity=False)

    assert len(kept.fixations) < len(all_rows.fixations)
    assert kept.fixations["duration_valid"].all()
    limit = kept.fixations["duration_s"].min()
    assert limit >= 0.050 - 1e-12
    assert not all_rows.fixations["duration_valid"].all()


def test_duration_threshold_is_higher_at_low_rates() -> None:
    from eyetracking_analytics import models as mdl

    low = mdl.min_saccade_minima_by_sampling(45.0, n_samples=const.N_SAMPLES_GAZE_SHIFT).t_fixation_s
    high = mdl.min_saccade_minima_by_sampling(250.0, n_samples=const.N_SAMPLES_GAZE_SHIFT).t_fixation_s
    assert low > high
    assert high == pytest.approx(const.MIN_FIXATION_DURATION_S)


# ─────────────────────── три режима измерения разброса
def test_spread_measures_are_ordered() -> None:
    """std <= p95 <= peak на одних и тех же интервалах."""
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    f = res.fixations
    assert (f["spread_std_deg"] <= f["spread_p95_deg"] + 1e-9).all()
    assert (f["spread_p95_deg"] <= f["spread_peak_deg"] + 1e-9).all()


def test_spread_measure_mode_selects_column() -> None:
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901)
    for measure in ("std", "p95", "peak"):
        res = mk.mark_gaze_events_direction(
            df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
            spread_measure=measure)
        chosen = res.fixations["spread_deg"].to_numpy()
        source = res.fixations[f"spread_{measure}_deg"].to_numpy()
        assert np.array_equal(chosen, source, equal_nan=True)
        assert f"мера «{measure}»" in res.params_used["допустимый разброс фиксации"]["пояснение"]


def test_unknown_spread_measure_rejected() -> None:
    df = synthetic.gaze_2d(fs_hz=90.0, duration_s=10.0, n_steps=10, seed=902)
    with pytest.raises(ValueError, match="spread_measure"):
        mk.mark_gaze_events_direction(
            df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
            spread_measure="iqr")


# ──────────────────────────────────────── устойчивые интервалы по скорости
def test_stable_intervals_by_velocity_finds_quiet_stretch() -> None:
    """Интервал берётся там, где скорость ниже порога, и нигде больше."""
    fs = 100.0
    t = np.arange(0, 2.0, 1 / fs)
    yaw = np.zeros_like(t)
    # быстрый участок в середине: 40 град/с при пороге 10
    moving = (t >= 0.8) & (t < 1.2)
    yaw[moving] = np.cumsum(np.full(int(moving.sum()), 40.0 / fs))
    yaw[t >= 1.2] = yaw[moving][-1]
    df = pd.DataFrame({"T_s": t, "x": yaw, "y": np.zeros_like(t)})

    intervals = gm.stable_intervals_by_velocity(
        df, col_horizontal="x", col_vertical="y", velocity_limit_deg_s=10.0, min_points=3)

    assert len(intervals) == 2, "спокойные участки до и после движения"
    assert intervals["t_start_s"].iloc[0] == pytest.approx(0.0)
    assert intervals["t_end_s"].iloc[0] < 0.8
    assert intervals["t_start_s"].iloc[1] >= 1.2 - 1 / fs
    assert (intervals["duration_s"] > 0).all()
    assert (intervals["n_points"] >= 3).all()


def test_stable_intervals_by_velocity_respects_min_points() -> None:
    """Короче min_points интервал не возвращается."""
    fs = 100.0
    t = np.arange(0, 1.0, 1 / fs)
    # знакопеременный зигзаг: спокойных серий длиннее двух отсчётов нет
    yaw = np.where(np.arange(t.size) % 2 == 0, 0.0, 1.0)
    df = pd.DataFrame({"T_s": t, "x": yaw, "y": np.zeros_like(t)})
    intervals = gm.stable_intervals_by_velocity(
        df, col_horizontal="x", col_vertical="y", velocity_limit_deg_s=10.0, min_points=3)
    assert intervals.empty
    assert list(intervals.columns)[:3] == ["t_start_s", "t_end_s", "duration_s"]


# ────────── единицы статистик картинной плоскости: всегда градусы
def test_three_parametrizations_give_the_same_fixation_statistics() -> None:
    """Разброс, площади и центр фиксации не должны зависеть от вида входа.

    Прежде пара статистик брала то, что пришло: градусы в угловой схеме,
    миллиметры в планарной, компоненты единичного вектора в векторной. Колонки
    при этом назывались `*_deg`, а порог модели -- градусный, и одна и та же
    запись признавалась невалидной по разбросу в одной схеме и валидной
    в другой. Здесь проверяется, что этого больше нет.
    """
    fs, n = 250.0, 1000
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    yaw, pitch = rng.normal(0, 6.0, n), rng.normal(0, 6.0, n)   # разброс заведомо больше 5 град

    ux = np.cos(np.radians(pitch)) * np.sin(np.radians(yaw))
    uy = np.sin(np.radians(pitch))
    uz = np.cos(np.radians(pitch)) * np.cos(np.radians(yaw))
    r = 12.0
    kw = dict(col_time="T_s", drop_short_fixations=False, homogeneity_split=False,
              trim_fixations_by_velocity=False, min_peak_velocity_deg_s=1e6)

    by_angles = mk.mark_gaze_events_direction(
        pd.DataFrame({"T_s": t, "h": yaw, "v": pitch}),
        col_horizontal="h", col_vertical="v", geometry="angles", **kw)
    by_plane = mk.mark_gaze_events_direction(
        pd.DataFrame({"T_s": t, "h": r * ux / uz, "v": r * uy / uz}),
        col_horizontal="h", col_vertical="v", geometry="plane", r=r, **kw)
    by_vector = mk.mark_gaze_events_direction(
        pd.DataFrame({"T_s": t, "h": ux, "v": uy, "f": uz}),
        col_horizontal="h", col_vertical="v", col_forward="f", geometry="vector", **kw)

    columns = ["mean_horizontal_deg", "mean_vertical_deg", "spread_std_deg", "spread_deg",
               "bcea_deg2", "hull_area_deg2", "cov_fro_norm_deg2", "spread_valid"]
    base = by_angles.fixations
    assert len(base) == 1, "запись задумана как один интервал"
    assert base["spread_deg"].iloc[0] > base["spread_limit_deg"].iloc[0], "разброс должен превышать порог"

    for name, other in (("plane", by_plane), ("vector", by_vector)):
        assert len(other.fixations) == len(base), name
        for col in columns:
            a, b = base[col].iloc[0], other.fixations[col].iloc[0]
            if isinstance(a, (bool, np.bool_)):
                assert a == b, f"{name}: {col}"
            else:
                assert a == pytest.approx(b, rel=1e-6, abs=1e-9), f"{name}: {col}"
