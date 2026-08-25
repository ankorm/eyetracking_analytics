"""Конвейер на реальной записи: то, чего синтетика не проверяет.

У синтетики известен правильный ответ, но нет того, чем реальная запись богата:
неравномерного шага, пропусков, выбросов выше физиологического предела скорости.
Здесь проверяется, что на таком материале код доходит до конца и что величины
остаются осмысленными, а не что они равны заранее известным числам.

Запись обезличена, см. `real_data`.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from eyetracking_analytics import constants as const
from eyetracking_analytics import direction as dr
from eyetracking_analytics import markup as mk
from eyetracking_analytics import models as mdl
from eyetracking_analytics import report as rp
from eyetracking_analytics import sampling as smp

from . import real_data

MARKUP_COLUMNS = dict(col_time="T_s", geometry="vector", col_horizontal="gaze_horizontal",
                      col_vertical="gaze_vertical", col_forward="gaze_forward")


def _gaze(with_head: bool = True):
    data = real_data.vr_head_eye()
    kw = {"eye": real_data.quaternions(data, "eye")}
    if with_head:
        kw["head"] = real_data.quaternions(data, "head")
    return data, dr.gaze_direction(**kw)


# ───────────────────────────────────────────────── свойства самой записи
def test_recording_is_uneven_and_gappy() -> None:
    """Реальная запись не равномерна -- ради этого она здесь и лежит."""
    data = real_data.vr_head_eye()
    info = smp.estimate_sampling_rate(data["T_s"].to_numpy(dtype=float))

    assert info.fs_hz == pytest.approx(90.0, abs=0.5)
    assert info.span_s == pytest.approx(20.0, abs=0.5)
    assert info.jitter_rel > 0.05, "шаг записи должен гулять: на синтетике этого нет"
    assert info.n_gaps > 0, "в записи должны быть пропуски"
    smp.require_seconds(info)          # ось времени в секундах, иначе исключение


def test_anonymised_columns_only() -> None:
    """В фикстуре только время и две ориентации: ничего, что указывает на человека."""
    data = real_data.vr_head_eye()
    expected = {"T_s"} | {f"{what}_q_{axis}" for what in ("head", "eye") for axis in "xyzw"}

    assert set(data.columns) == expected
    assert data["T_s"].iloc[0] == 0.0, "время идёт от нуля: дата записи не сохраняется"
    assert data.notna().all().all()


# ──────────────────────────────────────── взор как композиция головы и глаза
def test_pipeline_runs_end_to_end() -> None:
    """Кватернионы прибора -> направление взора -> разметка."""
    data, gaze = _gaze()
    assert gaze.is_gaze
    assert np.allclose(np.linalg.norm(gaze.direction, axis=1), 1.0)

    res = mk.mark_gaze_events_direction(gaze.to_frame(data["T_s"]), **MARKUP_COLUMNS)

    assert len(res.gaze_shifts) > 10, "за 20 с активной задачи переводов взора должно быть много"
    assert (res.gaze_shifts["duration_s"] > 0).all()
    assert (res.gaze_shifts["amplitude_deg"] > 0).all()
    assert np.isfinite(res.eccentricity_deg).all()
    assert res.fs_hz == pytest.approx(90.0, abs=0.5)


def test_thresholds_come_from_the_actual_rate() -> None:
    """Пороги выведены моделью по фактической частоте записи, а не заданы."""
    data, gaze = _gaze()
    res = mk.mark_gaze_events_direction(gaze.to_frame(data["T_s"]), **MARKUP_COLUMNS)
    model = mdl.min_saccade_minima_by_sampling(res.fs_hz, n_samples=const.N_SAMPLES_GAZE_SHIFT)

    p = res.params_used
    assert p["порог пика скорости"]["источник"] == "модель"
    assert p["порог пика скорости"]["значение"] == pytest.approx(model.v_peak_deg_s)
    assert p["минимальная амплитуда"]["значение"] == pytest.approx(model.a_min_deg)


def test_summary_is_plausible() -> None:
    """Интегральные характеристики попадают в разумный диапазон."""
    data, gaze = _gaze()
    res = mk.mark_gaze_events_direction(gaze.to_frame(data["T_s"]), **MARKUP_COLUMNS)
    s = res.summary()

    assert s.t_total_s == pytest.approx(20.0, abs=0.5)
    assert 0.5 < s.event_rate_hz < 10.0, "переводов взора у человека единицы в секунду"
    assert 0.0 < s.intervals_time_fraction < 1.0
    assert s.median_interval_duration_s >= const.MIN_FIXATION_DURATION_S


# ─────────────────────────── без головы это движение глаза, а не взор
def test_eye_alone_is_marked_as_eye_movement(caplog: pytest.LogCaptureFixture) -> None:
    """Та же запись без учёта головы: `is_gaze` ложен и в журнале предупреждение."""
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        data, eye_only = _gaze(with_head=False)

    assert not eye_only.is_gaze
    assert any("не взор" in rec.message for rec in caplog.records)

    res = mk.mark_gaze_events_direction(eye_only.to_frame(data["T_s"]), **MARKUP_COLUMNS)
    assert len(res.gaze_shifts) > 0


def test_head_changes_the_result() -> None:
    """Учёт головы меняет разметку: иначе различение взора и глаза было бы пустым."""
    data, gaze = _gaze()
    _, eye_only = _gaze(with_head=False)

    with_head = mk.mark_gaze_events_direction(gaze.to_frame(data["T_s"]), **MARKUP_COLUMNS)
    without = mk.mark_gaze_events_direction(eye_only.to_frame(data["T_s"]), **MARKUP_COLUMNS)

    assert len(with_head.gaze_shifts) != len(without.gaze_shifts), (
        "на записи с движущейся головой разметка взора и разметка глаза "
        "не могут совпасть")


# ──────────────────────────────────────────── артефакты реальной записи
def test_recording_contains_artifacts_above_the_physiological_limit() -> None:
    """Скорость местами выше предела: ради таких мест и заведён `MAX_EYE_VELOCITY_DEG_S`."""
    data, gaze = _gaze()
    res = mk.mark_gaze_events_direction(gaze.to_frame(data["T_s"]), **MARKUP_COLUMNS)

    assert res.speed_deg_s.max() > const.MAX_EYE_VELOCITY_DEG_S, (
        "в реальной записи есть выбросы -- на них и рассчитан физиологический предел")
    # Предел при этой частоте: максимальное правдоподобное приращение за отсчёт.
    assert mdl.max_increment_deg(res.fs_hz) == pytest.approx(
        const.MAX_EYE_VELOCITY_DEG_S / res.fs_hz)
