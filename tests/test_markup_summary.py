"""Интегральные характеристики разметки: длительность события, частота, доля времени.

Три величины, которые прежде выводились из таблиц вручную и потому у каждого
получались по-своему: длительность события отдельной колонкой, число переводов
взора в единицу времени и суммарная длительность фиксаций как доля записи.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk

from . import synthetic


def _record() -> pd.DataFrame:
    """Запись с длинными удержаниями: шум ниже порога фиксации, шесть переводов."""
    return synthetic.gaze_2d(fs_hz=250.0, duration_s=20.0, n_steps=6,
                             noise_deg=0.005, seed=11)


def _span_s(df: pd.DataFrame) -> float:
    t = df["T_s"].to_numpy(dtype=float)
    return float(t[-1] - t[0])


# ────────────────────────────────────────────── длительность события в таблице
def test_direction_events_have_duration_column() -> None:
    df = _record()
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")

    assert list(res.gaze_shifts.columns)[:3] == ["t_start_s", "t_end_s", "duration_s"]
    expected = res.gaze_shifts["t_end_s"] - res.gaze_shifts["t_start_s"]
    assert np.array_equal(res.gaze_shifts["duration_s"].to_numpy(), expected.to_numpy())
    assert (res.gaze_shifts["duration_s"] > 0).all()


def test_scalar_and_eye_branch_events_have_duration_column() -> None:
    """Колонка появляется всюду: событие строит один и тот же код."""
    df = _record()
    scalar = mk.mark_gaze_events_1d(
        pd.DataFrame({"T_s": df["T_s"], "angle": df["gaze_yaw"]}), col_gaze_angle="angle")
    _, saccades, _ = gm.mark_eye_rest_from_angles(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch")

    for events in (scalar.gaze_shifts, saccades):
        assert "duration_s" in events.columns
        expected = events["t_end_s"] - events["t_start_s"]
        assert np.array_equal(events["duration_s"].to_numpy(), expected.to_numpy())


def test_duration_column_present_when_no_events() -> None:
    """Пустая таблица обязана иметь ту же схему, иначе разбор результата ломается."""
    t = np.arange(0.0, 2.0, 1 / 250.0)
    df = pd.DataFrame({"T_s": t, "gaze_yaw": np.zeros_like(t), "gaze_pitch": np.zeros_like(t)})
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")

    assert res.gaze_shifts.empty
    assert "duration_s" in res.gaze_shifts.columns


# ─────────────────────────────────────────────────── частота и доля времени
def test_rate_and_fraction_match_tables() -> None:
    df = _record()
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    s = res.summary()

    assert s.t_total_s == pytest.approx(_span_s(df))
    assert s.n_events == len(res.gaze_shifts)
    assert s.event_rate_hz == pytest.approx(len(res.gaze_shifts) / s.t_total_s)
    assert s.intervals_total_duration_s == pytest.approx(
        float(res.fixations["duration_s"].sum()))
    assert s.intervals_time_fraction == pytest.approx(
        s.intervals_total_duration_s / s.t_total_s)
    assert 0.0 < s.intervals_time_fraction <= 1.0
    assert s.median_interval_duration_s > 0


def test_summary_is_logged() -> None:
    """Величина без условий получения не интерпретируется — журнал обязателен."""
    df = _record()
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    s = res.summary()

    assert s.params_used["частота событий"]["единицы"] == "1/с"
    assert s.params_used["доля времени удержания"]["значение"] == pytest.approx(
        s.intervals_time_fraction)


def test_only_valid_excludes_flagged_intervals() -> None:
    """Помеченные невалидными интервалы учитываются или нет — по явному флагу."""
    df = _record()
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles",
        drop_short_fixations=False, min_fixation_duration_s=0.5)

    flagged = int((~res.fixations["duration_valid"]).sum())
    assert flagged > 0, "запись должна давать хотя бы один короткий интервал"

    everything = res.summary()
    valid_only = res.summary(only_valid=True)

    assert everything.n_intervals_invalid == flagged
    assert valid_only.n_intervals_invalid == flagged
    assert valid_only.n_intervals == everything.n_intervals - flagged
    assert valid_only.intervals_total_duration_s < everything.intervals_total_duration_s


def test_eye_branch_summary_from_tables() -> None:
    """При анализе данных движения глаза возвращается тройка, а не объект: таблицы передаются напрямую."""
    df = _record()
    _, saccades, rest = gm.mark_eye_rest_from_angles(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch")
    span = _span_s(df)

    s = mk.describe_markup(events=saccades, intervals=rest, span_s=span)

    assert s.n_events == len(saccades)
    assert s.event_rate_hz == pytest.approx(len(saccades) / span)
    # В таблице покоя нет колонки duration_s: длительность берётся из границ.
    assert "duration_s" not in rest.columns
    assert s.intervals_total_duration_s == pytest.approx(
        float((rest["t_end_s"] - rest["t_start_s"]).sum()))


def test_explicit_tables_override_markup() -> None:
    df = _record()
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")

    half = res.gaze_shifts.iloc[: len(res.gaze_shifts) // 2]
    s = mk.describe_markup(res, events=half)

    assert s.n_events == len(half)
    assert s.n_intervals == len(res.fixations)


def test_requires_markup_or_all_tables() -> None:
    with pytest.raises(ValueError, match="span_s"):
        mk.describe_markup(events=pd.DataFrame(), intervals=pd.DataFrame())


def test_record_without_events() -> None:
    """Неподвижная запись: событий нет, частота нулевая, а не NaN."""
    t = np.arange(0.0, 5.0, 1 / 250.0)
    df = pd.DataFrame({"T_s": t, "gaze_yaw": np.zeros_like(t), "gaze_pitch": np.zeros_like(t)})
    res = mk.mark_gaze_events_direction(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles")
    s = res.summary()

    assert s.n_events == 0
    assert s.event_rate_hz == 0.0
    assert np.isnan(s.mean_event_duration_s)
    assert s.intervals_time_fraction == pytest.approx(1.0, abs=0.01)


def test_zero_span_gives_nan_rates() -> None:
    """Без длительности записи ни частота, ни доля не определены."""
    events = pd.DataFrame({"t_start_s": [0.0], "t_end_s": [0.1], "duration_s": [0.1]})
    intervals = pd.DataFrame({"t_start_s": [0.1], "t_end_s": [0.3], "duration_s": [0.2]})

    s = mk.describe_markup(events=events, intervals=intervals, span_s=0.0)

    assert np.isnan(s.event_rate_hz)
    assert np.isnan(s.intervals_time_fraction)
    assert s.intervals_total_duration_s == pytest.approx(0.2)
