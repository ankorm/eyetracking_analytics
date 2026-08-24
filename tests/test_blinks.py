"""Моргания: интервалы недостоверных данных.

Проверяется то, ради чего функции написаны: разрыв в потоке достоверных отсчётов
опознаётся как моргание, потеря одного кадра — нет, а границы расширяются до
точки, с которой сигнал начал портиться.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eyetracking_analytics import (
    blink_mask,
    describe_blinks,
    find_blinks_by_gaps,
    find_blinks_by_threshold,
    refine_blink_bounds,
)

FS_HZ = 90.0


def _record_with_blinks(blinks=((2.0, 2.18), (5.5, 5.62))) -> pd.DataFrame:
    """Запись с отметкой качества и «площадью зрачка», падающей на морганиях."""
    t = np.arange(0.0, 8.0, 1.0 / FS_HZ)
    good = np.ones(t.size, dtype=bool)
    pupil = np.full(t.size, 20.0)
    for t0, t1 in blinks:
        inside = (t >= t0) & (t <= t1)
        good[inside] = False
        pupil[inside] = 2.0
        falling = (t >= t0 - 0.05) & (t < t0)
        pupil[falling] = np.linspace(20.0, 3.0, int(falling.sum()))
        rising = (t > t1) & (t <= t1 + 0.05)
        pupil[rising] = np.linspace(3.0, 20.0, int(rising.sum()))
    return pd.DataFrame({"T_s": t, "good": good, "pupil": pupil})


def test_gaps_in_quality_are_blinks() -> None:
    df = _record_with_blinks()
    blinks = find_blinks_by_gaps(df, col_quality="good")
    assert len(blinks) == 2
    assert blinks["t_start_s"].iloc[0] < 2.0 <= blinks["t_end_s"].iloc[0]
    assert blinks["t_start_s"].iloc[1] < 5.5 <= blinks["t_end_s"].iloc[1]
    assert (blinks["duration_s"] > 0.1).all()


def test_single_lost_frame_is_not_a_blink() -> None:
    """Потеря одного кадра даёт разрыв в один шаг записи — это не моргание."""
    df = _record_with_blinks(blinks=())
    df.loc[100, "good"] = False
    assert find_blinks_by_gaps(df, col_quality="good").empty


def test_refine_widens_bounds_to_monotonicity_change() -> None:
    """Сигнал начинает падать раньше, чем прибор теряет глаз."""
    df = _record_with_blinks()
    blinks = find_blinks_by_gaps(df, col_quality="good")
    refined = refine_blink_bounds(df, blinks, col_value="pupil")
    assert (refined["t_start_s"] <= blinks["t_start_s"]).all()
    assert (refined["t_end_s"] >= blinks["t_end_s"]).all()
    assert (refined["duration_s"] > blinks["duration_s"]).all()


def test_threshold_finds_the_same_blinks() -> None:
    """По падению сигнала находятся те же моргания, что по отметке качества."""
    df = _record_with_blinks()
    by_quality = find_blinks_by_gaps(df, col_quality="good")
    by_signal = find_blinks_by_threshold(df, col_value="pupil", threshold=10.0)
    assert len(by_signal) == len(by_quality)
    for (_, a), (_, b) in zip(by_quality.iterrows(), by_signal.iterrows()):
        assert abs(a["t_start_s"] - b["t_start_s"]) < 0.06
        assert abs(a["t_end_s"] - b["t_end_s"]) < 0.06


def test_mask_covers_blink_samples() -> None:
    df = _record_with_blinks()
    blinks = find_blinks_by_gaps(df, col_quality="good")
    mask = blink_mask(df, blinks)
    assert mask.sum() > 0
    # внутри моргания отметка качества False, значит маска обязана его накрыть
    assert not df.loc[mask, "good"].all()
    assert df.loc[~mask, "good"].all()


def test_describe_reports_share_of_record() -> None:
    df = _record_with_blinks()
    blinks = find_blinks_by_gaps(df, col_quality="good")
    report = describe_blinks(blinks, span_s=float(df["T_s"].max()))
    assert "моргания" in str(report).lower() or "найдено" in report
    assert report["доля записи"]["значение"] > 0
