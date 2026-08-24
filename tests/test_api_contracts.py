"""Контракты публичного API, которые не должны нарушаться незаметно.

Пункт A-1 аудита: схема результата не может зависеть от того, как пользователь
назвал входные столбцы. Внутри расчёта оси называются канонически (`gaze_x`, `gaze_y`),
переименование в имена пользователя делается на границе.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyetracking_analytics import gaze_mark as gm

from . import synthetic

_MARK_KW = dict(
    col_time="T_s", min_peak_velocity_deg_s=30.0, min_amplitude_deg=0.5, max_axis_dt_s=0.012,
    dt_window_s=0.05, r2_min_keep=0.5,
)


def _record() -> pd.DataFrame:
    return synthetic.gaze_2d(fs_hz=250.0, duration_s=10.0, n_steps=15, seed=606)


def test_axis_column_map() -> None:
    assert gm.axis_column_map("gaze_yaw", "gaze_pitch") == {
        "gaze_x": "gaze_yaw", "gaze_y": "gaze_pitch",
    }


def test_rejects_equal_axis_keys() -> None:
    with pytest.raises(ValueError, match="должны различаться"):
        gm.axis_column_map("a", "a")

    df = _record()
    with pytest.raises(ValueError, match="должны различаться"):
        gm.mark_eye_data_with_sigmoid(df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_yaw", **_MARK_KW)


def test_rejects_unknown_columns_mode() -> None:
    df = _record()
    with pytest.raises(ValueError, match="columns"):
        gm.mark_eye_data_with_sigmoid(
            df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", columns="whatever", **_MARK_KW
        )


def test_canonical_schema_invariant_under_input_rename() -> None:
    """Переименование входных столбцов не меняет каноническую схему результата."""
    df = _record()
    df2 = df.rename(columns={"gaze_yaw": "azimuth [deg]", "gaze_pitch": "elevation [deg]"})

    _, s1, _ = gm.mark_eye_data_with_sigmoid(
        df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", columns="canonical", **_MARK_KW)
    _, s2, _ = gm.mark_eye_data_with_sigmoid(
        df2, col_gaze_x="azimuth [deg]", col_gaze_y="elevation [deg]", columns="canonical", **_MARK_KW)

    assert list(s1.columns) == list(s2.columns)
    assert len(s1) == len(s2)
    for col in s1.columns:
        a, b = s1[col].to_numpy(), s2[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            assert np.array_equal(a, b, equal_nan=True), col
        else:
            assert (a == b).all(), col


def test_source_schema_equals_declared_map() -> None:
    """Схема в режиме `source` ровно равна канонической, пропущенной через отображение."""
    df = _record()
    kw = dict(col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", **_MARK_KW)

    _, canonical, _ = gm.mark_eye_data_with_sigmoid(df, columns="canonical", **kw)
    _, source, _ = gm.mark_eye_data_with_sigmoid(df, columns="source", **kw)

    mapping = gm.axis_column_map("gaze_yaw", "gaze_pitch")
    expected = gm.rename_axis_columns(canonical, mapping, to="source")
    assert list(source.columns) == list(expected.columns)
    assert "gaze_yaw_L" in source.columns and "has_gaze_yaw" in source.columns
    assert "gaze_x_L" not in source.columns


def test_both_modes_numerically_identical() -> None:
    """Режим влияет только на имена: числа обязаны совпасть побитово."""
    df = _record()
    kw = dict(col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", **_MARK_KW)

    a1, canonical, f1 = gm.mark_eye_data_with_sigmoid(df, columns="canonical", **kw)
    a2, source, f2 = gm.mark_eye_data_with_sigmoid(df, columns="source", **kw)

    assert np.array_equal(a1, a2, equal_nan=True)
    assert f1.equals(f2)
    mapping = gm.axis_column_map("gaze_yaw", "gaze_pitch")
    back = gm.rename_axis_columns(source, mapping, to="canonical")
    assert list(back.columns) == list(canonical.columns)
    for col in canonical.columns:
        a, b = canonical[col].to_numpy(), back[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            assert np.array_equal(a, b, equal_nan=True), col
        else:
            assert (a == b).all(), col


def test_event_type_values_are_canonical() -> None:
    """`event_type` — код, а не подпись: значения не зависят от имён колонок."""
    df = _record()
    allowed = {"gaze_x+gaze_y", "gaze_x_only", "gaze_y_only", "unknown"}
    for columns in ("source", "canonical"):
        _, saccades, _ = gm.mark_eye_data_with_sigmoid(
            df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", columns=columns, **_MARK_KW)
        assert set(saccades["event_type"].unique()) <= allowed


def test_shared_columns_never_renamed() -> None:
    df = _record()
    _, saccades, _ = gm.mark_eye_data_with_sigmoid(
        df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", **_MARK_KW)
    for col in gm.SHARED_EVENT_COLUMNS:
        assert col in saccades.columns, col


def test_axis_columns_recorded_in_attrs() -> None:
    df = _record()
    _, saccades, fixations = gm.mark_eye_data_with_sigmoid(
        df, col_gaze_x="gaze_yaw", col_gaze_y="gaze_pitch", **_MARK_KW)
    expected = {"gaze_x": "gaze_yaw", "gaze_y": "gaze_pitch"}
    assert saccades.attrs["axis_columns"] == expected
    assert fixations.attrs["axis_columns"] == expected


def test_rename_axis_columns_handles_prefix_collision() -> None:
    """Имя одной оси может быть префиксом другой — обратное переименование не должно путать."""
    mapping = {"gaze_x": "x", "gaze_y": "x2"}
    df = pd.DataFrame({
        "x_L": [1.0], "x2_L": [2.0], "has_x": [True], "has_x2": [False], "t_event_s": [0.5],
    })
    canonical = gm.rename_axis_columns(df, mapping, to="canonical")
    assert set(canonical.columns) == {"gaze_x_L", "gaze_y_L", "has_gaze_x", "has_gaze_y", "t_event_s"}
    assert canonical["gaze_x_L"].iloc[0] == 1.0
    assert canonical["gaze_y_L"].iloc[0] == 2.0

    back = gm.rename_axis_columns(canonical, mapping, to="source")
    assert list(back.columns) == list(df.columns)
