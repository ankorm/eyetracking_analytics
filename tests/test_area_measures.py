"""Меры разброса взора по площади: эллипс рассеяния и выпуклая оболочка.

BCEA перенесена из работы Kruchinina, Polikanova, 2025
(DOI https://doi.org/10.17323/1813-8918-2025-3-473-488), где так считалась
площадь области прицеливания при стрельбе биатлонистов. Оболочка добавлена рядом: у мер разный смысл -- эллипс равной
плотности против границы фактически посещённых точек.

Обе применяются **только ко взору**: площадь имеет смысл как проекция
на неподвижное пространство, а для движения глаза относительно головы такой
проекции нет.
"""
from __future__ import annotations

import numpy as np
import pytest

from eyetracking_analytics import bcea_deg2, convex_hull_area_deg2
from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mk

from . import synthetic

_KW = dict(col_horizontal="gaze_yaw", col_vertical="gaze_pitch")


def test_matches_the_original_biathlon_formula() -> None:
    """S = 2*ln(10)*sqrt(Dx*Dy - cov^2)*pi -- та же формула, без перевода единиц."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, 500)
    y = 0.4 * x + rng.normal(0.0, 1.0, 500)

    cov = np.cov(x, y, ddof=1)
    original = 2 * np.log(10) * np.sqrt(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2) * np.pi

    # Не бит-в-бит: та же формула, записанная в другом порядке, на разных
    # машинах даёт последние биты по-разному. Здесь проверяется тождество
    # формул, а не воспроизводимость арифметики.
    assert bcea_deg2(x, y) == pytest.approx(original, rel=1e-12)


def test_scales_as_an_area() -> None:
    """Растяжение облака вдвое по обеим осям учетверяет площадь."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 400)
    y = rng.normal(0.0, 1.0, 400)
    assert bcea_deg2(2 * x, 2 * y) == pytest.approx(4 * bcea_deg2(x, y))


def test_sees_shape_where_radius_does_not() -> None:
    """Главное отличие от `spread_*`: площадь чувствительна к вытянутости.

    Два облака с одинаковым СКО расстояния от центра, но разной формой дают
    разную площадь -- вытянутое заметает меньше.
    """
    rng = np.random.default_rng(2)
    n = 4000
    round_x, round_y = rng.normal(0, 1.0, n), rng.normal(0, 1.0, n)
    long_x, long_y = rng.normal(0, np.sqrt(1.9), n), rng.normal(0, np.sqrt(0.1), n)

    r_round = np.sqrt(np.mean((round_x - round_x.mean()) ** 2 + (round_y - round_y.mean()) ** 2))
    r_long = np.sqrt(np.mean((long_x - long_x.mean()) ** 2 + (long_y - long_y.mean()) ** 2))
    assert r_round == pytest.approx(r_long, rel=0.05)          # радиус почти одинаков
    assert bcea_deg2(long_x, long_y) < 0.7 * bcea_deg2(round_x, round_y)


def test_probability_level() -> None:
    """`probability` задаёт долю накрываемых точек; 0.90 отвечает k = ln(10)."""
    rng = np.random.default_rng(3)
    x, y = rng.normal(0, 1.0, 300), rng.normal(0, 1.0, 300)
    ratio = bcea_deg2(x, y, probability=0.95) / bcea_deg2(x, y, probability=0.90)
    assert ratio == pytest.approx(np.log(20) / np.log(10))


def test_degenerate_and_short_inputs() -> None:
    assert np.isnan(bcea_deg2([1.0, 2.0], [1.0, 2.0]))          # по двум точкам площади нет
    assert bcea_deg2([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 0.0   # точки на прямой
    assert np.isnan(bcea_deg2([1.0, np.nan, 3.0], [1.0, 2.0, np.nan]))


def test_hull_area_of_a_known_polygon() -> None:
    """Площадь оболочки единичного квадрата -- единица; внутренняя точка не влияет."""
    x = [0.0, 1.0, 1.0, 0.0, 0.5]
    y = [0.0, 0.0, 1.0, 1.0, 0.5]
    assert convex_hull_area_deg2(x, y) == pytest.approx(1.0)


def test_hull_scales_as_an_area() -> None:
    rng = np.random.default_rng(4)
    x, y = rng.normal(0, 1.0, 300), rng.normal(0, 1.0, 300)
    assert convex_hull_area_deg2(3 * x, 3 * y) == pytest.approx(9 * convex_hull_area_deg2(x, y))


def test_hull_and_bcea_differ_on_an_outlier() -> None:
    """Смысл двух мер разный, и это видно на одиночном выбросе.

    Эллипс равной плотности выброс сдвигает умеренно, границу фактически
    посещённых точек -- сильно. Расхождение мер само по себе информативно.
    """
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 1.0, 300), rng.normal(0, 1.0, 300)
    xo, yo = np.append(x, 20.0), np.append(y, 20.0)

    bcea_growth = bcea_deg2(xo, yo) / bcea_deg2(x, y)
    hull_growth = convex_hull_area_deg2(xo, yo) / convex_hull_area_deg2(x, y)
    assert hull_growth > bcea_growth


def test_hull_degenerate_and_short_inputs() -> None:
    assert np.isnan(convex_hull_area_deg2([0.0, 1.0], [0.0, 1.0]))
    assert convex_hull_area_deg2([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 0.0
    assert np.isnan(convex_hull_area_deg2([1.0, np.nan, 3.0], [1.0, 2.0, np.nan]))


def test_column_present_for_gaze_only() -> None:
    """Колонки есть в разметке взора по направлению и больше нигде."""
    df = synthetic.gaze_2d(fs_hz=250.0, duration_s=6.0, n_steps=8, seed=1)

    area_cols = ("bcea_deg2", "hull_area_deg2")

    gaze = mk.mark_gaze_events_direction(df, **_KW, geometry="angles")
    assert all(c in gaze.fixations.columns for c in area_cols)

    scalar = mk.mark_gaze_events_1d(df.assign(a=df["gaze_yaw"]), col_gaze_angle="a")
    assert not any(c in scalar.fixations.columns for c in area_cols)

    _, _, eye_rest = gm.mark_eye_rest_from_angles(df, **_KW)
    assert not any(c in eye_rest.columns for c in area_cols)

    _, _, eye_int = gm.mark_eye_data_with_full_angle_from_angles(df, **_KW)
    assert not any(c in eye_int.columns for c in area_cols)


def test_column_agrees_with_the_standalone_function() -> None:
    """Колонка считается по той же ковариации, что и публичная функция."""
    df = synthetic.gaze_2d(fs_hz=250.0, duration_s=10.0, n_steps=6, seed=3)
    res = mk.mark_gaze_events_direction(
        df, **_KW, geometry="angles", drop_short_fixations=False)
    f = res.fixations
    assert len(f) > 0

    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)
    t = df["T_s"].to_numpy(dtype=float)
    for _, row in f.iterrows():
        m = (t >= row["t_start_s"]) & (t <= row["t_end_s"])
        assert row["bcea_deg2"] == pytest.approx(bcea_deg2(x[m], y[m]), rel=1e-9)
        assert row["hull_area_deg2"] == pytest.approx(
            convex_hull_area_deg2(x[m], y[m]), rel=1e-9)
