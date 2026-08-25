"""Вход: ориентация головы и глаза приводятся к единому направлению взора.

Система координат зафиксирована: правая, X вправо, Y вверх, Z вперёд,
нейтральное направление [0, 0, 1]. Кватернион -- поворот из локальной системы
в родительскую, порядок компонент (q_x, q_y, q_z, q_w).

Проверки сверены со `scipy.spatial.transform.Rotation`: он задаёт кватернионы
в том же порядке (скаляр последний) и с той же семантикой поворота вектора,
поэтому годится как независимый эталон.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from eyetracking_analytics import direction as dr
from eyetracking_analytics import report as rp

FORWARD = np.array([0.0, 0.0, 1.0])
RIGHT = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


def _quat_frame(rot: Rotation) -> pd.DataFrame:
    """Кватернионы scipy в порядке (q_x, q_y, q_z, q_w) -- как на входе библиотеки."""
    q = np.atleast_2d(rot.as_quat())
    return pd.DataFrame(q, columns=list(dr.DEFAULT_QUATERNION_COLUMNS))


# ───────────────────────────────────────────────── система координат
def test_identity_looks_forward() -> None:
    """Единичная ориентация -- взгляд вперёд, [0, 0, 1]."""
    res = dr.gaze_direction(head=_quat_frame(Rotation.identity()))
    assert np.allclose(res.direction[0], FORWARD)
    assert res.is_gaze


def test_axis_meaning_is_fixed() -> None:
    """Поворот вокруг Y ведёт вправо, вокруг X -- вниз: это конвенция модуля."""
    yaw = dr.gaze_direction(head=_quat_frame(Rotation.from_euler("y", 90, degrees=True)))
    pitch = dr.gaze_direction(head=_quat_frame(Rotation.from_euler("x", 90, degrees=True)))

    assert np.allclose(yaw.direction[0], RIGHT, atol=1e-12)
    assert np.allclose(pitch.direction[0], -UP, atol=1e-12)


def test_matches_scipy_on_random_rotations() -> None:
    """Направление совпадает с поворотом вперёд-вектора средствами scipy."""
    rng = np.random.default_rng(20260825)
    rot = Rotation.from_quat(rng.normal(size=(200, 4)))
    res = dr.gaze_direction(head=_quat_frame(rot))
    assert np.allclose(res.direction, rot.apply(FORWARD), atol=1e-12)


# ─────────────────────────────────────────── представления на входе
def test_vector_representation_normalizes() -> None:
    raw = pd.DataFrame({"x": [0.0, 3.0], "y": [0.0, 0.0], "z": [5.0, 4.0]})
    res = dr.gaze_direction(head=raw, head_representation="vector")

    assert np.allclose(res.direction[0], FORWARD)
    assert np.allclose(np.linalg.norm(res.direction, axis=1), 1.0)
    assert np.allclose(res.direction[1], [0.6, 0.0, 0.8])


def test_rotation_vector_matches_quaternion() -> None:
    """Вектор поворота и кватернион одного поворота дают одно направление."""
    rng = np.random.default_rng(7)
    rot = Rotation.from_rotvec(rng.normal(size=(50, 3)))
    by_rotvec = dr.gaze_direction(
        head=rot.as_rotvec(), head_representation="rotation_vector")
    by_quat = dr.gaze_direction(head=_quat_frame(rot))

    assert np.allclose(by_rotvec.direction, by_quat.direction, atol=1e-12)


def test_rotation_vector_in_degrees() -> None:
    rv_deg = np.array([[0.0, 90.0, 0.0]])
    res = dr.gaze_direction(head=rv_deg, head_representation="rotation_vector",
                            rotation_vector_in_degrees=True)
    assert np.allclose(res.direction[0], RIGHT, atol=1e-12)


def test_relative_quaternion_needs_initial() -> None:
    frame = _quat_frame(Rotation.identity())
    with pytest.raises(ValueError, match="initial"):
        dr.gaze_direction(head=frame, head_representation="quaternion_relative")


def test_relative_quaternion_composes_with_initial() -> None:
    """Поворот относительно исходного состояния приводится к неподвижному пространству."""
    initial = Rotation.from_euler("y", 30, degrees=True)
    relative = Rotation.from_euler("y", [[0.0], [60.0]], degrees=True)

    res = dr.gaze_direction(
        head=_quat_frame(relative), head_representation="quaternion_relative",
        head_initial=initial.as_quat())

    expected = (initial * relative).apply(FORWARD)
    assert np.allclose(res.direction, expected, atol=1e-12)


# ────────────────────────────────────────── композиция головы и глаза
def test_head_and_eye_compose() -> None:
    """Взор -- поворот глаза, затем поворот головы; сверено со scipy."""
    rng = np.random.default_rng(11)
    head = Rotation.from_quat(rng.normal(size=(30, 4)))
    eye = Rotation.from_quat(rng.normal(size=(30, 4)))

    res = dr.gaze_direction(head=_quat_frame(head), eye=_quat_frame(eye))

    assert np.allclose(res.direction, (head * eye).apply(FORWARD), atol=1e-12)
    assert np.allclose(res.head_direction, head.apply(FORWARD), atol=1e-12)
    assert np.allclose(res.eye_direction, eye.apply(FORWARD), atol=1e-12)


def test_eye_alone_is_not_gaze(caplog: pytest.LogCaptureFixture) -> None:
    """Без головы это движение глаза: признак `is_gaze` ложен, в журнале предупреждение."""
    eye = _quat_frame(Rotation.from_euler("y", 10, degrees=True))
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        res = dr.gaze_direction(eye=eye)

    assert not res.is_gaze
    assert any("не взор" in rec.message for rec in caplog.records)


def test_head_vector_with_eye_warns_about_torsion(caplog: pytest.LogCaptureFixture) -> None:
    """Вектор не задаёт торсию: композиция приближённая, и это сказано."""
    head = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [1.0]})
    eye = _quat_frame(Rotation.from_euler("y", 10, degrees=True))
    with caplog.at_level(logging.WARNING, logger=rp.LOGGER_NAME):
        res = dr.gaze_direction(head=head, head_representation="vector", eye=eye)

    assert any("торсия" in rec.message for rec in caplog.records)
    assert np.allclose(res.direction[0], Rotation.from_euler("y", 10, degrees=True).apply(FORWARD))


def test_lengths_must_match() -> None:
    head = _quat_frame(Rotation.from_euler("y", [[0.0], [10.0]], degrees=True))
    eye = _quat_frame(Rotation.identity())
    with pytest.raises(ValueError, match="одной длины"):
        dr.gaze_direction(head=head, eye=eye)


# ─────────────────────────────────────────────── контракт входа и выхода
def test_column_order_is_fixed_and_named() -> None:
    """Имена столбцов задаются, порядок -- нет: он всегда x, y, z."""
    frame = pd.DataFrame({"vx": [0.0], "vy": [0.0], "vz": [1.0]})
    res = dr.gaze_direction(head=frame, head_representation="vector",
                            head_cols=("vx", "vy", "vz"))
    assert np.allclose(res.direction[0], FORWARD)

    with pytest.raises(KeyError, match="нет столбцов"):
        dr.gaze_direction(head=frame, head_representation="vector")


def test_unknown_representation_rejected() -> None:
    with pytest.raises(ValueError, match="representation"):
        dr.gaze_direction(head=_quat_frame(Rotation.identity()),
                          head_representation="euler")


def test_nothing_given_rejected() -> None:
    with pytest.raises(ValueError, match="данные головы"):
        dr.gaze_direction()


def test_invalid_rows_give_nan() -> None:
    frame = pd.DataFrame({"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 0.0]})
    res = dr.gaze_direction(head=frame, head_representation="vector")

    assert np.allclose(res.direction[0], FORWARD)
    assert np.isnan(res.direction[1]).all()
    assert res.params_used["неполных отсчётов"]["значение"] == 1


def test_to_frame_orders_columns_for_markup() -> None:
    """Столбцы -- компоненты в единой системе: horizontal (X), vertical (Y), forward (Z)."""
    res = dr.gaze_direction(head=_quat_frame(Rotation.from_euler("y", 90, degrees=True)))
    frame = res.to_frame([0.0])

    assert list(frame.columns) == ["T_s", "gaze_horizontal", "gaze_vertical", "gaze_forward"]
    assert frame["gaze_horizontal"].iloc[0] == pytest.approx(1.0)
    assert frame["gaze_forward"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_to_frame_checks_time_length() -> None:
    res = dr.gaze_direction(head=_quat_frame(Rotation.identity()))
    with pytest.raises(ValueError, match="не совпадает"):
        res.to_frame([0.0, 0.1])


def test_markup_reads_the_result() -> None:
    """Сквозная проверка: результат подаётся в разметку и даёт ожидаемый азимут."""
    from eyetracking_analytics import mark_gaze_events_direction

    fs, n = 250.0, 500
    t = np.arange(n) / fs
    # Взгляд стоит вправо от нейтрали: поворот вокруг вертикали на 10 град.
    head = _quat_frame(Rotation.from_euler("y", np.full((n, 1), 10.0), degrees=True))
    res = dr.gaze_direction(head=head)
    frame = res.to_frame(t)

    markup = mark_gaze_events_direction(
        frame, col_time="T_s", geometry="vector",
        col_horizontal="gaze_horizontal", col_vertical="gaze_vertical", col_forward="gaze_forward")

    assert markup.eccentricity_deg.max() < 1e-9        # нейтраль -- само это направление
    assert np.allclose(markup.azimuth_deg[0], 0.0, atol=1e-9)


def test_head_turn_compensated_by_eye_keeps_gaze_still() -> None:
    """Тот самый пример методики: голова идёт, глаз компенсирует -- взор стоит.

    Фиксация при этом есть, а покоя глаза нет (см. `Sci_base-rus.md`). Проверка
    сквозная: композиция ориентаций подаётся в разметку и должна дать
    неподвижный взор, тогда как одна голова даёт уход на её же угол.
    """
    from eyetracking_analytics import mark_gaze_events_direction

    fs, n, turn_deg = 250.0, 500, 20.0
    t = np.arange(n) / fs
    angle = np.linspace(0.0, turn_deg, n).reshape(-1, 1)
    head = Rotation.from_euler("y", angle, degrees=True)
    eye = Rotation.from_euler("y", -angle, degrees=True)

    gaze = dr.gaze_direction(head=_quat_frame(head), eye=_quat_frame(eye))
    markup = mark_gaze_events_direction(
        gaze.to_frame(t), col_time="T_s", geometry="vector",
        col_horizontal="gaze_horizontal", col_vertical="gaze_vertical", col_forward="gaze_forward")

    assert markup.eccentricity_deg.max() < 1e-9, "взор должен стоять на месте"
    assert markup.gaze_shifts.empty, "переводов взора здесь быть не должно"

    # Без учёта глаза та же запись выглядит как уход взора на угол поворота головы.
    head_only = dr.gaze_direction(head=_quat_frame(head))
    markup_head = mark_gaze_events_direction(
        head_only.to_frame(t), col_time="T_s", geometry="vector",
        col_horizontal="gaze_horizontal", col_vertical="gaze_vertical", col_forward="gaze_forward")
    assert markup_head.eccentricity_deg.max() == pytest.approx(turn_deg / 2, abs=0.1)
