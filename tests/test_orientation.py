"""Пункт J аудита: кватернионы — внутренняя деталь, наружу только углы.

Проверяется два свойства:

* публичный слой `orientation` не принимает и не возвращает кватернионы,
  а его результаты совпадают со `scipy.spatial.transform`;
* исправленные дефекты приватного слоя не возвращаются (J-1 знаки компонент,
  J-2 семантика обратного элемента, J-3 отказ на неподдерживаемых операциях).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from eyetracking_analytics import _quaternion as q
from eyetracking_analytics import eyetracking_funs as ef
from eyetracking_analytics import orientation as ori

_COLS = ["q_x", "q_y", "q_z", "q_w"]


def _rotations(n: int, seed: int) -> tuple[Rotation, pd.DataFrame]:
    rots = Rotation.random(n, random_state=seed)
    return rots, pd.DataFrame(rots.as_quat(), columns=_COLS)


# ────────────────────────────── публичный слой против scipy
def test_angles_match_scipy() -> None:
    rots, df = _rotations(500, 7)
    angles = ori.angles_from_orientation(df)
    expected = rots.as_euler("xyz", degrees=True)
    got = angles[["roll_deg", "pitch_deg", "yaw_deg"]].to_numpy()
    assert np.allclose(got, expected, atol=1e-9)
    assert len(angles) == len(df)


def test_angles_in_radians() -> None:
    _, df = _rotations(20, 8)
    rad = ori.angles_from_orientation(df, to_degrees=False)
    deg = ori.angles_from_orientation(df, to_degrees=True)
    assert set(rad.columns) == {"roll_rad", "pitch_rad", "yaw_rad"}
    assert np.allclose(np.degrees(rad["roll_rad"]), deg["roll_deg"])


def test_rotate_points_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    rots, df = _rotations(300, 9)
    pts = rng.normal(0.0, 3.0, (300, 3))
    assert np.allclose(ori.rotate_points(pts, df), rots.apply(pts), atol=1e-9)


def test_rotate_single_point_and_broadcast() -> None:
    rots, df = _rotations(1, 10)
    point = np.array([1.0, 2.0, 3.0])
    got = ori.rotate_points(point, df)
    assert got.shape == (3,)
    assert np.allclose(got, rots.apply(point)[0], atol=1e-9)

    many = np.tile(point, (5, 1))
    assert np.allclose(ori.rotate_points(many, df), rots.apply(many[0]), atol=1e-9)


def test_angular_distance_matches_scipy() -> None:
    a, da = _rotations(300, 11)
    b, db = _rotations(300, 12)
    expected = np.degrees((a.inv() * b).magnitude())
    assert np.allclose(ori.angular_distance_deg(da, db), expected, atol=1e-9)


def test_angular_distance_requires_equal_length() -> None:
    _, da = _rotations(10, 1)
    _, db = _rotations(11, 2)
    with pytest.raises(ValueError, match="одной длины"):
        ori.angular_distance_deg(da, db)


def test_missing_columns_reported() -> None:
    _, df = _rotations(5, 1)
    with pytest.raises(KeyError, match="нет столбцов"):
        ori.angles_from_orientation(df, cols=("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="4 имени столбца"):
        ori.angles_from_orientation(df, cols=("q_x", "q_y"))


def test_unknown_axis_order_rejected() -> None:
    _, df = _rotations(5, 1)
    with pytest.raises(ValueError, match="axis_order"):
        ori.angles_from_orientation(df, axis_order="zyx")


def test_non_unit_quaternions_are_normalized() -> None:
    """Прибор пишет единичные кватернионы, но масштаб не должен влиять на углы."""
    rots, df = _rotations(50, 13)
    scaled = df * 3.0
    assert np.allclose(
        ori.angles_from_orientation(scaled).to_numpy(),
        ori.angles_from_orientation(df).to_numpy(),
        atol=1e-9,
    )


def test_parse_vector_column() -> None:
    values = ["(1.0,2.0,3.0)", "(-1.5, 0.5, 2)"]
    got = ori.parse_vector_column(values)
    assert np.allclose(got, [[1.0, 2.0, 3.0], [-1.5, 0.5, 2.0]])
    swapped = ori.parse_vector_column(values, axis_order="xzy")
    assert np.allclose(swapped, [[1.0, 3.0, 2.0], [-1.5, 2.0, 0.5]])


# ──────────────────────── J-1: знаки компонент при переходе от матрицы
def test_matrix_to_angles_restores_signs() -> None:
    """Прежняя реализация теряла знаки и в половине случаев давала другой поворот."""
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(200):
        rot = Rotation.random(random_state=int(rng.integers(1e9)))
        got = ori.quaternion_from_matrix_angles_deg(rot.as_matrix()).to_numpy()
        expected = rot.as_euler("xyz", degrees=True)
        worst = max(worst, float(np.max(np.abs(got - expected))))
    assert worst < 1e-9, f"расхождение с scipy: {worst:.3e} град"


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_matrix_to_quaternion_half_turns(axis: str) -> None:
    """Повороты на 180° — вырожденный случай, где след матрицы отрицателен."""
    rot = Rotation.from_euler(axis, 180.0, degrees=True)
    quat = q.rotation_to_quaternion(q.Matrix.from_rows(rot.as_matrix()))
    got = np.array([quat.w, quat.x, quat.y, quat.z])
    ref = rot.as_quat()
    ref = np.array([ref[3], ref[0], ref[1], ref[2]])
    if got[0] < 0:
        got = -got
    if ref[0] < 0:
        ref = -ref
    assert np.allclose(got, ref, atol=1e-9)


def test_matrix_to_quaternion_identity_and_empty() -> None:
    assert q.rotation_to_quaternion(q.Matrix.empty()) == q.Quaternion.empty()
    identity = q.rotation_to_quaternion(q.Matrix.from_rows(np.eye(3)))
    assert identity == q.Quaternion.empty()


# ──────────────────────── J-2 и J-3: семантика операций
def test_single_inverse_semantics() -> None:
    """Прежде было две реализации обратного элемента с разной семантикой."""
    quat = q.Quaternion(w=2.0, x=1.0, y=0.0, z=0.0)
    product = quat * quat.inverse
    assert np.allclose(np.array(product), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    with pytest.raises(ZeroDivisionError):
        _ = q.Quaternion(0.0, 0.0, 0.0, 0.0).inverse


def test_unsupported_operations_raise_not_silently_none() -> None:
    """`Vector3 * Quaternion` прежде молча возвращало None."""
    quat = q.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    with pytest.raises(TypeError):
        q.Vector3(1.0, 2.0, 3.0) * quat
    with pytest.raises(TypeError):
        quat + q.Vector3(1.0, 2.0, 3.0)


def test_rotate_vector_matches_scipy() -> None:
    rot = Rotation.from_euler("xyz", [20.0, 30.0, 40.0], degrees=True)
    xyzw = rot.as_quat()
    quat = q.Quaternion(w=xyzw[3], x=xyzw[0], y=xyzw[1], z=xyzw[2])
    got = np.array(q.rotate_vector(quat, q.Vector3(1.0, 2.0, 3.0)))
    assert np.allclose(got, rot.apply([1.0, 2.0, 3.0]), atol=1e-9)


def test_angle_between_matches_scipy() -> None:
    a = Rotation.random(random_state=31)
    b = Rotation.random(random_state=32)
    qa, qb = a.as_quat(), b.as_quat()
    got = q.angle_between(q.Quaternion(w=qa[3], x=qa[0], y=qa[1], z=qa[2]),
                          q.Quaternion(w=qb[3], x=qb[0], y=qb[1], z=qb[2]))
    assert got == pytest.approx(float((a.inv() * b).magnitude()), abs=1e-9)


# ──────────────────────── инкапсуляция: кватернионов нет в публичном API
def test_quaternion_layer_is_not_public() -> None:
    """Ни одно имя кватернионного слоя не должно быть доступно через фасад."""
    removed = ["Quaternion", "Vector3", "Matrix", "quat_scale", "quat_length",
               "quat_normalize", "quat_invert", "quat_mul_quat", "quat_mul_vector",
               "quat_transform_vector", "euler_from_quaternion", "get_euler_from_quats",
               "get_quat_from_list", "get_vec3_from_list", "isQuaternion", "isVector3",
               "isMatrix", "isNumber", "get_x", "get_y", "get_z", "get_w", "str2bool",
               "declist_from_vec", "floatlist_from_vec"]
    present = [name for name in removed if hasattr(ef, name)]
    assert not present, f"кватернионный слой всё ещё публичен: {present}"


def test_public_orientation_api_has_no_quaternion_types() -> None:
    """В подписях публичных функций не должно быть кватернионных типов."""
    import inspect

    for name in ("angles_from_orientation", "angular_distance_deg", "rotate_points",
                 "quaternion_from_matrix_angles_deg", "parse_vector_column"):
        func = getattr(ori, name)
        text = str(inspect.signature(func))
        for forbidden in ("Quaternion", "Vector3", "Matrix"):
            assert forbidden not in text, f"{name}: тип {forbidden} в подписи"
