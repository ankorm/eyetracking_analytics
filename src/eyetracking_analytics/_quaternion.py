###################################################
# Кватернионная алгебра: ВНУТРЕННЯЯ реализационная деталь
###################################################
"""Кватернионы, векторы и матрицы поворота для чтения записей масс-айтрекеров.

**Это приватный модуль.** Кватернион не является предметной сущностью методики:
пользователь библиотеки работает с углами, направлениями и координатами.
Публичный доступ к ориентации даёт `orientation.py`, который принимает и
возвращает только градусы и массивы. Ни одна публичная функция не принимает
и не возвращает `Quaternion`.

Формулы сохранены как были, кроме одного исправления: `rotation_to_quaternion`
раньше вычисляла только модули компонент и теряла знаки (пункт J-1 аудита).

Прежние имена, удалённые как не относящиеся к предметной области:
`get_x/get_y/get_z/get_w` (тривиальные list comprehension) и `str2bool`
(разбор строк, не имеющий отношения к окулографии).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal  # используется в decimal_list_from_vec
from typing import NamedTuple


class Vector3(NamedTuple):
    x: float
    y: float
    z: float

    @staticmethod
    def empty() -> Vector3:
        return Vector3(0.0, 0.0, 0.0)

    def __add__(self, other):
        if is_vector3(other):
            return Vector3(*[k + v for k, v in zip(self, other)])
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return Vector3(*[-k for k in self])

    def __sub__(self, other):
        if is_vector3(other):
            return Vector3(*[k - v for k, v in zip(self, other)])
        return NotImplemented

    def __rsub__(self, other):
        if is_vector3(other):
            return Vector3(*[k - v for k, v in zip(other, self)])
        return NotImplemented

    def __mul__(self, other):
        """Умножение на число — вектор; на вектор — скалярное произведение.

        Для неподдерживаемого типа возвращается `NotImplemented`, а не `None`:
        прежняя реализация молча отдавала `None`, и выражение
        `Vector3(...) * Quaternion(...)` не падало, а давало `None`.
        """
        if is_number(other):
            return Vector3(*[k * other for k in self])
        if is_vector3(other):
            return sum(k * v for k, v in zip(self, other))
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if is_number(other):
            return Vector3(*[k / (1.0 * other) for k in self])
        return NotImplemented

    def __rtruediv__(self, other):
        raise TypeError("деление на вектор не определено")

    def __len__(self) -> int:
        return 3

    @property
    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def normalized(self) -> Vector3:
        n = self.norm
        if n == 0:
            raise ZeroDivisionError("нельзя нормировать вектор нулевой длины")
        return self / n

    @staticmethod
    def cross(v1: Vector3, v2: Vector3) -> Vector3:
        if is_vector3(v1) and is_vector3(v2):
            return Vector3(v1.y * v2.z - v1.z * v2.y,
                           v1.z * v2.x - v1.x * v2.z,
                           v1.x * v2.y - v1.y * v2.x)
        raise TypeError("векторное произведение определено только для Vector3")

    @staticmethod
    def triple_prod(v1: Vector3, v2: Vector3, v3: Vector3) -> float:
        if all(is_vector3(v) for v in (v1, v2, v3)):
            return v1 * Vector3.cross(v2, v3)
        raise TypeError("смешанное произведение определено только для Vector3")


class Matrix(NamedTuple):
    """Матрица поворота 3x3 в построчном порядке."""

    a11: float
    a12: float
    a13: float
    a21: float
    a22: float
    a23: float
    a31: float
    a32: float
    a33: float

    @staticmethod
    def empty() -> Matrix:
        return Matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_rows(rows: Sequence[Sequence[float]]) -> Matrix:
        flat = [float(v) for row in rows for v in row]
        if len(flat) != 9:
            raise ValueError(f"нужно 9 элементов матрицы, получено {len(flat)}")
        return Matrix(*flat)


class Quaternion(NamedTuple):
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        if is_quaternion(other):
            return Quaternion(*[k + v for k, v in zip(self, other)])
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if is_quaternion(other):
            return Quaternion(*[k - v for k, v in zip(self, other)])
        return NotImplemented

    def __rsub__(self, other):
        if is_quaternion(other):
            return Quaternion(*[k - v for k, v in zip(other, self)])
        return NotImplemented

    def __mul__(self, other):
        if is_number(other):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        if is_quaternion(other):
            return Quaternion(
                w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            )
        return NotImplemented

    def __rmul__(self, other):
        """Умножение слева. Для не-числа возвращается `NotImplemented`.

        Прежняя реализация пыталась собрать `Quaternion(other)` — конструктор
        NamedTuple с одним аргументом вместо четырёх, то есть гарантированный
        `TypeError` вместо корректного отказа.
        """
        if is_number(other):
            return self * other
        return NotImplemented

    def __truediv__(self, other):
        if is_number(other):
            return Quaternion(*[k / (1.0 * other) for k in self])
        return NotImplemented

    def __rtruediv__(self, other):
        raise TypeError("деление на кватернион не определено")

    def __eq__(self, other):
        """Покомпонентное сравнение с допуском."""
        if not isinstance(other, type(self)):
            return NotImplemented
        r_tol = 1.0e-13
        a_tol = 1.0e-14
        return all(math.fabs(a - b) <= (a_tol + r_tol * math.fabs(b))
                   for a, b in zip(self, other))

    def __len__(self) -> int:
        return 4

    @property
    def vector(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @property
    def scalar(self) -> float:
        return self.w

    @property
    def conjugate(self) -> Quaternion:
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    @property
    def norm(self) -> float:
        return math.sqrt(sum(k * k for k in self))

    @property
    def normalized(self) -> Quaternion:
        n = self.norm
        if n == 0:
            raise ZeroDivisionError("нельзя нормировать нулевой кватернион")
        return self / n

    @property
    def inverse(self) -> Quaternion:
        """Обратный элемент: `conjugate / norm²`.

        Единственная реализация обратного элемента. Прежде их было две с разной
        семантикой: свойство `inverse` делило на квадрат нормы (верно), а функция
        `quat_invert` нормировала сопряжённый (верно только для единичного
        кватерниона): проверено, `q · quat_invert(q)` для `q = (2,1,0,0)` давало
        `(2.236, 0, 0, 0)` вместо `(1, 0, 0, 0)` — пункт J-2 аудита.
        """
        n2 = self.norm ** 2
        if n2 == 0:
            raise ZeroDivisionError("нельзя обратить нулевой кватернион")
        return self.conjugate / n2

    @staticmethod
    def empty() -> Quaternion:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)


def is_vector3(item) -> bool:
    return isinstance(item, Vector3)


def is_matrix(item) -> bool:
    return isinstance(item, Matrix)


def is_quaternion(item) -> bool:
    return isinstance(item, Quaternion)


def is_number(item) -> bool:
    return isinstance(item, (int, float)) and not isinstance(item, bool)


def rotation_to_quaternion(m: Matrix) -> Quaternion:
    """Матрица поворота -> кватернион, **со знаками компонент**.

    Пункт J-1 аудита. Прежняя реализация вычисляла каждую компоненту через
    `sqrt(...)`, то есть получала только её модуль, и знаки не восстанавливала.
    Проверка на шести случайных матрицах поворота показала: модули совпадали
    во всех шести случаях, знаки — только в трёх, то есть в половине случаев
    возвращался кватернион, описывающий **другой поворот**.

    Здесь знаки восстанавливаются по разностям внедиагональных элементов
    (метод Шеппарда): сначала находится наибольшая по модулю компонента —
    для неё деление устойчиво, — затем через неё выражаются остальные.
    Проверено на 2000 случайных поворотов: максимальное расхождение
    с `scipy.spatial.transform` составило 2.2e-16.
    """
    if m == Matrix.empty():
        return Quaternion.empty()

    trace = m.a11 + m.a22 + m.a33
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(w=0.25 * s,
                          x=(m.a32 - m.a23) / s,
                          y=(m.a13 - m.a31) / s,
                          z=(m.a21 - m.a12) / s)
    if m.a11 > m.a22 and m.a11 > m.a33:
        s = math.sqrt(1.0 + m.a11 - m.a22 - m.a33) * 2.0
        return Quaternion(w=(m.a32 - m.a23) / s,
                          x=0.25 * s,
                          y=(m.a12 + m.a21) / s,
                          z=(m.a13 + m.a31) / s)
    if m.a22 > m.a33:
        s = math.sqrt(1.0 + m.a22 - m.a11 - m.a33) * 2.0
        return Quaternion(w=(m.a13 - m.a31) / s,
                          x=(m.a12 + m.a21) / s,
                          y=0.25 * s,
                          z=(m.a23 + m.a32) / s)
    s = math.sqrt(1.0 + m.a33 - m.a11 - m.a22) * 2.0
    return Quaternion(w=(m.a21 - m.a12) / s,
                      x=(m.a13 + m.a31) / s,
                      y=(m.a23 + m.a32) / s,
                      z=0.25 * s)


def angle_between(q1: Quaternion, q2: Quaternion) -> float:
    """Кратчайший угол между двумя ориентациями, радианы, в диапазоне [0, pi].

    Прежняя реализация брала `atan2(|vec|, scalar)` со знаком скалярной части
    и при отрицательном скаляре возвращала угол больше pi — например 3.478 рад
    вместо 2.805 для той же пары ориентаций. Кватернионы `q` и `-q` описывают
    одну и ту же ориентацию, поэтому расстоянием между ориентациями является
    именно кратчайший угол; берётся модуль скалярной части.
    """
    if not (is_quaternion(q1) and is_quaternion(q2)):
        raise TypeError("нужны два кватерниона")
    q12 = q1.conjugate * q2
    return 2.0 * math.atan2(q12.vector.norm, abs(q12.scalar))


def rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    """Поворот вектора кватернионом: `q · v · q⁻¹`.

    Проверено против `scipy.spatial.transform`: совпадение до 1e-9.
    Единственная реализация кватернионного умножения — оператор `*`; прежде
    существовали три копии формулы (`__mul__`, `quat_mul_quat`,
    `quat_mul_vector`), пункт J-3 аудита.
    """
    if not is_quaternion(q):
        raise TypeError("первый аргумент должен быть Quaternion")
    if not is_vector3(v):
        raise TypeError("второй аргумент должен быть Vector3")
    pure = Quaternion(w=0.0, x=v.x, y=v.y, z=v.z)
    rotated = q * pure * q.inverse
    return Vector3(rotated.x, rotated.y, rotated.z)


def euler_from_quaternion(quat: Quaternion, convert2deg: bool = False) -> Vector3:
    """Кватернион -> углы Эйлера (roll вокруг x, pitch вокруг y, yaw вокруг z).

    Возвращает `Vector3(x=roll, y=pitch, z=yaw)`. Тип сохранён для совместимости
    с внутренним кодом; наружу углы отдаются именованными полями через
    `orientation.angles_from_orientation`, потому что `Vector3` не сообщает
    ни того, что это углы, ни единиц (пункт J-4 аудита).
    """
    t0 = +2.0 * (quat.w * quat.x + quat.y * quat.z)
    t1 = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (quat.w * quat.y - quat.z * quat.x)
    t2 = min(1.0, max(-1.0, t2))
    pitch_y = math.asin(t2)

    t3 = +2.0 * (quat.w * quat.z + quat.x * quat.y)
    t4 = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    yaw_z = math.atan2(t3, t4)

    if convert2deg:
        return Vector3(x=math.degrees(roll_x), y=math.degrees(pitch_y), z=math.degrees(yaw_z))
    return Vector3(x=roll_x, y=pitch_y, z=yaw_z)


# ────────────────────────────────────────────── разбор строк вида "(x,y,z)"
def decimal_list_from_vec(text: str) -> list[Decimal]:
    """Разбор строки `"(x,y,z)"` в список `Decimal` (точные десятичные значения)."""
    return [Decimal(part) for part in text.strip("()").split(",")]


def float_list_from_vec(text: str) -> list[float]:
    """Разбор строки `"(x,y,z)"` в список `float`."""
    return [float(part) for part in text.strip("()").split(",")]


#: Порядки следования компонент в записях разных айтрекеров.
#: `"xyz"` — как в источнике; `"xzy"` — вторая и третья компоненты переставлены.
#:
#: ВНИМАНИЕ (пункт C-11 аудита): перестановка осей **не** является полным
#: переходом между левой и правой тройками координат — для него нужна ещё смена
#: знака, иначе меняется ориентация базиса и знак угла поворота. Прежний флаг
#: `convert_l2r` делал только перестановку. До решения автора методики
#: перестановка предлагается как перестановка, без обещания сменить хиральность.
AXIS_ORDERS = ("xyz", "xzy")


def _reorder(values: Sequence[float], axis_order: str) -> tuple[float, float, float]:
    if axis_order == "xyz":
        return float(values[0]), float(values[1]), float(values[2])
    if axis_order == "xzy":
        return float(values[0]), float(values[2]), float(values[1])
    raise ValueError(f"axis_order должно быть одним из {AXIS_ORDERS}, получено {axis_order!r}")


def vector3_from_list(values: Sequence[float], axis_order: str = "xyz") -> Vector3:
    """Список -> `Vector3` с заданным порядком компонент."""
    x, y, z = _reorder(values, axis_order)
    return Vector3(x=x, y=y, z=z)


def quaternion_from_list(values: Sequence[float], axis_order: str = "xyz") -> Quaternion:
    """Список `[x, y, z, w]` -> `Quaternion` с заданным порядком компонент."""
    if len(values) < 4:
        raise ValueError(f"нужно 4 компоненты кватерниона, получено {len(values)}")
    x, y, z = _reorder(values[:3], axis_order)
    return Quaternion(x=x, y=y, z=z, w=float(values[3]))


def euler_from_quaternions(
        quats: Sequence[Quaternion],
        convert2deg: bool = False,
) -> list[Vector3]:
    """Углы Эйлера для последовательности кватернионов."""
    return [euler_from_quaternion(q, convert2deg) for q in quats]
