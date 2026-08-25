###################################################
# Вход: ориентация головы и глаза -> единое направление взора
###################################################
"""Приведение исходных данных прибора к единому направлению взора.

Прибор пишет ориентацию так, как ему удобно: вектором, кватернионом, поворотом
относительно исходного состояния, вектором поворота. Разметке нужен один вид --
направление взора в неподвижном пространстве. Этот модуль делает перевод, чтобы
он не повторялся в каждом проекте по-своему.

**Система координат.** Правая, единая для всего модуля:

* **X** -- вправо;
* **Y** -- вверх;
* **Z** -- вперёд.

Нейтральное направление (взгляд прямо перед собой) -- `[0, 0, 1]`. Порядок
столбцов у всех трёхмерных векторов на входе фиксирован: `x, y, z`.

**Кватернионы.** Порядок компонент фиксирован: `(q_x, q_y, q_z, q_w)`, скаляр
последний. Кватернион означает поворот **из локальной системы в родительскую**:
для головы -- из системы головы в неподвижное пространство, для глаза -- из
системы глаза в систему головы. Иначе одни и те же данные читались бы двумя
способами, и ошибка не проявлялась бы ничем, кроме зеркального результата.
Повороты правые: положительный поворот вокруг оси видно против часовой стрелки,
если смотреть с конца оси в начало. Отсюда, например: поворот на +90° вокруг
**Y** (вверх) переводит взгляд вперёд в направление **вправо**, а поворот
на +90° вокруг **X** (вправо) -- в направление **вниз**.

**Что откуда берётся.** Ориентация головы задаётся в неподвижном пространстве,
ориентация глаза -- относительно головы. Итоговое направление взора --
композиция: сначала применяется поворот глаза, затем поворот головы. Если
данных глаза нет, направление взора определяется по голове; если нет данных
головы, а есть только глаз, получается направление глаза относительно головы,
и это уже не взор (см. `docs/Sci_base-rus.md`).

Готовый результат подаётся в разметку `mark_gaze_events_direction`
с `geometry="vector"`; метод `GazeDirection.to_frame` раскладывает вектор
в нужном ей порядке -- вперёд, вправо, вверх, -- при котором азимут разметки
отсчитывается от направления «вправо» к «вверх».
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .report import COMPUTED, FROM_DATA, FROM_DEFAULT, FROM_USER, Param, log_computation

#: Нейтральное направление: взгляд прямо перед собой.
FORWARD_XYZ: tuple[float, float, float] = (0.0, 0.0, 1.0)

#: Имена столбцов вектора по умолчанию; порядок фиксирован: x, y, z.
DEFAULT_VECTOR_COLUMNS: tuple[str, str, str] = ("x", "y", "z")

#: Имена столбцов кватерниона по умолчанию; порядок фиксирован, скаляр последний.
DEFAULT_QUATERNION_COLUMNS: tuple[str, str, str, str] = ("q_x", "q_y", "q_z", "q_w")

#: Поддерживаемые представления ориентации на входе.
REPRESENTATIONS: tuple[str, ...] = (
    "vector",               # вектор направления
    "quaternion",           # кватернион ориентации
    "quaternion_relative",  # кватернион относительно исходного состояния
    "rotation_vector",      # вектор поворота: направление -- ось, длина -- угол
)


@dataclass(frozen=True)
class GazeDirection:
    """Единое направление взора и то, из чего оно получено.

    Все векторы единичные, `[N, 3]`, в порядке `x, y, z`: X вправо, Y вверх,
    Z вперёд. Строки с недостающими данными -- `NaN`.
    """

    direction: np.ndarray                  # итоговое направление взора
    head_direction: np.ndarray | None      # направление головы, если задана
    eye_direction: np.ndarray | None       # направление глаза в системе головы
    n_samples: int
    is_gaze: bool                          # учтено ли положение головы
    params_used: dict[str, Any] = field(default_factory=dict)

    def to_frame(
            self,
            t_s: Sequence[float] | np.ndarray | None = None,
            *,
            col_time: str = "T_s",
            prefix: str = "gaze",
    ) -> pd.DataFrame:
        """Таблица для `mark_gaze_events_direction(geometry="vector")`.

        Столбцы -- те же компоненты в той же системе координат, что и всюду:
        `horizontal` (X, вправо), `vertical` (Y, вверх), `forward` (Z, вперёд).
        Перестановки нет: разметка работает в этой же системе, поэтому азимут
        у неё значит то же самое -- 0° вправо, +90° вверх.

        Вызов после этого:

            res = mark_gaze_events_direction(
                frame, col_time="T_s", geometry="vector",
                col_horizontal="gaze_horizontal", col_vertical="gaze_vertical",
                col_forward="gaze_forward")
        """
        out: dict[str, Any] = {}
        if t_s is not None:
            t = np.asarray(t_s, dtype=float).ravel()
            if t.size != self.n_samples:
                raise ValueError(
                    f"длина t_s ({t.size}) не совпадает с числом отсчётов ({self.n_samples})")
            out[col_time] = t
        out[f"{prefix}_horizontal"] = self.direction[:, 0]   # X, вправо
        out[f"{prefix}_vertical"] = self.direction[:, 1]     # Y, вверх
        out[f"{prefix}_forward"] = self.direction[:, 2]      # Z, вперёд
        return pd.DataFrame(out)


# ────────────────────────────────────────────── кватернионная арифметика
def _normalize(v: np.ndarray) -> np.ndarray:
    """Нормировка строк; нулевые и неполные строки дают NaN."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(norm > 0, v / norm, np.nan)
    return np.asarray(out, dtype=float)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Произведение кватернионов `[N, 4]` в порядке `(w, x, y, z)`.

    Композиция поворотов: `a * b` означает «сначала b, потом a», то есть b
    задан в системе, которую поворачивает a. Для взора a -- голова, b -- глаз.
    """
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Поворот векторов `[N, 3]` кватернионами `[N, 4]` в порядке `(w, x, y, z)`."""
    w = q[:, 0:1]
    u = q[:, 1:4]
    t = np.cross(u, v) + w * v
    return v + 2.0 * np.cross(u, t)


def _quat_from_rotation_vector(rv: np.ndarray, in_degrees: bool) -> np.ndarray:
    """Вектор поворота -> кватернион: направление -- ось, длина -- угол."""
    rv = np.asarray(rv, dtype=float)
    if in_degrees:
        rv = np.radians(rv)
    theta = np.linalg.norm(rv, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(theta > 0, rv / theta, 0.0)
    half = theta / 2.0
    return np.column_stack([np.cos(half).ravel(), axis * np.sin(half)])


def _quat_shortest_arc(target: np.ndarray) -> np.ndarray:
    """Кратчайший поворот, переводящий `FORWARD_XYZ` в заданное направление.

    Нужен там, где ориентация задана только вектором: вектор не определяет
    поворот вокруг самого себя, поэтому торсия (`roll`) берётся нулевой.
    Для композиции головы с глазом это допущение, и в журнал оно пишется.
    """
    f = np.broadcast_to(np.asarray(FORWARD_XYZ, dtype=float), target.shape)
    dot = np.clip(np.sum(f * target, axis=1), -1.0, 1.0)
    axis = np.cross(f, target)
    q = np.column_stack([1.0 + dot, axis])

    # Противоположное направление: ось поворота не определена, берём любую
    # перпендикулярную -- поворот на 180° вокруг вертикали.
    opposite = dot < -1.0 + 1e-12
    if np.any(opposite):
        q[opposite] = np.array([0.0, 0.0, 1.0, 0.0])
    return _normalize(q)


def _column_array(
        data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        cols: Sequence[str] | None,
        default_cols: Sequence[str],
        what: str,
) -> np.ndarray:
    """Достаёт `[N, k]` из таблицы по именам столбцов либо из массива по порядку."""
    width = len(default_cols)
    if isinstance(data, pd.DataFrame):
        names = list(cols) if cols is not None else list(default_cols)
        if len(names) != width:
            raise ValueError(
                f"{what}: нужно {width} имён столбцов в порядке "
                f"{tuple(default_cols)}, получено {len(names)}")
        missing = [c for c in names if c not in data.columns]
        if missing:
            raise KeyError(
                f"{what}: в таблице нет столбцов {missing}. Порядок столбцов "
                f"фиксирован: {tuple(default_cols)}; имена задаются параметром cols")
        return data.loc[:, names].to_numpy(dtype=float)

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(
            f"{what}: массив должен иметь форму [N, {width}] в порядке "
            f"{tuple(default_cols)}, получено {arr.shape}")
    return arr


def _as_quaternion(
        data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        *,
        representation: str,
        cols: Sequence[str] | None,
        initial: Sequence[float] | None,
        rotation_vector_in_degrees: bool,
        what: str,
) -> tuple[np.ndarray, str]:
    """Любое поддерживаемое представление -> кватернион `[N, 4]` в `(w, x, y, z)`.

    Возвращает `(кватернионы, примечание для журнала)`.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"{what}: representation должно быть одним из {REPRESENTATIONS}, "
            f"получено {representation!r}")

    if representation == "vector":
        v = _normalize(_column_array(data, cols, DEFAULT_VECTOR_COLUMNS, what))
        return _quat_shortest_arc(v), "вектор направления; торсия принята нулевой"

    if representation == "rotation_vector":
        rv = _column_array(data, cols, DEFAULT_VECTOR_COLUMNS, what)
        units = "градусы" if rotation_vector_in_degrees else "радианы"
        return (_quat_from_rotation_vector(rv, rotation_vector_in_degrees),
                f"вектор поворота, угол в {units}")

    raw = _column_array(data, cols, DEFAULT_QUATERNION_COLUMNS, what)
    # Внутри работаем в порядке (w, x, y, z), наружу порядок остаётся (x, y, z, w).
    q = _normalize(np.column_stack([raw[:, 3], raw[:, 0], raw[:, 1], raw[:, 2]]))

    if representation == "quaternion":
        return q, "кватернион ориентации, поворот из локальной системы в родительскую"

    if initial is None:
        raise ValueError(
            f"{what}: для representation='quaternion_relative' нужен параметр "
            f"initial -- исходная ориентация в порядке "
            f"{DEFAULT_QUATERNION_COLUMNS}. Без неё поворот относительно "
            f"исходного состояния нельзя перевести в неподвижное пространство")
    init = np.asarray(initial, dtype=float).ravel()
    if init.size != 4:
        raise ValueError(
            f"{what}: initial -- 4 компоненты в порядке "
            f"{DEFAULT_QUATERNION_COLUMNS}, получено {init.size}")
    q0 = _normalize(np.array([[init[3], init[0], init[1], init[2]]]))
    q0 = np.repeat(q0, q.shape[0], axis=0)
    return _quat_multiply(q0, q), "кватернион относительно исходной ориентации"


def orientation_to_direction(
        data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        *,
        representation: str = "quaternion",
        cols: Sequence[str] | None = None,
        initial: Sequence[float] | None = None,
        rotation_vector_in_degrees: bool = False,
) -> np.ndarray:
    """Ориентация в любом поддерживаемом виде -> единичный вектор направления.

    Параметры:
      data           — таблица или массив: `[N, 3]` для вектора и вектора
                       поворота, `[N, 4]` для кватерниона
      representation — `"vector"`, `"quaternion"`, `"quaternion_relative"`
                       либо `"rotation_vector"`
      cols           — имена столбцов; порядок фиксирован: `x, y, z`
                       либо `q_x, q_y, q_z, q_w`
      initial        — исходная ориентация для `"quaternion_relative"`
      rotation_vector_in_degrees — длина вектора поворота в градусах, не в радианах

    Возвращает `[N, 3]`: X вправо, Y вверх, Z вперёд.
    """
    q, _ = _as_quaternion(
        data, representation=representation, cols=cols, initial=initial,
        rotation_vector_in_degrees=rotation_vector_in_degrees, what="ориентация")
    forward = np.broadcast_to(np.asarray(FORWARD_XYZ, dtype=float), (q.shape[0], 3))
    return _normalize(_quat_rotate(q, forward))


def gaze_direction(
        head: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
        eye: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
        *,
        head_representation: str = "quaternion",
        eye_representation: str = "quaternion",
        head_cols: Sequence[str] | None = None,
        eye_cols: Sequence[str] | None = None,
        head_initial: Sequence[float] | None = None,
        eye_initial: Sequence[float] | None = None,
        rotation_vector_in_degrees: bool = False,
) -> GazeDirection:
    """Направление взора из ориентации головы и, если есть, глаза.

    Ориентация головы задаётся в неподвижном пространстве, ориентация глаза --
    относительно головы. Итог -- композиция: сначала поворот глаза, затем
    поворот головы. Каждая из двух может быть задана в любом из представлений
    `REPRESENTATIONS`, и представления могут не совпадать.

    Параметры:
      head, eye — таблицы или массивы; любая из двух может отсутствовать
      *_representation — вид записи, см. `REPRESENTATIONS`
      *_cols    — имена столбцов; порядок фиксирован: `x, y, z` либо
                  `q_x, q_y, q_z, q_w`
      *_initial — исходная ориентация для `"quaternion_relative"`
      rotation_vector_in_degrees — длина вектора поворота в градусах

    Возвращает `GazeDirection`. Поле `is_gaze` говорит, учтено ли положение
    головы: если задан только глаз, это направление глаза относительно головы,
    и фиксацией его интервалы называть нельзя.

    Ограничение представления `"vector"`: вектор не определяет поворот вокруг
    самого себя, поэтому при композиции с глазом торсия головы принимается
    нулевой (кратчайший поворот от направления вперёд). В журнал это пишется.
    """
    if head is None and eye is None:
        raise ValueError(
            "нужны данные головы, глаза или обе величины: "
            "без них направление взора не из чего получить")

    forward = np.asarray(FORWARD_XYZ, dtype=float)
    q_head = q_eye = None
    notes: dict[str, str] = {}

    if head is not None:
        q_head, notes["голова"] = _as_quaternion(
            head, representation=head_representation, cols=head_cols,
            initial=head_initial,
            rotation_vector_in_degrees=rotation_vector_in_degrees, what="голова")
    if eye is not None:
        q_eye, notes["глаз"] = _as_quaternion(
            eye, representation=eye_representation, cols=eye_cols,
            initial=eye_initial,
            rotation_vector_in_degrees=rotation_vector_in_degrees, what="глаз")

    if q_head is not None and q_eye is not None and q_head.shape[0] != q_eye.shape[0]:
        raise ValueError(
            f"голова и глаз должны быть одной длины, получено "
            f"{q_head.shape[0]} и {q_eye.shape[0]}")

    n = int((q_head if q_head is not None else q_eye).shape[0])
    rows = np.broadcast_to(forward, (n, 3))

    head_dir = _normalize(_quat_rotate(q_head, rows)) if q_head is not None else None
    eye_dir = _normalize(_quat_rotate(q_eye, rows)) if q_eye is not None else None

    if q_head is not None and q_eye is not None:
        q_gaze = _quat_multiply(q_head, q_eye)
        source = "композиция: поворот глаза, затем поворот головы"
    elif q_head is not None:
        q_gaze = q_head
        source = "только голова: направление взора совпадает с направлением головы"
    else:
        q_gaze = q_eye
        source = "только глаз: это направление глаза относительно головы, а не взор"

    direction = _normalize(_quat_rotate(q_gaze, rows))
    n_bad = int(np.sum(~np.isfinite(direction).all(axis=1)))

    warnings_text = []
    if q_head is not None and q_eye is not None and head_representation == "vector":
        warnings_text.append(
            "ориентация головы задана вектором: торсия головы не определена "
            "и принята нулевой, поэтому композиция с глазом приближённая")
    if eye is not None and head is None:
        warnings_text.append(
            "положение головы не задано: результат -- движение глаза "
            "относительно головы, а не взор; фиксацией его интервалы "
            "называть нельзя (см. docs/Sci_base-rus.md)")

    params = {
        "отсчётов": Param(n, "шт.", FROM_DATA),
        "система координат": Param("правая: X вправо, Y вверх, Z вперёд",
                                   source=FROM_DEFAULT,
                                   note=f"нейтральное направление {FORWARD_XYZ}"),
        "итог": Param(source, source=COMPUTED),
    }
    if head is not None:
        params["голова"] = Param(head_representation, source=FROM_USER, note=notes["голова"])
    if eye is not None:
        params["глаз"] = Param(eye_representation, source=FROM_USER, note=notes["глаз"])
    if n_bad:
        params["неполных отсчётов"] = Param(n_bad, "шт.", COMPUTED, "направление NaN")

    params_used = log_computation(
        "направление взора из ориентации", params, warnings_text=warnings_text)

    return GazeDirection(
        direction=direction,
        head_direction=head_dir,
        eye_direction=eye_dir,
        n_samples=n,
        is_gaze=head is not None,
        params_used=params_used,
    )
