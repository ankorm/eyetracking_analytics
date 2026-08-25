###################################################
# Ориентация: публичный слой над кватернионной реализацией
###################################################
"""Работа с ориентацией головы и взора в предметных величинах.

**Зачем это в библиотеке про глаза.** Взор строится из двух слагаемых:
направления глаза относительно головы и ориентации головы в неподвижном
пространстве. Прибор обычно даёт первое в углах, а второе -- кватернионом
или строкой "(x,y,z)", и в таком виде оно в разметку не годится. Функции
этого модуля -- переходник: `angles_from_orientation` переводит ориентацию
головы в углы, `rotate_points` поворачивает направление глаза в мировую
систему, `angular_distance_deg` меряет расхождение двух ориентаций
(например головы и взора), `parse_vector_column` разбирает текстовый
формат вектора из CSV масс-айтрекеров.

Само сложение головы с глазом библиотека не делает: это зависит от того, как
прибор задаёт обе величины. Разметка принимает уже готовый взор, и если
положение головы не учтено, размечается движение глаза (см. `docs/Sci_base-rus.md`).

**Какая ось какому движению отвечает.** Система та же, что во всей библиотеке
(см. `direction`): правая, **X** вправо, **Y** вверх, **Z** вперёд.

* поворот вокруг **Y** (вертикальной оси) -- `yaw`, горизонтальное движение
  взора: влево/вправо. В разметке ему отвечает азимут 0 град (вправо)
  и 180 град (влево);
* поворот вокруг **X** (поперечной, «вправо») -- `pitch`, вертикальное
  движение взора: вверх/вниз. Азимут +90 град вверх, -90 град вниз;
* поворот вокруг **Z** (продольной, «вперёд») -- `roll`, торсия: наклон головы
  к плечу либо циклоторсия глаза. Направление взора при этом не меняется,
  и методика такого движения не размечает.

Ту же систему использует разметка направления во всех трёх параметризациях:
эксцентриситет -- угол между текущим направлением и нейтральным, азимут --
куда именно отведён взор, отсчитывается от направления «вправо» к «вверх».

Кватернион -- способ записи ориентации, а не понятие методики, поэтому наружу
он не выходит: функции этого модуля принимают таблицы и массивы, а возвращают
**градусы**. Сама алгебра лежит в приватном `_quaternion.py`.

Пункт J аудита: прежде весь кватернионный слой (25 имён) был публичным
и экспортировался через `import *`, хотя ни одна вычислительная функция
библиотеки его не вызывала.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import _quaternion as _q
from .report import FROM_DATA, FROM_USER, Param, log_computation

#: Как прибор упорядочивает компоненты вектора. Различие между приборами --
#: в наименовании осей (у одних вверх Z, у других Y), поэтому перестановки
#: достаточно; тройка координат при этом должна быть правой.
AXIS_ORDERS = _q.AXIS_ORDERS

#: Имена столбцов кватерниона по умолчанию.
DEFAULT_QUATERNION_COLUMNS = ("q_x", "q_y", "q_z", "q_w")


def _quaternion_array(
        data: pd.DataFrame,
        cols: Sequence[str],
        axis_order: str,
) -> np.ndarray:
    """Извлекает кватернионы как массив `[N, 4]` в порядке `(w, x, y, z)`."""
    if len(cols) != 4:
        raise ValueError(f"нужно 4 имени столбца (x, y, z, w), получено {len(cols)}")
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"в таблице нет столбцов {missing}")

    raw = data.loc[:, list(cols)].to_numpy(dtype=float)
    if axis_order == "xyz":
        xyz = raw[:, :3]
    elif axis_order == "xzy":
        xyz = raw[:, [0, 2, 1]]
    else:
        raise ValueError(f"axis_order должно быть одним из {AXIS_ORDERS}, получено {axis_order!r}")
    return np.column_stack([raw[:, 3], xyz])


def _normalize_rows(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return q / norm


def angles_from_orientation(
        data: pd.DataFrame,
        *,
        cols: Sequence[str] = DEFAULT_QUATERNION_COLUMNS,
        axis_order: str = "xyz",
        to_degrees: bool = True,
) -> pd.DataFrame:
    """Ориентация -> углы `roll`, `pitch`, `yaw`.

    Параметры:
      data       — таблица с компонентами ориентации
      cols       — имена столбцов в порядке `(x, y, z, w)`
      axis_order — как прибор упорядочивает компоненты вектора: `"xyz"`
                   или `"xzy"`. Приборы расходятся именно в наименовании осей
                   (у одних вверх смотрит Z, у других Y), поэтому достаточно
                   перестановки. Переходом между левой и правой тройками
                   координат это не является: тот требует ещё смены знака,
                   и такие записи нужно приводить к правой тройке до вызова
      to_degrees — вернуть градусы (иначе радианы)

    Возвращает:
      DataFrame с колонками `roll_deg`, `pitch_deg`, `yaw_deg`
      (или `*_rad` при `to_degrees=False`), той же длины что `data`.

    Кватернионы нормируются: ненормированная запись иначе давала бы неверные
    углы, а прибор всегда пишет единичные с точностью до округления.
    """
    q = _normalize_rows(_quaternion_array(data, cols, axis_order))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Те же формулы, что в _quaternion.euler_from_quaternion, но векторно.
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    suffix = "deg" if to_degrees else "rad"
    if to_degrees:
        roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    n_bad = int(np.sum(~np.isfinite(q).all(axis=1)))
    log_computation(
        "углы ориентации из кватернионов",
        {
            "отсчётов": Param(len(data), "шт.", FROM_DATA),
            "столбцы": Param(", ".join(cols), source=FROM_USER,
                             note="порядок (x, y, z, w)"),
            "порядок осей": Param(axis_order, source=FROM_USER,
                                  note="перестановка компонент; тройка координат должна быть правой"),
            "единицы": Param("градусы" if to_degrees else "радианы", source=FROM_USER),
        },
        warnings_text=(
            [f"неполных кватернионов: {n_bad} — углы для них NaN"] if n_bad else []
        ),
    )

    return pd.DataFrame({
        f"roll_{suffix}": roll,
        f"pitch_{suffix}": pitch,
        f"yaw_{suffix}": yaw,
    }, index=data.index)


def angular_distance_deg(
        data: pd.DataFrame,
        other: pd.DataFrame,
        *,
        cols: Sequence[str] = DEFAULT_QUATERNION_COLUMNS,
        other_cols: Sequence[str] | None = None,
        axis_order: str = "xyz",
) -> np.ndarray:
    """Угол между двумя последовательностями ориентаций, градусы.

    Полезно, например, для расхождения ориентации головы и направления взора.
    Длины таблиц должны совпадать.
    """
    q1 = _normalize_rows(_quaternion_array(data, cols, axis_order))
    q2 = _normalize_rows(_quaternion_array(other, other_cols or cols, axis_order))
    if q1.shape != q2.shape:
        raise ValueError(
            f"таблицы должны быть одной длины, получено {q1.shape[0]} и {q2.shape[0]}"
        )

    # Угол поворота, переводящего одну ориентацию в другую: 2·atan2(|vec|, scalar)
    # для q1⁻¹ · q2. Для единичных кватернионов это 2·arccos(|q1·q2|).
    dot = np.abs(np.sum(q1 * q2, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def rotate_points(
        points: np.ndarray,
        data: pd.DataFrame,
        *,
        cols: Sequence[str] = DEFAULT_QUATERNION_COLUMNS,
        axis_order: str = "xyz",
) -> np.ndarray:
    """Поворот векторов заданной ориентацией.

    Параметры:
      points — массив `[N, 3]` либо один вектор `[3]`
      data   — таблица ориентаций длины N (либо 1, тогда применяется ко всем)

    Возвращает массив той же формы, что `points`.
    """
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points должен иметь форму [N, 3] или [3], получено {pts.shape}")

    q = _normalize_rows(_quaternion_array(data, cols, axis_order))
    if q.shape[0] == 1 and pts.shape[0] > 1:
        q = np.repeat(q, pts.shape[0], axis=0)
    if q.shape[0] != pts.shape[0]:
        raise ValueError(
            f"число ориентаций ({q.shape[0]}) не совпадает с числом векторов ({pts.shape[0]})"
        )

    w = q[:, 0:1]
    u = q[:, 1:4]
    # v' = v + 2·u × (u × v + w·v) — форма, эквивалентная q·v·q⁻¹ для единичного q.
    t = np.cross(u, pts) + w * pts
    rotated = pts + 2.0 * np.cross(u, t)
    return rotated[0] if single else rotated


def quaternion_from_matrix_angles_deg(rows: Sequence[Sequence[float]]) -> pd.Series:
    """Матрица поворота 3x3 -> углы `roll`, `pitch`, `yaw` в градусах.

    Пункт J-1 аудита: прежняя реализация преобразования матрицы в кватернион
    теряла знаки компонент и примерно в половине случаев описывала другой
    поворот. Исправлено; здесь наружу отдаются сразу углы, а кватернион
    остаётся внутренней деталью.
    """
    quat = _q.rotation_to_quaternion(_q.Matrix.from_rows(rows))
    angles = _q.euler_from_quaternion(quat, convert2deg=True)
    return pd.Series({"roll_deg": angles.x, "pitch_deg": angles.y, "yaw_deg": angles.z})


def parse_vector_column(text_values: Sequence[str], axis_order: str = "xyz") -> np.ndarray:
    """Разбор столбца строк вида `"(x,y,z)"` в массив `[N, 3]`.

    Такой формат встречается в CSV масс-айтрекеров: вектор направления или
    положения записан одной текстовой ячейкой. `axis_order` переставляет
    компоненты под наименование осей прибора (см. шапку модуля).
    """
    rows = [_q.vector3_from_list(_q.float_list_from_vec(str(v)), axis_order)
            for v in text_values]
    return np.asarray([[r.x, r.y, r.z] for r in rows], dtype=float)
