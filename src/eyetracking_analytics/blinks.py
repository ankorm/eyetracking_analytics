###################################################
# Моргания: интервалы, когда глаза не видно
###################################################
"""Выделение морганий и работа с ними.

Моргание — не событие взора и не движение глаза: это интервал, на котором
измерения глаза недостоверны. Для методики важно уметь его найти и исключить,
иначе провал в данных превращается либо в мнимый перевод взора (координата
скачком уходит и возвращается), либо в мнимую фиксацию (прибор отдаёт последнее
значение).

Способа два, и они опираются на разные признаки:

* `find_blinks_by_gaps` — по отметке качества: прибор сам говорит, видит ли он
  глаз, и моргание — это разрыв в потоке достоверных отсчётов;
* `find_blinks_by_threshold` — по самому сигналу: площадь или диаметр зрачка
  падает ниже порога, когда веко закрывает зрачок.

Границы, найденные любым из способов, всегда занижены: веко закрывает зрачок
не мгновенно, и сигнал начинает портиться раньше, чем прибор теряет глаз.
`refine_blink_bounds` расширяет интервал до ближайшей смены монотонности
сигнала — точки, с которой началось падение и на которой закончился возврат.

Единицы: время в секундах, длительности в секундах.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DEFAULT_MIN_BLINK_GAP_S, DEFAULT_REFINE_SEARCH_S
from .report import COMPUTED, FROM_DATA, FROM_USER, Param, log_computation

BLINK_COLUMNS: tuple[str, ...] = ("t_start_s", "t_end_s", "duration_s")


def _empty_blinks() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=float) for name in BLINK_COLUMNS})


def _time_array(data: pd.DataFrame, col_time: str) -> np.ndarray:
    if col_time not in data.columns:
        raise KeyError(f"в таблице нет столбца времени {col_time!r}")
    return data[col_time].to_numpy(dtype=float)


def find_blinks_by_gaps(
        data: pd.DataFrame,
        *,
        col_quality: str,
        col_time: str = "T_s",
        min_gap_s: float = DEFAULT_MIN_BLINK_GAP_S,
) -> pd.DataFrame:
    """Моргания как разрывы в потоке достоверных отсчётов.

    `col_quality` — столбец, где True означает «глаз виден». Берутся отсчёты с
    True, и всякий разрыв между соседними такими отсчётами длиннее `min_gap_s`
    объявляется морганием: интервал от последнего достоверного отсчёта до
    первого следующего.

    Порог нужен именно потому, что не всякий разрыв — моргание: потеря одного
    кадра даёт разрыв в один шаг записи, и на 90 Гц это 11 мс, вчетверо меньше
    самого короткого моргания.

    Размерности: `col_time` `[s]`, `min_gap_s` `[s]`.
    Возвращает таблицу `t_start_s`, `t_end_s`, `duration_s`.
    """
    if col_quality not in data.columns:
        raise KeyError(f"в таблице нет столбца качества {col_quality!r}")
    t = _time_array(data, col_time)
    good = data[col_quality].to_numpy()
    good = np.asarray(good, dtype=bool) & np.isfinite(t)

    t_good = t[good]
    if t_good.size < 2:
        return _empty_blinks()

    order = np.argsort(t_good)
    t_good = t_good[order]
    gaps = np.diff(t_good)
    is_blink = gaps > float(min_gap_s)
    if not np.any(is_blink):
        return _empty_blinks()

    t_start = t_good[:-1][is_blink]
    t_end = t_good[1:][is_blink]
    return pd.DataFrame({
        "t_start_s": t_start.astype(float),
        "t_end_s": t_end.astype(float),
        "duration_s": (t_end - t_start).astype(float),
    })


def find_blinks_by_threshold(
        data: pd.DataFrame,
        *,
        col_value: str,
        col_time: str = "T_s",
        threshold: float,
        min_gap_s: float = DEFAULT_MIN_BLINK_GAP_S,
        min_duration_s: float = 0.0,
) -> pd.DataFrame:
    """Моргания как участки, где сигнал упал ниже порога.

    Работает по самому измерению: площадь или диаметр зрачка падает, когда веко
    его закрывает. Отсчёты ниже `threshold` собираются в серии; разрыв внутри
    серии короче `min_gap_s` нарушением не считается — на границе моргания
    сигнал дрожит около порога, и без этого допуска одно моргание распадалось бы
    на несколько.

    Размерности: `col_time` `[s]`, `threshold` — в единицах `col_value`,
    `min_gap_s` и `min_duration_s` `[s]`.
    Возвращает таблицу `t_start_s`, `t_end_s`, `duration_s`.
    """
    if col_value not in data.columns:
        raise KeyError(f"в таблице нет столбца {col_value!r}")
    t = _time_array(data, col_time)
    value = data[col_value].to_numpy(dtype=float)

    low = np.isfinite(t) & np.isfinite(value) & (value < float(threshold))
    idx = np.flatnonzero(low)
    if idx.size == 0:
        return _empty_blinks()

    rows: list[dict[str, float]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if float(t[i] - t[prev]) > float(min_gap_s):
            rows.append({"t_start_s": float(t[start]), "t_end_s": float(t[prev])})
            start = i
        prev = i
    rows.append({"t_start_s": float(t[start]), "t_end_s": float(t[prev])})

    out = pd.DataFrame(rows)
    out["duration_s"] = out["t_end_s"] - out["t_start_s"]
    out = out[out["duration_s"] >= float(min_duration_s)].reset_index(drop=True)
    return out if len(out) else _empty_blinks()


def _monotonicity_change_before(t: np.ndarray, value: np.ndarray, t_edge: float,
                                search_s: float) -> float:
    """Последняя точка смены монотонности перед `t_edge` (локальный максимум)."""
    window = np.isfinite(t) & np.isfinite(value) & (t >= t_edge - search_s) & (t < t_edge)
    idx = np.flatnonzero(window)
    if idx.size < 3:
        return float(t_edge)
    v = value[idx]
    peak = (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:])
    hits = idx[1:-1][peak]
    return float(t[hits[-1]]) if hits.size else float(t_edge)


def _monotonicity_change_after(t: np.ndarray, value: np.ndarray, t_edge: float,
                               search_s: float) -> float:
    """Первая точка смены монотонности после `t_edge` (локальный максимум)."""
    window = np.isfinite(t) & np.isfinite(value) & (t > t_edge) & (t <= t_edge + search_s)
    idx = np.flatnonzero(window)
    if idx.size < 3:
        return float(t_edge)
    v = value[idx]
    peak = (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:])
    hits = idx[1:-1][peak]
    return float(t[hits[0]]) if hits.size else float(t_edge)


def refine_blink_bounds(
        data: pd.DataFrame,
        blinks: pd.DataFrame,
        *,
        col_value: str,
        col_time: str = "T_s",
        search_s: float = DEFAULT_REFINE_SEARCH_S,
) -> pd.DataFrame:
    """Расширяет границы морганий до ближайшей смены монотонности сигнала.

    Найденная граница моргания — это момент, когда прибор потерял глаз, а сигнал
    начал портиться раньше: веко идёт вниз, зрачок сужается по измерению. Поэтому
    слева граница сдвигается к последнему локальному максимуму сигнала перед ней,
    справа — к первому локальному максимуму после: это точки, с которых началось
    падение и на которых закончился возврат.

    Если в окне `search_s` смены монотонности нет, граница остаётся прежней.

    Размерности: `col_time` `[s]`, `search_s` `[s]`.
    """
    if blinks.empty:
        return blinks.copy()
    if col_value not in data.columns:
        raise KeyError(f"в таблице нет столбца {col_value!r}")

    t = _time_array(data, col_time)
    value = data[col_value].to_numpy(dtype=float)

    starts = blinks["t_start_s"].to_numpy(dtype=float)
    ends = blinks["t_end_s"].to_numpy(dtype=float)
    new_starts = np.array([_monotonicity_change_before(t, value, s, float(search_s))
                           for s in starts], dtype=float)
    new_ends = np.array([_monotonicity_change_after(t, value, e, float(search_s))
                         for e in ends], dtype=float)

    out = blinks.copy()
    out["t_start_s"] = np.minimum(new_starts, starts)
    out["t_end_s"] = np.maximum(new_ends, ends)
    out["duration_s"] = out["t_end_s"] - out["t_start_s"]
    return out


def blink_mask(
        data: pd.DataFrame,
        blinks: pd.DataFrame,
        *,
        col_time: str = "T_s",
        pad_s: float = 0.0,
) -> np.ndarray:
    """Булева маска отсчётов, попавших в моргание (True = моргание).

    `pad_s` расширяет каждый интервал в обе стороны: полезно, когда нужно
    выбросить и переходные отсчёты по краям.
    """
    t = _time_array(data, col_time)
    mask = np.zeros(t.size, dtype=bool)
    if blinks.empty:
        return mask
    for t0, t1 in zip(blinks["t_start_s"].to_numpy(dtype=float),
                      blinks["t_end_s"].to_numpy(dtype=float)):
        mask |= (t >= t0 - float(pad_s)) & (t <= t1 + float(pad_s))
    return mask


def describe_blinks(
        blinks: pd.DataFrame,
        *,
        span_s: float | None = None,
        source: str = "разрывы в потоке достоверных отсчётов",
) -> dict:
    """Журнал по найденным морганиям: сколько, какой длительности, какая доля записи."""
    duration = blinks["duration_s"].to_numpy(dtype=float) if len(blinks) else np.array([])
    total = float(np.nansum(duration)) if duration.size else 0.0
    params = {
        "признак моргания": Param(source, source=FROM_USER),
        "найдено": Param(f"{len(blinks)} морганий", source=COMPUTED),
        "суммарная длительность": Param(total, "с", COMPUTED),
    }
    if duration.size:
        params["длительность моргания"] = Param(
            f"медиана {float(np.median(duration)) * 1000:.0f} мс, "
            f"от {float(np.min(duration)) * 1000:.0f} до {float(np.max(duration)) * 1000:.0f} мс",
            source=COMPUTED)
    if span_s is not None and span_s > 0:
        params["доля записи"] = Param(total / float(span_s), "", COMPUTED,
                                      f"из {float(span_s):.2f} с записи")
        params["запись"] = Param(float(span_s), "с", FROM_DATA)
    return log_computation("моргания", params)
