###################################################
# Оценка частоты дискретизации и приведение записи к другой частоте
###################################################
"""Единственное место, где оценивается частота записи.

До появления этого модуля частота оценивалась тремя разными способами в трёх
местах: `len(t)/(max−min)`, `median(diff(t))` и `len(t)/max(t)`. Формулы дают
разные результаты на записях с пропусками и на записях, начинающихся не с нуля,
поэтому пороги, выведенные из частоты, зависели от того, какая функция их считала.

Контракт частоты (решение автора методики):

* отсчёты времени **обязательны** всегда, и они **в секундах**;
* `fs_hz=None` — параметры считаются на частоте записи, пересчёта нет;
* `fs_hz` задана — выполняется **децимация**: берутся реальные отсчёты, ближайшие
  к узлам сетки нужной частоты. Это эмуляция того, что записал бы прибор с этой
  частотой. Дополнительного осреднения не вводится: сглаживание остаётся только
  там, где оно уже есть в методике (медианный фильтр, гауссово, star-фильтр);
* запросить частоту **выше** частоты записи нельзя: прибор быстрее не эмулируется.
  Допуск — два СКО длины шага исходных данных (дрожание метки времени).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingInfo:
    """Характеристики временной сетки записи. Все времена в секундах."""

    fs_hz: float                    # 1 / медианный шаг
    n_samples: int
    span_s: float
    dt_median_s: float
    dt_mean_s: float
    dt_sd_s: float
    dt_min_s: float
    dt_max_s: float
    n_duplicated_timestamps: int    # шагов, равных нулю или отрицательных
    n_gaps: int                     # шагов больше двух медианных
    max_gap_s: float
    jitter_rel: float               # dt_sd / dt_mean
    fs_max_requestable_hz: float    # предел по правилу «два СКО шага»

    @property
    def looks_like_milliseconds(self) -> bool:
        """Признак того, что время подано в миллисекундах, а не в секундах.

        Условие узкое: медианный шаг больше секунды, оценка частоты ниже 5 Гц
        и запись длится больше 1000 единиц. Для окулографии такой записи не бывает,
        а ось в миллисекундах даёт ровно такую картину.
        """
        return bool(self.dt_median_s > 1.0 and self.fs_hz < 5.0 and self.span_s > 1000.0)


def estimate_sampling_rate(t_s: Sequence[float] | np.ndarray) -> SamplingInfo:
    """Оценивает частоту записи и характеристики дрожания метки времени.

    Параметры:
      t_s — отсчёты времени в секундах, неубывающие (проверяется мягко:
            неположительные шаги считаются дублями и исключаются из статистики шага)

    Возвращает:
      SamplingInfo
    """
    t = np.asarray(t_s, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("estimate_sampling_rate: нужно минимум 2 отсчёта времени")
    if not np.isfinite(t).all():
        raise ValueError("estimate_sampling_rate: в отсчётах времени есть NaN или inf")

    dt_all = np.diff(t)
    positive = dt_all[dt_all > 0]
    n_dup = int(np.sum(dt_all <= 0))
    if positive.size == 0:
        raise ValueError(
            "estimate_sampling_rate: все шаги времени неположительные — "
            "метки времени не возрастают"
        )

    dt_median = float(np.median(positive))
    dt_mean = float(np.mean(positive))
    dt_sd = float(np.std(positive))
    dt_min = float(np.min(positive))
    dt_max = float(np.max(positive))

    # Предел запрашиваемой частоты: шаг не может быть короче, чем средний минус
    # два СКО дрожания, и не короче фактически наблюдавшегося минимального шага.
    dt_floor = max(dt_mean - 2.0 * dt_sd, dt_min)
    if dt_floor <= 0:
        dt_floor = dt_min

    return SamplingInfo(
        fs_hz=1.0 / dt_median,
        n_samples=int(t.size),
        span_s=float(t[-1] - t[0]),
        dt_median_s=dt_median,
        dt_mean_s=dt_mean,
        dt_sd_s=dt_sd,
        dt_min_s=dt_min,
        dt_max_s=dt_max,
        n_duplicated_timestamps=n_dup,
        n_gaps=int(np.sum(positive > 2.0 * dt_median)),
        max_gap_s=dt_max,
        jitter_rel=(dt_sd / dt_mean) if dt_mean > 0 else np.nan,
        fs_max_requestable_hz=1.0 / dt_floor,
    )


def require_seconds(info: SamplingInfo) -> None:
    """Отвергает временную ось, похожую на миллисекунды.

    Без этой проверки ошибка проходит молча: скорости получаются в тысячу раз
    меньше порогов, ни один пик не проходит отбор, и на выходе оказывается не
    исключение, а правдоподобная пустая таблица событий.
    """
    if info.looks_like_milliseconds:
        raise ValueError(
            f"похоже, время задано в миллисекундах: медианный шаг "
            f"{info.dt_median_s:.4g}, оценка частоты {info.fs_hz:.4g} Гц, "
            f"длительность {info.span_s:.4g}. Приведите отсчёты времени к секундам "
            f"(разделите на 1000)."
        )


def resolve_target_rate(info: SamplingInfo, fs_hz: float | None) -> tuple[float, bool]:
    """Разрешает запрошенную частоту в фактическую и признак необходимости децимации.

    Возвращает `(fs_used_hz, need_decimation)`.

    Запрос частоты выше частоты записи отвергается: допуск — два СКО длины шага
    исходных данных, то есть в пределах дрожания метки времени запрос считается
    совпадающим с частотой записи.
    """
    if fs_hz is None:
        return info.fs_hz, False

    fs_req = float(fs_hz)
    if not np.isfinite(fs_req) or fs_req <= 0:
        raise ValueError(f"fs_hz должна быть положительным числом, получено {fs_hz!r}")

    if fs_req > info.fs_max_requestable_hz:
        raise ValueError(
            f"недопустимое значение fs_hz={fs_req:.4g} Гц: частота записи "
            f"{info.fs_hz:.4g} Гц, предел с учётом дрожания шага (два СКО) — "
            f"{info.fs_max_requestable_hz:.4g} Гц. Прибор с большей частотой "
            f"не эмулируется: интерполяция не добавляет информации."
        )

    # В пределах дрожания запрос совпадает с частотой записи — децимация не нужна.
    if fs_req >= 1.0 / max(info.dt_mean_s + 2.0 * info.dt_sd_s, 1e-12):
        return info.fs_hz, False

    return fs_req, True


def decimate_to_rate(
        t_s: np.ndarray,
        fs_hz: float,
        *arrays: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Децимация: выбор реальных отсчётов, ближайших к узлам сетки `fs_hz`.

    Ни интерполяции, ни осреднения: возвращаются сами измерения и их
    фактические метки времени. Это эмуляция записи прибором с частотой `fs_hz`.

    Возвращает `(t_decimated, decimated_arrays, indices)`.
    """
    t = np.asarray(t_s, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("decimate_to_rate: нужно минимум 2 отсчёта времени")

    step = 1.0 / float(fs_hz)
    n_nodes = int(np.floor((t[-1] - t[0]) / step)) + 1
    if n_nodes < 2:
        raise ValueError(
            f"decimate_to_rate: при fs_hz={fs_hz:.4g} на записи длительностью "
            f"{t[-1] - t[0]:.4g} с получается меньше двух узлов"
        )

    nodes = t[0] + step * np.arange(n_nodes, dtype=float)
    # Ближайший реальный отсчёт к каждому узлу.
    pos = np.searchsorted(t, nodes, side="left")
    pos = np.clip(pos, 1, t.size - 1)
    left = pos - 1
    take_left = (nodes - t[left]) <= (t[pos] - nodes)
    idx = np.where(take_left, left, pos)
    idx = np.unique(idx)

    return t[idx], tuple(np.asarray(a, dtype=float).ravel()[idx] for a in arrays), idx
