###################################################
# Разметка событий: три отдельных механизма по типу записи
###################################################
"""Публичная разметка переводов взора и фиксаций.

Термин «саккада» относится к движению глаза относительно черепа и в этом
модуле не используется: здесь размечается взор, то есть ориентация
относительно окружающего пространства, и событие называется **переводом взора**.

Три раздельные сущности — по типу записи, а не по способу вычисления:

* `mark_gaze_events_1d`        — одномерная запись: одна угловая координата
                                 (окулограмма по оси, готовый эксцентриситет);
* `mark_gaze_events_direction` — запись направления: x-y или x-y-z.

Во входных столбцах разметки взора ожидается направление относительно
**неподвижного пространства**: положение головы плюс положение глаза. Прибор
обычно даёт только второе слагаемое — направление глаза в системе, связанной с
головой, — поэтому движение головы нужно добавить (углы поворота головы) либо
явно принять голову неподвижной: подбородник, фиксированная посадка, короткий
интервал без движения головы. Если голова двигалась, а её движение не учтено,
размечается движение глаза, и события правильнее называть саккадами.

Критерий простой: величина, заданная в координатах головы или самого окулографа,
относится к глазу, как бы прибор её ни называл. Например, Pupil Invisible пишет
один глаз, и его `azimuth/elevation` заданы в системе камеры сцены, связанной с
головой, то есть это движение глаза.

Библиотека работает только с данными глаза и взора. Разметка сигналов другой
природы (например стабилограммы) в область применимости не входит: у них нет
ни модели глазного яблока, ни расстояния до плоскости, а значит нет и углового
эксцентриситета, на котором построена методика.

Пункт M-1 аудита. Прежняя планарная функция считала `alpha = atan2(dy, dx)` —
**полярный угол** направления смещения, а скорость как `|dα/dt|`. Следствия
измерены: движение строго «от центра» не детектировалось вообще (0 град/с при
восьми радиальных переводах взора), а прохождение траектории вблизи нейтральной точки
давало скачок α на 180° и пик скорости 4043 град/с на медленном монотонном
движении. Решение автора методики: правильная величина — **эксцентриситет**.

Буквальное восстановление унаследованной схемы (скорость как производная
эксцентриситета) дало бы зеркальную ошибку: тангенциальное движение при
постоянном эксцентриситете не детектируется. Поэтому скорость считается как
**полная угловая** — модуль производной единичного вектора направления. На одних
и тех же направлениях взора эта схема совпадает со сферической
(`geometry="angles"`) до 1e-12, то есть плоская и сферическая параметризации
перестают быть двумя разными методиками.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import gaze_mark as gm
from .gaze_mark import (
    _build_fixations_between_saccades,
    _build_saccades,
)
from .models import (
    DEFAULT_MIN_FIXATION_DURATION_S,
    IDEAL_FIXATION_SPREAD_DEG,
    N_SAMPLES_GAZE_SHIFT,
    min_saccade_minima_by_sampling,
)
from .report import (
    COMPUTED,
    FROM_DATA,
    FROM_DEFAULT,
    FROM_MODEL,
    FROM_USER,
    Param,
    log_computation,
)
from .sampling import estimate_sampling_rate, require_seconds

GEOMETRY_ANGLES = "angles"
GEOMETRY_PLANE = "plane"
GEOMETRY_VECTOR = "vector"


@dataclass(frozen=True)
class ScalarMarkup:
    """Результат разметки одномерной записи."""

    angle_deg: np.ndarray            # исходная угловая координата, [град]
    speed_deg_s: np.ndarray          # |da/dt|, [град/с]
    gaze_shifts: pd.DataFrame        # переводы взора
    fixations: pd.DataFrame
    fs_hz: float
    params_used: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectionMarkup:
    """Результат разметки записи направления (2D или 3D)."""

    eccentricity_deg: np.ndarray     # угол между направлением и нейтральным, [град]
    azimuth_deg: np.ndarray          # направление отклонения в картинной плоскости, [град]
    speed_deg_s: np.ndarray          # полная угловая скорость, [град/с]
    gaze_shifts: pd.DataFrame        # переводы взора
    fixations: pd.DataFrame
    fs_hz: float
    neutral_direction: tuple[float, float, float]
    params_used: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────── общие помощники
def _columns(data: pd.DataFrame, *keys: str | None) -> list[np.ndarray]:
    out = []
    for key in keys:
        if key is None:
            out.append(None)
            continue
        if key not in data.columns:
            raise KeyError(f"в таблице нет столбца {key!r}")
        out.append(data[key].to_numpy(dtype=float))
    return out


def _min_peak_width_points(min_event_duration_s: float, dt_median_s: float) -> float:
    return max(1.0, float(np.ceil(float(min_event_duration_s) / max(dt_median_s, 1e-12))))


def _resolve_speed(
        speed: np.ndarray,
        data: pd.DataFrame,
        col_speed: str | None,
        smoothing_points: int,
) -> tuple[np.ndarray, str, str, str]:
    """Скорость взора, по которой ведётся разметка: вычисленная или своя.

    Способ получения скорости — часть методики, а не деталь реализации.
    Производная положения без сглаживания на шумной записи завышает скорость:
    на биатлонной записи 192 Гц медиана скорости взора вышла 79.8 град/с против
    42.9 град/с у того же взора со сглаженной производной, а доля времени под
    порогом фиксации 20 град/с упала с 15.4 % до 4.8 %. Пороги при этом начинают
    отбирать шум, а не движение, поэтому источник скорости сделан явным:

      * `col_speed` — готовая угловая скорость взора, `[град/с]`: например сумма
        измеренной скорости головы и сглаженной производной глаза. Берётся модуль;
      * `smoothing_points` — медианный фильтр по скорости, нечётное число
        отсчётов; 0 и 1 означают «без сглаживания».

    Возвращает `(скорость, чем получена, источник для журнала, примечание)`.
    """
    if col_speed is not None:
        if col_speed not in data.columns:
            raise KeyError(f"в таблице нет столбца скорости {col_speed!r}")
        speed = np.abs(data[col_speed].to_numpy(dtype=float))
        kind = f"столбец {col_speed!r}"
        source = FROM_USER
    else:
        kind = "производная положения"
        source = FROM_DEFAULT

    points = int(smoothing_points)
    if points < 0:
        raise ValueError(f"speed_smoothing_points не может быть отрицательным, получено {points}")
    if points > 1:
        if points % 2 == 0:
            raise ValueError(
                "speed_smoothing_points должно быть нечётным: окно медианы "
                f"центрируется на отсчёте, получено {points}"
            )
        speed = (pd.Series(speed).rolling(points, center=True, min_periods=1)
                 .median().to_numpy(dtype=float))
        note = f"медиана по {points} отсчётам"
    else:
        note = "без сглаживания"
    return speed, kind, source, note


def _windowed_speed(
        t: np.ndarray,
        displacement_deg: Callable[[np.ndarray, np.ndarray], np.ndarray],
        window_s: float,
) -> tuple[np.ndarray, int]:
    """Угловая скорость как смещение за окно: `|Δp| / w`, `[град/с]`.

    Порог скорости на фиксации — физиологическая величина: он говорит, что взор
    за время порядка длительности события смещается не более чем на столько-то.
    Разность соседних отсчётов такой величиной не является: её шум растёт с
    частотой записи (при шуме положения 0.05 град на 250 Гц — 18 град/с), тогда
    как порог модели с частотой падает, до предела 5 град/с. Поэтому критерий
    считается по окну: шум смещения делится на длину окна, а не на шаг записи.

    `displacement_deg(i, j)` — угловое расстояние между отсчётами, `[град]`.
    Окно центрируется на отсчёте; при `window_s` меньше шага записи берётся
    один шаг. Возвращает `(скорость, длина окна в отсчётах)`.
    """
    n = int(len(t))
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    k = max(1, int(round(float(window_s) / dt))) if dt > 0 else 1
    if n < 2:
        return np.full(n, np.nan), k
    j1 = np.clip(np.arange(n) - k // 2 + k, 0, n - 1)
    j0 = np.clip(j1 - k, 0, n - 1)
    span = t[j1] - t[j0]
    ang = displacement_deg(j0, j1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(span > 0, ang / span, np.nan)
    return np.asarray(v, dtype=float), k


def _resolve_durations(
        min_event_duration_s: float,
        min_subprocess_duration_s: float | None,
        max_merge_gap_s: float | None,
) -> tuple[float, float]:
    sub = float(min_event_duration_s if min_subprocess_duration_s is None
                else min_subprocess_duration_s)
    gap = float(min_event_duration_s / 2.0 if max_merge_gap_s is None else max_merge_gap_s)
    return sub, gap


@dataclass(frozen=True)
class _Thresholds:
    """Разрешённые пороги вместе с источником каждого — для журнала."""

    min_peak_velocity_deg_s: float
    fixation_velocity_limit_deg_s: float
    min_amplitude_deg: float
    sources: dict[str, str]
    model_note: str
    fixation_note: str
    spread_max_deg: float
    spread_source: str
    min_fixation_duration_s: float


def _resolve_thresholds(
        fs_hz: float,
        *,
        min_peak_velocity_deg_s: float | None,
        fixation_velocity_limit_deg_s: float | None,
        min_amplitude_deg: float | None,
        n_samples: int,
        min_fixation_duration_s: float,
        ideal_fixation_spread_deg: float,
        spread_max_deg: float | None,
) -> _Thresholds:
    """Пороги по модели главной последовательности; заданные пользователем важнее.

    Прежде значения по умолчанию были `min_peak_velocity_deg_s = 15` и `fixation_velocity_limit_deg_s = 5` —
    замороженные константы вырожденной ветки унаследованной `eye_params`, никак
    не зависящие от частоты. Теперь по умолчанию все три порога выводятся
    из модели при фактической частоте записи.
    """
    minima = min_saccade_minima_by_sampling(
        fs_hz, n_samples=n_samples, min_fixation_duration_s=min_fixation_duration_s,
        ideal_fixation_spread_deg=ideal_fixation_spread_deg)

    resolved_peak = float(minima.v_peak_deg_s if min_peak_velocity_deg_s is None else min_peak_velocity_deg_s)
    resolved_fix = float(
        minima.v_fixation_deg_s if fixation_velocity_limit_deg_s is None else fixation_velocity_limit_deg_s)
    resolved_amp = float(minima.a_min_deg if min_amplitude_deg is None else min_amplitude_deg)

    if not resolved_fix < resolved_peak:
        detail = ""
        if min_peak_velocity_deg_s is None and fixation_velocity_limit_deg_s is None:
            detail = (
                f" Оба порога взяты из модели при {fs_hz:.4g} Гц и n_samples={n_samples}, "
                f"а она гарантирует это соотношение, поэтому такое сочетание означает "
                f"ошибку в самой модели — сообщите о нём."
            )
        raise ValueError(
            f"порог скорости на фиксации ({resolved_fix:.4g} град/с) должен быть строго "
            f"ниже порога пика перевода взора ({resolved_peak:.4g} град/с): иначе одно и то же "
            f"движение проходит и как перевод взора, и как фиксация." + detail
        )

    return _Thresholds(
        min_peak_velocity_deg_s=resolved_peak,
        fixation_velocity_limit_deg_s=resolved_fix,
        min_amplitude_deg=resolved_amp,
        sources={
            "min_peak_velocity_deg_s": FROM_USER if min_peak_velocity_deg_s is not None else FROM_MODEL,
            "fixation_velocity_limit_deg_s": (
                FROM_USER if fixation_velocity_limit_deg_s is not None else FROM_MODEL),
            "min_amplitude_deg": FROM_USER if min_amplitude_deg is not None else FROM_MODEL,
        },
        spread_max_deg=float(
            minima.spread_max_deg if spread_max_deg is None else spread_max_deg),
        spread_source=FROM_USER if spread_max_deg is not None else FROM_MODEL,
        min_fixation_duration_s=float(minima.t_fixation_s),
        fixation_note=(
            f"физиологический предел идеальной фовеальной фиксации; модель при этой "
            f"частоте даёт {minima.v_fixation_by_model_deg_s:.4g} град/с"
            if minima.v_fixation_at_floor else
            f"A_min / max(T_min, {min_fixation_duration_s * 1000:.0f} мс)"
        ),
        model_note=(
            f"модель при {fs_hz:.4g} Гц, n_samples={n_samples}: "
            f"A_min={minima.a_min_deg:.4g} град, T_min={minima.t_min_s * 1000:.1f} мс, "
            f"V_peak={minima.v_peak_deg_s:.4g} град/с, "
            f"v_фикс={minima.v_fixation_deg_s:.4g} град/с "
            f"(по интервалу {minima.t_fixation_s * 1000:.0f} мс)"
        ),
    )



def _apply_fixation_criteria(
        fixations: pd.DataFrame,
        *,
        spread_limit_deg: float,
        spread_measure: str,
        drop_invalid_spread: bool,
        min_fixation_duration_s: float,
        drop_short_fixations: bool,
) -> tuple[pd.DataFrame, int, int]:
    """Два дополнительных критерия валидности фиксации.

    **Минимальная длительность (пункт C-4, решение принято).** Интервал короче
    минимальной длительности фиксации не является фиксацией. Порог зависит от
    частоты: он равен `max(t_min, 50 мс)` из модели, то есть на низких частотах
    выше. По умолчанию такие интервалы отбрасываются.

    **Пространственный разброс (пункт C-19, подход в разработке).** Измеряется
    в одном из трёх режимов и сравнивается с допустимым. По умолчанию только
    помечается: `drop_invalid_spread=True` включает фильтрацию явно.

    Возвращает `(таблица, число_коротких, число_широких)`.
    """
    if spread_measure not in gm.SPREAD_MEASURES:
        raise ValueError(
            f"spread_measure должно быть одним из {gm.SPREAD_MEASURES}, "
            f"получено {spread_measure!r}"
        )

    column = f"spread_{spread_measure}_deg"
    if fixations.empty:
        out = fixations.copy()
        for name, dtype in (("duration_s", float), ("duration_valid", bool),
                            ("spread_deg", float), ("spread_limit_deg", float),
                            ("spread_valid", bool)):
            out[name] = pd.Series(dtype=dtype)
        return out, 0, 0

    duration = (fixations["t_end_s"] - fixations["t_start_s"]).to_numpy(dtype=float)
    duration_valid = np.isfinite(duration) & (duration >= float(min_fixation_duration_s))

    spread = fixations[column].to_numpy(dtype=float)
    spread_valid = np.isfinite(spread) & (spread <= float(spread_limit_deg))

    out = fixations.assign(
        duration_s=duration,
        duration_valid=duration_valid,
        spread_deg=spread,
        spread_limit_deg=float(spread_limit_deg),
        spread_valid=spread_valid,
    )
    n_short = int((~duration_valid).sum())
    n_wide = int((~spread_valid).sum())

    keep = np.ones(len(out), dtype=bool)
    if drop_short_fixations:
        keep &= duration_valid
    if drop_invalid_spread:
        keep &= spread_valid
    if not keep.all():
        out = pd.DataFrame(out.loc[keep].reset_index(drop=True))
    return out, n_short, n_wide


# ═════════════════════════════════════════════════════ одномерная запись
def mark_gaze_events_1d(
        data: pd.DataFrame,
        *,
        col_gaze_angle: str,
        col_time: str = "T_s",
        col_speed: str | None = None,
        speed_smoothing_points: int = 0,
        min_peak_velocity_deg_s: float | None = None,
        fixation_velocity_limit_deg_s: float | None = None,
        min_event_duration_s: float = 0.006,
        min_amplitude_deg: float | None = None,
        n_samples: int = N_SAMPLES_GAZE_SHIFT,
        min_fixation_duration_s: float = DEFAULT_MIN_FIXATION_DURATION_S,
        ideal_fixation_spread_deg: float = IDEAL_FIXATION_SPREAD_DEG,
        spread_limit_deg: float | None = None,
        spread_measure: str = "std",
        drop_invalid_spread: bool = False,
        drop_short_fixations: bool = True,
        min_subprocess_duration_s: float | None = None,
        max_merge_gap_s: float | None = None,
        trend_alpha: float = 0.05,
        trend_slope_min: float | None = None,
        trim_fixations_by_velocity: bool = True,
        fixation_velocity_window_s: float = 0.010,
        homogeneity_split: bool = True,
        split_alpha: float = gm.SPLIT_ALPHA,
        split_min_points: int = gm.SPLIT_MIN_POINTS,
        split_score_min_with_pvalue: float = gm.SPLIT_SCORE_MIN_WITH_PVALUE,
        split_score_min_alone: float = gm.SPLIT_SCORE_MIN_ALONE,
) -> ScalarMarkup:
    """Разметка одномерной угловой записи.

    Применимо к окулограмме по одной оси и к любому готовому угловому ряду.
    Скорость — `|da/dt|` `[град/с]`, амплитуда события — `|Δa|` `[град]`.

    Параметры:
      col_gaze_angle                 — столбец угловой координаты, `[град]`
      col_time                     — столбец времени, `[с]`
      col_speed                 — готовая угловая скорость взора, `[град/с]`;
                                  None = производная положения. Нужен, когда
                                  скорость измерена (гироскоп) или посчитана
                                  сглаженной производной: на шумной записи
                                  несглаженная производная завышает скорость
      speed_smoothing_points    — медианный фильтр по скорости, нечётное число
                                  отсчётов; 0 = без сглаживания
      min_peak_velocity_deg_s                  — минимальная высота пика скорости, `[град/с]`
      fixation_velocity_limit_deg_s       — порог «скорость уже фиксационная», `[град/с]`
      min_event_duration_s      — минимальная длительность события, `[с]`
      min_amplitude_deg                 — минимальная амплитуда события, `[град]`
      min_subprocess_duration_s — минимум подпроцесса при разбиении фиксации, `[с]`;
                                  None = `min_event_duration_s`
      max_merge_gap_s           — максимальный разрыв при склейке фиксаций, `[с]`;
                                  None = `min_event_duration_s / 2`
      trend_alpha               — уровень значимости критерия тренда
      trend_slope_min           — минимальный по модулю наклон, `[град/с]`;
                                  None = `fixation_velocity_limit_deg_s`
      trim_fixations_by_velocity — урезать ли фиксацию до отсчётов со скоростью
                                  ниже `fixation_velocity_limit_deg_s`; False =
                                  фиксацией берётся весь интервал между переводами
      fixation_velocity_window_s — окно, по которому считается скорость для этого
                                  критерия: `|Δp| / w`, `[с]`. Поотсчётная разность
                                  не годится: её шум растёт с частотой записи, а
                                  порог модели с частотой падает. Если задан
                                  `col_speed`, критерий считается по нему
      homogeneity_split         — требовать ли постоянства положения на фиксации:
                                  True дробит неоднородный интервал на подпроцессы,
                                  False снимает требование целиком
      split_alpha               — уровень значимости тестов однородности; меньше =
                                  слабее требование
      split_min_points          — минимум отсчётов в каждой половине при разбиении
      split_score_min_with_pvalue, split_score_min_alone — меры различия средних и
                                  ковариаций, при которых разбиение обосновано;
                                  больше = слабее требование к постоянству положения

    Возвращает `ScalarMarkup`. Таблица фиксаций содержит статистику только по
    одной координате: `e_angle` (среднее), `var_angle` (дисперсия),
    `slope_angle`, `p_angle`.
    """
    (a, t) = _columns(data, col_gaze_angle, col_time)
    sampling = estimate_sampling_rate(t)
    require_seconds(sampling)
    sub_dur, merge_gap = _resolve_durations(
        min_event_duration_s, min_subprocess_duration_s, max_merge_gap_s)
    th = _resolve_thresholds(
        sampling.fs_hz,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_amplitude_deg=min_amplitude_deg,
        n_samples=n_samples,
        min_fixation_duration_s=min_fixation_duration_s,
        ideal_fixation_spread_deg=ideal_fixation_spread_deg,
        spread_max_deg=spread_limit_deg,
    )
    trend_slope_min = float(th.fixation_velocity_limit_deg_s if trend_slope_min is None else trend_slope_min)

    speed = np.abs(np.gradient(a, t))                       # [град/с]
    speed, speed_kind, speed_source, speed_note = _resolve_speed(
        speed, data, col_speed, speed_smoothing_points)
    if col_speed is None:
        v_criterion, window_points = _windowed_speed(
            t, lambda i, j: np.abs(a[j] - a[i]), fixation_velocity_window_s)
        criterion_note = (f"смещение за {fixation_velocity_window_s * 1000:.0f} мс "
                          f"({window_points} отсч.)")
    else:
        v_criterion, criterion_note = speed, f"по столбцу {col_speed!r}"

    def amplitude_fn(i0: int, i1: int) -> float:
        return float(abs(a[i1] - a[i0]))

    gaze_shifts = _build_saccades(
        t=t,
        alpha_deg=a,
        v_ang=speed,
        min_peak_velocity_deg_s=th.min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=th.fixation_velocity_limit_deg_s,
        min_peak_width=_min_peak_width_points(min_event_duration_s, sampling.dt_median_s),
        min_amplitude_deg=th.min_amplitude_deg,
        amplitude_fn=amplitude_fn,
        with_amplitude=True,
    )

    # Вторая координата постоянна: статистика вырождается в одномерную
    # (ковариация имеет только один ненулевой элемент, тесты по второй оси
    # возвращают p = 1 и на критерий не влияют).
    zero = np.zeros_like(a)
    fixations = _build_fixations_between_saccades(
        t=t,
        x=a,
        y=zero,
        alpha_deg=a,
        v_ang=speed,
        events=gaze_shifts,
        fixation_velocity_limit_deg_s=th.fixation_velocity_limit_deg_s,
        min_amplitude_deg=th.min_amplitude_deg,
        min_interval_duration_s=float(min_event_duration_s),
        min_subprocess_duration_s=sub_dur,
        max_merge_gap_s=merge_gap,
        trend_alpha=trend_alpha,
        trend_slope_min=trend_slope_min,
        trim_by_velocity=trim_fixations_by_velocity,
        v_criterion=v_criterion,
        homogeneity_split=homogeneity_split,
        split_alpha=split_alpha,
        split_min_points=split_min_points,
        split_score_min_with_pvalue=split_score_min_with_pvalue,
        split_score_min_alone=split_score_min_alone,
    )
    fixations, n_short, n_wide = _apply_fixation_criteria(
        fixations,
        spread_limit_deg=th.spread_max_deg,
        spread_measure=spread_measure,
        drop_invalid_spread=drop_invalid_spread,
        min_fixation_duration_s=th.min_fixation_duration_s,
        drop_short_fixations=drop_short_fixations,
    )
    fixations = _to_scalar_fixation_schema(fixations)

    split_changed = (
        not homogeneity_split
        or split_alpha != gm.SPLIT_ALPHA
        or split_min_points != gm.SPLIT_MIN_POINTS
        or split_score_min_with_pvalue != gm.SPLIT_SCORE_MIN_WITH_PVALUE
        or split_score_min_alone != gm.SPLIT_SCORE_MIN_ALONE
    )

    params_used = log_computation(
        "разметка одномерной угловой записи",
        {
            "запись": Param(f"{sampling.span_s:.2f} с, {sampling.n_samples} отсчётов",
                            source=FROM_DATA),
            "частота записи": Param(sampling.fs_hz, "Гц", FROM_DATA),
            "схема расчёта": Param("|da/dt|, амплитуда |Δa|", source=FROM_DEFAULT),
            "источник скорости": Param(speed_kind, "", speed_source, speed_note),
            "критерий скорости на фиксации": Param(
                "интервал урезан по порогу" if trim_fixations_by_velocity
                else "интервал взят целиком",
                "", FROM_DEFAULT if trim_fixations_by_velocity else FROM_USER,
                f"скорость критерия: {criterion_note}"
                if trim_fixations_by_velocity
                else "порог участвует только в запрете склейки"),
            "постоянство положения": Param(
                "разбиение на подпроцессы" if homogeneity_split else "разбиение отключено",
                "", FROM_USER if split_changed else FROM_DEFAULT,
                f"alpha={split_alpha:g}, score>{split_score_min_with_pvalue:g} при значимом p, "
                f"score>{split_score_min_alone:g} без него, минимум {split_min_points} отсчётов"),
            "порог пика скорости": Param(
                th.min_peak_velocity_deg_s, "град/с", th.sources["min_peak_velocity_deg_s"], th.model_note),
            "порог скорости на фиксации": Param(
                th.fixation_velocity_limit_deg_s, "град/с", th.sources["fixation_velocity_limit_deg_s"],
                th.fixation_note if th.sources["fixation_velocity_limit_deg_s"] == FROM_MODEL else ""),
            "минимальная амплитуда": Param(
                th.min_amplitude_deg, "град", th.sources["min_amplitude_deg"]),
            "минимальная длительность события": Param(
                float(min_event_duration_s) * 1000, "мс", FROM_USER),
            "минимальная длительность фиксации": Param(
                th.min_fixation_duration_s * 1000, "мс", FROM_MODEL,
                ("короткие отброшены" if drop_short_fixations else "только помечаются")
                + f": {n_short} шт."),
            "допустимый разброс фиксации": Param(
                th.spread_max_deg, "град", th.spread_source,
                f"мера «{spread_measure}»; критерий в разработке: "
                + ("невалидные отброшены" if drop_invalid_spread
                   else "только помечаются")
                + f": {n_wide} шт."),
            "найдено": Param(f"{len(gaze_shifts)} переводов взора, {len(fixations)} фиксаций",
                             source=COMPUTED,
                             note=f"коротких: {n_short}, широких: {n_wide}"),
        },
    )
    gaze_shifts.attrs["params_used"] = params_used
    fixations.attrs["params_used"] = params_used

    return ScalarMarkup(
        angle_deg=a,
        speed_deg_s=speed,
        gaze_shifts=gaze_shifts,
        fixations=fixations,
        fs_hz=float(sampling.fs_hz),
        params_used=params_used,
    )


def _to_scalar_fixation_schema(fixations: pd.DataFrame) -> pd.DataFrame:
    """Приводит таблицу фиксаций к одномерной схеме: без колонок второй оси."""
    if fixations.empty:
        columns = ["t_start_s", "t_end_s", "mean_speed_deg_s", "mean_angle_deg", "has_trend",
                   "slope_angle_deg_s", "p_angle", "n_valid", "mean_angle_position_deg",
                   "var_angle_deg2", "heterogeneous_parent", "split_score",
                   "split_pvalue", "split_depth"]
        return pd.DataFrame(columns=columns)

    out = fixations.rename(columns={
        "mean_eccentricity_deg": "mean_angle",
        "slope_gaze_x_deg_s": "slope_angle_deg_s",
        "p_gaze_x": "p_angle",
        "mean_gaze_x": "mean_angle_position_deg",
        "cov_fro_norm_deg2": "var_angle_deg2",
    })
    # Колонки второй (постоянной) оси смысла не имеют.
    return pd.DataFrame(out.drop(columns=[c for c in ("slope_gaze_y_deg_s", "p_gaze_y", "mean_gaze_y") if c in out.columns]))


# ═════════════════════════════════════════════════ запись направления
def mark_gaze_events_direction(
        data: pd.DataFrame,
        *,
        col_gaze_x: str,
        col_gaze_y: str,
        col_gaze_z: str | None = None,
        col_time: str = "T_s",
        col_speed: str | None = None,
        speed_smoothing_points: int = 0,
        geometry: str = GEOMETRY_ANGLES,
        r: float | None = None,
        k: float = 1.0,
        gaze_x_neutral: float = 0.0,
        gaze_y_neutral: float = 0.0,
        deg: bool = True,
        min_peak_velocity_deg_s: float | None = None,
        fixation_velocity_limit_deg_s: float | None = None,
        min_event_duration_s: float = 0.006,
        min_amplitude_deg: float | None = None,
        n_samples: int = N_SAMPLES_GAZE_SHIFT,
        min_fixation_duration_s: float = DEFAULT_MIN_FIXATION_DURATION_S,
        ideal_fixation_spread_deg: float = IDEAL_FIXATION_SPREAD_DEG,
        spread_limit_deg: float | None = None,
        spread_measure: str = "std",
        drop_invalid_spread: bool = False,
        drop_short_fixations: bool = True,
        min_subprocess_duration_s: float | None = None,
        max_merge_gap_s: float | None = None,
        trend_alpha: float = 0.05,
        trend_slope_min: float | None = None,
        trim_fixations_by_velocity: bool = True,
        fixation_velocity_window_s: float = 0.010,
        homogeneity_split: bool = True,
        split_alpha: float = gm.SPLIT_ALPHA,
        split_min_points: int = gm.SPLIT_MIN_POINTS,
        split_score_min_with_pvalue: float = gm.SPLIT_SCORE_MIN_WITH_PVALUE,
        split_score_min_alone: float = gm.SPLIT_SCORE_MIN_ALONE,
) -> DirectionMarkup:
    """Разметка записи направления взора: x-y или x-y-z.

    Три параметризации входа приводятся к одному и тому же расчёту, поэтому
    результаты сопоставимы между собой:

    * `geometry="angles"` — `c1/c2` это yaw/pitch (`deg=True` — в градусах).
      Нейтральное направление — (0, 0);
    * `geometry="plane"` — `c1/c2` это координаты точки на плоскости,
      удалённой на `r` от центра вращения (обязательный параметр, в тех же
      единицах, что координаты после умножения на `k`). Нейтральное положение —
      `gaze_x_neutral`, `gaze_y_neutral`;
    * `geometry="vector"` — `c1/c2/c3` это компоненты вектора направления;
      нейтральное направление берётся как нормированное медианное направление
      записи (допущение методики: медиана = наиболее комфортное направление).

    Во всех случаях считаются:

    * `eccentricity_deg` — угол между текущим и нейтральным направлением;
    * `azimuth_deg` — направление отклонения (прежний `alpha` планарной схемы);
    * `speed_deg_s` — **полная** угловая скорость, модуль производной единичного
      вектора направления: она видит и радиальное, и тангенциальное движение;
    * амплитуда события — угол большого круга между направлениями на границах.

    Источник скорости (`col_speed`, `speed_smoothing_points`), критерий скорости
    на фиксации (`trim_fixations_by_velocity`) и требование
    постоянства положения на фиксации (`homogeneity_split`, `split_alpha`,
    `split_min_points`, `split_score_min_with_pvalue`, `split_score_min_alone`)
    работают так же, как в `mark_gaze_events_1d`.

    Возвращает `DirectionMarkup`.
    """
    if geometry not in (GEOMETRY_ANGLES, GEOMETRY_PLANE, GEOMETRY_VECTOR):
        raise ValueError(
            f"geometry должно быть {GEOMETRY_ANGLES!r}, {GEOMETRY_PLANE!r} "
            f"или {GEOMETRY_VECTOR!r}, получено {geometry!r}"
        )
    if geometry == GEOMETRY_VECTOR and col_gaze_z is None:
        raise ValueError("для geometry='vector' нужен третий столбец col_gaze_z")
    if geometry == GEOMETRY_PLANE and (r is None or not np.isfinite(r) or r <= 0):
        raise ValueError(
            "для geometry='plane' нужен параметр r > 0 — расстояние от центра "
            "вращения до плоскости в тех же единицах, что координаты. "
            "Для координат зрачка это радиус глаза, для точки взора на экране — "
            "дистанция наблюдения. Без r угловой эксцентриситет не определён."
        )

    c1, c2, c3, t = _columns(data, col_gaze_x, col_gaze_y, col_gaze_z, col_time)
    sampling = estimate_sampling_rate(t)
    require_seconds(sampling)
    sub_dur, merge_gap = _resolve_durations(
        min_event_duration_s, min_subprocess_duration_s, max_merge_gap_s)
    th = _resolve_thresholds(
        sampling.fs_hz,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_amplitude_deg=min_amplitude_deg,
        n_samples=n_samples,
        min_fixation_duration_s=min_fixation_duration_s,
        ideal_fixation_spread_deg=ideal_fixation_spread_deg,
        spread_max_deg=spread_limit_deg,
    )
    trend_slope_min = float(th.fixation_velocity_limit_deg_s if trend_slope_min is None else trend_slope_min)

    if geometry == GEOMETRY_ANGLES:
        yaw = np.radians(c1) if deg else np.asarray(c1, dtype=float)
        pitch = np.radians(c2) if deg else np.asarray(c2, dtype=float)
        # Ось «вперёд» — X, как в сферической схеме.
        ux = np.cos(pitch) * np.cos(yaw)
        uy = np.cos(pitch) * np.sin(yaw)
        uz = np.sin(pitch)
        neutral = (1.0, 0.0, 0.0)
        stat_x, stat_y = np.degrees(yaw), np.degrees(pitch)
        scheme = "yaw/pitch на единичной сфере, нейтраль (0, 0)"
    elif geometry == GEOMETRY_PLANE:
        dx = (np.asarray(c1, dtype=float) - float(gaze_x_neutral)) * float(k)
        dy = (np.asarray(c2, dtype=float) - float(gaze_y_neutral)) * float(k)
        norm = np.sqrt(dx * dx + dy * dy + float(r) ** 2)
        ux = float(r) / norm
        uy = dx / norm
        uz = dy / norm
        neutral = (1.0, 0.0, 0.0)
        stat_x, stat_y = dx, dy
        scheme = f"плоскость на расстоянии r = {float(r):.4g}, нейтраль ({gaze_x_neutral}, {gaze_y_neutral})"
    else:
        vx = np.asarray(c1, dtype=float)
        vy = np.asarray(c2, dtype=float)
        vz = np.asarray(c3, dtype=float)
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux, uy, uz = vx / norm, vy / norm, vz / norm
        med = np.array([np.nanmedian(ux), np.nanmedian(uy), np.nanmedian(uz)], dtype=float)
        med_norm = float(np.linalg.norm(med))
        if not np.isfinite(med_norm) or med_norm == 0:
            raise ValueError("не удалось определить нейтральное направление: медиана нулевая")
        med = med / med_norm
        # Поворачиваем систему так, чтобы нейтральное направление стало осью X.
        neutral = (float(med[0]), float(med[1]), float(med[2]))
        stat_x, stat_y = uy, uz
        scheme = "вектор направления, нейтраль = медианное направление записи"

    if geometry == GEOMETRY_VECTOR:
        dot = np.clip(ux * neutral[0] + uy * neutral[1] + uz * neutral[2], -1.0, 1.0)
    else:
        dot = np.clip(ux, -1.0, 1.0)
    eccentricity = np.degrees(np.arccos(dot))                # [град]
    azimuth = np.degrees(np.arctan2(uz, uy))                 # [град]

    # Полная угловая скорость: модуль производной единичного вектора.
    speed = np.degrees(np.sqrt(
        np.gradient(ux, t) ** 2 + np.gradient(uy, t) ** 2 + np.gradient(uz, t) ** 2))
    speed, speed_kind, speed_source, speed_note = _resolve_speed(
        speed, data, col_speed, speed_smoothing_points)
    def _great_circle_deg(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        d = np.clip(ux[i] * ux[j] + uy[i] * uy[j] + uz[i] * uz[j], -1.0, 1.0)
        return np.degrees(np.arccos(d))

    if col_speed is None:
        v_criterion, window_points = _windowed_speed(
            t, _great_circle_deg, fixation_velocity_window_s)
        criterion_note = (f"смещение за {fixation_velocity_window_s * 1000:.0f} мс "
                          f"({window_points} отсч.)")
    else:
        v_criterion, criterion_note = speed, f"по столбцу {col_speed!r}"

    def amplitude_fn(i0: int, i1: int) -> float:
        d = ux[i0] * ux[i1] + uy[i0] * uy[i1] + uz[i0] * uz[i1]
        return float(np.degrees(np.arccos(np.clip(d, -1.0, 1.0))))

    gaze_shifts = _build_saccades(
        t=t,
        alpha_deg=eccentricity,
        v_ang=speed,
        min_peak_velocity_deg_s=th.min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=th.fixation_velocity_limit_deg_s,
        min_peak_width=_min_peak_width_points(min_event_duration_s, sampling.dt_median_s),
        min_amplitude_deg=th.min_amplitude_deg,
        amplitude_fn=amplitude_fn,
        with_amplitude=True,
    )
    # Прежняя схема писала одно и то же значение в две пары колонок (пункт I-1).
    # Теперь эксцентриситет и азимут -- разные величины с явными именами.
    if not gaze_shifts.empty:
        idx_start = gaze_shifts["i_start"].to_numpy(dtype=int)
        idx_end = gaze_shifts["i_end"].to_numpy(dtype=int)
        gaze_shifts = gaze_shifts.assign(
            azimuth_start_deg=azimuth[idx_start],
            azimuth_end_deg=azimuth[idx_end],
        )

    fixations_raw = _build_fixations_between_saccades(
        # Площади заметаемой области -- характеристики взора и только его:
        # они имеют смысл как проекция на неподвижное пространство. В глазной
        # ветке и в одномерной записи колонок нет.
        with_area=True,
        t=t,
        x=stat_x,
        y=stat_y,
        alpha_deg=eccentricity,
        v_ang=speed,
        events=gaze_shifts,
        fixation_velocity_limit_deg_s=th.fixation_velocity_limit_deg_s,
        min_amplitude_deg=th.min_amplitude_deg,
        min_interval_duration_s=float(min_event_duration_s),
        min_subprocess_duration_s=sub_dur,
        max_merge_gap_s=merge_gap,
        trend_alpha=trend_alpha,
        trend_slope_min=trend_slope_min,
        trim_by_velocity=trim_fixations_by_velocity,
        v_criterion=v_criterion,
        homogeneity_split=homogeneity_split,
        split_alpha=split_alpha,
        split_min_points=split_min_points,
        split_score_min_with_pvalue=split_score_min_with_pvalue,
        split_score_min_alone=split_score_min_alone,
    )
    fixations, n_short, n_wide = _apply_fixation_criteria(
        fixations_raw,
        spread_limit_deg=th.spread_max_deg,
        spread_measure=spread_measure,
        drop_invalid_spread=drop_invalid_spread,
        min_fixation_duration_s=th.min_fixation_duration_s,
        drop_short_fixations=drop_short_fixations,
    )

    split_changed = (
        not homogeneity_split
        or split_alpha != gm.SPLIT_ALPHA
        or split_min_points != gm.SPLIT_MIN_POINTS
        or split_score_min_with_pvalue != gm.SPLIT_SCORE_MIN_WITH_PVALUE
        or split_score_min_alone != gm.SPLIT_SCORE_MIN_ALONE
    )

    params_used = log_computation(
        "разметка переводов взора по направлению",
        {
            "запись": Param(f"{sampling.span_s:.2f} с, {sampling.n_samples} отсчётов",
                            source=FROM_DATA),
            "частота записи": Param(sampling.fs_hz, "Гц", FROM_DATA),
            "параметризация": Param(geometry, source=FROM_USER, note=scheme),
            "нейтральное направление": Param(
                f"({neutral[0]:.4f}, {neutral[1]:.4f}, {neutral[2]:.4f})",
                source=COMPUTED if geometry == GEOMETRY_VECTOR else FROM_DEFAULT),
            "скорость": Param("полная угловая, |d(единичный вектор)/dt|",
                              source=FROM_DEFAULT,
                              note="видит и радиальное, и тангенциальное движение"),
            "источник скорости": Param(speed_kind, "", speed_source, speed_note),
            "критерий скорости на фиксации": Param(
                "интервал урезан по порогу" if trim_fixations_by_velocity
                else "интервал взят целиком",
                "", FROM_DEFAULT if trim_fixations_by_velocity else FROM_USER,
                f"скорость критерия: {criterion_note}"
                if trim_fixations_by_velocity
                else "порог участвует только в запрете склейки"),
            "постоянство положения": Param(
                "разбиение на подпроцессы" if homogeneity_split else "разбиение отключено",
                "", FROM_USER if split_changed else FROM_DEFAULT,
                f"alpha={split_alpha:g}, score>{split_score_min_with_pvalue:g} при значимом p, "
                f"score>{split_score_min_alone:g} без него, минимум {split_min_points} отсчётов"),
            "порог пика скорости": Param(
                th.min_peak_velocity_deg_s, "град/с", th.sources["min_peak_velocity_deg_s"], th.model_note),
            "порог скорости на фиксации": Param(
                th.fixation_velocity_limit_deg_s, "град/с", th.sources["fixation_velocity_limit_deg_s"],
                th.fixation_note if th.sources["fixation_velocity_limit_deg_s"] == FROM_MODEL else ""),
            "минимальная амплитуда": Param(
                th.min_amplitude_deg, "град", th.sources["min_amplitude_deg"],
                "угол большого круга между границами события"),
            "минимальная длительность события": Param(
                float(min_event_duration_s) * 1000, "мс", FROM_USER),
            "минимальный наклон тренда": Param(
                trend_slope_min, "град/с",
                FROM_USER if trend_slope_min is not None else FROM_DEFAULT),
            "минимальная длительность фиксации": Param(
                th.min_fixation_duration_s * 1000, "мс", FROM_MODEL,
                ("короткие отброшены" if drop_short_fixations else "только помечаются")
                + f": {n_short} шт."),
            "допустимый разброс фиксации": Param(
                th.spread_max_deg, "град", th.spread_source,
                f"мера «{spread_measure}»; критерий в разработке: "
                + ("невалидные отброшены" if drop_invalid_spread
                   else "только помечаются")
                + f": {n_wide} шт."),
            "найдено": Param(f"{len(gaze_shifts)} переводов, {len(fixations)} фиксаций",
                             source=COMPUTED,
                             note=f"коротких: {n_short}, широких: {n_wide}"),
        },
    )
    gaze_shifts.attrs["params_used"] = params_used
    fixations.attrs["params_used"] = params_used

    return DirectionMarkup(
        eccentricity_deg=eccentricity,
        azimuth_deg=azimuth,
        speed_deg_s=speed,
        gaze_shifts=gaze_shifts,
        fixations=fixations,
        fs_hz=float(sampling.fs_hz),
        neutral_direction=neutral,
        params_used=params_used,
    )
