############
# Разметка взора
# Оптимизированная версия
############
from __future__ import annotations

import logging as _logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, QhullError
from scipy.special import betainc, expit

from .models import N_SAMPLES_SACCADE, min_saccade_minima_by_sampling
from .report import (
    COMPUTED,
    FROM_DATA,
    FROM_DEFAULT,
    FROM_MODEL,
    FROM_USER,
    Param,
    get_logger,
    log_computation,
)
from .sampling import estimate_sampling_rate, require_seconds

_LOG = get_logger()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False

# Numba / Joblib are optional accelerators.
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

#: Канонические имена осей внутри расчёта. Наружу они переименовываются в имена
#: пользовательских колонок (см. `axis_column_map` и `rename_axis_columns`),
#: но вся механика внутри работает только с ними — иначе схема результата
#: зависела бы от того, как пользователь назвал столбцы (пункт A-1 аудита).
#:
#: Имена предметные, а не порядковые: по колонке `gaze_x_L` сразу видно, что это
#: амплитуда сигмоиды по первой координате взора, а не «что-то по оси номер один».
AXIS_IDS: tuple[str, str] = ("gaze_x", "gaze_y")

#: Колонки результата разметки, которые НЕ привязаны к осям и всегда называются одинаково.
#: Константы критерия разбиения интервала на подпроцессы и критерия склейки.
#: Пункт C-9 аудита: их происхождение не задокументировано, обоснование ожидается
#: от автора методики. До тех пор они выведены в параметры функций разбиения
#: и склейки, чтобы их можно было менять и чтобы было видно, где они применяются.
#:
#: SPLIT_SCORE_MIN_WITH_PVALUE — минимальная мера различия при значимом p-value;
#: SPLIT_SCORE_MIN_ALONE      — мера различия, достаточная сама по себе;
#: SPLIT_COV_WEIGHT           — вес ковариационного члена в мере различия;
#: SPLIT_ALPHA                — уровень значимости для четырёх статистических тестов;
#: SPLIT_MIN_POINTS           — минимум валидных точек в каждой половине;
#: MERGE_STAT_TOLERANCE       — допустимое относительное отличие статистик при склейке.
SPLIT_SCORE_MIN_WITH_PVALUE: float = 1.0
SPLIT_SCORE_MIN_ALONE: float = 2.5
SPLIT_COV_WEIGHT: float = 0.5
SPLIT_ALPHA: float = 0.01
SPLIT_MIN_POINTS: int = 5
MERGE_STAT_TOLERANCE: float = 0.2

#: Прежние значения окон в ОТСЧЁТАХ. Используются, когда пользователь не задал
#: окно в секундах: так дефолтное поведение остаётся тем же, что до перехода
#: на секунды, без пересчёта через частоту (пункт M-3 аудита).
LEGACY_MIN_PEAK_WIDTH_POINTS: int = 3
LEGACY_GAUSSIAN_SIGMA_POINTS: float = 1.5

SHARED_EVENT_COLUMNS: tuple[str, ...] = (
    "t_event_s", "event_type", "t_start_s", "t_end_s", "duration_s", "amplitude_deg",
)


def axis_column_map(col_gaze_x: str, col_gaze_y: str) -> dict[str, str]:
    """Отображение «каноническая ось -> имя колонки пользователя».

    Публичная функция: по ней можно построить ожидаемую схему результата,
    не запуская разметку.

    >>> axis_column_map("gaze_yaw", "gaze_pitch")
    {'gaze_x': 'gaze_yaw', 'gaze_y': 'gaze_pitch'}
    """
    if str(col_gaze_x) == str(col_gaze_y):
        raise ValueError(
            f"col_gaze_x и col_gaze_y должны различаться, получено обе '{col_gaze_x}': "
            "иначе колонки результата обеих осей совпадут и перезапишут друг друга"
        )
    return {AXIS_IDS[0]: str(col_gaze_x), AXIS_IDS[1]: str(col_gaze_y)}


def rename_axis_columns(
        df: pd.DataFrame,
        mapping: dict[str, str],
        *,
        to: str = "source",
) -> pd.DataFrame:
    """Переименовывает пер-осевые колонки между канонической и пользовательской схемой.

    `to="source"`   : `gaze_x_L` -> `<col_gaze_x>_L`, `has_gaze_x` -> `has_<col_gaze_x>`;
    `to="canonical"`: обратное преобразование.

    Общие колонки (`SHARED_EVENT_COLUMNS`) не затрагиваются никогда.
    """
    if to not in ("source", "canonical"):
        raise ValueError(f"to должно быть 'source' или 'canonical', получено {to!r}")

    if to == "source":
        pairs = [(axis, mapping[axis]) for axis in AXIS_IDS if axis in mapping]
    else:
        # Сначала более длинные имена: иначе имя, являющееся префиксом другого,
        # перехватит переименование не своей колонки.
        pairs = sorted(
            ((mapping[axis], axis) for axis in AXIS_IDS if axis in mapping),
            key=lambda p: len(p[0]),
            reverse=True,
        )

    renames: dict[str, str] = {}
    for src, dst in pairs:
        for col in df.columns:
            if col in SHARED_EVENT_COLUMNS or col in renames:
                continue
            if col == f"has_{src}":
                renames[col] = f"has_{dst}"
            elif col.startswith(f"{src}_"):
                renames[col] = f"{dst}_{col[len(src) + 1:]}"

    return df.rename(columns=renames) if renames else df


@dataclass
class SigmoidFitResult:
    params: dict[str, float]  # {'L','k','t0','y0'}
    cov: np.ndarray  # 4x4 covariance matrix (NaN if failed)
    y_hat: np.ndarray  # fitted values on the SAME t passed to fit()
    r2: float  # R^2 on valid points
    ss_res: float  # sum of squared residuals (on valid points)
    success: bool
    message: str


# вспомогательная для: _build_slow_movements_between_events, build_stable_intervals_between_events, merge_fits, peaks_to_sigmoid_fits, resample_uniform_Ts_nan_event_ffill, stable_intervals_between_events
def _sort_frame(df: Any, by: str) -> pd.DataFrame:
    """Сортирует таблицу по столбцу `by` и сбрасывает индекс."""
    return cast(pd.DataFrame, cast(Any, df).sort_values(by).reset_index(drop=True))


# вспомогательная для: interval_stats_by_time, merge_fits, resample_uniform_Ts_nan_event_ffill
def _float_array(values: Any) -> np.ndarray:
    """Преобразует вход в `float`-массив `np.ndarray`."""
    return np.asarray(pd.Series(values, copy=False), dtype=float)


# вспомогательная для: _show_sigmoid_peak_debug_plot, merge_fits, _build_slow_movements_between_events
def _float_or_nan(value: Any) -> float:
    """Преобразует значение к `float`, а при ошибке возвращает `np.nan`."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# вспомогательная для: peaks_to_sigmoid_fits, _mark_eye_data_with_sigmoid_planar_impl
def _progress_print(verbose: bool | int, message: str) -> None:
    """Пишет сообщение о ходе расчёта в журнал библиотеки.

    Библиотека не печатает в stdout: при `verbose` сообщение идёт уровнем INFO,
    иначе DEBUG. Чтобы увидеть их в блокноте, вызовите
    `eyetracking_analytics.report.enable_report_logging()`.
    """
    _LOG.log(_logging.INFO if verbose else _logging.DEBUG, message)


# вспомогательная для: _mark_eye_data_with_sigmoid_planar_impl
def _show_sigmoid_peak_debug_plot(
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        dg_x: np.ndarray,
        dg_y: np.ndarray,
        gaze_x_fits: pd.DataFrame,
        gaze_y_fits: pd.DataFrame,
        events: pd.DataFrame,
) -> None:
    """Показывает отладочный график координат `[deg]`, времени `[s]` и скоростей `[deg/s]`.

    Используется только в sigmoid-ветке при `debug_plot_peaks=True`.
    """
    if not HAS_PLOTLY or go is None or make_subplots is None:
        _LOG.warning("отладочный график пропущен: plotly не установлен")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Coordinates", "Absolute derivatives and peaks"),
    )

    fig.add_trace(
        go.Scatter(x=t, y=x, mode="lines", name="x", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", name="y", line=dict(color="#ff7f0e")),
        row=1,
        col=1,
    )

    abs_dg_x = np.abs(np.asarray(dg_x, dtype=float))  # [deg/s]
    abs_dg_y = np.abs(np.asarray(dg_y, dtype=float))  # [deg/s]
    fig.add_trace(
        go.Scatter(x=t, y=abs_dg_x, mode="lines", name="|dx/dt|", line=dict(color="#2ca02c")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=abs_dg_y, mode="lines", name="|dy/dt|", line=dict(color="#d62728")),
        row=2,
        col=1,
    )

    for fits, coord_name, color in (
        (gaze_x_fits, "x peaks", "#17becf"),
        (gaze_y_fits, "y peaks", "#e377c2"),
    ):
        if fits.empty or "i_peak" not in fits.columns:
            continue
        peak_idx = _float_array(pd.to_numeric(fits["i_peak"], errors="coerce"))  # [points]
        peak_idx = peak_idx[np.isfinite(peak_idx)].astype(int)
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < len(t))]
        if peak_idx.size == 0:
            continue
        source = abs_dg_x if coord_name.startswith("x") else abs_dg_y  # [deg/s]
        coord_source = x if coord_name.startswith("x") else y  # [deg]
        fig.add_trace(
            go.Scatter(
                x=t[peak_idx],
                y=coord_source[peak_idx],
                mode="markers",
                name=f"{coord_name} markup",
                marker=dict(color=color, size=11, symbol="star"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t[peak_idx],
                y=source[peak_idx],
                mode="markers",
                name=coord_name,
                marker=dict(color=color, size=10, symbol="star"),
            ),
            row=2,
            col=1,
        )

    if not events.empty:
        for _, row in events.iterrows():
            t0 = _float_or_nan(row.get("t_start_s"))
            t1 = _float_or_nan(row.get("t_end_s"))
            if not np.isfinite(t0) or not np.isfinite(t1):
                continue
            for subplot_row in (1, 2):
                cast(Any, fig).add_vrect(
                    x0=min(t0, t1),
                    x1=max(t0, t1),
                    fillcolor="rgba(128, 128, 128, 0.15)",
                    line_width=0,
                    row=subplot_row,
                    col=1,
                )

    fig.update_xaxes(title_text="t, s", row=2, col=1)
    fig.update_yaxes(title_text="coord", row=1, col=1)
    fig.update_yaxes(title_text="speed", row=2, col=1)
    fig.update_layout(
        title="Sigmoid peak debug view",
        template="plotly_white",
        hovermode="x unified",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.show()


# вспомогательная для: fit_sigmoid_logistic
@jit(nopython=True)
def _logistic_numba(t, L, k, t0, y0):
    val = -k * (t - t0)
    # Clip to avoid overflow in exp
    val = np.minimum(val, 700.0)
    val = np.maximum(val, -700.0)
    return y0 + L / (1.0 + np.exp(val))


# вспомогательная для: fit_sigmoid_logistic
def _logistic(t: np.ndarray, L: float, k: float, t0: float, y0: float) -> np.ndarray:
    """
    Вычисляет значение логистической функции (сигмоиды) в точках t.
    """
    return y0 + L * expit(k * (t - t0))


# вспомогательная для: _fit_single_peak_candidate
def fit_sigmoid_logistic(
        t: np.ndarray,
        y: np.ndarray,
        *,
        maxfev: int = 12_000,
        bounds: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
        n_starts: int = 6,  # сколько стартов пробовать
        r2_min: float | None = None,  # если задано, то success=True только при r2>=r2_min
) -> SigmoidFitResult:
    """
    Подбирает параметры логистической сигмоиды по точкам `(t, y)`.

    Размерности: `t` `[s]`, `y` `[deg]`, параметры сигмоиды:
    `L` `[deg]`, `k` `[1/s]`, `t0` `[s]`, `y0` `[deg]`.

    Использует несколько стартовых приближений:
    сначала пробует "умную" начальную оценку по данным, затем
    дополняет ее сеткой стартов и выбирает решение с минимальным `SS_res`.
    Если задан `r2_min`, то флаг `success` выставляется только при
    достаточном качестве аппроксимации.
    """

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError("t and y must have the same length.")

    # Сначала упорядочиваем точки по времени.
    order = np.argsort(t)
    t_fit = t[order]
    y_fit = y[order]

    t_min, t_max = float(np.min(t_fit)), float(np.max(t_fit))
    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))
    y_span = max(y_max - y_min, 1e-6)

    # Границы параметров по умолчанию подстраиваются под масштаб данных.
    if bounds is None:
        lower = (0.0, -np.inf, t_min, y_min - 2.0 * y_span)
        upper = (10.0 * y_span, np.inf, t_max, y_max + 2.0 * y_span)
        bounds = (lower, upper)

    # Формируем набор стартовых приближений для multi-start оптимизации.
    p0_list: list[tuple[float, float, float, float]] = []

    # Start from a robust heuristic guess before multi-start refinement.
    try:
        y_start_est = float(np.median(y_fit[:3])) if len(y_fit) > 3 else float(y_fit[0])
        y_end_est = float(np.median(y_fit[-3:])) if len(y_fit) > 3 else float(y_fit[-1])
        L_smart = y_end_est - y_start_est
        y0_smart = y_start_est

        mid_val = y0_smart + 0.5 * L_smart
        idx_mid = int(np.abs(y_fit - mid_val).argmin())
        t0_smart = float(t_fit[idx_mid])

        k_smart = 1.0
        if abs(L_smart) > 1e-6:
            i0 = max(0, idx_mid - 1)
            i1 = min(len(t_fit) - 1, idx_mid + 1)
            if i1 > i0:
                slope = (y_fit[i1] - y_fit[i0]) / (t_fit[i1] - t_fit[i0])
                k_est = 4.0 * slope / L_smart
                if np.isfinite(k_est) and abs(k_est) > 1e-3 and abs(k_est) < 1000.0:
                    k_smart = k_est

        p0_list.append((L_smart, k_smart, t0_smart, y0_smart))
    except Exception:
        pass

    if r2_min is not None:
        n_starts = min(int(n_starts), 4)

    # Далее добавляем более широкую сетку стартовых точек.
    y_lo = float(np.nanpercentile(y_fit, 10))
    y_hi = float(np.nanpercentile(y_fit, 90))
    L_base = max(y_hi - y_lo, 1e-6)
    y0_base = y_lo

    q = np.linspace(0.15, 0.85, max(n_starts // 2, 4))
    t0_candidates = [float(np.quantile(t_fit, qq)) for qq in q]

    t_span = max(t_max - t_min, 1e-6)
    k_scales = np.array([2.0, 4.0, 8.0]) / t_span
    k_candidates: list[float] = []
    for ks in k_scales:
        k_candidates.extend([float(ks), float(-ks)])
    k_candidates.extend([float(16.0 / t_span), float(-16.0 / t_span)])

    L_candidates = [L_base, 0.7 * L_base, 1.3 * L_base]
    y0_candidates = [y0_base, y0_base - 0.2 * y_span, y0_base + 0.2 * y_span]

    grid_p0 = []
    for t0c in t0_candidates:
        for kc in k_candidates:
            for Lc in L_candidates:
                for y0c in y0_candidates:
                    grid_p0.append((float(Lc), float(kc), float(t0c), float(y0c)))

    if len(grid_p0) > n_starts:
        idx = np.linspace(0, len(grid_p0) - 1, n_starts, dtype=int)
        grid_p0 = [grid_p0[i] for i in idx]

    p0_list.extend(grid_p0)

    best = None  # (ss_res, r2, popt, pcov)
    best_err = None

    EARLY_EXIT_R2 = float(r2_min) if r2_min is not None else 0.99

    for p0 in p0_list:
        try:
            popt, pcov = curve_fit(
                _logistic,
                t_fit,
                y_fit,
                p0=p0,
                bounds=bounds,
                maxfev=maxfev,
            )
            L, k, t0_est, y0_est = map(float, popt)

            if HAS_NUMBA:
                y_hat_fit = _logistic_numba(t_fit, L, k, t0_est, y0_est)
            else:
                y_hat_fit = _logistic(t_fit, L, k, t0_est, y0_est)

            resid = y_fit - y_hat_fit
            ss_res = float(np.sum(resid * resid))

            ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            if (best is None) or (ss_res < best[0]):
                best = (ss_res, r2, popt, pcov)

            if r2 >= EARLY_EXIT_R2:
                break

        except Exception as e:
            best_err = e
            continue

    if best is None:
        return SigmoidFitResult(
            params={"L": np.nan, "k": np.nan, "t0": np.nan, "y0": np.nan},
            cov=np.full((4, 4), np.nan),
            y_hat=np.full_like(t, np.nan, dtype=float),
            r2=np.nan,
            ss_res=np.inf,
            success=False,
            message=f"Fit failed for all starts: {type(best_err).__name__}: {best_err}" if best_err else "Fit failed for all starts.",
        )

    ss_res, r2, popt, pcov = best
    L, k, t0_est, y0_est = map(float, popt)

    y_hat = np.full_like(t, np.nan, dtype=float)
    if HAS_NUMBA:
        y_hat = _logistic_numba(t, L, k, t0_est, y0_est)
    else:
        y_hat = _logistic(t, L, k, t0_est, y0_est)

    ok = True
    msg = "OK (best SS_res)"
    if r2_min is not None:
        ok = bool(np.isfinite(r2) and (r2 >= float(r2_min)))
        if not ok:
            msg = f"Best SS_res found, but low R2: {r2:.3f} < {float(r2_min):.3f}"

    return SigmoidFitResult(
        params={"L": float(L), "k": float(k), "t0": float(t0_est), "y0": float(y0_est)},
        cov=pcov,
        y_hat=y_hat,
        r2=float(r2) if np.isfinite(r2) else np.nan,
        ss_res=float(ss_res),
        success=ok,
        message=msg,
    )


# вспомогательная для: _build_saccades, _build_sigmoid_peak_records
def _nearest_peak_bounds_by_threshold(arr: np.ndarray, i: int, th: float) -> tuple[int, int]:
    """Возвращает ближайшие к пику границы, где сигнал все еще выше порога."""
    n = len(arr)
    if n == 0:
        return i, i

    i = max(0, min(int(i), n - 1))
    if not np.isfinite(arr[i]):
        return i, i

    left = i
    while left > 0:
        prev_val = arr[left - 1]
        if not np.isfinite(prev_val) or prev_val < th:
            break
        left -= 1

    right = i
    while right < n - 1:
        next_val = arr[right + 1]
        if not np.isfinite(next_val) or next_val < th:
            break
        right += 1

    return left, right
# вспомогательная для: _make_fixation_record
def _interval_stats_by_indices(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        i_start: int,
        i_end: int,
        *,
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.0,
):
    """
    Считает статистики на полуинтервале индексов `[i_start, i_end]`.

    Размерности: `t` `[s]`, `x/y` `[deg]`, индексы `[points]`,
    `slope_x/slope_y` `[deg/s]`.

    Возвращает средние, ковариацию, число валидных точек и признаки тренда
    по координатам `x` и `y`.
    """
    if i_end < i_start:
        return (np.nan, np.nan, np.full((2, 2), np.nan), 0,
                np.nan, np.nan, np.nan, np.nan, True)

    t_seg = np.asarray(t[i_start:i_end + 1], dtype=float)
    x_seg = np.asarray(x[i_start:i_end + 1], dtype=float)
    y_seg = np.asarray(y[i_start:i_end + 1], dtype=float)

    m = np.isfinite(t_seg) & np.isfinite(x_seg) & np.isfinite(y_seg)
    n_valid = int(m.sum())
    if n_valid < 3:
        return (np.nan, np.nan, np.full((2, 2), np.nan), n_valid,
                np.nan, np.nan, np.nan, np.nan, True)

    t_seg = t_seg[m]
    x_seg = x_seg[m]
    y_seg = y_seg[m]

    ex = float(np.mean(x_seg))
    ey = float(np.mean(y_seg))
    cov = np.cov(x_seg, y_seg, ddof=1)

    slope_x, p_x, _ = ols_slope_pvalue(t_seg, x_seg)
    slope_y, p_y, _ = ols_slope_pvalue(t_seg, y_seg)

    has_trend_x = (np.isfinite(p_x) and (p_x < trend_alpha) and (abs(slope_x) > trend_slope_min))
    has_trend_y = (np.isfinite(p_y) and (p_y < trend_alpha) and (abs(slope_y) > trend_slope_min))
    has_trend = bool(has_trend_x or has_trend_y)

    return ex, ey, cov, n_valid, slope_x, p_x, slope_y, p_y, has_trend


def _f_sf(w: float, dof1: float, dof2: float) -> float:
    """Правый хвост распределения Фишера через `betainc`.

    `F.sf(w; d1, d2) = I_{d2/(d2 + d1·w)}(d2/2, d1/2)` — тождество, позволяющее
    обойтись без `scipy.stats`, диспетчер которого доминировал в профиле разметки.
    """
    if not np.isfinite(w) or dof1 <= 0 or dof2 <= 0:
        return np.nan
    if w <= 0:
        return 1.0
    return float(betainc(0.5 * dof2, 0.5 * dof1, dof2 / (dof2 + dof1 * w)))


def _degenerate_scale(a: np.ndarray, b: np.ndarray) -> float:
    """Масштаб для допуска вырожденных случаев (как в прежней реализации)."""
    return max(
        float(np.ptp(a)) if a.size > 0 else 0.0,
        float(np.ptp(b)) if b.size > 0 else 0.0,
        abs(float(np.mean(a))) if a.size > 0 else 0.0,
        abs(float(np.mean(b))) if b.size > 0 else 0.0,
        1.0,
    )


# вспомогательная для: _best_homogeneity_split
def _safe_ttest_ind_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """p-value двустороннего t-теста Уэлча (неравные дисперсии).

    Замкнутая форма вместо `scipy.stats.ttest_ind(equal_var=False)`. Значения
    совпадают до 1e-12 на 3000 случайных выборок и побитово воспроизводят прежнее
    поведение на вырожденных и квантованных данных, но без накладных расходов
    диспетчера scipy, который занимал большую часть времени разметки.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1 = a.size
    n2 = b.size
    if n1 < 2 or n2 < 2:
        return np.nan

    atol = 1e-12 * _degenerate_scale(a, b)
    m1 = float(np.mean(a))
    m2 = float(np.mean(b))
    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))

    if np.allclose(m1, m2, atol=atol, rtol=1e-9) and np.var(a) <= atol and np.var(b) <= atol:
        return 1.0

    s1 = v1 / n1
    s2 = v2 / n2
    denom = s1 + s2
    if denom <= 0:
        # Обе выборки постоянны: различие есть тогда и только тогда,
        # когда различаются сами значения.
        return 1.0 if m1 == m2 else 0.0

    t_stat = (m1 - m2) / np.sqrt(denom)
    dof_den = (s1 * s1) / (n1 - 1) + (s2 * s2) / (n2 - 1)
    if dof_den <= 0:
        return np.nan
    return _student_two_sided_pvalue(float(t_stat), denom * denom / dof_den)


# вспомогательная для: _best_homogeneity_split
def _safe_levene_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """p-value теста Левена (центрирование по медиане) для двух выборок.

    Замкнутая форма вместо `scipy.stats.levene(center="median")`: тест Левена —
    однофакторный дисперсионный анализ абсолютных отклонений от медианы группы,
    поэтому статистика и её распределение выписываются напрямую.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1 = a.size
    n2 = b.size
    if n1 < 2 or n2 < 2:
        return np.nan

    atol = 1e-12 * _degenerate_scale(a, b)
    if float(np.var(a)) <= atol and float(np.var(b)) <= atol:
        return 1.0

    z1 = np.abs(a - float(np.median(a)))
    z2 = np.abs(b - float(np.median(b)))
    zbar1 = float(np.mean(z1))
    zbar2 = float(np.mean(z2))
    n_total = n1 + n2
    zbar = (zbar1 * n1 + zbar2 * n2) / n_total

    numer = (n_total - 2) * (n1 * (zbar1 - zbar) ** 2 + n2 * (zbar2 - zbar) ** 2)
    denom = float(np.sum((z1 - zbar1) ** 2) + np.sum((z2 - zbar2) ** 2))
    if denom <= 0:
        return 1.0 if numer <= 0 else 0.0

    return _f_sf(numer / denom, 1.0, float(n_total - 2))


# вспомогательная для: _build_fixations_between_saccades
def _best_homogeneity_split(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        i_start: int,
        i_end: int,
        *,
        min_duration_s: float,
        min_points: int = SPLIT_MIN_POINTS,
        alpha_split: float = SPLIT_ALPHA,
        score_min_with_pvalue: float = SPLIT_SCORE_MIN_WITH_PVALUE,
        score_min_alone: float = SPLIT_SCORE_MIN_ALONE,
        cov_weight: float = SPLIT_COV_WEIGHT,
) -> tuple[int | None, float, float]:
    """
    Ищет лучший индекс разбиения неоднородного интервала на два подпроцесса.

    Размерности: `t` `[s]`, `x/y` `[deg]`, индексы `[points]`,
    `min_duration_s` `[s]`.

    В качестве критерия использует различие средних и ковариаций,
    а также p-value статистических тестов для левой и правой частей:

        score = ‖Δμ‖ / (√tr cov_L + √tr cov_R)
              + cov_weight · ‖Δcov‖_F / (‖cov_L‖_F + ‖cov_R‖_F)

    Разбиение считается значимым, если минимальное p-value четырёх тестов
    (Уэлч и Левен по каждой координате) ниже `alpha_split` при
    `score > score_min_with_pvalue`, либо если `score > score_min_alone`
    независимо от p-value.

    Пункт C-9 аудита: происхождение значений `score_min_with_pvalue = 1.0`,
    `score_min_alone = 2.5`, `cov_weight = 0.5`, `alpha_split = 0.01`
    и `min_points = 5` не задокументировано. Они вынесены в параметры,
    чтобы было видно, где применяются, и чтобы их можно было изменить
    без правки кода; обоснование ожидается от автора методики.
    """
    if i_end - i_start + 1 < 2 * int(min_points):
        return None, np.nan, np.nan

    best_k = None
    best_score = -np.inf
    best_p = np.nan

    for k in range(i_start + int(min_points) - 1, i_end - int(min_points) + 1):
        if float(t[k] - t[i_start]) < float(min_duration_s):
            continue
        if float(t[i_end] - t[k + 1]) < float(min_duration_s):
            continue

        xl = np.asarray(x[i_start:k + 1], dtype=float)
        yl = np.asarray(y[i_start:k + 1], dtype=float)
        xr = np.asarray(x[k + 1:i_end + 1], dtype=float)
        yr = np.asarray(y[k + 1:i_end + 1], dtype=float)

        ml = np.isfinite(xl) & np.isfinite(yl)
        mr = np.isfinite(xr) & np.isfinite(yr)
        if int(ml.sum()) < int(min_points) or int(mr.sum()) < int(min_points):
            continue

        xl = xl[ml]
        yl = yl[ml]
        xr = xr[mr]
        yr = yr[mr]

        mu_l = np.array([np.mean(xl), np.mean(yl)], dtype=float)
        mu_r = np.array([np.mean(xr), np.mean(yr)], dtype=float)
        cov_l = np.cov(xl, yl, ddof=1)
        cov_r = np.cov(xr, yr, ddof=1)

        mean_diff = float(np.linalg.norm(mu_l - mu_r))
        cov_scale = max(float(np.linalg.norm(cov_l, ord="fro") + np.linalg.norm(cov_r, ord="fro")), 1e-9)
        cov_diff = float(np.linalg.norm(cov_l - cov_r, ord="fro"))
        spread = max(float(np.sqrt(max(np.trace(cov_l), 0.0)) + np.sqrt(max(np.trace(cov_r), 0.0))), 1e-9)

        score = mean_diff / spread + cov_weight * cov_diff / cov_scale

        # Оба теста считаются замкнутыми формами, поэтому критерий больше
        # не зависит от наличия scipy.stats (пункт N-2 аудита: раньше без него
        # он молча деградировал до чисто эвристического score > 2.5).
        pvals = [
            _safe_ttest_ind_pvalue(xl, xr),
            _safe_ttest_ind_pvalue(yl, yr),
            _safe_levene_pvalue(xl, xr),
            _safe_levene_pvalue(yl, yr),
        ]

        pvals = [p for p in pvals if np.isfinite(p)]
        p_min = float(min(pvals)) if len(pvals) > 0 else np.nan

        is_significant = (
            (np.isfinite(p_min) and p_min < float(alpha_split)
             and score > float(score_min_with_pvalue))
            or (score > float(score_min_alone))
        )
        if not is_significant:
            continue

        if score > best_score:
            best_k = k
            best_score = score
            best_p = p_min

    if best_k is None:
        return None, np.nan, np.nan

    return int(best_k), float(best_score), float(best_p) if np.isfinite(best_p) else np.nan


# вспомогательная для: _can_merge_fixation_records
def _relative_change_within(value_a: float, value_b: float, tol: float = 0.2) -> bool:
    """Проверяет, что относительное отличие двух величин не превышает `tol`."""
    if not np.isfinite(value_a) and not np.isfinite(value_b):
        return True
    if not np.isfinite(value_a) or not np.isfinite(value_b):
        return False

    scale = max(abs(float(value_a)), abs(float(value_b)), 1e-9)
    return abs(float(value_a) - float(value_b)) <= float(tol) * scale


#: Режимы измерения пространственного разброса фиксации.
SPREAD_MEASURES: tuple[str, ...] = ("std", "p95", "peak")


#: Уровень покрытия эллипса рассеяния по умолчанию. 0.90 отвечает k = ln(10)
#: в исходной формуле проекта по биатлону, откуда мера перенесена.
DEFAULT_BCEA_PROBABILITY: float = 0.90


def _bcea_from_cov(cov: np.ndarray, probability: float) -> float:
    """Площадь эллипса рассеяния по готовой матрице ковариации, град^2."""
    if cov is None or not np.isfinite(cov).all():
        return float("nan")
    det = float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0])
    if det <= 0.0:
        # Вырожденное облако: точки лежат на прямой либо det ушёл в минус
        # на численной погрешности. Площади у такой фигуры нет, но и ошибкой
        # это не является -- интервал просто не даёт эллипса.
        return 0.0
    k = -float(np.log1p(-probability))
    return float(2.0 * k * np.pi * np.sqrt(det))


def bcea_deg2(
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
        *,
        probability: float = DEFAULT_BCEA_PROBABILITY,
) -> float:
    """Площадь эллипса рассеяния взора (BCEA), град^2.

    Мера разброса, дополняющая `spread_*`: те дают радиус облака в градусах,
    эта -- площадь области, которую заметает точка взора, в квадратных градусах.
    В отличие от радиуса, площадь учитывает вытянутость и наклон облака: два
    интервала с одинаковым СКО расстояния, но разной формой дают разную площадь.

        BCEA = 2 * k * pi * sqrt(det Cov),   k = -ln(1 - probability)

    `probability` -- доля точек, которую эллипс должен накрывать при нормальном
    распределении: 0.90 по умолчанию, что отвечает `k = ln(10)`.

    Ковариация несмещённая (`ddof=1`). Меньше трёх валидных точек -- `NaN`:
    по двум точкам площадь не определена. Вырожденное облако (точки на прямой)
    даёт 0.

    Перенесено из проекта по биатлону, где так считалась площадь области
    прицеливания; формула та же с точностью до перевода единиц -- там
    сантиметры на плоскости, здесь градусы на сфере.
    """
    xa = np.asarray(gaze_x, dtype=float)
    ya = np.asarray(gaze_y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if int(m.sum()) < 3:
        return float("nan")
    return _bcea_from_cov(np.cov(xa[m], ya[m], ddof=1), probability)


def convex_hull_area_deg2(gaze_x: np.ndarray, gaze_y: np.ndarray) -> float:
    """Площадь выпуклой оболочки облака точек взора, град^2.

    Вторая мера площади, рядом с `bcea_deg2`, и смысл у них разный:

    * BCEA -- эллипс равной плотности: модельная область, куда попадает
      заданная доля точек при нормальном распределении. Одиночный выброс
      сдвигает её слабо, зато на ненормальном облаке она врёт;
    * оболочка -- граница **фактически посещённых** точек. Ничего
      о распределении не предполагает, но одиночный выброс раздувает её
      сколь угодно сильно.

    Поэтому обе считаются и ни одна не заменяет другую: расхождение между ними
    само по себе информативно -- оболочка много больше эллипса означает выбросы
    либо облако, далёкое от нормального.

    Меньше трёх валидных точек -- `NaN`. Вырожденное облако (все точки
    на одной прямой) даёт 0: у отрезка площади нет.
    """
    xa = np.asarray(gaze_x, dtype=float)
    ya = np.asarray(gaze_y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if int(m.sum()) < 3:
        return float("nan")
    pts = np.column_stack([xa[m], ya[m]])
    try:
        # В двумерном случае ConvexHull.volume -- это площадь, а .area -- периметр.
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def _spread_measures(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Три меры пространственного разброса интервала, все в градусах.

    * `std`  — СКО расстояния от среднего положения (1σ облака точек);
    * `p95`  — 95-й процентиль расстояния от среднего (эмпирический, без
               предположения о нормальности распределения);
    * `peak` — максимальное расстояние от среднего.

    Возвращаются все три: выбор режима — вопрос методики (пункт C-19), а считать
    их вместе дешевле, чем пересчитывать при смене режима.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    if int(m.sum()) < 3:
        return np.nan, np.nan, np.nan
    xa = xa[m]
    ya = ya[m]
    r = np.hypot(xa - xa.mean(), ya - ya.mean())
    return (
        float(np.sqrt(np.mean(r * r))),
        float(np.percentile(r, 95.0)),
        float(np.max(r)),
    )


# вспомогательная для: _build_fixations_between_saccades, _can_merge_fixation_records
def _make_fixation_record(
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        alpha_deg: np.ndarray,
        v_ang: np.ndarray,
        i_start: int,
        i_end: int,
        trend_alpha: float,
        trend_slope_min: float,
        heterogeneous_parent: bool,
        split_score: float,
        split_pvalue: float,
        split_depth: int,
        with_area: bool = False,
) -> dict:
    mean_speed = float(np.nanmean(v_ang[i_start:i_end + 1]))
    mean_alpha = float(np.nanmean(alpha_deg[i_start:i_end + 1]))
    ex, ey, cov, n_valid, slope_x, p_x, slope_y, p_y, has_trend = _interval_stats_by_indices(
        t, x, y, i_start, i_end, trend_alpha=trend_alpha, trend_slope_min=trend_slope_min
    )
    fro_cov = float(np.linalg.norm(cov, ord="fro")) if np.isfinite(cov).all() else np.nan
    spread_std, spread_p95, spread_peak = _spread_measures(
        x[i_start:i_end + 1], y[i_start:i_end + 1])

    record = {
        "_i_start": int(i_start),
        "_i_end": int(i_end),
        "t_start_s": float(t[i_start]),
        "t_end_s": float(t[i_end]),
        "mean_speed_deg_s": mean_speed,
        "mean_eccentricity_deg": mean_alpha,
        "has_trend": bool(has_trend),
        "slope_gaze_x_deg_s": float(slope_x) if np.isfinite(slope_x) else np.nan,
        "p_gaze_x": float(p_x) if np.isfinite(p_x) else np.nan,
        "slope_gaze_y_deg_s": float(slope_y) if np.isfinite(slope_y) else np.nan,
        "p_gaze_y": float(p_y) if np.isfinite(p_y) else np.nan,
        "n_valid": int(n_valid),
        "mean_gaze_x": float(ex) if np.isfinite(ex) else np.nan,
        "mean_gaze_y": float(ey) if np.isfinite(ey) else np.nan,
        "cov_fro_norm_deg2": fro_cov,
        "spread_std_deg": spread_std,
        "spread_p95_deg": spread_p95,
        "spread_peak_deg": spread_peak,
        "heterogeneous_parent": bool(heterogeneous_parent),
        "split_score": float(split_score) if np.isfinite(split_score) else np.nan,
        "split_pvalue": float(split_pvalue) if np.isfinite(split_pvalue) else np.nan,
        "split_depth": int(split_depth),
    }
    if with_area:
        # Площади заметаемой области -- характеристики ВЗОРА: они имеют смысл
        # как проекция на неподвижное пространство. Для движения глаза
        # относительно черепа такой проекции нет, поэтому в глазной ветке
        # колонок нет вовсе, а не NaN.
        record["bcea_deg2"] = _bcea_from_cov(cov, DEFAULT_BCEA_PROBABILITY)
        record["hull_area_deg2"] = convex_hull_area_deg2(
            x[i_start:i_end + 1], y[i_start:i_end + 1])
    return record


# вспомогательная для: _merge_adjacent_fixation_records
def _can_merge_fixation_records(
        left: dict,
        right: dict,
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        alpha_deg: np.ndarray,
        v_ang: np.ndarray,
        fixation_velocity_limit_deg_s: float,
        min_amplitude_deg: float,
        max_merge_gap_s: float,
        trend_alpha: float,
        trend_slope_min: float,
        stat_tol: float = MERGE_STAT_TOLERANCE,
        with_area: bool = False,
) -> tuple[bool, dict | None]:
    gap = float(right["t_start_s"]) - float(left["t_end_s"])
    if gap > float(max_merge_gap_s):
        return False, None

    index_gap = int(right["_i_start"]) - int(left["_i_end"])
    if index_gap <= 1:
        low_left = np.isfinite(left["mean_eccentricity_deg"]) and float(left["mean_eccentricity_deg"]) < float(min_amplitude_deg) / 2.0
        low_right = np.isfinite(right["mean_eccentricity_deg"]) and float(right["mean_eccentricity_deg"]) < float(min_amplitude_deg) / 2.0
        if low_left and low_right:
            split_scores = [v for v in [left["split_score"], right["split_score"]] if np.isfinite(v)]
            split_pvalues = [v for v in [left["split_pvalue"], right["split_pvalue"]] if np.isfinite(v)]
            merged = _make_fixation_record(
                t=t,
                x=x,
                y=y,
                alpha_deg=alpha_deg,
                v_ang=v_ang,
                i_start=int(left["_i_start"]),
                i_end=int(right["_i_end"]),
                trend_alpha=trend_alpha,
                trend_slope_min=trend_slope_min,
                heterogeneous_parent=bool(left["heterogeneous_parent"]) or bool(right["heterogeneous_parent"]),
                split_score=max(split_scores) if len(split_scores) > 0 else np.nan,
                split_pvalue=min(split_pvalues) if len(split_pvalues) > 0 else np.nan,
                split_depth=min(int(left["split_depth"]), int(right["split_depth"])),
                with_area=with_area,
            )
            return True, merged

    split_scores = [v for v in [left["split_score"], right["split_score"]] if np.isfinite(v)]
    split_pvalues = [v for v in [left["split_pvalue"], right["split_pvalue"]] if np.isfinite(v)]

    merged = _make_fixation_record(
        t=t,
        x=x,
        y=y,
        alpha_deg=alpha_deg,
        v_ang=v_ang,
        i_start=int(left["_i_start"]),
        i_end=int(right["_i_end"]),
        trend_alpha=trend_alpha,
        trend_slope_min=trend_slope_min,
        heterogeneous_parent=bool(left["heterogeneous_parent"]) or bool(right["heterogeneous_parent"]),
        split_score=max(split_scores) if len(split_scores) > 0 else np.nan,
        split_pvalue=min(split_pvalues) if len(split_pvalues) > 0 else np.nan,
        split_depth=min(int(left["split_depth"]), int(right["split_depth"])),
        with_area=with_area,
    )

    if np.isfinite(merged["mean_speed_deg_s"]) and merged["mean_speed_deg_s"] >= float(fixation_velocity_limit_deg_s):
        return False, None

    keys_to_compare = ["mean_speed_deg_s", "mean_gaze_x", "mean_gaze_y", "cov_fro_norm_deg2"]
    for key in keys_to_compare:
        if not _relative_change_within(merged[key], left[key], tol=stat_tol):
            return False, None
        if not _relative_change_within(merged[key], right[key], tol=stat_tol):
            return False, None

    return True, merged


# вспомогательная для: _build_fixations_between_saccades
def _merge_adjacent_fixation_records(
        records: list[dict],
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        alpha_deg: np.ndarray,
        v_ang: np.ndarray,
        fixation_velocity_limit_deg_s: float,
        min_amplitude_deg: float,
        max_merge_gap_s: float,
        trend_alpha: float,
        trend_slope_min: float,
        stat_tol: float = MERGE_STAT_TOLERANCE,
        with_area: bool = False,
) -> list[dict]:
    if len(records) <= 1:
        return records

    records = sorted(records, key=lambda r: (r["t_start_s"], r["t_end_s"]))
    merged_records: list[dict] = []
    current = records[0]

    for nxt in records[1:]:
        can_merge, merged = _can_merge_fixation_records(
            current,
            nxt,
            t=t,
            x=x,
            y=y,
            alpha_deg=alpha_deg,
            v_ang=v_ang,
            fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
            min_amplitude_deg=min_amplitude_deg,
            max_merge_gap_s=max_merge_gap_s,
            trend_alpha=trend_alpha,
            trend_slope_min=trend_slope_min,
            stat_tol=stat_tol,
            with_area=with_area,
        )
        if can_merge and merged is not None:
            current = merged
        else:
            merged_records.append(current)
            current = nxt

    merged_records.append(current)
    return merged_records


# вспомогательная для: _mark_eye_data_with_full_angle_planar_impl, mark_eye_data_with_full_angle_from_angles
def _trim_candidates_by_velocity(
        candidates: list[tuple[int, int]],
        *,
        t: np.ndarray,
        v_ang: np.ndarray,
        fixation_velocity_limit_deg_s: float,
        max_merge_gap_s: float,
) -> list[tuple[int, int]]:
    """Оставляет в интервале только отсчёты со скоростью ниже порога фиксации.

    Без этого фиксацией становится весь интервал между переводами взора целиком,
    вместе с отсчётами, где скорость в разы выше порога: на биатлонной записи
    192 Гц средняя скорость всех найденных фиксаций оказалась 20…120 град/с при
    пороге 20 град/с, а суммарное время фиксаций — вдвое больше, чем даёт отбор
    по скорости. При этом сам порог участвовал только в запрете склейки.

    Разрыв короче `max_merge_gap_s` нарушением не считается: одиночный выброс
    скорости не делит фиксацию надвое — то же правило, что при склейке.

    Скорость для критерия приходит отдельным массивом (`v_criterion` вызывающей
    функции): порог фиксации относится к смещению за физиологическое окно, а не к
    разности соседних отсчётов. Разность соседних отсчётов на высоких частотах —
    это шум прибора: при шуме положения 0.05 град на 250 Гц она даёт 18 град/с
    при пороге модели 5 град/с, то есть отвергала бы любую фиксацию.

    Размерности: `t` `[s]`, `v_ang` `[deg/s]`, порог `[deg/s]`, разрыв `[s]`.
    """
    limit = float(fixation_velocity_limit_deg_s)
    gap = float(max_merge_gap_s)
    out: list[tuple[int, int]] = []
    for i0, i1 in candidates:
        v = np.asarray(v_ang[i0:i1 + 1], dtype=float)
        idx = np.flatnonzero(np.isfinite(v) & (v <= limit)) + int(i0)
        if idx.size < 2:
            continue
        start = int(idx[0])
        prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            # Разрывом считается только пропуск отсчётов: между соседними отсчётами
            # разрыва нет по определению, иначе при шаге записи крупнее
            # `max_merge_gap_s` (низкие частоты) рвался бы каждый интервал.
            if i - prev > 1 and float(t[i] - t[prev]) >= gap:
                if prev > start:
                    out.append((start, prev))
                start = i
            prev = i
        if prev > start:
            out.append((start, prev))
    return out


def _build_fixations_between_saccades(
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        alpha_deg: np.ndarray,
        v_ang: np.ndarray,
        events: pd.DataFrame,
        fixation_velocity_limit_deg_s: float,
        min_amplitude_deg: float,
        min_interval_duration_s: float,
        min_subprocess_duration_s: float,
        max_merge_gap_s: float,
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.1,
        trim_by_velocity: bool = False,
        v_criterion: np.ndarray | None = None,
        homogeneity_split: bool = True,
        split_alpha: float = SPLIT_ALPHA,
        split_min_points: int = SPLIT_MIN_POINTS,
        split_score_min_with_pvalue: float = SPLIT_SCORE_MIN_WITH_PVALUE,
        split_score_min_alone: float = SPLIT_SCORE_MIN_ALONE,
        merge_stat_tolerance: float = MERGE_STAT_TOLERANCE,
        with_area: bool = False,
) -> pd.DataFrame:
    if len(t) < 2:
        return pd.DataFrame(columns=[
            't_start_s', 't_end_s', 'mean_speed_deg_s', 'mean_eccentricity_deg', 'has_trend',
            'slope_gaze_x_deg_s', 'p_gaze_x', 'slope_gaze_y_deg_s', 'p_gaze_y', 'n_valid', 'mean_gaze_x', 'mean_gaze_y', 'cov_fro_norm_deg2',
            *(['bcea_deg2', 'hull_area_deg2'] if with_area else []),
            'heterogeneous_parent', 'split_score', 'split_pvalue', 'split_depth'
        ])

    candidates: list[tuple[int, int]] = []

    if events.empty:
        candidates.append((0, len(t) - 1))
    else:
        ev = events.sort_values("t_start_s").reset_index(drop=True)

        first_start = float(ev.loc[0, "t_start_s"])
        i_end = int(np.searchsorted(t, first_start, side="right") - 1)
        i_end = max(0, min(i_end, len(t) - 1))
        if i_end > 0:
            candidates.append((0, i_end))

        for i in range(len(ev) - 1):
            t0 = float(ev.loc[i, "t_end_s"])
            t1 = float(ev.loc[i + 1, "t_start_s"])
            i0 = int(np.searchsorted(t, t0, side="left"))
            i1 = int(np.searchsorted(t, t1, side="right") - 1)
            i0 = max(0, min(i0, len(t) - 1))
            i1 = max(0, min(i1, len(t) - 1))
            if i1 > i0:
                candidates.append((i0, i1))

        last_finish = float(ev.loc[len(ev) - 1, "t_end_s"])
        i_start = int(np.searchsorted(t, last_finish, side="left"))
        i_start = max(0, min(i_start, len(t) - 1))
        if i_start < len(t) - 1:
            candidates.append((i_start, len(t) - 1))

    if trim_by_velocity:
        candidates = _trim_candidates_by_velocity(
            candidates,
            t=t,
            v_ang=v_ang if v_criterion is None else v_criterion,
            fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
            max_merge_gap_s=max_merge_gap_s,
        )

    fixation_records: list[dict] = []
    for i0, i1 in candidates:
        if float(t[i1] - t[i0]) < float(min_interval_duration_s):
            continue

        stack = [(i0, i1, 0, False, np.nan, np.nan)]
        while len(stack) > 0:
            j0, j1, split_depth, heterogeneous_parent, local_split_score, local_split_p = stack.pop()
            if j1 < j0:
                continue
            if float(t[j1] - t[j0]) < float(min_interval_duration_s):
                continue

            if homogeneity_split:
                split_k, split_score, split_p = _best_homogeneity_split(
                    t, x, y, j0, j1,
                    min_duration_s=min_subprocess_duration_s,
                    min_points=split_min_points,
                    alpha_split=split_alpha,
                    score_min_with_pvalue=split_score_min_with_pvalue,
                    score_min_alone=split_score_min_alone,
                )
            else:
                # Требование постоянства положения снято пользователем: интервал
                # между переводами взора не дробится на подпроцессы.
                split_k, split_score, split_p = None, np.nan, np.nan
            if split_k is not None:
                # Кладем сначала правый, потом левый, чтобы обход был слева направо.
                stack.append((split_k + 1, j1, split_depth + 1, True, split_score, split_p))
                stack.append((j0, split_k, split_depth + 1, True, split_score, split_p))
                continue

            fixation_records.append(_make_fixation_record(
                t=t,
                x=x,
                y=y,
                alpha_deg=alpha_deg,
                v_ang=v_ang,
                i_start=j0,
                i_end=j1,
                trend_alpha=trend_alpha,
                trend_slope_min=trend_slope_min,
                heterogeneous_parent=bool(heterogeneous_parent),
                with_area=with_area,
                split_score=float(local_split_score) if np.isfinite(local_split_score) else np.nan,
                split_pvalue=float(local_split_p) if np.isfinite(local_split_p) else np.nan,
                split_depth=int(split_depth),
            ))

    fixation_records = _merge_adjacent_fixation_records(
        fixation_records,
        t=t,
        x=x,
        y=y,
        alpha_deg=alpha_deg,
        v_ang=v_ang,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_amplitude_deg=min_amplitude_deg,
        max_merge_gap_s=max_merge_gap_s,
        trend_alpha=trend_alpha,
        trend_slope_min=trend_slope_min,
        stat_tol=merge_stat_tolerance,
        with_area=with_area,
    )

    if len(fixation_records) == 0:
        return pd.DataFrame(columns=[
            't_start_s', 't_end_s', 'mean_speed_deg_s', 'mean_eccentricity_deg', 'has_trend',
            'slope_gaze_x_deg_s', 'p_gaze_x', 'slope_gaze_y_deg_s', 'p_gaze_y', 'n_valid', 'mean_gaze_x', 'mean_gaze_y', 'cov_fro_norm_deg2',
            *(['bcea_deg2', 'hull_area_deg2'] if with_area else []),
            'heterogeneous_parent', 'split_score', 'split_pvalue', 'split_depth'
        ])

    out = pd.DataFrame(fixation_records)
    if "_i_start" in out.columns:
        out = out.drop(columns=["_i_start", "_i_end"])
    cols = [
        't_start_s', 't_end_s', 'mean_speed_deg_s', 'mean_eccentricity_deg', 'has_trend',
        'slope_gaze_x_deg_s', 'p_gaze_x', 'slope_gaze_y_deg_s', 'p_gaze_y', 'n_valid', 'mean_gaze_x', 'mean_gaze_y', 'cov_fro_norm_deg2',
        *(['bcea_deg2', 'hull_area_deg2'] if with_area else []),
        'spread_std_deg', 'spread_p95_deg', 'spread_peak_deg',
        'heterogeneous_parent', 'split_score', 'split_pvalue', 'split_depth'
    ]
    return pd.DataFrame(out.loc[:, cols].copy())


# вспомогательная для: _mark_eye_data_with_full_angle_planar_impl, mark_eye_data_with_full_angle_from_angles
def _build_saccades(
        *,
        t: np.ndarray,
        alpha_deg: np.ndarray,
        v_ang: np.ndarray,
        min_peak_velocity_deg_s: float,
        fixation_velocity_limit_deg_s: float,
        min_peak_width: float,
        min_amplitude_deg: float,
        amplitude_fn: Callable[[int, int], float],
        with_amplitude: bool = False,
) -> pd.DataFrame:
    """
    Строит таблицу быстрых движений по пикам полной угловой скорости.

    Пик считается кандидатом в событие, если он выше `min_peak_velocity_deg_s`, достаточно
    широк по времени и даёт суммарную амплитуду не меньше `min_amplitude_deg`. Границы
    события уточняются по падению скорости ниже порога фиксации.

    Терминология: для глаза относительно черепа такое событие — **саккада**,
    для взора относительно пространства — **перевод взора**. Функция
    предметно-нейтральна и используется в обоих случаях.
    """
    # Прежде колонки A_s/alpha_s и A_f/alpha_f содержали одно и то же значение
    # (пункт I-1 аудита). Дубликаты убраны, имена несут единицы.
    base_columns = ['t_start_s', 't_end_s', 'peak_speed_deg_s',
                    'eccentricity_start_deg', 'eccentricity_end_deg']
    columns = base_columns + (['amplitude_deg', 'i_start', 'i_end'] if with_amplitude else [])

    peak_ind, _ = find_peaks(v_ang, height=min_peak_velocity_deg_s, width=min_peak_width)

    if len(peak_ind) == 0:
        return pd.DataFrame(columns=columns)

    saccade_rows = []
    seen_bounds: set[tuple[int, int]] = set()

    for idx in peak_ind:
        left, right = _nearest_peak_bounds_by_threshold(
            v_ang, int(idx), float(fixation_velocity_limit_deg_s)
        )
        if right < left:
            continue

        t_start = float(t[left])
        t_finish = float(t[right])
        max_vel = float(v_ang[idx])
        val_start = float(alpha_deg[left])
        val_finish = float(alpha_deg[right])

        amplitude = float(amplitude_fn(left, right))
        if amplitude < float(min_amplitude_deg):
            continue

        row = [t_start, t_finish, max_vel, val_start, val_finish]
        if with_amplitude:
            # Дедупликация: два пика внутри одного движения дают одинаковые границы
            # (пункт B-3 аудита). В прежней схеме такие строки дублировались.
            if (int(left), int(right)) in seen_bounds:
                continue
            seen_bounds.add((int(left), int(right)))
            row += [amplitude, int(left), int(right)]
        saccade_rows.append(row)

    if len(saccade_rows) == 0:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(saccade_rows, columns=columns)


def _fit_single_peak_candidate(
        candidate: dict[str, Any],
        t: np.ndarray,
        y: np.ndarray,
        coord_name: str,
        r2_min_keep: float,
):
    """Строит sigmoid-fit для заранее подготовленного кандидата.

    `t` задается в `[s]`, `y` в `[deg]`.
    Параметры сигмоиды: `L` `[deg]`, `k` `[1/s]`, `t0` `[s]`, `y0` `[deg]`.
    """
    left = int(candidate["window_left"])  # [points]
    right = int(candidate["window_right"])  # [points]
    t_win = t[left:right]  # [s]
    y_win = y[left:right]  # [deg]
    valid = np.isfinite(t_win) & np.isfinite(y_win)
    t_fit = t_win[valid]  # [s]
    y_fit = y_win[valid]  # [deg]
    max_fit_points = 121
    if t_fit.size > max_fit_points:
        idx = np.linspace(0, t_fit.size - 1, max_fit_points, dtype=int)  # [points]
        t_fit = t_fit[idx]
        y_fit = y_fit[idx]
    res = fit_sigmoid_logistic(t_fit, y_fit, r2_min=r2_min_keep)

    r2 = float(res.r2) if np.isfinite(res.r2) else np.nan
    return {
        "coord": coord_name,
        "i_peak": int(candidate["i_peak"]),  # [points]
        "t_peak": float(candidate["t_peak"]),  # [s]
        "amp_peak_abs_dg": float(candidate["amp_peak_abs_dg"]),  # [deg/s]
        "i_start": int(candidate["i_start"]),  # [points]
        "t_start_s": float(candidate["t_start_s"]),  # [s]
        "i_end": int(candidate["i_end"]),  # [points]
        "t_end_s": float(candidate["t_end_s"]),  # [s]
        "delta": float(candidate["delta"]),  # [deg]
        "delta_xy": float(candidate["delta_xy"]),  # [deg]
        "success": bool(res.success),
        "r2": r2,
        "L": float(res.params.get("L", np.nan)),  # [deg]
        "k": float(res.params.get("k", np.nan)),  # [1/s]
        "t0": float(res.params.get("t0", np.nan)),  # [s]
        "y0": float(res.params.get("y0", np.nan)),  # [deg]
    }


# вспомогательная для: _mark_eye_data_with_sigmoid_planar_impl
def _build_sigmoid_peak_records(
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        speed_xy: np.ndarray,
        peak_indices: np.ndarray,
        left_ips: np.ndarray,
        right_ips: np.ndarray,
        prominences: np.ndarray,
        min_amplitude_deg: float,
) -> list[dict[str, Any]]:
    """Строит разметку пиков с их границами и 2D-амплитудой.

    Используется в `_mark_eye_data_with_sigmoid_planar_impl` перед sigmoid-fit.
    Размерности: `t` `[s]`, `x/y` `[deg]`, `speed_xy` `[deg/s]`,
    `peak_indices/left_ips/right_ips` `[points]`, `min_amplitude_deg` `[deg]`.
    """
    t = np.asarray(t, dtype=float)  # [s]
    x = np.asarray(x, dtype=float)  # [deg]
    y = np.asarray(y, dtype=float)  # [deg]
    speed_xy = np.asarray(speed_xy, dtype=float)  # [deg/s]
    peak_indices = np.asarray(peak_indices, dtype=int)  # [points]
    left_ips = np.asarray(left_ips, dtype=float)  # [points]
    right_ips = np.asarray(right_ips, dtype=float)  # [points]
    prominences = np.asarray(prominences, dtype=float)  # [deg/s]

    peak_records: list[dict[str, Any]] = []
    for idx, i_peak in enumerate(peak_indices):
        if not (0 <= int(i_peak) < len(t)):
            continue
        if not (
            np.isfinite(t[int(i_peak)]) and np.isfinite(x[int(i_peak)])
            and np.isfinite(y[int(i_peak)]) and np.isfinite(speed_xy[int(i_peak)])
        ):
            continue

        left_ip = left_ips[idx] if idx < len(left_ips) and np.isfinite(left_ips[idx]) else float(i_peak)
        right_ip = right_ips[idx] if idx < len(right_ips) and np.isfinite(right_ips[idx]) else float(i_peak)
        peak_height = float(speed_xy[int(i_peak)])  # [deg/s]
        prominence = prominences[idx] if idx < len(prominences) and np.isfinite(prominences[idx]) else peak_height
        dynamic_level = max(0.0, peak_height - 0.5 * max(0.0, prominence))  # [deg/s]
        dyn_start, dyn_end = _nearest_peak_bounds_by_threshold(speed_xy, int(i_peak), dynamic_level)  # [points], [points]

        i_start = max(0, min(min(int(np.floor(left_ip)), int(dyn_start)), len(t) - 1))  # [points]
        i_end = max(0, min(max(int(np.ceil(right_ip)), int(dyn_end)), len(t) - 1))  # [points]
        if i_end < i_start:
            i_start, i_end = i_end, i_start
        if i_end < i_start:
            continue
        if not (
            np.isfinite(x[i_start]) and np.isfinite(x[i_end])
            and np.isfinite(y[i_start]) and np.isfinite(y[i_end])
        ):
            continue

        delta_x = float(x[i_end] - x[i_start])  # [deg]
        delta_y = float(y[i_end] - y[i_start])  # [deg]
        delta_xy = float(np.hypot(delta_x, delta_y))  # [deg]
        if delta_xy < float(min_amplitude_deg):
            continue

        peak_records.append({
            "i_peak": int(i_peak),  # [points]
            "t_peak": float(t[int(i_peak)]),  # [s]
            "i_start": int(i_start),  # [points]
            "t_start_s": float(min(t[i_start], t[i_end])),  # [s]
            "i_end": int(i_end),  # [points]
            "t_end_s": float(max(t[i_start], t[i_end])),  # [s]
            "delta_x": float(delta_x),  # [deg]
            "delta_y": float(delta_y),  # [deg]
            "delta_xy": float(delta_xy),  # [deg]
            "amp_peak_speed_xy": float(speed_xy[int(i_peak)]),  # [deg/s]
        })

    return peak_records


def peaks_to_sigmoid_fits(
        *,
        t: np.ndarray,
        y: np.ndarray,
        dg: np.ndarray,
        peak_records: list[dict[str, Any]],
        coord_name: str,
        dt_window_s: float = 0.15,
        r2_min_keep: float = 0.96,
        verbose: bool | int = False,
) -> pd.DataFrame:
    """
    Аппроксимирует движения вокруг уже размеченных пиков логистическими сигмоидами.

    Используется в `_mark_eye_data_with_sigmoid_planar_impl`.
    Размерности: `t` `[s]`, `y` `[deg]`, `dg` `[deg/s]`,
    `dt_window_s` `[s]`, `peak_records[*].i_peak/i_start/i_end` `[points]`.
    """
    fit_columns = [
        "coord", "i_peak", "t_peak", "amp_peak_abs_dg",
        "i_start", "t_start_s", "i_end", "t_end_s",
        "delta", "delta_xy", "success", "r2", "L", "k", "t0", "y0",
    ]
    t = np.asarray(t, dtype=float)  # [s]
    y = np.asarray(y, dtype=float)  # [deg]
    dg = np.asarray(dg, dtype=float)  # [deg/s]
    if len(peak_records) == 0:
        _progress_print(verbose, f"[sigmoid:{coord_name}] peaks found: 0")
        return pd.DataFrame(columns=fit_columns)

    _progress_print(verbose, f"[sigmoid:{coord_name}] peaks found: {len(peak_records)}")

    candidates = []
    for record in peak_records:
        i = int(record["i_peak"])
        i_start = int(record["i_start"])
        i_end = int(record["i_end"])
        if not (0 <= i < len(t) and 0 <= i_start < len(t) and 0 <= i_end < len(t)):
            continue
        if not (np.isfinite(t[i]) and np.isfinite(y[i]) and np.isfinite(dg[i])):
            continue
        if not (np.isfinite(y[i_start]) and np.isfinite(y[i_end])):
            continue

        delta = float(y[i_end] - y[i_start])  # [deg]
        delta_xy = float(record["delta_xy"])  # [deg]
        t_start = float(min(t[i_start], t[i_end]))  # [s]
        t_end = float(max(t[i_start], t[i_end]))  # [s]
        plateau_pad = max(float(dt_window_s), 0.5 * max(0.0, t_end - t_start))  # [s]

        left = int(np.searchsorted(t, t_start - plateau_pad, side="left"))  # [points]
        right = int(np.searchsorted(t, t_end + plateau_pad, side="right"))  # [points]
        if right <= left:
            continue

        candidates.append({
            "i_peak": int(i),  # [points]
            "t_peak": float(t[i]),  # [s]
            "amp_peak_abs_dg": float(abs(dg[i])),  # [deg/s]
            "i_start": int(i_start),  # [points]
            "t_start_s": float(t_start),  # [s]
            "i_end": int(i_end),  # [points]
            "t_end_s": float(t_end),  # [s]
            "delta": float(delta),  # [deg]
            "delta_xy": float(delta_xy),  # [deg]
            "window_left": int(left),  # [points]
            "window_right": int(right),  # [points]
        })

    if len(candidates) == 0:
        return pd.DataFrame(columns=fit_columns)

    # Дедупликация кандидатов по (i_start, i_end) здесь была вычислена и
    # выброшена: цикл ниже идёт по исходному `candidates`. Удалено как мёртвый
    # код — та же дедупликация выполняется ниже уже на фитах (`df_fits`), так
    # что результат не меняется; экономия на подгонке дублей осталась неполученной.
    fits = []
    for candidate in candidates:
        fit = _fit_single_peak_candidate(
            candidate=candidate,
            t=t,
            y=y,
            coord_name=coord_name,
            r2_min_keep=r2_min_keep,
        )
        if fit is not None:
            fits.append(fit)
#        _progress_print(verbose, f"[sigmoid:{coord_name}] fitted {idx}/{len(candidates)}")

    fits = [f for f in fits if f is not None]
    _progress_print(verbose, f"[sigmoid:{coord_name}] successful fits before final dedup: {len(fits)}")
    if len(fits) == 0:
        return pd.DataFrame(columns=fit_columns)
    df_fits = pd.DataFrame(fits).sort_values("t_peak").reset_index(drop=True)
    if not df_fits.empty:
        df_fits = (
            df_fits.sort_values(["amp_peak_abs_dg", "r2"], ascending=[False, False])
            .drop_duplicates(subset=["i_start", "i_end"], keep="first")
            .sort_values("t_peak")
            .reset_index(drop=True)
        )
        df_fits = df_fits[df_fits["r2"] >= float(r2_min_keep)].reset_index(drop=True)
    _progress_print(verbose, f"[sigmoid:{coord_name}] final kept fits: {len(df_fits)}")
    return pd.DataFrame(df_fits.loc[:, fit_columns]) if not df_fits.empty else pd.DataFrame(columns=fit_columns)


# вспомогательная для: merge_fits
def sigmoid_time_at_fraction_vec(t0: Any, k: Any, p: float) -> np.ndarray:
    """Возвращает время достижения доли `p` сигмоиды: `t0` `[s]`, `k` `[1/s]`, результат `[s]`."""
    t0 = np.asarray(t0, dtype=float)
    k = np.asarray(k, dtype=float)
    out = np.full_like(t0, np.nan, dtype=float)
    m = np.isfinite(t0) & np.isfinite(k) & (k != 0.0)
    out[m] = t0[m] + (1.0 / k[m]) * np.log(p / (1.0 - p))
    return out


# вспомогательная для: _mark_eye_data_with_sigmoid_planar_impl
def merge_fits(
        gaze_x_fits: pd.DataFrame,
        gaze_y_fits: pd.DataFrame,
        *,
        axis_x_id: str = AXIS_IDS[0],
        axis_y_id: str = AXIS_IDS[1],
        max_axis_dt_s: float = 0.05,
        p_lo: float = 0.03,
        p_hi: float = 0.97,
) -> pd.DataFrame:
    """
    Сливает fits по времени.

    Используется в `_mark_eye_data_with_sigmoid_planar_impl`.
    Размерности: времена `t_*` `[s]`, амплитуды `delta*` `[deg]`,
    скорости пиков `amp_peak_abs_dg` `[deg/s]`, `max_axis_dt_s` `[s]`.
    `axis_x_id`, `axis_y_id` — КАНОНИЧЕСКИЕ идентификаторы осей (`gaze_x`, `gaze_y`),
    которыми префиксуются пер-осевые колонки результата. Имена колонок
    пользователя сюда не попадают: переименование делается на границе API.
    `p_lo`, `p_hi` — доли сигмоиды для обрезки границ (0..1): левая и правая.
    """

    p = _sort_frame(gaze_x_fits.copy(), "t_peak")
    q = _sort_frame(gaze_y_fits.copy(), "t_peak")

    def rename(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "coord"]
        out = df[cols].copy()
        out = out.add_prefix(f"{prefix}_")
        out_df: pd.DataFrame = pd.DataFrame(out)
        return out_df

    p2 = rename(p, axis_x_id)
    q2 = rename(q, axis_y_id)
    has_x_col = f"has_{axis_x_id}"
    has_y_col = f"has_{axis_y_id}"
    empty_cols = [has_x_col, has_y_col, "t_event_s", "event_type", "t_start_s", "t_end_s", "duration_s"]
    if p2.empty and q2.empty:
        return pd.DataFrame(columns=empty_cols)

    t_peak_x = f"{axis_x_id}_t_peak"
    t_peak_y = f"{axis_y_id}_t_peak"
    if t_peak_x in p2.columns:
        p2[t_peak_x] = pd.to_numeric(p2[t_peak_x], errors="coerce")
    if t_peak_y in q2.columns:
        q2[t_peak_y] = pd.to_numeric(q2[t_peak_y], errors="coerce")

    def _row_interval(row: pd.Series, prefix: str) -> tuple[float, float]:
        t_peak = _float_or_nan(row.get(f"{prefix}_t_peak"))
        t_start = _float_or_nan(row.get(f"{prefix}_t_start"))
        t_end = _float_or_nan(row.get(f"{prefix}_t_end"))
        if not np.isfinite(t_start):
            t_start = t_peak
        if not np.isfinite(t_end):
            t_end = t_peak
        return float(min(t_start, t_end)), float(max(t_start, t_end))

    def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    merged_rows: list[dict[str, Any]] = []
    matched_y_idx: set[int] = set()
    y_records = [cast(dict[str, Any], row.to_dict()) for _, row in q2.iterrows()]

    for _, p_row in p2.iterrows():
        p_series = cast(pd.Series, p_row)
        p_record = cast(dict[str, Any], p_series.to_dict())
        p0, p1 = _row_interval(p_series, axis_x_id)
        p_peak = _float_or_nan(p_series.get(t_peak_x))

        best_idx: int | None = None
        best_overlap = -1.0
        best_dt = np.inf

        for idx, y_record in enumerate(y_records):
            if idx in matched_y_idx:
                continue
            y_series = pd.Series(y_record)
            y0, y1 = _row_interval(y_series, axis_y_id)
            y_peak = _float_or_nan(y_series.get(t_peak_y))
            overlap = _interval_overlap(p0, p1, y0, y1)
            dt_peak = abs(float(p_peak - y_peak)) if np.isfinite(p_peak) and np.isfinite(y_peak) else np.inf

            if overlap > 0.0:
                if overlap > best_overlap or (np.isclose(overlap, best_overlap) and dt_peak < best_dt):
                    best_idx = idx
                    best_overlap = overlap
                    best_dt = dt_peak
            elif best_overlap <= 0.0 and dt_peak <= float(max_axis_dt_s) and dt_peak < best_dt:
                best_idx = idx
                best_dt = dt_peak

        merged_record = dict(p_record)
        if best_idx is not None:
            matched_y_idx.add(best_idx)
            for key, value in y_records[best_idx].items():
                merged_record[key] = value

        merged_rows.append(merged_record)

    merged = pd.DataFrame(merged_rows) if len(merged_rows) > 0 else pd.DataFrame(columns=list(p2.columns) + list(q2.columns))

    merged[has_x_col] = True
    merged[has_y_col] = merged[t_peak_y].notna() if t_peak_y in merged.columns else False

    merged["t_event_s"] = merged[t_peak_x]
    both = merged[has_y_col]
    merged.loc[both, "t_event_s"] = 0.5 * (merged.loc[both, t_peak_x] + merged.loc[both, t_peak_y])

    y_only = pd.DataFrame([row for idx, row in enumerate(y_records) if idx not in matched_y_idx])
    if not y_only.empty:
        y_only[has_x_col] = False
        y_only[has_y_col] = True
        y_only["t_event_s"] = y_only[t_peak_y]

        for c in p2.columns:
            if c not in y_only.columns:
                y_only[c] = np.nan

        for c in merged.columns:
            if c not in y_only.columns:
                y_only[c] = np.nan

        y_only = pd.DataFrame(y_only[merged.columns])
        merged = pd.DataFrame(pd.concat([merged, y_only], ignore_index=True))

    merged = _sort_frame(merged, "t_event_s")

    merged["event_type"] = np.select(
        [
            merged[has_x_col] & merged[has_y_col],
            merged[has_x_col] & ~merged[has_y_col],
            ~merged[has_x_col] & merged[has_y_col],
        ],
        [f"{axis_x_id}+{axis_y_id}", f"{axis_x_id}_only", f"{axis_y_id}_only"],
        default="unknown",
    )

    for coord in [axis_x_id, axis_y_id]:
        col_t0 = f"{coord}_t0"
        col_k = f"{coord}_k"

        if col_t0 not in merged.columns or col_k not in merged.columns:
            continue

        t0_arr = _float_array(merged[col_t0])
        k_arr = _float_array(merged[col_k])

        t_3 = sigmoid_time_at_fraction_vec(t0_arr, k_arr, p_lo)
        t_97 = sigmoid_time_at_fraction_vec(t0_arr, k_arr, p_hi)

        merged[f"{coord}_t_start"] = np.minimum(t_3, t_97)
        merged[f"{coord}_t_end"] = np.maximum(t_3, t_97)

    hp = merged[has_x_col].to_numpy(dtype=bool)
    hy = merged[has_y_col].to_numpy(dtype=bool)

    def get_col(name: str) -> np.ndarray:
        if name in merged.columns:
            return _float_array(merged[name])
        return np.full(len(merged), np.nan)

    p_start = get_col(f"{axis_x_id}_t_start")
    q_start = get_col(f"{axis_y_id}_t_start")
    p_end = get_col(f"{axis_x_id}_t_end")
    q_end = get_col(f"{axis_y_id}_t_end")

    def combine_start_vec(hp: np.ndarray, hy: np.ndarray, vp: np.ndarray, vy: np.ndarray) -> np.ndarray:
        out = np.full_like(vp, np.nan)
        fin_p = np.isfinite(vp)
        fin_y = np.isfinite(vy)

        both = hp & hy & fin_p & fin_y
        only_p = hp & fin_p & (~both)
        only_y = hy & fin_y & (~both)

        out[both] = np.minimum(vp[both], vy[both])
        out[only_p] = vp[only_p]
        out[only_y] = vy[only_y]
        return out

    def combine_end_vec(hp: np.ndarray, hy: np.ndarray, vp: np.ndarray, vy: np.ndarray) -> np.ndarray:
        out = np.full_like(vp, np.nan)
        fin_p = np.isfinite(vp)
        fin_y = np.isfinite(vy)

        both = hp & hy & fin_p & fin_y
        only_p = hp & fin_p & (~both)
        only_y = hy & fin_y & (~both)

        out[both] = np.maximum(vp[both], vy[both])
        out[only_p] = vp[only_p]
        out[only_y] = vy[only_y]
        return out

    merged["t_start_s"] = combine_start_vec(hp, hy, p_start, q_start)
    merged["t_end_s"] = combine_end_vec(hp, hy, p_end, q_end)
    merged["duration_s"] = merged["t_end_s"] - merged["t_start_s"]

    return merged


# для внешнего вызова
def median_filter_dataframe(
        df: pd.DataFrame,
        *,
        kernel_size: int = 5,
        skip_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Применяет медианный фильтр к числовым столбцам таблицы.

    Внешняя функция подготовки данных.
    Используется в `_mark_eye_data_with_sigmoid_planar_impl`.
    Размерности: `kernel_size` `[points]`, время в `skip_cols` обычно хранится в `[s]`.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    out = df.copy()
    cols_to_filter = [
        c for c in out.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not cols_to_filter:
        return out

    out[cols_to_filter] = out[cols_to_filter].rolling(
        window=kernel_size,
        center=True,
        min_periods=1,
    ).median()
    return out


# для внешнего вызова
def resample_uniform_Ts_nan_event_ffill(
        df: pd.DataFrame,
        *,
        col_time: str = "T_s",
        cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Перекладывает данные на равномерную временную сетку и возвращает шаг `dt`.

    Используется как внешняя функция подготовки данных.
    Размерности: `col_time` `[s]`, возвращаемый `dt` `[s]`.
    """
    if col_time not in df.columns:
        raise ValueError(f"'{col_time}' not found in df.")

    work = _sort_frame(df.copy(), col_time)

    t = _float_array(pd.to_numeric(work[col_time], errors="coerce"))
    if not np.isfinite(t).all():
        raise ValueError(f"{col_time} has non-finite values; clean it first.")
    if t.size < 2:
        raise ValueError("Need at least 2 rows to build uniform grid.")

    N = int(t.size)
    t0 = float(t[0])
    t1 = float(t[-1])

    dt = (t1 - t0) / (N - 1)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot compute dt from length: invalid time range or N.")

    grid_t = t0 + dt * np.arange(N, dtype=float)

    if cols is None:
        cols = [c for c in work.columns if c != col_time]
    cols = list(cols)
    if len(cols) == 0:
        raise ValueError("No columns to resample (cols is empty).")

    out = pd.DataFrame({col_time: grid_t})

    j = np.rint((t - t0) / dt).astype(int)
    j = np.clip(j, 0, N - 1)
    dist = np.abs(t - grid_t[j])

    for c in cols:
        v = _float_array(pd.to_numeric(work[c], errors="coerce"))

        order = np.lexsort((
            ~np.isnan(v),
            dist,
            j
        ))

        jj = j[order]
        vv = v[order]

        first_mask = np.ones_like(jj, dtype=bool)
        first_mask[1:] = jj[1:] != jj[:-1]

        jj_best = jj[first_mask]
        vv_best = vv[first_mask]

        grid_vals = np.full(N, np.nan, dtype=float)
        grid_vals[jj_best] = vv_best

        out[c] = _float_array(pd.Series(grid_vals).ffill())

    return out, float(dt)


# вспомогательная для: _ols_core
@jit(nopython=True)
def _ols_core_numba(t, y):
    """Арифметика МНК: наклон, t-статистика, число степеней свободы, число точек.

    p-value здесь НЕ считается: оно вычисляется одним способом снаружи, чтобы
    ускоренная и обычная ветки не могли разойтись (пункт N-1 аудита).
    """
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    n = len(t)

    if n < 3:
        return np.nan, np.nan, 0, n

    t0m = np.mean(t)
    y0m = np.mean(y)
    tt = t - t0m
    denom = np.sum(tt * tt)
    if denom <= 0:
        return np.nan, np.nan, 0, n

    slope = np.sum(tt * (y - y0m)) / denom
    intercept = y0m - slope * t0m

    resid = y - (intercept + slope * t)
    dof = n - 2
    if dof <= 0:
        return slope, np.nan, 0, n

    s2 = np.sum(resid * resid) / dof
    se = np.sqrt(s2 / denom)

    if not np.isfinite(se) or se == 0.0:
        return slope, np.nan, dof, n

    return slope, slope / se, dof, n


# вспомогательная для: _ols_core
def _ols_core_python(t, y):
    """Та же арифметика без numba. Порядок операций совпадает с `_ols_core_numba`."""
    m = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t, dtype=float)[m]
    y = np.asarray(y, dtype=float)[m]
    n = int(t.size)

    if n < 3:
        return np.nan, np.nan, 0, n

    t0m = np.mean(t)
    y0m = np.mean(y)
    tt = t - t0m
    denom = np.sum(tt * tt)
    if denom <= 0:
        return np.nan, np.nan, 0, n

    slope = np.sum(tt * (y - y0m)) / denom
    intercept = y0m - slope * t0m

    resid = y - (intercept + slope * t)
    dof = n - 2
    if dof <= 0:
        return slope, np.nan, 0, n

    s2 = np.sum(resid * resid) / dof
    se = np.sqrt(s2 / denom)

    if not np.isfinite(se) or se == 0.0:
        return slope, np.nan, dof, n

    return slope, slope / se, dof, n


# вспомогательная для: ols_slope_pvalue
def _student_two_sided_pvalue(t_stat: float, dof: int) -> float:
    """Двусторонний p-value распределения Стьюдента.

    Использует тождество `P(|T| > t) = I_{dof/(dof+t^2)}(dof/2, 1/2)`, то есть
    `betainc` из `scipy.special`. Результат совпадает с `2 * stats.t.sf(|t|, dof)`
    (проверено до 1e-12), но вызов дешевле и векторизуем.
    """
    if dof <= 0 or not np.isfinite(t_stat):
        return np.nan
    return float(betainc(0.5 * dof, 0.5, dof / (dof + t_stat * t_stat)))


# вспомогательная для: _interval_stats_by_indices, interval_stats_by_time
def ols_slope_pvalue(t: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Возвращает наклон OLS, p-value и число валидных точек.

    Размерности: `t` `[s]`, `y` `[deg]`, наклон `[deg/s]`.

    p-value считается по распределению Стьюдента с `dof = n - 2` независимо от
    того, установлена ли numba: ускоренная ветка отвечает только за арифметику.
    """
    if HAS_NUMBA:
        slope, t_stat, dof, n = _ols_core_numba(np.asarray(t, dtype=float),
                                                np.asarray(y, dtype=float))
    else:
        slope, t_stat, dof, n = _ols_core_python(t, y)

    if not np.isfinite(t_stat):
        return slope, np.nan, int(n)

    return float(slope), _student_two_sided_pvalue(float(t_stat), int(dof)), int(n)


# вспомогательная для: _build_slow_movements_between_events, analyze_stable_intervals
def interval_stats_by_time(
        data: pd.DataFrame,
        t_start_s: float,
        t_end_s: float,
        *,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = "T_s",
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.0,
):
    """Считает статистики сигнала на интервале времени `[t_start_s, t_end_s]`.

    Размерности: `t_start_s/t_end_s` `[s]`, `col_gaze_x/col_gaze_y` `[deg]`,
    наклоны `[deg/s]`. Границы включаются с двух сторон, поэтому соседние
    интервалы, стыкующиеся по времени, разделяют общий отсчёт.
    """
    mask = (data[col_time] >= t_start_s) & (data[col_time] <= t_end_s)

    t = _float_array(data.loc[mask, col_time])
    x = _float_array(data.loc[mask, col_gaze_x])
    y = _float_array(data.loc[mask, col_gaze_y])

    m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    n_valid = int(m.sum())
    if n_valid < 3:
        return (np.nan, np.nan, np.full((2, 2), np.nan), n_valid,
                np.nan, np.nan, np.nan, np.nan, True)

    t = t[m]
    x = x[m]
    y = y[m]

    ex = float(np.mean(x))
    ey = float(np.mean(y))
    cov = np.cov(x, y, ddof=1)

    slope_x, p_x, _ = ols_slope_pvalue(t, x)
    slope_y, p_y, _ = ols_slope_pvalue(t, y)

    has_trend_x = (np.isfinite(p_x) and (p_x < trend_alpha) and (abs(slope_x) > trend_slope_min))
    has_trend_y = (np.isfinite(p_y) and (p_y < trend_alpha) and (abs(slope_y) > trend_slope_min))
    has_trend = bool(has_trend_x or has_trend_y)

    return ex, ey, cov, n_valid, slope_x, p_x, slope_y, p_y, has_trend


# вспомогательная для: analyze_stable_intervals
def stable_intervals_by_velocity(
        data: pd.DataFrame,
        *,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = "T_s",
        velocity_limit_deg_s: float,
        min_points: int = 3,
        deg: bool = True,
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.0,
) -> pd.DataFrame:
    """Устойчивые интервалы: серии отсчётов со скоростью ниже порога.

    Третья величина рядом с покоем и интервалом между саккадами, и самая
    простая из трёх: событий она не знает вовсе, интервал не дробит и не
    склеивает, а просто берёт подряд идущие отсчёты, на которых скорость не
    превышает порога. Годится там, где нужна «стабильность» сама по себе, без
    решения о том, покой это или медленное движение.

    Скорость считается по шагу записи: угол большого круга между соседними
    отсчётами, делённый на фактическую длину шага, и приписывается левому
    отсчёту шага. Плоское приближение `hypot(Δyaw, Δpitch)` не используется:
    на сфере `ds² = dpitch² + cos²(pitch)·dyaw²`, поэтому оно **завышает**
    скорость тем сильнее, чем дальше от нейтрали, -- вдвое уже при
    `pitch = 60` град.

    Размерности: `col_gaze_x/col_gaze_y` `[deg]` при `deg=True`, иначе `[rad]`;
    `col_time` `[s]`, `velocity_limit_deg_s` `[deg/s]`, наклоны `[deg/s]`.

    Параметры:
      velocity_limit_deg_s — порог скорости, выше которого отсчёт не устойчив
      min_points           — минимум отсчётов в интервале
      trend_alpha, trend_slope_min — критерий значимого дрейфа внутри интервала

    Возвращает таблицу с границами, длительностью, числом отсчётов, средними
    координатами, наклонами МНК и их p-value.
    """
    columns = [
        "t_start_s", "t_end_s", "duration_s", "n_points",
        "mean_gaze_x", "mean_gaze_y", "mean_speed_deg_s",
        "slope_gaze_x_deg_s", "p_gaze_x", "slope_gaze_y_deg_s", "p_gaze_y", "has_trend",
    ]
    for column in (col_time, col_gaze_x, col_gaze_y):
        if column not in data.columns:
            raise KeyError(f"в таблице нет столбца {column!r}")

    work = data[[col_time, col_gaze_x, col_gaze_y]].apply(pd.to_numeric, errors="coerce").dropna()
    work = _sort_frame(work, col_time)
    t = _float_array(work[col_time])
    x = _float_array(work[col_gaze_x])
    y = _float_array(work[col_gaze_y])
    if t.size < max(2, int(min_points)):
        return pd.DataFrame(columns=columns)

    yaw = np.radians(x) if deg else x
    pitch = np.radians(y) if deg else y
    unit = np.stack([np.cos(pitch) * np.cos(yaw),
                     np.cos(pitch) * np.sin(yaw),
                     np.sin(pitch)], axis=1)
    dot = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    step = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(step > 0, np.degrees(np.arccos(dot)) / step, np.inf)  # [deg/s]

    stable = np.r_[np.isfinite(speed) & (speed <= float(velocity_limit_deg_s)), False]
    rows: list[dict[str, Any]] = []
    start: int | None = None
    for index, flag in enumerate(stable):
        if flag and start is None:
            start = index
            continue
        if flag or start is None:
            continue
        if index - start >= int(min_points):
            i0, i1 = start, index - 1
            ex, ey, _cov, n_valid, slope_x, p_x, slope_y, p_y, has_trend = _interval_stats_by_indices(
                t, np.degrees(yaw) if not deg else x, np.degrees(pitch) if not deg else y,
                i0, i1, trend_alpha=trend_alpha, trend_slope_min=trend_slope_min)
            rows.append({
                "t_start_s": float(t[i0]),
                "t_end_s": float(t[i1]),
                "duration_s": float(t[i1] - t[i0]),
                "n_points": int(i1 - i0 + 1),
                "mean_gaze_x": ex,
                "mean_gaze_y": ey,
                "mean_speed_deg_s": float(np.nanmean(speed[i0:i1])) if i1 > i0 else np.nan,
                "slope_gaze_x_deg_s": slope_x,
                "p_gaze_x": p_x,
                "slope_gaze_y_deg_s": slope_y,
                "p_gaze_y": p_y,
                "has_trend": bool(has_trend),
            })
        start = None

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def build_stable_intervals_between_events(
        df_events: pd.DataFrame,
        *,
        col_t_start: str = "t_start_s",
        col_t_end: str = "t_end_s",
        min_gap_s: float = 0.0,
) -> pd.DataFrame:
    """Строит интервалы стабильности между соседними событиями.

    Размерности: `col_t_start/col_t_end` `[s]`, `min_gap_s` `[s]`.
    """
    ev = df_events.copy()
    ev[col_t_start] = pd.to_numeric(ev[col_t_start], errors="coerce")
    ev[col_t_end] = pd.to_numeric(ev[col_t_end], errors="coerce")
    ev = ev[np.isfinite(ev[col_t_start]) & np.isfinite(ev[col_t_end])].copy()
    ev = _sort_frame(ev, col_t_start)

    rows = []
    for i in range(len(ev) - 1):
        t_start = float(ev.loc[i, col_t_end])
        t_end = float(ev.loc[i + 1, col_t_start])
        if t_end <= t_start:
            continue
        if (t_end - t_start) < min_gap_s:
            continue
        rows.append({
            "i_event_left": int(i),
            "i_event_right": int(i + 1),
            "t_stable_start": t_start,
            "t_stable_end": t_end,
            "dt_stable": float(t_end - t_start),
        })
    return pd.DataFrame(rows)


# вспомогательная для: _build_slow_movements_between_events
def stable_intervals_between_events(
        df_events: pd.DataFrame,
        t_grid: np.ndarray,
        *,
        col_t_start: str = "t_start_s",
        col_t_end: str = "t_end_s",
        min_gap_s: float = 0.0,
        clip_to_grid: bool = True,
) -> pd.DataFrame:
    """Ищет интервалы стабильности между событиями и при необходимости привязывает их к сетке.

    Размерности: `t_grid` `[s]`, `col_t_start/col_t_end` `[s]`, `min_gap_s` `[s]`.
    """

    ev = df_events.copy()
    ev[col_t_start] = pd.to_numeric(ev[col_t_start], errors="coerce")
    ev[col_t_end] = pd.to_numeric(ev[col_t_end], errors="coerce")

    ev = ev[np.isfinite(ev[col_t_start]) & np.isfinite(ev[col_t_end])].copy()
    ev = _sort_frame(ev, col_t_start)

    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.size < 2 or not np.isfinite(t_grid).all():
        raise ValueError("t_grid must be a finite 1D array with length >= 2.")

    rows = []
    for i in range(len(ev) - 1):
        t_start = float(ev.loc[i, col_t_end])
        t_end = float(ev.loc[i + 1, col_t_start])

        if not (np.isfinite(t_start) and np.isfinite(t_end)):
            continue
        if t_end <= t_start:
            continue

        dt = t_end - t_start
        if dt < min_gap_s:
            continue

        r = {
            "i_event_left": int(i),
            "i_event_right": int(i + 1),
            "t_stable_start": t_start,
            "t_stable_end": t_end,
            "dt_stable": dt,
        }

        if clip_to_grid:
            i0 = int(np.searchsorted(t_grid, t_start, side="left"))
            i1 = int(np.searchsorted(t_grid, t_end, side="right") - 1)
            i0 = max(0, min(i0, len(t_grid) - 1))
            i1 = max(0, min(i1, len(t_grid) - 1))

            if i1 <= i0:
                continue

            r["i_stable_start"] = i0
            r["i_stable_end"] = i1
            r["t_stable_start_grid"] = float(t_grid[i0])
            r["t_stable_end_grid"] = float(t_grid[i1])
            r["dt_stable_grid"] = float(t_grid[i1] - t_grid[i0])

        rows.append(r)

    return pd.DataFrame(rows)


#############################
# выделение пиков скорости (саккад/переводы взора)
#######################################################

# вспомогательная для: _mark_eye_data_with_sigmoid_planar_impl
def _build_slow_movements_between_events(
        *,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        events: pd.DataFrame,
        min_gap_s: float = 0.0,
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.0,
        col_time: str = "T_s",
) -> pd.DataFrame:
    """
    Строит интервалы **медленного движения** взора между переводами взора.

    Это иное определение, чем в схеме по полной угловой скорости: здесь интервал
    не проверяется ни на однородность, ни на разброс, и не разбивается на
    подпроцессы — берётся всё, что лежит между двумя переводами взора. Для взора
    такой интервал корректен, но фиксацией не является: между переводами взор
    может медленно смещаться (компенсаторные движения, следящее движение),
    поэтому величина называется медленным движением, а не фиксацией.

    Размерности: `t` `[s]`, `x/y` `[deg]`, `min_gap_s` `[s]`,
    `slope_x/slope_y` `[deg/s]`.
    """
    cols = [
        "t_start_s", "t_end_s", "mean_speed_deg_s", "mean_eccentricity_deg", "has_trend",
        "slope_gaze_x_deg_s", "p_gaze_x", "slope_gaze_y_deg_s", "p_gaze_y", "n_valid", "mean_gaze_x", "mean_gaze_y", "cov_fro_norm_deg2",
        "spread_std_deg", "spread_p95_deg", "spread_peak_deg",
        "i_start", "i_end", "i_event_left", "i_event_right",
        "t_stable_start", "t_stable_end", "dt_stable",
        "t_stable_start_grid", "t_stable_end_grid", "dt_stable_grid",
    ]
    if t.size == 0:
        return pd.DataFrame(columns=cols)

    event_rows = [{"t_start_s": float(t[0]), "t_end_s": float(t[0])}]
    if not events.empty:
        event_rows.extend(
            pd.DataFrame(events.loc[:, ["t_start_s", "t_end_s"]]).to_dict(orient="records")
        )
    event_rows.append({"t_start_s": float(t[-1]), "t_end_s": float(t[-1])})
    events = _sort_frame(pd.DataFrame(event_rows), "t_start_s")

    stable = stable_intervals_between_events(
        events,
        t,
        col_t_start="t_start_s",
        col_t_end="t_end_s",
        min_gap_s=min_gap_s,
        clip_to_grid=True,
    )
    if stable.empty:
        return pd.DataFrame(columns=cols)

    data = pd.DataFrame({
        col_time: t,
        "_x": np.asarray(x, dtype=float),
        "_y": np.asarray(y, dtype=float),
    })
    vx = np.gradient(np.asarray(x, dtype=float), t)
    vy = np.gradient(np.asarray(y, dtype=float), t)
    v_xy = np.hypot(vx, vy)

    rows: list[dict[str, Any]] = []
    for _, r in stable.iterrows():
        row = cast(dict[str, Any], r.to_dict())
        t0 = float(row["t_stable_start"])
        tf = float(row["t_stable_end"])
        has_i0 = "i_stable_start" in row and np.isfinite(_float_or_nan(row["i_stable_start"]))
        has_i1 = "i_stable_end" in row and np.isfinite(_float_or_nan(row["i_stable_end"]))
        i0 = int(row["i_stable_start"]) if has_i0 else int(np.searchsorted(t, t0, side="left"))
        i1 = int(row["i_stable_end"]) if has_i1 else int(np.searchsorted(t, tf, side="right") - 1)
        i0 = max(0, min(i0, len(t) - 1))
        i1 = max(0, min(i1, len(t) - 1))
        if i1 < i0:
            continue

        ex, ey, cov, n_valid, sx, px, sy, py, has_trend = interval_stats_by_time(
            data,
            t0,
            tf,
            col_gaze_x="_x",
            col_gaze_y="_y",
            col_time=col_time,
            trend_alpha=trend_alpha,
            trend_slope_min=trend_slope_min,
        )

        speed_seg = np.asarray(v_xy[i0:i1 + 1], dtype=float)
        mean_speed = float(np.nanmean(speed_seg)) if speed_seg.size > 0 else np.nan
        fro_cov = float(np.linalg.norm(cov, ord="fro")) if np.isfinite(cov).all() else np.nan
        spread_std, spread_p95, spread_peak = _spread_measures(
            np.asarray(x, dtype=float)[i0:i1 + 1], np.asarray(y, dtype=float)[i0:i1 + 1])

        rows.append({
            "t_start_s": t0,
            "t_end_s": tf,
            "mean_speed_deg_s": mean_speed,
            "mean_eccentricity_deg": np.nan,
            "has_trend": bool(has_trend),
            "slope_gaze_x_deg_s": sx,
            "p_gaze_x": px,
            "slope_gaze_y_deg_s": sy,
            "p_gaze_y": py,
            "n_valid": n_valid,
            "mean_gaze_x": ex,
            "mean_gaze_y": ey,
            "cov_fro_norm_deg2": fro_cov,
            "spread_std_deg": spread_std,
            "spread_p95_deg": spread_p95,
            "spread_peak_deg": spread_peak,
            "i_start": i0,
            "i_end": i1,
            "i_event_left": int(row["i_event_left"]),
            "i_event_right": int(row["i_event_right"]),
            "t_stable_start": t0,
            "t_stable_end": tf,
            "dt_stable": float(row["dt_stable"]),
            "t_stable_start_grid": float(row["t_stable_start_grid"]) if "t_stable_start_grid" in row else np.nan,
            "t_stable_end_grid": float(row["t_stable_end_grid"]) if "t_stable_end_grid" in row else np.nan,
            "dt_stable_grid": float(row["dt_stable_grid"]) if "dt_stable_grid" in row else np.nan,
        })

    return pd.DataFrame(rows, columns=cols)


# вспомогательная для: mark_eye_data_with_sigmoid
def _mark_eye_data_with_sigmoid_planar_impl(
        data: pd.DataFrame,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = "T_s",
        min_peak_velocity_deg_s: float = 15,
        max_axis_dt_s: float = 0.006,
        min_peak_width_s: float | None = None,
        min_amplitude_deg: float = 1.0,
        dt_window_s: float = 0.15,
        peak_prominence: float = 0.0,
        r2_min_keep: float = 0.96,
        min_fixation_gap_s: float = 0.0,
        fixation_alpha: float = 0.05,
        fixation_slope_min: float = 0.0,
        median_window_s: float | None = None,
        gaussian_sigma_s: float | None = None,
        sigmoid_p_lo: float = 0.03,
        sigmoid_p_hi: float = 0.97,
        debug_plot_peaks: bool = False,
        verbose: bool | int = False,
        columns: str = "source",
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Размечает саккады и фиксации по угловым координатам через sigmoid-fit.

    Саккады ищутся отдельно по координатам `x` и `y` как сигмоидальные переходы,
    затем объединяются по времени. Фиксации выделяются между объединенными
    событиями и анализируются только по координатам `x/y`.

    Используется в `mark_eye_data_with_sigmoid`.
    Размерности: `x/y` `[deg]`, `t` `[s]`, `dg_x/dg_y` `[deg/s]`,
    `min_peak_velocity_deg_s` `[deg/s]`, `max_axis_dt_s` `[s]`, `min_amplitude_deg` `[deg]`, `dt_window_s` `[s]`.
    Окна `min_peak_width_s`, `median_window_s`, `gaussian_sigma_s` задаются в
    секундах и переводятся в отсчёты по фактической частоте; `None` означает
    прежние значения в отсчётах (3, выключено, 1.5 соответственно).
    `sigmoid_p_lo`, `sigmoid_p_hi` — доли сигмоиды для обрезки границ (0..1).
    """
    if columns not in ("source", "canonical"):
        raise ValueError(
            f"columns должно быть 'source' или 'canonical', получено {columns!r}"
        )
    # Проверяет заодно, что имена осей различаются.
    axis_map = axis_column_map(col_gaze_x, col_gaze_y)

    total_started = perf_counter()
    _progress_print(verbose, f"[sigmoid] start: {len(data)} samples")

    stage_started = perf_counter()
    x = data[col_gaze_x].to_numpy(dtype=float)  # [deg]
    y = data[col_gaze_y].to_numpy(dtype=float)  # [deg]
    t = data[col_time].to_numpy(dtype=float)  # [s]
    finite_mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    dropped = int(len(t) - int(finite_mask.sum()))  # [points]
    if dropped > 0:
        x = x[finite_mask]
        y = y[finite_mask]
        t = t[finite_mask]
        _progress_print(verbose, f"[sigmoid] dropped non-finite samples: {dropped}")
    if t.size < 2:
        raise ValueError("Need at least 2 finite samples in t/x/y for sigmoid markup.")

    sampling = estimate_sampling_rate(t)
    require_seconds(sampling)
    fs_hz = sampling.fs_hz

    # Окна задаются в секундах и переводятся в отсчёты по фактической частоте.
    # None означает «как было до перехода на секунды» и берёт значение в отсчётах
    # напрямую, без пересчёта: иначе округление round(1.5/fs*fs) могло бы дать
    # не ровно 1.5 и незаметно сдвинуть результат.
    if min_peak_width_s is None:
        min_peak_width = LEGACY_MIN_PEAK_WIDTH_POINTS
    else:
        min_peak_width = max(1, int(round(float(min_peak_width_s) * fs_hz)))

    if median_window_s is None:
        median_window_points = 1                      # прежний дефолт: фильтр выключен
    else:
        median_window_points = max(1, int(round(float(median_window_s) * fs_hz)))
        if median_window_points % 2 == 0:             # ядро медианы обязано быть нечётным
            median_window_points += 1

    if gaussian_sigma_s is None:
        gaussian_sigma_points = LEGACY_GAUSSIAN_SIGMA_POINTS
    else:
        gaussian_sigma_points = float(gaussian_sigma_s) * fs_hz

    if median_window_points > 1:
        smoothed = median_filter_dataframe(
            pd.DataFrame({
                "_t": t,
                AXIS_IDS[0]: x,
                AXIS_IDS[1]: y,
            }),
            kernel_size=int(median_window_points),  # [points]
            skip_cols=("_t",),
        )
        x = smoothed[AXIS_IDS[0]].to_numpy(dtype=float)  # [deg]
        y = smoothed[AXIS_IDS[1]].to_numpy(dtype=float)  # [deg]
        _progress_print(
            verbose,
            f"[sigmoid] median smoothing applied, kernel={int(median_window_points)} "
            f"points ({median_window_points / fs_hz * 1000:.1f} ms)",
        )

    if gaussian_sigma_points > 0:
        x = gaussian_filter1d(x, sigma=gaussian_sigma_points)  # [deg]
        y = gaussian_filter1d(y, sigma=gaussian_sigma_points)  # [deg]

    dg_x = np.gradient(x, t)  # [deg/s]
    dg_y = np.gradient(y, t)  # [deg/s]
    speed_xy = np.hypot(dg_x, dg_y)  # [deg/s] модуль 2D-угловой скорости
    _progress_print(verbose, f"[sigmoid] prepared signals in {perf_counter() - stage_started:.2f}s")
    if dt_window_s > 1.0 and max_axis_dt_s > 0 and dt_window_s > max_axis_dt_s:
        warnings.warn(
            "dt_window_s выглядит слишком большим по сравнению с max_axis_dt_s. "
            "Если порог задан в миллисекундах, переведите его в секунды.",
            RuntimeWarning,
            stacklevel=2,
        )
        _progress_print(
            verbose,
            f"[sigmoid] предупреждение: dt_window_s={dt_window_s:.4f} с — проверьте единицы",
        )

    stage_started = perf_counter()
    if debug_plot_peaks and HAS_PLOTLY and go is not None:
        go.Figure(
            data=[
                go.Scatter(
                    x=t,
                    y=dg_x,
                    mode="lines",
                    name="dg_x",
                    line=dict(color="#2ca02c"),
                ),
                go.Scatter(
                    x=t,
                    y=dg_y,
                    mode="lines",
                    name="dg_y",
                    line=dict(color="#d62728"),
                ),
                go.Scatter(
                    x=t,
                    y=speed_xy,
                    mode="lines",
                    name="|w|",
                    line=dict(color="#9467bd"),
                ),
            ]
        ).update_layout(
            title="Angular velocities over time",
            template="plotly_white",
            xaxis_title="t, s",
            yaxis_title="speed, deg/s",
            hovermode="x unified",
        ).show()

    if peak_prominence > 0:
        peaks, peak_props = find_peaks(
            speed_xy,
            prominence=peak_prominence,
            height=min_peak_velocity_deg_s,
            width=min_peak_width,
        )
    else:
        peaks, peak_props = find_peaks(
            speed_xy,
            height=min_peak_velocity_deg_s,
            width=min_peak_width,
        )
    _progress_print(verbose, f"[sigmoid] 2D speed peaks found: {len(peaks)}")
    peak_records = _build_sigmoid_peak_records(
        t=t,
        x=x,
        y=y,
        speed_xy=speed_xy,
        peak_indices=peaks,
        left_ips=np.asarray(peak_props.get("left_ips", np.asarray(peaks, dtype=float)), dtype=float),
        right_ips=np.asarray(peak_props.get("right_ips", np.asarray(peaks, dtype=float)), dtype=float),
        prominences=np.asarray(peak_props.get("prominences", np.full(len(peaks), np.nan)), dtype=float),
        min_amplitude_deg=min_amplitude_deg,
    )

    gaze_x_fits = peaks_to_sigmoid_fits(
        t=t,
        y=x,
        dg=dg_x,
        peak_records=peak_records,
        coord_name=AXIS_IDS[0],
        dt_window_s=dt_window_s,
        r2_min_keep=r2_min_keep,
        verbose=verbose,
    )
    _progress_print(verbose, f"[sigmoid] x fits: {len(gaze_x_fits)} in {perf_counter() - stage_started:.2f}s")

    stage_started = perf_counter()
    gaze_y_fits = peaks_to_sigmoid_fits(
        t=t,
        y=y,
        dg=dg_y,
        peak_records=peak_records,
        coord_name=AXIS_IDS[1],
        dt_window_s=dt_window_s,
        r2_min_keep=r2_min_keep,
        verbose=verbose,
    )
    _progress_print(verbose, f"[sigmoid] y fits: {len(gaze_y_fits)} in {perf_counter() - stage_started:.2f}s")

    stage_started = perf_counter()
    events = merge_fits(
        gaze_x_fits, gaze_y_fits,
        max_axis_dt_s=max_axis_dt_s,
        axis_x_id=AXIS_IDS[0],
        axis_y_id=AXIS_IDS[1],
        p_lo=sigmoid_p_lo,
        p_hi=sigmoid_p_hi,
    )
    if not events.empty:

        has_x_col = f"has_{AXIS_IDS[0]}"
        has_y_col = f"has_{AXIS_IDS[1]}"
        x_delta_col = f"{AXIS_IDS[0]}_delta"
        y_delta_col = f"{AXIS_IDS[1]}_delta"
        has_x = events[has_x_col].to_numpy(dtype=bool) if has_x_col in events.columns else np.zeros(len(events), dtype=bool)
        has_y = events[has_y_col].to_numpy(dtype=bool) if has_y_col in events.columns else np.zeros(len(events), dtype=bool)
        x_delta = _float_array(events[x_delta_col]) if x_delta_col in events.columns else np.full(len(events), np.nan)  # [deg]
        y_delta = _float_array(events[y_delta_col]) if y_delta_col in events.columns else np.full(len(events), np.nan)  # [deg]
        amp_xy = np.full(len(events), np.nan)  # [deg]
        both = has_x & has_y & np.isfinite(x_delta) & np.isfinite(y_delta)
        only_x = has_x & np.isfinite(x_delta) & (~both)
        only_y = has_y & np.isfinite(y_delta) & (~both)
        amp_xy[both] = np.hypot(x_delta[both], y_delta[both])
        amp_xy[only_x] = np.abs(x_delta[only_x])
        amp_xy[only_y] = np.abs(y_delta[only_y])
        events["amplitude_deg"] = amp_xy
        events = pd.DataFrame(
            events.loc[
                np.isfinite(events["amplitude_deg"]) & (events["amplitude_deg"] >= float(min_amplitude_deg))
            ].reset_index(drop=True)
        )
    _progress_print(verbose, f"[sigmoid] объединено переводов взора: {len(events)} за {perf_counter() - stage_started:.2f} с")

    stage_started = perf_counter()
    slow_movements = _build_slow_movements_between_events(
        t=t,
        x=x,
        y=y,
        events=events,
        min_gap_s=min_fixation_gap_s,
        trend_alpha=fixation_alpha,
        trend_slope_min=fixation_slope_min,
        col_time=col_time,
    )
    _progress_print(verbose, f"[sigmoid] медленных движений: {len(slow_movements)} за {perf_counter() - stage_started:.2f} с")

    if debug_plot_peaks:
        _show_sigmoid_peak_debug_plot(
            t=t,
            x=x,
            y=y,
            dg_x=dg_x,
            dg_y=dg_y,
            gaze_x_fits=gaze_x_fits,
            gaze_y_fits=gaze_y_fits,
            events=events,
        )

    _progress_print(verbose, f"[sigmoid] done in {perf_counter() - total_started:.2f}s")

    warnings_text = []
    if dropped:
        warnings_text.append(
            f"отброшено неконечных отсчётов: {dropped} из {len(data)} "
            f"(длина возвращаемого alpha равна числу оставшихся)"
        )

    params_used = log_computation(
        "разметка переводов взора (сигмоидальный метод)",
        {
            "запись": Param(f"{sampling.span_s:.2f} с, {sampling.n_samples} отсчётов",
                            source=FROM_DATA),
            "частота записи": Param(fs_hz, "Гц", FROM_DATA,
                                    f"дрожание шага {sampling.jitter_rel * 100:.1f} %"),
            "медианное сглаживание": Param(
                f"{median_window_points} отсчётов ({median_window_points / fs_hz * 1000:.1f} мс)"
                if median_window_points > 1 else "выключено",
                source=FROM_USER if median_window_s is not None else FROM_DEFAULT),
            "гауссово сглаживание": Param(
                f"sigma {gaussian_sigma_points:.3g} отсчёта "
                f"({gaussian_sigma_points / fs_hz * 1000:.1f} мс)"
                if gaussian_sigma_points > 0 else "выключено",
                source=FROM_USER if gaussian_sigma_s is not None else FROM_DEFAULT),
            "порог пика скорости": Param(float(min_peak_velocity_deg_s), "град/с", FROM_USER),
            "минимальная амплитуда": Param(float(min_amplitude_deg), "град", FROM_USER,
                                           "применяется дважды: к 2D-амплитуде пика и к событию"),
            "минимальная ширина пика": Param(
                f"{min_peak_width} отсчётов ({min_peak_width / fs_hz * 1000:.1f} мс)",
                source=FROM_USER if min_peak_width_s is not None else FROM_DEFAULT),
            "выраженность пика": Param(float(peak_prominence), "град/с",
                                       FROM_USER if peak_prominence > 0 else FROM_DEFAULT,
                                       "0 = не применяется" if peak_prominence <= 0 else ""),
            "порог качества подгонки": Param(float(r2_min_keep), "", FROM_USER, "r² не ниже"),
            "обрезка границ события": Param(
                f"{sigmoid_p_lo:.0%} и {sigmoid_p_hi:.0%} сигмоиды", source=FROM_USER),
            "допуск сопоставления осей": Param(float(max_axis_dt_s) * 1000, "мс", FROM_USER),
            "найдено": Param(
                f"{len(events)} переводов взора, "
                f"{len(slow_movements)} интервалов медленного движения",
                source=COMPUTED),
            "подгонок по осям": Param(f"{len(gaze_x_fits)} и {len(gaze_y_fits)} прошли отбор по r²",
                                      source=COMPUTED,
                                      note=f"пиков 2D-скорости: {len(peaks)}"),
        },
        warnings_text=warnings_text,
    )

    if columns == "source":
        events = rename_axis_columns(events, axis_map, to="source")
    events.attrs["axis_columns"] = dict(axis_map)
    slow_movements.attrs["axis_columns"] = dict(axis_map)
    events.attrs["params_used"] = params_used
    slow_movements.attrs["params_used"] = params_used

    alpha = np.degrees(np.arctan2(y, x))
    return alpha, events, slow_movements


# для внешнего вызова
def mark_eye_data_with_sigmoid(
        data: pd.DataFrame,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = "T_s",
        min_peak_velocity_deg_s: float = 15,
        max_axis_dt_s: float = 0.006,
        min_peak_width_s: float | None = None,
        min_amplitude_deg: float = 1.0,
        dt_window_s: float = 0.15,
        peak_prominence: float = 0.0,
        r2_min_keep: float = 0.96,
        min_fixation_gap_s: float = 0.0,
        fixation_alpha: float = 0.05,
        fixation_slope_min: float = 0.0,
        median_window_s: float | None = None,
        gaussian_sigma_s: float | None = None,
        sigmoid_p_lo: float = 0.03,
        sigmoid_p_hi: float = 0.97,
        debug_plot_peaks: bool = False,
        verbose: bool | int = False,
        columns: str = "source",
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Размечает переводы взора через sigmoid-fit по координатам `x/y`,
    а интервалы медленного движения — через `stable_intervals_between_events`.

    Третья возвращаемая таблица — **не фиксации**: интервал между переводами взора
    не проверяется ни на однородность, ни на разброс и не разбивается на
    подпроцессы. Между переводами взор может медленно смещаться, поэтому величина
    называется медленным движением. Определение фиксации, включающее проверку
    однородности и разброса, реализовано в `markup.mark_gaze_events_direction`.

    Внешняя функция.
    Размерности: `x/y` `[deg]`, `col_time` `[s]`, `min_peak_velocity_deg_s` `[deg/s]`,
    `min_amplitude_deg` `[deg]`. Все временные параметры — в СЕКУНДАХ:
    `max_axis_dt_s` (допуск сопоставления пиков по осям), `dt_window_s`
    (запас плато для подгонки), `min_fixation_gap_s`.
    Окна сглаживания и минимальная ширина пика тоже в секундах
    (`median_window_s`, `gaussian_sigma_s`, `min_peak_width_s`) и переводятся
    в отсчёты по фактической частоте записи; `None` сохраняет прежние значения
    в отсчётах: 3 отсчёта, фильтр выключен, sigma 1.5 отсчёта.
    `sigmoid_p_lo`, `sigmoid_p_hi` — доли сигмоиды для обрезки границ (0..1).

    Возвращает массив направлений `alpha` в градусах, таблицу переводов взора
    `events` и таблицу интервалов медленного движения `slow_movements`.

    Схема таблицы событий. Внутри расчёта оси называются канонически
    (`gaze_x`, `gaze_y`), поэтому механика не зависит от имён пользовательских
    колонок. Наружу:

    * `columns="source"` (по умолчанию) — пер-осевые колонки получают имена
      пользовательских столбцов: `<col_gaze_x>_L`, `<col_gaze_x>_k`, `has_<col_gaze_x>` и т.д.;
    * `columns="canonical"` — сохраняются канонические имена `gaze_x_L`, `has_gaze_x`:
      схема одинакова для любых входных данных.

    Общие колонки (`t_event`, `event_type`, `t_start`, `t_end`, `t_finish`,
    `duration`, `A_deg`) не зависят от режима. Значения `event_type` —
    **всегда канонические** (`gaze_x+gaze_y`, `gaze_x_only`, `gaze_y_only`), потому что
    это код, а не подпись: иначе фильтрация по нему была бы непереносимой.
    Отображение осей доступно как `events.attrs["axis_columns"]`
    и через `axis_column_map(col_gaze_x, col_gaze_y)`.
    """
    return _mark_eye_data_with_sigmoid_planar_impl(
        data=data,
        col_gaze_x=col_gaze_x,
        col_gaze_y=col_gaze_y,
        col_time=col_time,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        max_axis_dt_s=max_axis_dt_s,
        min_peak_width_s=min_peak_width_s,
        min_amplitude_deg=min_amplitude_deg,
        dt_window_s=dt_window_s,
        peak_prominence=peak_prominence,
        r2_min_keep=r2_min_keep,
        min_fixation_gap_s=min_fixation_gap_s,
        fixation_alpha=fixation_alpha,
        fixation_slope_min=fixation_slope_min,
        median_window_s=median_window_s,
        gaussian_sigma_s=gaussian_sigma_s,
        sigmoid_p_lo=sigmoid_p_lo,
        sigmoid_p_hi=sigmoid_p_hi,
        debug_plot_peaks=debug_plot_peaks,
        verbose=verbose,
        columns=columns,
    )


_PLANAR_WITHDRAWN = (
    "Планарная разметка по полярному углу изъята (пункт M-1 аудита). Прежняя схема "
    "считала alpha = atan2(dy, dx) — направление смещения, а не его величину, поэтому "
    "движение строго «от центра» не детектировалось вообще, а прохождение траектории "
    "вблизи нейтральной точки давало скачок угла на 180° и ложное событие.\n"
    "Замена:\n"
    "  • взор на плоскости  -> markup.mark_gaze_events_direction(..., geometry='plane', r=...)\n"
    "    где r — расстояние от центра вращения до плоскости (радиус глаза либо дистанция "
    "наблюдения);\n"
    "  • одна угловая координата -> markup.mark_gaze_events_1d(...).\n"
    "Сигналы другой природы (например стабилограмма) в область применимости "
    "библиотеки не входят: у них нет ни модели глазного яблока, ни расстояния "
    "до плоскости, а значит нет и углового эксцентриситета."
)


# изъято: см. _PLANAR_WITHDRAWN
def mark_eye_data_with_full_angle(*args: Any, **kwargs: Any):
    """Изъято. Схема по полярному углу признана неверной — см. `_PLANAR_WITHDRAWN`."""
    raise NotImplementedError(_PLANAR_WITHDRAWN)


# для внешнего вызова
def mark_eye_data_with_full_angle_from_angles(
        data: pd.DataFrame,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = 'T_s',
        min_peak_velocity_deg_s: float | None = None,
        fixation_velocity_limit_deg_s: float | None = None,
        min_event_duration_s: float = 0.006,
        min_amplitude_deg: float | None = None,
        n_samples: int = N_SAMPLES_SACCADE,
        deg: bool = True,
        min_subprocess_duration_s: float | None = None,
        max_merge_gap_s: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Саккады и **интервалы между саккадами** по углам `yaw/pitch`.

    Внешняя функция.
    Третья таблица -- это всё, что лежит между двумя саккадами: интервал
    проверяется на однородность, разброс и длительность, но НЕ на скорость.
    Покоем он поэтому не является: между саккадами глаз может медленно
    смещаться. Если нужны отсчёты со скоростью ниже порога -- берите
    `mark_eye_rest_from_angles`.

    Фиксацией ни одна из этих таблиц не является в принципе: фиксация --
    состояние взора относительно неподвижного пространства, а здесь углы
    заданы относительно головы.

    Возвращает `alpha` в градусах, таблицу саккад и таблицу интервалов
    между саккадами.
    """
    return _mark_eye_data_from_angles_impl(
        data, col_gaze_x, col_gaze_y, col_time,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_event_duration_s=min_event_duration_s,
        min_amplitude_deg=min_amplitude_deg,
        n_samples=n_samples,
        deg=deg,
        min_subprocess_duration_s=min_subprocess_duration_s,
        max_merge_gap_s=max_merge_gap_s,
        trim_by_velocity=False,
    )


_EYE_FIXATIONS_RENAMED = (
    "mark_eye_fixations_from_angles переименована в mark_eye_rest_from_angles. "
    "Фиксация -- состояние взора относительно неподвижного пространства: взор "
    "удерживается на объекте, а голова и глаз при этом могут двигаться, "
    "компенсируя друг друга. Эта функция работает с углами относительно головы, "
    "то есть с движением глаза, и её интервалы -- покой глаза, а не фиксация. "
    "Поведение и все параметры прежние, изменилось только имя."
)


def mark_eye_fixations_from_angles(*args: Any, **kwargs: Any):
    """Переименована: интервалы глаза относительно головы -- покой, не фиксация."""
    raise NotImplementedError(_EYE_FIXATIONS_RENAMED)


def mark_eye_rest_from_angles(
        data: pd.DataFrame,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = 'T_s',
        min_peak_velocity_deg_s: float | None = None,
        fixation_velocity_limit_deg_s: float | None = None,
        min_event_duration_s: float = 0.006,
        min_amplitude_deg: float | None = None,
        n_samples: int = N_SAMPLES_SACCADE,
        deg: bool = True,
        min_subprocess_duration_s: float | None = None,
        max_merge_gap_s: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Саккады и **интервалы покоя глаза** по углам `yaw/pitch`.

    Не фиксации. Фиксация -- состояние **взора**: взор удерживается
    на объекте неподвижного пространства, и при этом голова и глаз вполне могут
    двигаться, компенсируя друг друга. Здесь же углы заданы относительно головы,
    то есть речь о движении глаза, и неподвижность глаза относительно черепа
    фиксацией не является. Поэтому интервал называется покоем.

    Отличие от `mark_eye_data_with_full_angle_from_angles` одно: интервал
    урезается до отсчётов со скоростью ниже `fixation_velocity_limit_deg_s`,
    то есть выполняется проверка скорости. Разрыв короче `max_merge_gap_s`
    нарушением не считается -- одиночный выброс не делит интервал покоя надвое.

    Скорость здесь -- полная угловая на единичной сфере, посчитанная по
    положению; на шумной записи её стоит сгладить до вызова.

    Возвращает `alpha` в градусах, таблицу саккад и таблицу интервалов покоя.
    """
    return _mark_eye_data_from_angles_impl(
        data, col_gaze_x, col_gaze_y, col_time,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_event_duration_s=min_event_duration_s,
        min_amplitude_deg=min_amplitude_deg,
        n_samples=n_samples,
        deg=deg,
        min_subprocess_duration_s=min_subprocess_duration_s,
        max_merge_gap_s=max_merge_gap_s,
        trim_by_velocity=True,
    )


def _mark_eye_data_from_angles_impl(
        data: pd.DataFrame,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time: str = 'T_s',
        *,
        min_peak_velocity_deg_s: float | None = None,
        fixation_velocity_limit_deg_s: float | None = None,
        min_event_duration_s: float = 0.006,
        min_amplitude_deg: float | None = None,
        n_samples: int = N_SAMPLES_SACCADE,
        deg: bool = True,
        min_subprocess_duration_s: float | None = None,
        max_merge_gap_s: float | None = None,
        trim_by_velocity: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Общая реализация разметки глаза по углам `yaw/pitch`.

    Вызывается двумя внешними функциями и отличает их одним флагом:
    `trim_by_velocity=False` даёт интервалы между саккадами,
    `trim_by_velocity=True` -- фиксации.

    Размерности: `yaw/pitch` `[deg]` при `deg=True`, иначе `[rad]`;
    `col_time` `[s]`, `min_peak_velocity_deg_s/fixation_velocity_limit_deg_s` `[deg/s]`,
    `min_event_duration_s` `[s]`, `min_amplitude_deg` `[deg]`.
    """

    yaw = data[col_gaze_x].to_numpy(dtype=float)
    pitch = data[col_gaze_y].to_numpy(dtype=float)
    t = data[col_time].to_numpy(dtype=float)

    # Единственная оценка частоты в проекте; заодно отвергает ось времени в миллисекундах.
    sampling = estimate_sampling_rate(t)
    require_seconds(sampling)

    # Пороги выводятся из частоты записи -- как в разметке взора. Прежде здесь
    # стояли константы 15 и 5 град/с и 1 град: значения вырожденной ветки старой
    # eye_params, не зависевшие от частоты. Событие здесь -- саккада, движение
    # глаза относительно черепа, поэтому модель берётся при N_SAMPLES_SACCADE.
    minima = min_saccade_minima_by_sampling(sampling.fs_hz, n_samples=n_samples)
    peak_source = FROM_USER if min_peak_velocity_deg_s is not None else FROM_MODEL
    fix_source = FROM_USER if fixation_velocity_limit_deg_s is not None else FROM_MODEL
    amp_source = FROM_USER if min_amplitude_deg is not None else FROM_MODEL
    if min_peak_velocity_deg_s is None:
        min_peak_velocity_deg_s = float(minima.v_peak_deg_s)
    if fixation_velocity_limit_deg_s is None:
        fixation_velocity_limit_deg_s = float(minima.v_fixation_deg_s)
    if min_amplitude_deg is None:
        min_amplitude_deg = float(minima.a_min_deg)

    # Три роли, которые раньше играл один параметр max_delta_time (пункт C-5 аудита).
    # По умолчанию значения выводятся из min_event_duration_s так, чтобы результат
    # совпадал с прежним поведением.
    sub_dur = float(min_event_duration_s if min_subprocess_duration_s is None
                    else min_subprocess_duration_s)
    merge_gap = float(min_event_duration_s / 2.0 if max_merge_gap_s is None
                      else max_merge_gap_s)

    if deg:
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)

    # угол отклонения от направления (0,0)
    c = np.cos(pitch) * np.cos(yaw)
    c = np.clip(c, -1.0, 1.0)
    alpha = np.arccos(c)

    # Полная угловая скорость
    # Переходим в декартовы координаты на единичной сфере
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)

    v_ang_rad = np.sqrt(vx**2 + vy**2 + vz**2)
    v_ang = np.degrees(v_ang_rad)

    alpha = np.degrees(alpha)

    def amplitude_fn(idx_start: int, idx_finish: int) -> float:
        dot_turn = (
            np.cos(pitch[idx_start]) * np.cos(pitch[idx_finish]) * np.cos(yaw[idx_start] - yaw[idx_finish])
            + np.sin(pitch[idx_start]) * np.sin(pitch[idx_finish])
        )
        dot_turn = np.clip(dot_turn, -1.0, 1.0)
        return float(np.degrees(np.arccos(dot_turn)))

    events = _build_saccades(
        t=t,
        alpha_deg=alpha,
        v_ang=v_ang,
        min_peak_velocity_deg_s=min_peak_velocity_deg_s,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_peak_width=max(
            1.0,
            float(np.ceil(min_event_duration_s / max(sampling.dt_median_s, 1e-12)))
        ),
        min_amplitude_deg=min_amplitude_deg,
        amplitude_fn=amplitude_fn,
    )

    intervals = _build_fixations_between_saccades(
        t=t,
        x=np.degrees(yaw),
        y=np.degrees(pitch),
        alpha_deg=alpha,
        v_ang=v_ang,
        events=events,
        fixation_velocity_limit_deg_s=fixation_velocity_limit_deg_s,
        min_amplitude_deg=min_amplitude_deg,
        min_interval_duration_s=float(min_event_duration_s),
        min_subprocess_duration_s=sub_dur,
        max_merge_gap_s=merge_gap,
        trend_slope_min=fixation_velocity_limit_deg_s,
        trim_by_velocity=trim_by_velocity,
    )
    kind = "интервалов покоя" if trim_by_velocity else "интервалов между саккадами"

    params_used = log_computation(
        "разметка саккад (сферическая схема)",
        {
            "запись": Param(f"{sampling.span_s:.2f} с, {sampling.n_samples} отсчётов",
                            source=FROM_DATA),
            "частота записи": Param(sampling.fs_hz, "Гц", FROM_DATA,
                                    f"дрожание шага {sampling.jitter_rel * 100:.1f} %"),
            "схема расчёта": Param("эксцентриситет от (0,0), скорость = полная угловая на единичной сфере", source=FROM_DEFAULT),
            "порог пика скорости": Param(
                float(min_peak_velocity_deg_s), "град/с", peak_source,
                f"модель при {sampling.fs_hz:.4g} Гц, n_samples={n_samples}"
                if peak_source == FROM_MODEL else None),
            "порог скорости покоя": Param(
                float(fixation_velocity_limit_deg_s), "град/с", fix_source,
                "интервал урезан по порогу" if trim_by_velocity
                else "интервал взят целиком: это интервалы между саккадами, не покой"),
            "минимальная амплитуда": Param(float(min_amplitude_deg), "град", amp_source),
            "минимальная длительность события": Param(
                float(min_event_duration_s) * 1000, "мс", FROM_USER,
                f"ширина пика не менее "
                f"{max(1.0, float(np.ceil(min_event_duration_s / max(sampling.dt_median_s, 1e-12)))):.0f}"
                f" отсчётов"),
            "минимальная длительность подпроцесса": Param(
                sub_dur * 1000, "мс",
                FROM_USER if min_subprocess_duration_s is not None else FROM_DEFAULT,
                "разбиение по однородности"),
            "разрыв при склейке": Param(
                merge_gap * 1000, "мс",
                FROM_USER if max_merge_gap_s is not None else FROM_DEFAULT),
            "найдено": Param(
                f"{len(events)} саккад, {len(intervals)} {kind}",
                source=COMPUTED),
        },
    )
    events.attrs["params_used"] = params_used
    intervals.attrs["params_used"] = params_used

    return alpha, events, intervals
