"""Единое описание того, что именно фиксируется как эталон методики.

Один и тот же список используется генератором фикстур (`generate_golden.py`)
и regression-тестами (`test_golden.py`), поэтому эталон и проверка не могут разойтись.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eyetracking_analytics import constants as const
from eyetracking_analytics import eyetracking_funs as ef
from eyetracking_analytics import for_raw_data as frd
from eyetracking_analytics import gaze_mark as gm
from eyetracking_analytics import markup as mkp
from eyetracking_analytics import models as mdl
from eyetracking_analytics import orientation as ori
from eyetracking_analytics import sampling as smp

from . import real_data, synthetic

DATA_DIR = Path(__file__).parent / "data"

# Пороги, выведенные из модели по частоте записи: так их задаёт методика.
_N_SAMPLES = const.N_SAMPLES_GAZE_SHIFT


def _thresholds(fs_hz: float) -> dict[str, float]:
    m = mdl.min_saccade_minima_by_sampling(fs_hz, n_samples=_N_SAMPLES)
    return {
        "min_amplitude_deg": m.a_min_deg,
        "min_peak_velocity_deg_s": m.v_peak_deg_s,
        "fixation_velocity_limit_deg_s": m.v_fixation_deg_s,
        "min_event_duration_s": m.t_min_s / 2.0,
    }


def _fs_of(df: pd.DataFrame) -> float:
    dt = np.diff(df["T_s"].to_numpy(dtype=float))
    return float(1.0 / np.median(dt[dt > 0]))


# ─────────────────────────────────────────────────────────── эталонные вычисления
# Каждый элемент: имя -> функция, возвращающая DataFrame или Series.

def _case_spherical(record: str) -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS[record]()
    th = _thresholds(_fs_of(df))
    # При анализе данных движения глаза порог скорости называется иначе:
    # интервалы там -- покой, а не фиксация. Значение то же, эталон не меняется.
    th_eye = {("rest_velocity_limit_deg_s" if k == "fixation_velocity_limit_deg_s" else k): v
              for k, v in th.items()}
    alpha, gaze_shifts, fixations = gm.mark_eye_data_with_full_angle_from_angles(
        df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s", deg=True, **th_eye
    )
    return {
        "alpha": pd.DataFrame({"alpha_deg": alpha}),
        "gaze_shifts": gaze_shifts,
        "fixations": fixations,
    }


def _case_sigmoid(record: str) -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS[record]()
    th = _thresholds(_fs_of(df))
    alpha, gaze_shifts, slow_movements = gm.mark_eye_data_with_sigmoid(
        df,
        col_horizontal="gaze_yaw",
        col_vertical="gaze_pitch",
        col_time="T_s",
        min_peak_velocity_deg_s=th["min_peak_velocity_deg_s"],
        min_amplitude_deg=th["min_amplitude_deg"],
        max_axis_dt_s=th["min_event_duration_s"],
        dt_window_s=0.05,
        r2_min_keep=0.5,
    )
    return {
        "alpha": pd.DataFrame({"alpha_deg": alpha}),
        "gaze_shifts": gaze_shifts,
        "slow_movements": slow_movements,
    }


def _case_direction(record: str, geometry: str) -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS[record]()
    th = _thresholds(_fs_of(df))
    if geometry == "plane":
        # Гномоническая проекция направлений на плоскость: то же самое движение,
        # заданное плоскими координатами.
        yaw = np.radians(df["gaze_yaw"].to_numpy(dtype=float))
        pitch = np.radians(df["gaze_pitch"].to_numpy(dtype=float))
        ux = np.cos(pitch) * np.cos(yaw)
        uy = np.cos(pitch) * np.sin(yaw)
        uz = np.sin(pitch)
        r = 12.0
        frame = pd.DataFrame({"T_s": df["T_s"], "X": r * uy / ux, "Y": r * uz / ux})
        res = mkp.mark_gaze_events_direction(
            frame, col_horizontal="X", col_vertical="Y", geometry="plane", r=r, **th)
    else:
        res = mkp.mark_gaze_events_direction(
            df, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", geometry="angles", **th)
    return {
        "eccentricity": pd.DataFrame({"deg": res.eccentricity_deg}),
        "azimuth": pd.DataFrame({"deg": res.azimuth_deg}),
        "speed": pd.DataFrame({"deg_s": res.speed_deg_s}),
        "gaze_shifts": res.gaze_shifts,
        "fixations": res.fixations,
    }


def _case_scalar() -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS["gaze_2d_250hz"]()
    th = _thresholds(_fs_of(df))
    res = mkp.mark_gaze_events_1d(
        pd.DataFrame({"T_s": df["T_s"], "angle": df["gaze_yaw"]}), col_gaze_angle="angle", **th)
    return {
        "speed": pd.DataFrame({"deg_s": res.speed_deg_s}),
        "gaze_shifts": res.gaze_shifts,
        "fixations": res.fixations,
    }


def _case_orientation() -> dict[str, pd.DataFrame]:
    """Слой ориентации: углы, поворот векторов, расхождение ориентаций."""
    df = synthetic.orientation_quaternions()
    angles = ori.angles_from_orientation(df)
    points = np.column_stack([
        np.linspace(-1.0, 1.0, len(df)),
        np.full(len(df), 0.5),
        np.linspace(1.0, -1.0, len(df)),
    ])
    rotated = ori.rotate_points(points, df)
    shifted = pd.DataFrame(
        np.roll(df.to_numpy(), 1, axis=0), columns=list(df.columns))
    return {
        "angles": angles,
        "rotated": pd.DataFrame(rotated, columns=["x", "y", "z"]),
        "angular_distance": pd.DataFrame(
            {"deg": ori.angular_distance_deg(df, shifted)}),
    }


def _case_models() -> dict[str, pd.DataFrame]:
    rows = []
    for fs in (31, 45, 60, 70, 90, 100, 123, 200, 250, 500, 1000, 1250, 2000):
        for n in (2, 3, 4, 10):
            r = mdl.min_saccade_minima_by_sampling(float(fs), n_samples=n)
            rows.append({
                "fs_hz": fs, "n_samples": n, "a_min_deg": r.a_min_deg,
                "t_min_s": r.t_min_s, "v_peak_deg_s": r.v_peak_deg_s,
                "v_fixation_deg_s": r.v_fixation_deg_s, "t_fixation_s": r.t_fixation_s,
                "v_fixation_by_model_deg_s": r.v_fixation_by_model_deg_s,
                "v_fixation_at_floor": r.v_fixation_at_floor,
                "a_min_by_model_deg": r.a_min_by_model_deg,
                "a_min_at_floor": r.a_min_at_floor,
                "spread_max_deg": r.spread_max_deg,
                "T0_ms": r.model.T0_ms, "m": r.model.m, "V0": r.model.V0, "n": r.model.n,
                "max_increment_deg": mdl.max_increment_deg(fs),
            })
    return {"models": pd.DataFrame(rows)}


def _case_metrics() -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS["gaze_2d_250hz"]()
    t = df["T_s"].to_numpy(dtype=float)
    x = df["gaze_yaw"].to_numpy(dtype=float)
    y = df["gaze_pitch"].to_numpy(dtype=float)

    # Контракт частоты: None = частота записи; заданная частота -> децимация;
    # выше частоты записи -> ошибка (проверяется отдельным тестом, не фикстурой).
    rows = []
    for fs_hz in (None, 250.0, 125.0, 60.0):
        for v0_deg_s in (None, 10.0):
            r = ef.ksv(t, x, y, v0_deg_s=v0_deg_s, fs_hz=fs_hz)
            rows.append({
                "fs_hz_requested": np.nan if fs_hz is None else fs_hz,
                "v0_requested": np.nan if v0_deg_s is None else v0_deg_s,
                "ksv": r.ksv,
                "t_stab_s": r.t_stab_s,
                "t_total_s": r.t_total_s,
                "fs_hz_used": r.fs_hz_used,
                "fs_hz_recording": r.fs_hz_recording,
                "resampled": r.resampled,
                "v0_deg_s_used": r.v0_deg_s_used,
                "v0_deg_s_model": r.v0_deg_s_model,
                "v0_source": r.v0_source,
            })

    windows = [(0.0, 5.0), (5.0, 10.0), (2.5, 3.5)]
    interval_rows = []
    for t1, t2 in windows:
        ex, ey, cov, n_valid, sx, px, sy, py, trend = gm.interval_stats_by_time(
            df, t1, t2, col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s"
        )
        interval_rows.append({
            "t1": t1, "t2": t2,
            "tr_length": ef.tr_length(df, t1, t2, "gaze_yaw", "gaze_pitch"),
            "std_x": ef.find_std_fx(df, t1, t2, "gaze_yaw"),
            "mean_horizontal_deg": ex, "mean_vertical_deg": ey, "cov_fro_norm_deg2": float(np.linalg.norm(cov, ord="fro")),
            "n_valid": n_valid, "slope_horizontal_deg_s": sx, "p_horizontal": px, "slope_vertical_deg_s": sy, "p_vertical": py,
            "has_trend": trend,
        })

    vel_native = frd.angular_velocity_at_rate(t, x, y)              # частота записи
    vel_dec = frd.angular_velocity_at_rate(t, x, y, fs_hz=100.0)       # децимация
    sampling_rows = []
    for name, record in synthetic.RECORDS.items():
        d = record()
        info = smp.estimate_sampling_rate(d["T_s"].to_numpy(dtype=float))
        sampling_rows.append({
            "запись": name, "fs_hz": info.fs_hz, "n_samples": info.n_samples,
            "span_s": info.span_s, "dt_median_s": info.dt_median_s,
            "dt_mean_s": info.dt_mean_s, "dt_sd_s": info.dt_sd_s,
            "dt_min_s": info.dt_min_s, "dt_max_s": info.dt_max_s,
            "jitter_rel": info.jitter_rel, "n_gaps": info.n_gaps,
            "n_duplicated": info.n_duplicated_timestamps,
            "fs_max_requestable_hz": info.fs_max_requestable_hz,
        })

    return {
        "ksv": pd.DataFrame(rows),
        "intervals": pd.DataFrame(interval_rows),
        "velocity_native": pd.DataFrame({
            "t": vel_native.t_s, "v_abs": vel_native.v_abs,
            "v_x": vel_native.v_x, "v_y": vel_native.v_y,
        }),
        "velocity_decimated_100hz": pd.DataFrame({
            "t": vel_dec.t_s, "v_abs": vel_dec.v_abs,
            "v_x": vel_dec.v_x, "v_y": vel_dec.v_y,
        }),
        "sampling": pd.DataFrame(sampling_rows),
    }


def _case_preprocess() -> dict[str, pd.DataFrame]:
    df = synthetic.RECORDS["gaze_2d_200hz_jitter"]()
    filtered = gm.median_filter_dataframe(df, kernel_size=3, skip_cols=("T_s",))
    uniform, dt = gm.resample_uniform_Ts_nan_event_ffill(
        df, col_time="T_s", cols=["gaze_yaw", "gaze_pitch"]
    )
    uniform = uniform.copy()
    uniform.attrs.clear()
    return {
        "median_filtered": filtered,
        "resampled": uniform,
        "resample_dt": pd.DataFrame({"dt": [dt]}),
    }


def _case_eye_layer() -> dict[str, pd.DataFrame]:
    a = synthetic.oculogram_1d_mm()
    # Порог задаётся скоростью и учитывает фактический шаг времени (пункт C-2).
    t = np.arange(a.size, dtype=float) / 120.0
    smoothed = frd.star_eye_smoothing(a, t, max_velocity=60.0, eps=0.1)
    x_c, y_c, _ = frd.find_central_position(a, a)
    okulo = frd.make_okulogram(a, neutral=None, r_eye=12)
    return {
        "star_smoothing": pd.DataFrame({"value": np.asarray(smoothed, dtype=float)}),
        "okulogram": pd.DataFrame({"angle_deg": np.asarray(okulo, dtype=float)}),
        "central": pd.DataFrame({"x_c": [x_c], "y_c": [y_c]}),
    }



def _case_real_vr() -> dict[str, pd.DataFrame]:
    """Реальная запись: кватернионы головы и глаза -> взор -> разметка.

    Синтетика проверяет методику там, где известен правильный ответ; этот
    случай фиксирует поведение на том, что выдаёт прибор, -- с неравномерным
    шагом, пропусками и выбросами. Запись обезличена, см. `real_data`.
    """
    from eyetracking_analytics import direction as dr

    data = real_data.vr_head_eye()
    gaze = dr.gaze_direction(head=real_data.quaternions(data, "head"),
                             eye=real_data.quaternions(data, "eye"))
    res = mkp.mark_gaze_events_direction(
        gaze.to_frame(data["T_s"]), col_time="T_s", geometry="vector",
        col_horizontal="gaze_horizontal", col_vertical="gaze_vertical",
        col_forward="gaze_forward")
    return {
        "direction": pd.DataFrame(gaze.direction, columns=["x", "y", "z"]),
        "eccentricity": pd.DataFrame({"deg": res.eccentricity_deg}),
        "speed": pd.DataFrame({"deg_s": res.speed_deg_s}),
        "gaze_shifts": res.gaze_shifts,
        "fixations": res.fixations,
    }


#: Полный набор эталонов: имя случая -> функция без аргументов.
CASES: dict[str, Callable[[], dict[str, pd.DataFrame]]] = {
    "spherical_250hz": lambda: _case_spherical("gaze_2d_250hz"),
    "spherical_90hz": lambda: _case_spherical("gaze_2d_90hz"),
    "spherical_jitter": lambda: _case_spherical("gaze_2d_200hz_jitter"),
    "sigmoid_250hz": lambda: _case_sigmoid("gaze_2d_250hz"),
    "sigmoid_90hz": lambda: _case_sigmoid("gaze_2d_90hz"),
    "direction_angles_250hz": lambda: _case_direction("gaze_2d_250hz", "angles"),
    "direction_angles_90hz": lambda: _case_direction("gaze_2d_90hz", "angles"),
    "direction_plane": lambda: _case_direction("gaze_2d_250hz", "plane"),
    "scalar_1d": _case_scalar,
    "orientation": _case_orientation,
    "models": _case_models,
    "metrics": _case_metrics,
    "preprocess": _case_preprocess,
    "eye_layer": _case_eye_layer,
    "real_vr": _case_real_vr,
}


# ─────────────────────────────────────────────────────────────── ввод/вывод CSV
#: '%.17g' обеспечивает точное сохранение float64, поэтому сравнение может быть строгим.
FLOAT_FORMAT = "%.17g"


@contextmanager
def reference_branches():
    """Считать случай на python-ветках — они объявлены эталоном методики (N-1).

    Ускоренные ветки алгоритмически совпадают с python-ветками, но не бит-в-бит:
    `_ols_core_numba` суммирует циклом, а numpy — попарно, поэтому наклон и
    p-value тренда расходятся в последних битах (до 1e-14 на текущем наборе).
    Если снимать эталон на ускоренной ветке, фикстуры перестают быть проверкой
    методики и начинают зависеть от того, установлена ли numba и какая версия
    LLVM у неё внутри.

    Поэтому эталон всегда снимается и проверяется на python-ветках, а согласие
    ускоренных ветвей с ними проверяет `test_numba_parity.py` — с допуском.
    """
    saved = (gm.HAS_NUMBA, frd.HAS_NUMBA)
    gm.HAS_NUMBA = False
    frd.HAS_NUMBA = False
    try:
        yield
    finally:
        gm.HAS_NUMBA, frd.HAS_NUMBA = saved


def run_case(case: str) -> dict[str, pd.DataFrame]:
    """Единая точка запуска: генератор и тест обязаны считать одинаково."""
    with reference_branches():
        return CASES[case]()


def path_for(case: str, name: str) -> Path:
    return DATA_DIR / f"{case}__{name}.csv"


def dump(case: str, name: str, frame: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(case, name)
    frame.to_csv(p, index=False, float_format=FLOAT_FORMAT)
    return p


def load(case: str, name: str) -> pd.DataFrame:
    # float_precision="round_trip" обязателен: быстрый парсер pandas округляет
    # с относительной погрешностью ~1e-16, из-за чего строгое сравнение ломается.
    return pd.read_csv(path_for(case, name), float_precision="round_trip")


# Столбцы решений методики: разметка, границы, число событий. Здесь расхождение
# означает, что методика повела себя иначе, поэтому сравнение только точное.
DECISION_PREFIXES = (
    "t_start", "t_end", "i_start", "i_end", "duration", "n_", "index", "count",
)

# Допуск для непрерывных величин. Он на порядки меньше любого методически
# значимого эффекта, но покрывает то, что нельзя воспроизвести бит-в-бит между
# версиями библиотек: numpy 2.x считает тригонометрию другими SIMD-ядрами,
# чем 1.24, и arctan2/sqrt расходятся в последних битах (замерено: до 1.3e-11
# при значениях в сотни град/с, то есть ~1e-14 относительно).
# Допуск для непрерывных величин. Он на порядки меньше любого методически
# значимого эффекта, но должен покрывать то, что нельзя воспроизвести между
# машинами: на arm64 и x86-64 тригонометрия считается разными ядрами. Особенно
# чувствителен модуль скорости: он идёт через `arccos` от скалярного произведения
# единичных векторов, а `arccos` вблизи единицы усиливает потерю значащих цифр --
# на эталонных записях расхождение между машинами доходит до 1.3e-08
# относительно при значениях в сотни град/с.
RTOL = 1e-7
ATOL = 1e-12


@dataclass(frozen=True)
class Tolerance:
    """Допуск для одного случая и признак того, точны ли границы событий."""

    rtol: float = RTOL
    atol: float = ATOL
    exact_decisions: bool = True


# Разметка на базе аппроксимации движения сигмоидой — единственная, где событие
# описывается не порогом, а фитом:
# `curve_fit` (gaze_mark.py) подгоняет логистическую кривую, а границы события
# вычисляются из её параметров. Поэтому результат наследует точность
# оптимизатора: между scipy 1.10 и 1.17 параметр наклона `k` расходится на
# 2.3e-7 относительно, а границы события — на 5 нс (при шаге отсчётов 11 мс,
# то есть на шесть порядков меньше разрешения записи).
#
# Цена решения: для sigmoid-случаев regression-тест грубее и мелкий дрейф
# поймает только `test_case_matches_golden_bit_exact` в окружении эталона.
TOLERANCES: dict[str, Tolerance] = {
    "sigmoid_90hz": Tolerance(rtol=1e-5, atol=1e-6, exact_decisions=False),
    "sigmoid_250hz": Tolerance(rtol=1e-5, atol=1e-6, exact_decisions=False),
}


def tolerance_for(case: str) -> Tolerance:
    return TOLERANCES.get(case, Tolerance())


def is_decision_column(col: str) -> bool:
    return col.startswith(DECISION_PREFIXES)


def compare(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    exact: bool = False,
    tol: Tolerance | None = None,
) -> list[str]:
    """Возвращает список расхождений; пустой список означает совпадение.

    Схема, число строк, позиции NaN и целочисленные столбцы сравниваются точно
    всегда. Столбцы решений (границы, длительности, счётчики) — точно, если
    `tol.exact_decisions`; при аппроксимации сигмоидой они вычисляются из параметров фита
    и потому сравниваются с допуском. Непрерывные величины — с допуском
    `tol.rtol`/`tol.atol`. При `exact=True` всё сравнивается точно: так
    проверяется окружение, в котором эталон и снимался.
    """
    tol = tol or Tolerance()
    problems: list[str] = []
    if list(actual.columns) != list(expected.columns):
        problems.append(
            f"схема: получено {list(actual.columns)}, ожидалось {list(expected.columns)}"
        )
        return problems
    if len(actual) != len(expected):
        problems.append(f"строк: получено {len(actual)}, ожидалось {len(expected)}")
        return problems

    for col in expected.columns:
        got = actual[col].to_numpy()
        exp = expected[col].to_numpy()
        if not (np.issubdtype(exp.dtype, np.number) and np.issubdtype(got.dtype, np.number)):
            if not (pd.Series(got).astype(str) == pd.Series(exp).astype(str)).all():
                problems.append(f"{col}: значения отличаются")
            continue

        g = got.astype(float)
        e = exp.astype(float)
        nan_mismatch = np.isnan(g) != np.isnan(e)
        if nan_mismatch.any():
            problems.append(f"{col}: NaN в разных позициях ({int(nan_mismatch.sum())} шт.)")
            continue
        m = ~np.isnan(g)
        if not m.any():
            continue

        strict = (
            exact
            or np.issubdtype(exp.dtype, np.integer)
            or (is_decision_column(col) and tol.exact_decisions)
        )
        if strict:
            if not np.array_equal(g[m], e[m]):
                worst = float(np.max(np.abs(g[m] - e[m])))
                kind = "решение методики" if is_decision_column(col) else "точное сравнение"
                problems.append(f"{col}: max |разница| = {worst:.3e} ({kind})")
        elif not np.allclose(g[m], e[m], rtol=tol.rtol, atol=tol.atol):
            worst = float(np.max(np.abs(g[m] - e[m])))
            scale = np.maximum(np.abs(e[m]), np.abs(g[m]))
            rel = float(np.max(np.abs(g[m] - e[m])[scale > 0] / scale[scale > 0])) if (scale > 0).any() else 0.0
            problems.append(
                f"{col}: max |разница| = {worst:.3e}, относительная = {rel:.3e} "
                f"(допуск rtol={tol.rtol:g}, atol={tol.atol:g})"
            )
    return problems


def environment() -> dict[str, Any]:
    import platform

    import scipy
    info = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        # Архитектура и ОС: тригонометрия и порядок суммирования в последних
        # битах зависят от машины, а не только от версий библиотек. Без этой
        # записи бит-в-бит проверка пыталась бы выполниться на чужой машине
        # и падала бы на расхождениях порядка 1e-13.
        "машина": platform.machine(),
        "система": platform.system(),
        # Не «установлена ли numba», а на какой ветке снят эталон: фикстуры
        # обязаны быть независимы от наличия ускорителя.
        "эталонная_ветка": "python",
    }
    try:
        import numba
        info["numba"] = numba.__version__
    except ImportError:
        info["numba"] = None
    return info
