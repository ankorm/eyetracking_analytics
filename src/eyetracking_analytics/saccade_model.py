###################################################
# Модель траектории саккады: семейство кривых и поиск похожих в записи
###################################################
"""Модель саккады как управляемого движения и подбор её под запись.

Зачем. Разметка находит границы события по скорости, но ничего не говорит о том,
как движение шло внутри: разгон, переключение, торможение. Модель даёт форму
траектории целиком, а дальше работает та же идея, что и с сигмоидами: строится
семейство кривых, и в записи ищутся похожие.

Модель. Глаз описан звеном второго порядка с двумя собственными числами
`lambda1` и `lambda2 = 3 * lambda1` и релейным управлением амплитуды `u`:
разгон, реверс, торможение -- три участка, разделённые двумя моментами
переключения. Моменты переключения не подбираются, а находятся из условия
попадания: к концу движения угол равен заданной амплитуде, а скорость -- нулю.
Отсюда и форма: она определена параметрами звена, а не подгонкой.

Безразмерность. Решение считается один раз в безразмерном виде (амплитуда 1,
время в единицах модели), а под конкретную саккаду масштабируется двумя числами:
амплитудой `[град]` и длительностью `[с]`. Поэтому семейство траекторий -- это
одна форма на сетке амплитуд и длительностей, а подбор сводится к поиску пары
(амплитуда, длительность), при которой кривая ложится на запись.

Единицы во всём публичном API: секунды, градусы, градусы в секунду.

Источник модели. Постановка задачи оптимального быстродействия, параметры звена
и вывод моментов переключения -- работы автора методики:

* Kruchinina A., Yakushev A. G. Parametrization of Saccade Trajectories //
  Moscow University Mechanics Bulletin. 2018. Vol. 73, no. 4. P. 97-100.
  DOI: https://doi.org/10.3103/S0027133018340052
* Kruchinina A. Optimal time task in saccadic eye movement // Russian Journal
  of Biomechanics. 2020. Vol. 24, no. 1. P. 33-39.
  DOI: https://doi.org/10.15593/RJBiomech/2020.1.04

При использовании этого модуля в исследовании ссылаться нужно на обе (раздел
`references` в `CITATION.cff`) -- дополнительно к статье главной
последовательности, на которой стоят пороги разметки (`models.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .constants import DEFAULT_SACCADE_MODEL_DT, DEFAULT_SACCADE_TEMPLATE_FS_HZ

__all__ = [
    "SaccadeModel",
    "switch_times",
    "trajectory",
    "saccade_template",
    "saccade_template_family",
    "match_saccade_templates",
]


@dataclass(frozen=True)
class SaccadeModel:
    """Параметры звена и управления.

    Значения по умолчанию -- из работ 2018 и 2020 (см. шапку модуля): `a1`, `a2`
    -- параметры звена, `u` -- амплитуда релейного управления, `lambda1` --
    собственное число (второе равно `3 * lambda1`), `amplitude` -- безразмерная
    целевая амплитуда.
    """

    a1: float = 2.8
    a2: float = 1.47
    u: float = 1.0
    lambda1: float = -0.7
    amplitude: float = 1.0

    @property
    def lambda2(self) -> float:
        return 3.0 * self.lambda1

    @property
    def m_phi(self) -> float:
        return self.a2 * self.amplitude

    @property
    def m(self) -> float:
        return float(np.exp(self.m_phi))


def _switch_condition(y: Any, model: SaccadeModel, b: float = 1.0) -> Any:
    """Условие на второй момент переключения (аналитическая выкладка автора).

    Ноль этой функции даёт `z2`; `z1` выражается через него замкнутой формулой.
    Формула перенесена без изменений из рабочего блокнота `saccad_model.ipynb`
    и намеренно не упрощалась: любое переписывание здесь меняет числа.
    """
    fy=(1/(model.a2**2*model.lambda1))*b*model.u*(-1+y)*y**6*(-2*model.a1*model.lambda1+model.a2*y+
        model.a1*model.lambda1*y+model.a2*y**2+model.a1*model.lambda1*y**2-(4*model.a2*b**2*(model.a2+
        3*model.a1*model.lambda1)**3*model.u**2*(model.a2*model.amplitude*model.lambda1+b*model.u)*(-1+y)**2)/
        (3*model.a2**2*model.amplitude*model.lambda1+model.a2*b*model.u+3*model.a1*b*model.lambda1*model.u-
        b*(model.a2+3*model.a1*model.lambda1)*model.m**(model.lambda1/model.u)*model.u*y**2-3*model.a2*b*model.lambda1*np.log(model.m))**3+((model.a2+
        3*model.a1*model.lambda1)*(27*model.a2**6*model.amplitude**3*model.lambda1**3+27*model.a2**5*model.amplitude**2*b*model.lambda1**2*model.u+
        81*model.a1*model.a2**4*model.amplitude**2*b*model.lambda1**3*model.u+9*model.a2**4*model.amplitude*b**2*model.lambda1*model.u**2+
        54*model.a1*model.a2**3*model.amplitude*b**2*model.lambda1**2*model.u**2+81*model.a1**2*model.a2**2*model.amplitude*b**2*model.lambda1**3*model.u**2+
        model.a2**3*b**3*model.u**3+5*model.a1*model.a2**2*b**3*model.lambda1*model.u**3+3*model.a1**2*model.a2*b**3*model.lambda1**2*model.u**3-
        9*model.a1**3*b**3*model.lambda1**3*model.u**3+8*model.a1*model.a2**2*b**3*model.lambda1*model.u**3*y+
        48*model.a1**2*model.a2*b**3*model.lambda1**2*model.u**3*y+72*model.a1**3*b**3*model.lambda1**3*model.u**3*y-
        4*model.a1*model.a2**2*b**3*model.lambda1*model.u**3*y**2-24*model.a1**2*model.a2*b**3*model.lambda1**2*model.u**3*y**2-
        36*model.a1**3*b**3*model.lambda1**3*model.u**3*y**2-3*b*(model.a2+3*model.a1*model.lambda1)*model.m**(model.lambda1/model.u)
        *model.u*(3*model.a2**2*model.amplitude*model.lambda1+b*(model.a2+3*model.a1*model.lambda1)*model.u)**2*y**2+
        3*b**2*(model.a2+3*model.a1*model.lambda1)**2*model.m**((2*model.lambda1)/model.u)*model.u**2*(3*model.a2**2*model.amplitude*model.lambda1+
        b*(model.a2+3*model.a1*model.lambda1)*model.u)*y**4+b**3*(model.a2+3*model.a1*model.lambda1)**2*model.m**((3*model.lambda1)/model.u)*
        model.u**3*y**6*(model.a1*model.lambda1*(1+4*(-2+y)*y)+model.a2*(3+4*(-2+y)*y))
        +model.a2*b*model.lambda1*np.log(model.m)*(-(9*model.a2**2*model.amplitude*model.lambda1+model.a2*b*model.u+3*model.a1*b*model.lambda1*model.u+
        2*model.a2*b*model.u*y+6*model.a1*b*model.lambda1*model.u*y-3*b*(model.a2+3*model.a1*model.lambda1)*model.m**(model.lambda1/model.u)*
        model.u*y**2)*(9*model.a2**2*model.amplitude*model.lambda1-model.a2*b*model.u*(-5+y*(2+3*model.m**(model.lambda1/model.u)*y))-
        3*model.a1*b*model.lambda1*model.u*(-5+y*(2+3*model.m**(model.lambda1/model.u)*y)))+
        27*model.a2*b*model.lambda1*np.log(model.m)*(3*model.a2**2*model.amplitude*model.lambda1+model.a2*b*model.u+3*model.a1*b*model.lambda1*model.u-
        b*(model.a2+3*model.a1*model.lambda1)*model.m**(model.lambda1/model.u)*model.u*y**2-model.a2*b*model.lambda1*np.log(model.m)))))/(3*model.a2**2*model.amplitude*model.lambda1+
        model.a2*b*model.u+3*model.a1*b*model.lambda1*model.u-b*(model.a2+3*model.a1*model.lambda1)*model.m**(model.lambda1/model.u)*model.u*y**2-3*model.a2*b*model.lambda1*np.log(model.m))**3)
    return fy


def switch_times(model: SaccadeModel | None = None, *, y0: float = 0.7,
                 b: float = 1.0) -> tuple[float, float, float]:
    """Моменты переключения управления, безразмерные.

    Возвращает `(t1, t2, t3)`: конец разгона, конец реверса, конец движения.
    `y0` -- начальное приближение для корня условия переключения.
    """
    model = model or SaccadeModel()
    z2 = float(np.asarray(fsolve(_switch_condition, y0, args=(model, b))).ravel()[0])
    z1 = float(
        (2 * b * (model.a2 + 3 * model.a1 * model.lambda1) * model.m ** (model.lambda1 / model.u)
         * model.u * (-1 + z2) * z2 ** 2)
        / (-3 * model.a2 ** 2 * model.amplitude * model.lambda1 - model.a2 * b * model.u
           - 3 * model.a1 * b * model.lambda1 * model.u
           + model.a2 * b * model.m ** (model.lambda1 / model.u) * model.u * z2 ** 2
           + 3 * model.a1 * b * model.lambda1 * model.m ** (model.lambda1 / model.u) * model.u * z2 ** 2
           + 3 * model.a2 * b * model.lambda1 * np.log(model.m))
    )
    w1 = z1 ** (1.0 / model.lambda1)
    w2 = z2 ** (1.0 / model.lambda1)
    n1, n2 = w1 / w2, w1
    n3 = model.m * n2 ** 2 / n1 ** 2
    return float(np.log(n1)), float(np.log(n2)), float(np.log(n3))


def trajectory(
        model: SaccadeModel | None = None,
        *,
        switch: tuple[float, float, float] | None = None,
        dt: float = DEFAULT_SACCADE_MODEL_DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Безразмерная траектория: угол, скорость, управление.

    Три участка -- разгон до `t1`, реверс до `t2`, торможение до `t3`; на стыках
    решение сшивается по непрерывности угла и скорости.
    """
    model = model or SaccadeModel()
    mt1, mt2, mt3 = switch if switch is not None else switch_times(model)
    a1, a2, u = model.a1, model.a2, model.u
    lam1, lam2 = model.lambda1, model.lambda2

    c1_i = -u * (a2 + a1 * lam2) / (a2 ** 2 * (lam1 - lam2))
    c2_i = u * (a2 + a1 * lam1) / (a2 ** 2 * (lam1 - lam2))
    c1_ii = (np.exp(-lam1 * mt1)
             * (a2 ** 2 * c1_i * np.exp(lam1 * mt1) * lam1 - a2 ** 2 * c1_i * np.exp(lam1 * mt1) * lam2
                + 2 * a2 * u + 2 * a1 * lam2 * u) / (a2 ** 2 * (lam1 - lam2)))
    c2_ii = -(np.exp(-lam2 * mt1)
              * (-a2 ** 2 * c2_i * np.exp(lam2 * mt1) * lam1 + a2 ** 2 * c2_i * np.exp(lam2 * mt1) * lam2
                 + 2 * a2 * u + 2 * a1 * lam1 * u) / (a2 ** 2 * (lam1 - lam2)))
    c1_iii = (np.exp(-lam1 * mt2)
              * (a2 ** 2 * c1_ii * np.exp(lam1 * mt2) * lam1 - a2 ** 2 * c1_ii * np.exp(lam1 * mt2) * lam2
                 - 2 * a2 * u - 2 * a1 * lam2 * u) / (a2 ** 2 * (lam1 - lam2)))
    c2_iii = -(np.exp(-lam2 * mt2)
               * (-a2 ** 2 * c2_ii * np.exp(lam2 * mt2) * lam1 + a2 ** 2 * c2_ii * np.exp(lam2 * mt2) * lam2
                  - 2 * a2 * u - 2 * a1 * lam1 * u) / (a2 ** 2 * (lam1 - lam2)))

    t1 = np.arange(0, mt1 - dt, dt)
    t2 = np.arange(mt1, mt2 - dt, dt)
    t3 = np.arange(mt2, mt3, dt)

    control = np.concatenate((u * t1,
                              -u * t2 + 2 * u * mt1 * np.ones_like(t2),
                              u * t3 + 2 * u * (mt1 - mt2)))
    angle_1 = c1_i * np.exp(lam1 * t1) + c2_i * np.exp(lam2 * t1) + u / a2 * t1 - u * a1 / a2 ** 2
    speed_1 = c1_i * lam1 * np.exp(lam1 * t1) + c2_i * lam2 * np.exp(lam2 * t1) + u / a2

    d1, d2 = -u, 2 * mt1 * u
    speed_2 = c1_ii * lam1 * np.exp(lam1 * t2) + c2_ii * lam2 * np.exp(lam2 * t2) + d1 / a2
    angle_2 = (c1_ii * np.exp(lam1 * t2) + c2_ii * np.exp(lam2 * t2)
               + d1 / a2 * t2 + (a2 * d2 - a1 * d1) / a2 ** 2)

    d1, d2 = u, 2 * u * (mt1 - mt2)
    speed_3 = c1_iii * lam1 * np.exp(lam1 * t3) + c2_iii * lam2 * np.exp(lam2 * t3) + d1 / a2
    angle_3 = (c1_iii * np.exp(lam1 * t3) + c2_iii * np.exp(lam2 * t3)
               + d1 / a2 * t3 + (a2 * d2 - a1 * d1) / a2 ** 2)

    angle = np.concatenate((angle_1, angle_2, angle_3))
    speed = np.concatenate((speed_1, speed_2, speed_3))
    return angle, speed, control


def _canonical_shape(model: SaccadeModel, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Форма саккады в нормированном виде: время 0…1, угол 0…1."""
    angle, _speed, _control = trajectory(model, dt=dt)
    span = angle[-1] - angle[0]
    if not np.isfinite(span) or span == 0:
        raise ValueError("модель не дала ненулевого перемещения: проверьте параметры")
    shape = (angle - angle[0]) / span
    tau = np.linspace(0.0, 1.0, shape.size)
    return tau, shape


def saccade_template(
        amplitude_deg: float,
        duration_s: float,
        *,
        fs_hz: float = DEFAULT_SACCADE_TEMPLATE_FS_HZ,
        model: SaccadeModel | None = None,
        dt: float = DEFAULT_SACCADE_MODEL_DT,
) -> pd.DataFrame:
    """Одна траектория семейства: угол и скорость во времени.

    Размерности: `amplitude_deg` `[град]`, `duration_s` `[с]`, `fs_hz` `[Гц]`.
    Возвращает таблицу `t_s`, `angle_deg`, `speed_deg_s`.
    """
    model = model or SaccadeModel()
    tau, shape = _canonical_shape(model, dt)
    n = max(2, int(round(float(duration_s) * float(fs_hz))) + 1)
    t_s = np.linspace(0.0, float(duration_s), n)
    angle = float(amplitude_deg) * np.interp(t_s / float(duration_s), tau, shape)
    speed = np.gradient(angle, t_s)
    return pd.DataFrame({"t_s": t_s, "angle_deg": angle, "speed_deg_s": speed})


def saccade_template_family(
        amplitudes_deg: Any,
        durations_s: Any,
        *,
        fs_hz: float = DEFAULT_SACCADE_TEMPLATE_FS_HZ,
        model: SaccadeModel | None = None,
        dt: float = DEFAULT_SACCADE_MODEL_DT,
) -> pd.DataFrame:
    """Семейство траекторий: одна форма на сетке амплитуд и длительностей.

    Возвращает длинную таблицу `amplitude_deg`, `duration_s`, `t_s`, `angle_deg`,
    `speed_deg_s` -- по строке на отсчёт каждой траектории.
    """
    model = model or SaccadeModel()
    frames = []
    for amplitude in np.atleast_1d(np.asarray(amplitudes_deg, dtype=float)):
        for duration in np.atleast_1d(np.asarray(durations_s, dtype=float)):
            frame = saccade_template(float(amplitude), float(duration),
                                     fs_hz=fs_hz, model=model, dt=dt)
            frame.insert(0, "duration_s", float(duration))
            frame.insert(0, "amplitude_deg", float(amplitude))
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["amplitude_deg", "duration_s", "t_s", "angle_deg", "speed_deg_s"])
    return pd.concat(frames, ignore_index=True)


def match_saccade_templates(
        data: pd.DataFrame,
        events: pd.DataFrame,
        *,
        col_gaze_angle: str,
        col_time: str = "T_s",
        model: SaccadeModel | None = None,
        durations_s: Any = None,
        dt: float = DEFAULT_SACCADE_MODEL_DT,
        pad_s: float = 0.0,
) -> pd.DataFrame:
    """Ищет в записи кривые, похожие на траектории семейства.

    Для каждого события из `events` (нужны `t_start_s` и `t_end_s`) перебираются
    длительности семейства; при каждой длительности амплитуда находится методом
    наименьших квадратов -- форма модели одна, поэтому подбирать её не нужно.
    Выбирается длительность с наименьшим остатком.

    Это тот же приём, что и подгонка сигмоид (`peaks_to_sigmoid_fits`), но кривая
    другая: не логистическая ступенька, а решение модели с релейным управлением.
    Сравнивать их между собой имеет смысл именно по остатку.

    Размерности: `col_time` `[s]`, `col_gaze_angle` `[град]`, `pad_s` `[с]`.
    Возвращает по строке на событие: подобранные амплитуда и длительность,
    пиковая скорость модели, доля объяснённой дисперсии `r2` и СКО остатка.
    """
    model = model or SaccadeModel()
    columns = ["t_start_s", "t_end_s", "amplitude_deg", "duration_s",
               "peak_speed_deg_s", "r2", "rmse_deg", "n_points"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    t = data[col_time].to_numpy(dtype=float)
    angle = data[col_gaze_angle].to_numpy(dtype=float)
    tau, shape = _canonical_shape(model, dt)

    rows: list[dict[str, float]] = []
    for t_start, t_end in zip(events["t_start_s"].to_numpy(dtype=float),
                              events["t_end_s"].to_numpy(dtype=float)):
        mask = (t >= t_start - pad_s) & (t <= t_end + pad_s)
        t_win, y_win = t[mask], angle[mask]
        good = np.isfinite(t_win) & np.isfinite(y_win)
        t_win, y_win = t_win[good], y_win[good]
        if t_win.size < 3:
            continue

        observed = y_win - y_win[0]
        grid = (np.asarray(durations_s, dtype=float) if durations_s is not None
                else np.array([t_end - t_start], dtype=float))
        best: dict[str, float] | None = None
        for duration in np.atleast_1d(grid):
            if not np.isfinite(duration) or duration <= 0:
                continue
            template = np.interp((t_win - t_win[0]) / float(duration), tau, shape,
                                 left=shape[0], right=shape[-1])
            denom = float(np.dot(template, template))
            if denom <= 0:
                continue
            amplitude = float(np.dot(template, observed) / denom)   # МНК по амплитуде
            residual = observed - amplitude * template
            ss_res = float(np.dot(residual, residual))
            ss_tot = float(np.dot(observed - observed.mean(), observed - observed.mean()))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if best is None or ss_res < best["ss_res"]:
                fitted = saccade_template(abs(amplitude), float(duration),
                                          fs_hz=max(1.0 / max(dt, 1e-9), 1.0), model=model, dt=dt)
                best = {
                    "t_start_s": float(t_start),
                    "t_end_s": float(t_end),
                    "amplitude_deg": float(amplitude),
                    "duration_s": float(duration),
                    "peak_speed_deg_s": float(np.nanmax(np.abs(fitted["speed_deg_s"].to_numpy()))),
                    "r2": float(r2),
                    "rmse_deg": float(np.sqrt(ss_res / t_win.size)),
                    "n_points": int(t_win.size),
                    "ss_res": ss_res,
                }
        if best is not None:
            best.pop("ss_res", None)
            rows.append(best)

    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
