from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .for_raw_data import angular_velocity_at_rate
from .gaze_mark import build_stable_intervals_between_events, interval_stats_by_time
from .models import N_SAMPLES_SACCADE, min_saccade_minima_by_sampling
from .report import (
    COMPUTED,
    FROM_DATA,
    FROM_DEFAULT,
    FROM_MODEL,
    FROM_USER,
    Param,
    log_computation,
)

# matplotlib и seaborn нужны только классу biline_regression и импортируются
# по месту использования: библиотека не должна тянуть графический стек при импорте.


###################################################
# фичи
###################################################

# для анализа скоростей глаз при длительном рассматривании
# логнормальное распределение
def lognormal_distribution(x, mu, sigma=0.8):
    """
    Вычисляет значение логнормального распределения.
    Параметры:
      x     - значение аргумента
      mu    - среднее (в логарифмическом масштабе)
      sigma - стандартное отклонение (по умолчанию 0.8)

    Прежде рядом существовала `lognormal_distribution_no_sigma` с зашитой
    sigma = 0.78 — то же распределение с другой константой. Удалена
    как дубликат: значение задаётся параметром.
    """
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))


@dataclass(frozen=True)
class BilinearFit:
    """Результат подгонки ломаной из двух отрезков.

    Нужен потому, что по двум наклонам нельзя понять ни где излом, ни стоило ли
    вообще брать ломаную вместо прямой. Прежний класс отдавал наружу только
    наклоны, а точку излома хранил как долю от числа отсчётов — величину,
    по которой абсциссу излома не восстановить.
    """

    slope_left: float        # наклон левого отрезка
    slope_right: float       # наклон правого отрезка
    intercept: float         # свободный член левого отрезка
    break_x: float           # абсцисса излома, в единицах X
    break_index: int         # индекс излома в очищенных данных
    n_left: int              # отсчётов слева от излома
    n_right: int             # отсчётов справа
    n_valid: int             # отсчётов после отбрасывания NaN и inf
    r2: float                # доля объяснённой дисперсии ломаной
    rmse: float              # СКО остатка, в единицах Y
    r2_single_line: float    # то же для одной прямой: с чем сравнивать r2
    params_used: dict        # журнал: что и откуда взято

    @property
    def slopes(self) -> tuple[float, float]:
        """Пара наклонов — та самая величина, ради которой строится ломаная."""
        return (self.slope_left, self.slope_right)


#класс для аппроксимации ломаной
class BilinearRegression:
    """
    Кусочно-линейная регрессия: аппроксимация ломаной из двух отрезков.

    Применяется к распределению интервалов в осях `ln(Δt)` — `ln(f(Δt))`:
    наклоны двух отрезков и есть показатели, по которым оценивается Хёрст.
    Как именно наклон переводится в показатель Хёрста, зависит от способа
    построения оси и остаётся за вызывающим: класс отдаёт наклоны, а не
    интерпретацию.

    Требование к данным: `X` отсортирован по возрастанию. Излом ищется
    по индексу, поэтому на неупорядоченном `X` отрезки окажутся бессмысленными.
    """

    def __init__(self, threshold=0.5):
        """
        Инициализация.

        Параметры:
          threshold - положение точки излома как доля от размаха X, по умолчанию 0.5
                      (середина). Подгонкой не уточняется, см. `fit`.
        """
        self.threshold = threshold
        self.par_0 = [0, 0, 0]
        self.a_0, self.a_1, self.b_0 = *(self.par_0),
        #: результат последней подгонки
        self.result: BilinearFit | None = None
        # данные последней подгонки: нужны отрисовке, которая теперь отдельно
        self._x = None
        self._y = None
        self._n_break = None

    def fit(self, X, Y, *, plot: bool = False) -> BilinearFit:
        """
        Подгонка модели под данные X, Y методом минимизации квадратов ошибок.

        Подбираются **только три параметра прямых** `(a_0, a_1, b_0)`. Точка
        излома не оптимизируется: она берётся из `threshold` и во время
        минимизации не меняется. То есть `threshold` -- не начальное
        приближение, а само положение излома, заданное долей от размаха X;
        значение по умолчанию 0.5 ставит излом в середину размаха независимо
        от данных. Это поведение прежней версии, сохранённое без изменений:
        подбор излома был бы изменением методики и требовал бы перегенерации
        эталона.

        График по умолчанию **не строится**: подгонка -- расчёт, и открывать
        окно как побочный эффект она не должна, иначе класс нельзя вызвать
        в цикле по записям или в окружении без дисплея. Отрисовка вынесена
        в `plot_fit()`; прежнее поведение возвращает `fit(X, Y, plot=True)`.

        Возвращает `BilinearFit`: наклоны, абсциссу излома и меры качества.
        `threshold` больше не портится подгонкой, поэтому повторный `fit` тем же
        объектом даёт тот же результат, что и первый.
        """
        X = np.array(X)
        Y = np.array(Y)
        X = X[~np.isnan(Y)]
        Y = Y[~np.isnan(Y)]
        X = X[~np.isinf(Y)]
        Y = Y[~np.isinf(Y)]
        X = X[1:]
        Y = Y[1:]
        base_tr = min(X) + self.threshold * (max(X) - min(X))
        base_tr = np.argmin(abs(X - base_tr))
        # доля излома -- локальная величина: раньше она затирала self.threshold,
        # и второй fit тем же объектом стартовал не оттуда, что первый
        threshold = base_tr / len(X)
        res = minimize(
            lambda x: np.sum((X[:int(len(X) * threshold)] * x[0] + x[2] - Y[:int(len(X) * threshold)]) ** 2)
                      + np.sum((x[1] * X[int(len(X) * threshold):] + X[int(len(X) * threshold)] * (
                        x[0] - x[1]) + x[2] - Y[int(len(X) * threshold):]) ** 2),
            (self.a_0, self.a_1, self.b_0), method='nelder-mead')
        self.a_0, self.a_1, self.b_0 = *(res.x),

        n_break = int(len(X) * threshold)
        self._x = X
        self._y = Y
        self._n_break = n_break
        self.result = self._make_result(X, Y, n_break)

        if plot:
            self.plot_fit()
        return self.result

    def predict(self, X=None, n_break: int | None = None):
        """
        Значения ломаной. Без аргументов -- на данных последней подгонки.

        Разбиение идёт по индексу, а не по абсциссе: так же, как в подгонке,
        иначе предсказание разошлось бы с моделью на повторяющихся значениях X.
        """
        if X is None:
            if self._x is None:
                raise RuntimeError("сначала fit(): предсказывать не по чему")
            X, n_break = self._x, self._n_break
        if n_break is None:
            raise ValueError("для произвольного X нужен n_break")
        X = np.asarray(X, dtype=float)
        left = self.a_0 * X[:n_break] + self.b_0
        right = self.a_1 * X[n_break:] + X[n_break] * (self.a_0 - self.a_1) + self.b_0
        return np.concatenate([left, right])

    def _make_result(self, X, Y, n_break: int) -> BilinearFit:
        """Показатели подгонки: без них нельзя судить, оправдана ли ломаная."""
        resid = Y - self.predict(X, n_break)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(ss_res / len(Y))) if len(Y) else float("nan")

        # одна прямая -- база сравнения: ломаная должна её заметно превзойти
        slope, intercept = np.polyfit(X, Y, 1)
        ss_res_1 = float(np.sum((Y - (slope * X + intercept)) ** 2))
        r2_1 = 1.0 - ss_res_1 / ss_tot if ss_tot > 0 else float("nan")

        params = log_computation(
            "билинейная регрессия",
            {
                "отсчётов после отбрасывания NaN и inf": Param(len(X), "шт", FROM_DATA),
                "положение излома": Param(self.threshold, "доля размаха X", FROM_USER,
                                          note="подгонкой не уточняется"),
                "абсцисса излома": Param(float(X[n_break]), "", COMPUTED),
                "наклон слева": Param(float(self.a_0), "", COMPUTED),
                "наклон справа": Param(float(self.a_1), "", COMPUTED),
                "r2 ломаной": Param(r2, "", COMPUTED),
                "r2 одной прямой": Param(r2_1, "", COMPUTED,
                                         note="база сравнения: насколько излом вообще нужен"),
            },
        )
        return BilinearFit(
            slope_left=float(self.a_0),
            slope_right=float(self.a_1),
            intercept=float(self.b_0),
            break_x=float(X[n_break]),
            break_index=int(n_break),
            n_left=int(n_break),
            n_right=int(len(X) - n_break),
            n_valid=int(len(X)),
            r2=r2,
            rmse=rmse,
            r2_single_line=r2_1,
            params_used=params,
        )

    def plot_fit(self):
        """
        График подгонки: точки и два отрезка ломаной.

        Требует `matplotlib` и `seaborn` (extra `viz`); они импортируются
        по месту, чтобы библиотека не тянула графический стек при импорте.
        """
        if self._x is None:
            raise RuntimeError("сначала fit(): рисовать нечего")

        import matplotlib.pyplot as plt
        import seaborn as sns

        X, Y, n_break = self._x, self._y, self._n_break

        sns.set(rc={'figure.figsize': (14, 5)})  ## здесь размер картинки
        sns.set(font_scale=1)  ## а здесь размер подписей
        fig, ax = plt.subplots()
        ax.set_xlabel('ln(Δt), c')
        ax.set_ylabel('ln(f(Δt))')

        # Рисуем на текущих осях: возвращаемое значение seaborn не нужно.
        sns.scatterplot(x=X, y=Y)
        sns.lineplot(x=X[:n_break + 1], y=self.a_0 * X[:n_break + 1] + self.b_0)
        sns.lineplot(
            x=X[n_break:],
            y=self.a_1 * X[n_break:] + X[n_break] * (self.a_0 - self.a_1) + self.b_0,
        )
        return ax

    def get_der(self):
        """Пара наклонов. Оставлена ради прежнего кода; см. `BilinearFit.slopes`."""
        return (self.a_0, self.a_1)


#: Прежнее имя класса. Оставлено рабочим: класс вызывается из блокнотов автора,
#: а имя в snake_case выбивалось из остальных классов пакета.
biline_regression = BilinearRegression


@dataclass(frozen=True)
class KsvResult:
    """Коэффициент стабилизации взора и условия, при которых он получен.

    Без `fs_hz_used` и `v0_deg_s_used` по одному числу нельзя понять, на каком
    пороге и на какой сетке оно посчитано, а порог по модели меняется на порядок
    в диапазоне частот 60…500 Гц.
    """

    ksv: float                      # доля времени, 0..1
    t_stab_s: float
    t_total_s: float
    fs_hz_used: float
    fs_hz_recording: float
    resampled: bool
    v0_deg_s_used: float
    v0_deg_s_model: float           # что даёт модель на использованной частоте
    v0_source: str                  # "модель" | "пользователь"
    params_used: dict               # журнал: что и откуда взято


def ksv(
        t_s,
        gaze_x,
        gaze_y,
        v0_deg_s: float | None = None,
        fs_hz: float | None = None,
) -> KsvResult:
    """
    Коэффициент стабилизации взора: доля времени, в течение которой угловая
    скорость не превышает порог.

    Зависимость от частоты — часть методики: порог `v0_deg_s` по умолчанию берётся из
    модели главной последовательности, а она даёт тем меньший порог, чем выше
    частота (60 Гц — около 440 град/с, 250 Гц — около 28 град/с). Поэтому
    значения KSV сопоставимы только между записями одной частоты либо после
    приведения записей к общей частоте через `fs_hz`.

    Порог — пиковая скорость **саккады**, поэтому модель берётся при
    `N_SAMPLES_SACCADE`, а не при `N_SAMPLES_GAZE_SHIFT`, с которым работает
    разметка переводов взора. Ниже 222 Гц это разные числа (на 90 Гц — 178
    против 94 град/с), и разница здесь намеренная.

    Контракт частоты:
      * `fs_hz=None` — расчёт на частоте записи;
      * `fs_hz` задана — децимация к ней; порог модели берётся при
        `min(fs_hz, частота записи)`, потому что интерполяция не добавляет
        разрешающей способности;
      * `fs_hz` выше частоты записи (допуск два СКО шага) — ошибка.

    Параметры:
      t_s     — отсчёты времени в СЕКУНДАХ (обязательны)
      gaze_x — координата x (град или пикс)
      gaze_y — координата y
      v0_deg_s         — порог скорости; None = из модели по эффективной частоте
      fs_hz         — частота расчёта, Гц; None = частота записи

    Возвращает:
      KsvResult
    """
    vel = angular_velocity_at_rate(t_s, gaze_x, gaze_y, fs_hz=fs_hz)

    # Эффективная разрешающая способность: децимация её снижает, интерполяция
    # не повышает, поэтому порог модели берём по минимуму из двух частот.
    fs_effective = min(vel.fs_hz_used, vel.fs_hz_recording)
    v0_model = min_saccade_minima_by_sampling(
        fs_effective, n_samples=N_SAMPLES_SACCADE).v_peak_deg_s

    if v0_deg_s is None:
        v0_used = float(v0_model)
        v0_source = "модель"
    else:
        v0_used = float(v0_deg_s)
        v0_source = "пользователь"

    v = np.asarray(vel.v_abs, dtype=float)
    dt = np.diff(np.asarray(vel.t_s, dtype=float), prepend=np.nan)
    # Первому значению скорости соответствует шаг между двумя первыми отсчётами.
    if vel.t_s.size > 0:
        dt[0] = dt[1] if dt.size > 1 else np.nan

    minima = min_saccade_minima_by_sampling(fs_effective, n_samples=N_SAMPLES_SACCADE)

    warnings_text = []
    if v0_deg_s is not None and np.isfinite(v0_model) and v0_model > 0:
        ratio = max(v0_used / v0_model, v0_model / v0_used) if v0_used > 0 else np.inf
        if ratio > 2.0:
            warnings_text.append(
                f"задан порог v0_deg_s = {v0_used:.4g} град/с, модель для "
                f"{fs_effective:.4g} Гц даёт {v0_model:.4g} град/с "
                f"(отличие в {ratio:.1f} раза). Используется заданное значение."
            )

    m = np.isfinite(v) & np.isfinite(dt)
    t_total = float(vel.sampling.span_s)

    def _log(ksv_value: float, t_stab_value: float) -> dict:
        return log_computation(
            "коэффициент стабилизации взора",
            {
                "запись": Param(
                    f"{t_total:.2f} с, {vel.sampling.n_samples} отсчётов", source=FROM_DATA),
                "частота записи": Param(vel.fs_hz_recording, "Гц", FROM_DATA),
                "частота расчёта": Param(
                    vel.fs_hz_used, "Гц",
                    FROM_USER if fs_hz is not None else FROM_DEFAULT,
                    "децимация выполнена" if vel.resampled else "пересчёт не требовался"),
                "минимальная саккада": Param(
                    f"{minima.a_min_deg:.4g} град, {minima.t_min_s * 1000:.1f} мс",
                    source=FROM_MODEL,
                    note=f"таблица главной последовательности при {fs_effective:.4g} Гц"),
                "порог скорости фиксации": Param(
                    v0_used, "град/с", v0_source,
                    "пиковая скорость минимальной саккады" if v0_deg_s is None
                    else f"модель даёт {v0_model:.4g}"),
                "коэффициент стабилизации": Param(
                    ksv_value, "", COMPUTED,
                    f"стабильно {t_stab_value:.3f} с из {t_total:.3f} с"),
            },
            warnings_text=warnings_text,
        )

    if not np.any(m) or t_total <= 0:
        return KsvResult(
            ksv=np.nan, t_stab_s=np.nan, t_total_s=t_total,
            fs_hz_used=vel.fs_hz_used, fs_hz_recording=vel.fs_hz_recording,
            resampled=vel.resampled, v0_deg_s_used=v0_used,
            v0_deg_s_model=float(v0_model), v0_source=v0_source,
            params_used=_log(np.nan, np.nan),
        )

    # Время стабильности как сумма фактических шагов, а не как «число отсчётов,
    # делённое на частоту»: так результат корректен и на неравномерной сетке,
    # и при любой децимации (пункт B-1 аудита).
    t_stab = float(np.sum(dt[m][v[m] <= v0_used]))

    return KsvResult(
        ksv=t_stab / t_total,
        t_stab_s=t_stab,
        t_total_s=t_total,
        fs_hz_used=vel.fs_hz_used,
        fs_hz_recording=vel.fs_hz_recording,
        resampled=vel.resampled,
        v0_deg_s_used=v0_used,
        v0_deg_s_model=float(v0_model),
        v0_source=v0_source,
        params_used=_log(t_stab / t_total, t_stab),
    )


def tr_length(data, t_start_s, t_end_s, col_gaze_x, col_gaze_y, col_time='T_s'):
    """
    Вычисляет длину траектории (суммарное перемещение) на интервале [t1, t2].
    Параметры:
      data - DataFrame с данными
      t_start_s, t_end_s - границы временного интервала, с
      col_gaze_x     - имя колонки X
      col_gaze_y     - имя колонки Y
      col_time    - имя колонки времени (по умолчанию 'T_s')
    Возвращает:
      total_length - суммарная длина траектории
    """
    filtered_data = data[(data[col_time] >= t_start_s) & (data[col_time] <= t_end_s)]
    if filtered_data.shape[0] < 2:
        return 0.0

    coords = filtered_data[[col_gaze_x, col_gaze_y]].to_numpy(dtype=float)
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])

    return float(segment_lengths.sum())


def find_std_fx(data, t_start_s, t_end_s, col_value, col_time='T_s'):
    """
    Вычисляет стандартное отклонение (std) для выбранной характеристики на интервале.
    Параметры:
      data - DataFrame с данными
      t_start_s, t_end_s - границы временного интервала, с
      col_value  - имя колонки для анализа
      col_time    - имя колонки времени
    Возвращает:
      std      - стандартное отклонение
    """
    filtered_data = data[(data[col_time] >= t_start_s) & (data[col_time] <= t_end_s)].copy()
    return np.std(filtered_data[col_value])


def analyze_stable_intervals(
        df_events: pd.DataFrame,
        data: pd.DataFrame,
        *,
        col_gaze_x: str,
        col_gaze_y: str,
        col_time="T_s",
        col_t_start: str = "t_start_s",
        col_t_end: str = "t_end_s",
        min_gap_s: float = 0.0,
        trend_alpha: float = 0.05,
        trend_slope_min: float = 0.0,
) -> pd.DataFrame:
    """
    1) строит интервалы стабильности между событиями
    2) на каждом интервале считает статистику
    """
    stable = build_stable_intervals_between_events(
        df_events,
        col_t_start=col_t_start,
        col_t_end=col_t_end,
        min_gap_s=min_gap_s,
    )
    if stable.empty:
        return stable

    # Можно распараллелить и этот цикл, если он долгий
    # Но обычно интервалов не так много, как пиков.

    raw = []
    for _, r in stable.iterrows():
        t0 = float(r["t_stable_start"])
        tf = float(r["t_stable_end"])

        ex, ey, cov, n_valid, sx, px, sy, py, has_trend = interval_stats_by_time(
            data, t0, tf, col_gaze_x=col_gaze_x, col_gaze_y=col_gaze_y, col_time=col_time,
            trend_alpha=trend_alpha, trend_slope_min=trend_slope_min
        )

        fro_cov = float(np.linalg.norm(cov, ord="fro")) if np.isfinite(cov).all() else np.nan

        raw.append({
            **r.to_dict(),
            "mean_gaze_x": ex,
            "mean_gaze_y": ey,
            "fro_cov": fro_cov,
            "n_valid": n_valid,
            "slope_gaze_x_deg_s": sx,
            "p_gaze_x": px,
            "slope_gaze_y_deg_s": sy,
            "p_gaze_y": py,
            "has_trend": bool(has_trend),
        })

    return pd.DataFrame(raw)
