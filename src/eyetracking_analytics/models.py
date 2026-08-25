###################################################
# модели для вычисдления параметров для анализа взора/движения глаза
###################################################

from dataclasses import dataclass
from math import log
from typing import NamedTuple, NoReturn

# Все заданные константы объявлены в одном месте (`constants.py`) и только
# импортируются сюда: одно объявление на величину.
from . import constants as _c
from .constants import (
    IDEAL_FIXATION_SPREAD_DEG,
    IDEAL_FIXATION_VELOCITY_DEG_S,
    MAX_EYE_VELOCITY_DEG_S,
    MIN_FIXATION_DURATION_S,
    N_SAMPLES_SACCADE,
)


def __getattr__(name: str) -> NoReturn:
    """Прежние имена констант объясняют замену, а не молча исчезают."""
    return _c.renamed_constant(name)


## параметры саккадических движений, которые можно определеить на заданной частоте
@dataclass(frozen=True)
class PowerDurationModel:
    """
    T(A) = T0_ms * A_deg^m
    V(A) = V0    * A_deg^n
    """
    T0_ms: float
    m: float
    V0: float
    n: float


# Таблица 4 статьи: Kruchinina A., Xu X. Analysis of saccadic main sequence
# relationship from a time-optimal control perspective // Russian Journal of
# Biomechanics. 2026. Vol. 30, no. 1. P. 108-114.
# DOI: https://doi.org/10.15593/RJBiomech/2026.1.10
_POWER_BY_FS: dict[int, PowerDurationModel] = {
    30:   PowerDurationModel(T0_ms=27.10, m=0.25, V0=4.05,  n=1.43),
    60:   PowerDurationModel(T0_ms=22.12, m=0.38, V0=45.68, n=0.78),
    90:   PowerDurationModel(T0_ms=27.29, m=0.32, V0=60.40, n=0.71),
    100:  PowerDurationModel(T0_ms=25.30, m=0.35, V0=62.74, n=0.70),
    250:  PowerDurationModel(T0_ms=28.69, m=0.32, V0=71.27, n=0.67),
    500:  PowerDurationModel(T0_ms=30.39, m=0.31, V0=72.59, n=0.66),
    1000: PowerDurationModel(T0_ms=30.73, m=0.30, V0=73.04, n=0.66),
    1250: PowerDurationModel(T0_ms=30.71, m=0.31, V0=72.86, n=0.66),
}

def _interp_logx(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Линейная интерполяция по log(x)."""
    if x0 == x1:
        return y0
    t = (log(x) - log(x0)) / (log(x1) - log(x0))
    return y0 + (y1 - y0) * t

def approx_params_by_fs(fs_hz: float) -> PowerDurationModel:
    """
    Кусочно-логарифмическая интерполяция параметров таблицы по частоте fs_hz.
    На узлах таблицы возвращает точные табличные значения.
    Частота 30 Гц отвергается: это нижняя граница таблицы, интерполировать ниже нечем.
    """
    if fs_hz <= 30:
        raise ValueError(f"fs_hz должна быть строго больше 30 Гц, получено {fs_hz}")
    freqs = sorted(_POWER_BY_FS.keys())

    # точное совпадение на узлах
    fs_int = int(fs_hz)
    if fs_hz == fs_int and fs_int in _POWER_BY_FS:
        return _POWER_BY_FS[fs_int]

    # за пределами — насыщаем краями
    if fs_hz <= freqs[0]:
        return _POWER_BY_FS[freqs[0]]
    if fs_hz >= freqs[-1]:
        return _POWER_BY_FS[freqs[-1]]

    # соседние узлы
    lo = max(f for f in freqs if f <= fs_hz)
    hi = min(f for f in freqs if f >= fs_hz)
    if lo == hi:
        return _POWER_BY_FS[lo]

    p_lo = _POWER_BY_FS[lo]
    p_hi = _POWER_BY_FS[hi]

    T0 = _interp_logx(fs_hz, lo, p_lo.T0_ms, hi, p_hi.T0_ms)
    m  = _interp_logx(fs_hz, lo, p_lo.m,     hi, p_hi.m)
    V0 = _interp_logx(fs_hz, lo, p_lo.V0,    hi, p_hi.V0)
    n  = _interp_logx(fs_hz, lo, p_lo.n,     hi, p_hi.n)

    return PowerDurationModel(T0_ms=T0, m=m, V0=V0, n=n)


class MinDetectableSaccade(NamedTuple):
    """Граница разрешающей способности записи при данной частоте.

    Все времена — **в секундах**: остальное API работает в секундах, а прежний
    возврат в миллисекундах приводил к ручному делению на 1000 в каждом вызове.
    Табличные константы модели (`T0_ms`) остаются в миллисекундах, как в статье.
    """

    a_min_deg: float          # минимальная детектируемая амплитуда саккады, град
    t_min_s: float            # минимальная длительность саккады, с
    v_peak_deg_s: float       # пиковая скорость саккады амплитуды a_min, град/с
    a_min_by_model_deg: float # амплитуда по критерию дискретизации, до предела
    a_min_at_floor: bool      # амплитуда упёрлась в физиологический предел
    v_fixation_deg_s: float   # порог скорости на фиксации, град/с
    t_fixation_s: float       # интервал, по которому получен порог фиксации, с
    v_fixation_by_model_deg_s: float  # то же до применения физиологического предела
    v_fixation_at_floor: bool # порог упёрся в физиологический предел
    spread_max_deg: float     # допустимый пространственный разброс фиксации, град
    model: PowerDurationModel # параметры главной последовательности на этой частоте


def min_saccade_minima_by_sampling(
    fs_hz: float,
    n_samples: int = N_SAMPLES_SACCADE,
    return_strict_sampling_duration: bool = False,
    min_fixation_duration_s: float = MIN_FIXATION_DURATION_S,
    ideal_fixation_velocity_deg_s: float = IDEAL_FIXATION_VELOCITY_DEG_S,
    ideal_fixation_spread_deg: float = IDEAL_FIXATION_SPREAD_DEG,
) -> MinDetectableSaccade:
    """
    Минимум по критерию дискретизации: >= n_samples отсчётов на саккаду.

    Заменяет прежнюю eye_params(fr, n). Параметры (T0, m, V0, n) берутся как
    approx_params_by_fs(fs_hz).

    Параметры:
      fs_hz    — частота записи, Гц; строго больше 30 (нижняя граница таблицы)
      n_samples — минимальное число отсчётов на саккаду, >= 2
      return_strict_sampling_duration — вернуть строго n_samples/fs вместо
                                        длительности, согласованной с моделью
      min_fixation_duration_s — минимальная длительность фиксации, с; нижняя
                                граница интервала, по которому считается порог
                                скорости на фиксации
      ideal_fixation_velocity_deg_s — скорость идеальной фовеальной фиксации,
                                град/с; нижняя граница порога
      ideal_fixation_spread_deg — пространственный разброс идеальной фовеальной
                                фиксации, град

    Возвращает:
      MinDetectableSaccade(a_min_deg [град], t_min_s [с], v_peak_deg_s [град/с],
                           v_fixation_deg_s [град/с], t_fixation_s [с], model)

    Порог скорости на фиксации выводится из тех же величин:
    `v_fix = max(a_min / max(t_min, min_fixation_duration_s), v_ideal)` — «если за
    минимальную длительность фиксации смещение меньше минимальной детектируемой
    амплитуды, это фиксация, а не движение», но не строже физиологического предела.

    Физиологический предел применяется к **амплитуде**, а пороги скорости выводятся
    уже из неё. Минимальная осмысленная амплитуда следует из тех же двух величин:

        a_floor = ideal_fixation_velocity_deg_s * min_fixation_duration_s = 0.25 град

    это амплитуда, дающая ровно скорость идеальной фиксации за минимальную
    длительность фиксации. Ниже неё движение неотличимо от фиксационного дрейфа,
    поэтому саккадой считаться не может. Критерий дискретизации даёт амплитуду
    меньше этой границы примерно с 222 Гц (`n_samples = 4`, саккада) и с 174 Гц
    (`n_samples = 3`, перевод взора), и без предела модель
    выдавала бы физически бессмысленные пороги: пиковую скорость саккады
    0.32 град/с при 1250 Гц. С пределом пиковая скорость выходит на 26–29 град/с,
    а длительность — на ~20 мс, то есть на правдоподобные значения микросаккады.

    Обе нижние границы содержательны и работают в одну сторону: чем выше частота
    записи, тем точнее определяется фиксация, поэтому модельная оценка убывает
    с ростом частоты — но опуститься ниже скорости идеальной фовеальной фиксации
    (`IDEAL_FIXATION_VELOCITY_DEG_S`) она не должна: медленнее взор на фиксации
    не бывает, и более строгий порог просто отвергал бы настоящие фиксации.
    Флаг `v_fixation_at_floor` показывает, какая из двух границ сработала.

    Допустимый пространственный разброс фиксации строится по тому же принципу:
    `spread_max = max(a_min_deg, ideal_fixation_spread_deg)`. Разброс меньше
    минимальной детектируемой амплитуды неотличим от шума измерения, поэтому на
    низких частотах, где `a_min_deg` велика, допустимый разброс выше; при высоких
    частотах он выходит на разброс идеальной фовеальной фиксации.

    Прежде порог фиксации модели не имел вовсе: унаследованная `eye_params`
    при любых реальных частотах возвращала константу 5 град/с, а в рабочих
    расчётах он задавался вручную как `v_peak/2` или `v_peak/100` — расхождение
    в 50 раз между двумя анализами одной методики.
    """
    if fs_hz <= 30:
        raise ValueError(f"fs_hz должна быть строго больше 30 Гц, получено {fs_hz}")
    if n_samples < 2:
        raise ValueError("n_samples should be >= 2")

    params = approx_params_by_fs(fs_hz)

    if min_fixation_duration_s <= 0:
        raise ValueError(
            f"min_fixation_duration_s должна быть положительной, "
            f"получено {min_fixation_duration_s}"
        )
    if ideal_fixation_velocity_deg_s <= 0:
        raise ValueError(
            f"ideal_fixation_velocity_deg_s должна быть положительной, "
            f"получено {ideal_fixation_velocity_deg_s}"
        )

    T_req_ms = 1000.0 * n_samples / fs_hz
    A_by_model = (T_req_ms / params.T0_ms) ** (1.0 / params.m)

    # Физиологический предел амплитуды: ниже неё движение неотличимо от
    # фиксационного дрейфа. Все пороги выводятся уже из итоговой амплитуды.
    A_floor = float(ideal_fixation_velocity_deg_s) * float(min_fixation_duration_s)
    min_amplitude_deg = max(A_by_model, A_floor)

    T_model_ms = params.T0_ms * (min_amplitude_deg ** params.m)
    V_peak_deg_s = params.V0 * (min_amplitude_deg ** params.n)

    T_out_ms = T_req_ms if return_strict_sampling_duration else T_model_ms
    t_min_s = T_out_ms / 1000.0

    t_fixation_s = max(t_min_s, float(min_fixation_duration_s))
    V_fix_model = min_amplitude_deg / t_fixation_s
    V_fixation_deg_s = max(V_fix_model, float(ideal_fixation_velocity_deg_s))

    return MinDetectableSaccade(
        a_min_deg=min_amplitude_deg,
        t_min_s=t_min_s,
        v_peak_deg_s=V_peak_deg_s,
        a_min_by_model_deg=A_by_model,
        a_min_at_floor=bool(min_amplitude_deg > A_by_model),
        v_fixation_deg_s=V_fixation_deg_s,
        t_fixation_s=t_fixation_s,
        v_fixation_by_model_deg_s=V_fix_model,
        v_fixation_at_floor=bool(V_fixation_deg_s > V_fix_model),
        spread_max_deg=max(min_amplitude_deg, float(ideal_fixation_spread_deg)),
        model=params,
    )

def max_increment_deg(fs_hz, v_max=MAX_EYE_VELOCITY_DEG_S):
    """
    Вычисляет максимальное допустимое приращение координат (в градусах) за один кадр.
    Параметры:
      fs_hz - частота съемки (Гц)
      v_max - максимальная физиологическая скорость глаза (град/с),
              по умолчанию MAX_EYE_VELOCITY_DEG_S
    Возвращает:
      max_delta - максимальное смещение за 1/fs_hz сек
    """
    return v_max / fs_hz
