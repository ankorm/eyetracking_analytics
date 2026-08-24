###################################################
# функции работы с исходными даными
###################################################

from dataclasses import dataclass

import numpy as np

from .report import (
    COMPUTED,
    FROM_DATA,
    FROM_DEFAULT,
    FROM_USER,
    Param,
    log_computation,
)
from .sampling import (
    SamplingInfo,
    decimate_to_rate,
    estimate_sampling_rate,
    require_seconds,
    resolve_target_rate,
)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


###################################################
# универсальные функции
###################################################
@dataclass(frozen=True)
class VelocityResult:
    """Мгновенная скорость и условия, при которых она получена.

    Все три компоненты скорости — в единицах входных координат **в секунду**.
    До введения контракта частоты `velocity_x`/`velocity_y` возвращались
    в «единицах за отсчёт», а `velocity_abs` — в «единицах в секунду»
    (пункт B-2 аудита).
    """

    t_s: np.ndarray                 # метки времени, к которым отнесены скорости
    v_abs: np.ndarray               # [град/с], угол большого круга за шаг
    v_x: np.ndarray                 # [град/с], производная первой координаты
    v_y: np.ndarray                 # [град/с], производная второй координаты
    fs_hz_used: float               # частота, на которой выполнен расчёт
    fs_hz_recording: float          # частота самой записи
    resampled: bool                 # выполнялась ли децимация
    n_samples_used: int
    sampling: SamplingInfo
    params_used: dict               # журнал: что и откуда взято


# расчет скорости с приведением к выбранной частоте
def angular_velocity_at_rate(
        t_s,
        gaze_x,
        gaze_y,
        fs_hz: float | None = None,
) -> VelocityResult:
    """
    Мгновенная скорость движения по двум координатам.

    Координаты угловые, в градусах: модуль скорости считается как угол большого
    круга между соседними направлениями, делённый на фактическую длину шага.
    Плоское приближение `hypot(dx, dy)` не используется: оно завышает скорость
    тем сильнее, чем дальше от нейтрали, -- вдвое уже при `pitch = 60` град,
    потому что меридианы сходятся.
    Покоординатные `v_x` и `v_y` остаются производными самих координат
    и в сумме `hypot` уже не дают `v_abs`.

    Контракт частоты:
      * `fs_hz=None` (по умолчанию) — расчёт на частоте записи, без пересчёта;
      * `fs_hz` задана — децимация к этой частоте: берутся реальные отсчёты,
        ближайшие к узлам сетки. Ни интерполяции, ни осреднения не вводится;
      * `fs_hz` выше частоты записи (с допуском два СКО длины шага) — ошибка.

    Параметры:
      t_s     — отсчёты времени в СЕКУНДАХ (обязательны)
      gaze_x — первая угловая координата взора, град
      gaze_y — вторая угловая координата взора, град
      fs_hz         — частота расчёта, Гц; None = частота записи

    Возвращает:
      VelocityResult — скорости и фактические условия расчёта
      (использованная частота, частота записи, признак децимации)
    """
    t = np.asarray(t_s, dtype=float).ravel()
    x = np.asarray(gaze_x, dtype=float).ravel()
    y = np.asarray(gaze_y, dtype=float).ravel()
    if not (t.size == x.size == y.size):
        raise ValueError(
            f"t_s, gaze_x и gaze_y должны быть одной длины, "
            f"получено {t.size}, {x.size}, {y.size}"
        )

    info = estimate_sampling_rate(t)
    require_seconds(info)
    fs_used, need_decimation = resolve_target_rate(info, fs_hz)

    if need_decimation:
        t_use, (x_use, y_use), _ = decimate_to_rate(t, fs_used, x, y)
    else:
        t_use, x_use, y_use = t, x, y

    dt = np.diff(t_use)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_x = np.diff(x_use) / dt
        v_y = np.diff(y_use) / dt

    # Модуль скорости -- по углу большого круга, а не по hypot(v_x, v_y).
    # На сфере ds^2 = dpitch^2 + cos^2(pitch)*dyaw^2, поэтому hypot(dyaw, dpitch)
    # всегда не меньше истинного углового пути: меридианы сходятся, и приращение
    # yaw на широте pitch стоит меньше, чем само по себе. То есть плоское
    # приближение скорость ЗАВЫШАЕТ -- вдвое уже при pitch = 60 град. Прежде
    # здесь стоял hypot, из-за чего одна и та же величина считалась
    # в библиотеке двумя разными способами: `stable_intervals_by_velocity`
    # уже брала большой круг. Покоординатные `v_x`, `v_y` остаются
    # производными самих координат.
    yaw = np.radians(x_use)
    pitch = np.radians(y_use)
    unit = np.stack([np.cos(pitch) * np.cos(yaw),
                     np.cos(pitch) * np.sin(yaw),
                     np.sin(pitch)], axis=1)
    dot = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_abs = np.degrees(np.arccos(dot)) / dt

    warnings_text = []
    if info.n_duplicated_timestamps:
        warnings_text.append(
            f"в метках времени {info.n_duplicated_timestamps} неположительных шагов "
            f"(дубли) — они исключены из оценки шага"
        )
    if info.n_gaps:
        warnings_text.append(
            f"пропусков в записи: {info.n_gaps}, наибольший {info.max_gap_s * 1000:.1f} мс"
        )

    params_used = log_computation(
        "мгновенная скорость",
        {
            "запись": Param(f"{info.span_s:.2f} с, {info.n_samples} отсчётов",
                            source=FROM_DATA),
            "частота записи": Param(info.fs_hz, "Гц", FROM_DATA,
                                    f"медианный шаг {info.dt_median_s * 1000:.3f} мс, "
                                    f"дрожание {info.jitter_rel * 100:.1f} %"),
            "частота расчёта": Param(
                float(fs_used), "Гц",
                FROM_USER if fs_hz is not None else FROM_DEFAULT,
                "задана пользователем" if fs_hz is not None else "не задана: частота записи",
            ),
            "приведение": Param(
                f"децимация, отобрано {t_use.size} из {info.n_samples} отсчётов"
                if need_decimation else "не требуется",
                source=COMPUTED,
                note="берутся реальные отсчёты, без осреднения" if need_decimation else "",
            ),
            "точек скорости": Param(int(v_abs.size), "шт.", COMPUTED),
        },
        warnings_text=warnings_text,
    )

    return VelocityResult(
        t_s=t_use[1:],
        v_abs=v_abs,
        v_x=v_x,
        v_y=v_y,
        fs_hz_used=float(fs_used),
        fs_hz_recording=float(info.fs_hz),
        resampled=bool(need_decimation),
        n_samples_used=int(t_use.size),
        sampling=info,
        params_used=params_used,
    )


###################################################
# данные в пикселях/миллиметрах
###################################################

#################################################
# сглаживание данных для метрических данных о положении зрачка/точки падения взора на плоскость
#################################################

@njit
def _nanmedian3(v0, v1, v2):
    """Медиана трёх значений с игнорированием NaN — как `np.nanmedian([v0, v1, v2])`."""
    vals = np.empty(3, dtype=np.float64)
    c = 0
    if not np.isnan(v0):
        vals[c] = v0
        c += 1
    if not np.isnan(v1):
        vals[c] = v1
        c += 1
    if not np.isnan(v2):
        vals[c] = v2
        c += 1

    if c == 0:
        return np.nan
    if c == 1:
        return vals[0]
    if c == 2:
        return 0.5 * (vals[0] + vals[1])

    x = vals[0]
    y = vals[1]
    z = vals[2]
    if (x <= y and y <= z) or (z <= y and y <= x):
        return y
    if (y <= x and x <= z) or (z <= x and x <= y):
        return x
    return z


@njit
def _star_eye_smoothing_numba(a, t, max_velocity, eps, max_data, min_data):
    """
    Внутренняя numba-реализация: принимает 1D ndarray и возвращает 1D ndarray.

    Точная копия python-ветки, включая рекуррентность: медиана берётся от
    ПРЕДЫДУЩЕГО ВЫХОДА `b[idx]`, а не от предыдущего входа `a[idx]`
    (`a[idx]` используется только в проверке «это быстрое движение»).
    Пункт N-1 аудита: до исправления numba-ветка реализовывала нерекуррентный
    фильтр, из-за чего результат зависел от наличия numba в окружении.

    Невалидные отсчёты дают NaN, а не повтор предыдущего значения (пункт C-2):
    потеря трекинга не должна выглядеть как идеальная фиксация.

    Порог приращения масштабируется фактическим шагом времени: при пропуске
    отсчётов соседи оказываются дальше друг от друга, и допустимое приращение
    должно быть больше.
    """
    n = a.shape[0]
    out_len = n - 2
    if out_len <= 0:
        tail = np.empty(1, dtype=np.float64)
        tail[0] = a[0]
        return tail

    b = np.empty(out_len, dtype=np.float64)
    b[0] = a[0]

    for idx in range(out_len - 1):
        k = a[idx]
        i = a[idx + 1]
        j = a[idx + 2]
        dt = t[idx + 1] - t[idx]

        in_range = (i > min_data + eps) and (i < max_data - eps)
        if not in_range or np.isnan(i):
            b[idx + 1] = np.nan
            continue

        if np.isnan(k) or np.isnan(dt) or dt <= 0:
            # Предыдущий отсчёт неизвестен: проверить приращение нечем.
            b[idx + 1] = _nanmedian3(b[idx], i, j)
            continue

        if abs(i - k) < max_velocity * dt:
            b[idx + 1] = _nanmedian3(b[idx], i, j)
        else:
            # Быстрое движение сглаживать нельзя: значение остаётся как есть.
            b[idx + 1] = i

    return b


def star_eye_smoothing(
    eye_coord,
    t_s,
    max_velocity,
    eps,
    max_data=None,
    min_data=None,
):
    """
    Сглаживает окулографический ряд медианным фильтром, не размывая быстрые движения.

    Параметры:
      eye_coord    - входной массив координаты глаза; единицы любые (мм, px,
                     градусы), важно лишь, чтобы `max_velocity` был в тех же
                     единицах за секунду. Единицы фиксируются ниже
                     по конвейеру, в `make_okulogram` и `smooth_eye_position_mm`
      t_s          - отсчёты времени в СЕКУНДАХ, той же длины
      max_velocity - максимальная физиологическая скорость в единицах `a` за секунду;
                     допустимое приращение между соседями = max_velocity * dt,
                     то есть порог учитывает фактический шаг времени
      eps          - отступ от границ области поиска зрачка
      max_data, min_data - границы валидных значений; по умолчанию берутся
                     из самих данных

    Возвращает:
      list[float] длины N-2. Невалидные отсчёты дают **NaN**: потеря трекинга
      не заменяется предыдущим значением, иначе она выглядит как идеальная
      фиксация и завышает все метрики стабильности (пункт C-2 аудита).

    Быстрое движение (приращение больше допустимого) не сглаживается: значение
    остаётся исходным, а не повторяет предыдущее.
    """
    a_np = np.asarray(eye_coord, dtype=float)
    t_np = np.asarray(t_s, dtype=float)
    if a_np.size == 0:
        raise ValueError("star_eye_smoothing: пустой входной массив")
    if t_np.size != a_np.size:
        raise ValueError(
            f"star_eye_smoothing: длины eye_coord и t_s должны совпадать, "
            f"получено {a_np.size} и {t_np.size}"
        )

    if max_data is None:
        max_data = float(np.nanmax(a_np))
    if min_data is None:
        min_data = float(np.nanmin(a_np))

    if HAS_NUMBA:
        b_arr = _star_eye_smoothing_numba(
            a_np, t_np, float(max_velocity), float(eps),
            float(max_data), float(min_data),
        )
        return b_arr.tolist()

    # Python-реализация (fallback, если нет numba). Обязана совпадать побитово.
    b = [float(a_np[0])]
    for idx in range(a_np.size - 3 + 1 - 1):
        k = a_np[idx]
        i = a_np[idx + 1]
        j = a_np[idx + 2]
        dt = t_np[idx + 1] - t_np[idx]

        if not (min_data + eps < i < max_data - eps) or np.isnan(i):
            b.append(np.nan)
            continue
        if np.isnan(k) or np.isnan(dt) or dt <= 0:
            b.append(float(np.nanmedian([b[-1], i, j])))
            continue
        if abs(i - k) < max_velocity * dt:
            b.append(float(np.nanmedian([b[-1], i, j])))
        else:
            b.append(float(i))
    return b


_VIVE_SMOOTHING_RENAMED = (
    "vive_eye_smoothing переименована в smooth_eye_position_mm. Функция ни от "
    "какого прибора не зависит, а единицы у неё не свободны: результат идёт "
    "в make_okulogram, где угол считается при r_eye = 12 мм. Проверьте, что "
    "координаты зрачка в МИЛЛИМЕТРАХ: метры дадут молча неверный угол, "
    "заниженный примерно в тысячу раз. Поведение и параметры прежние."
)


def vive_eye_smoothing(*args, **kwargs):
    """Переименована: единицы вынесены в имя, привязка к прибору убрана."""
    raise NotImplementedError(_VIVE_SMOOTHING_RENAMED)


def smooth_eye_position_mm(eye_x, eye_y, t_s, max_velocity, eps, eye_z=None, max_data=None):
    """
    Покоординатное сглаживание метрических координат зрачка. Единицы -- **миллиметры**.

    Единицы вынесены в имя, потому что дальше по конвейеру они уже не свободны:
    результат идёт в `make_okulogram`, где угол считается как
    `atan2(смещение, r_eye)` с `r_eye = 12` **мм**. Подача метров даёт молча
    неверный угол, заниженный примерно в тысячу раз, -- домножайте на 1000.
    Прежнее имя `vive_eye_smoothing` единиц не называло и привязывало функцию
    к прибору, хотя она ни от какого прибора не зависит.

    **Про бинокулярные записи.** Для приборов вроде Vive Pro Eye, которые дают
    положение каждого глаза отдельно, эта функция и `make_okulogram`
    применяются к каждому глазу **по отдельности**: у левого и правого своё
    нейтральное положение, и усреднять координаты до перевода в углы нельзя --
    межзрачковое расстояние вошло бы в результат как постоянное смещение.
    Порядок такой: сгладить координаты каждого глаза, найти нейтраль каждого
    (`find_central_position`), перевести каждый в углы (`make_okulogram`),
    и только потом, если нужен один ряд, объединять уже углы.

    Параметры:
      eye_x, eye_y, eye_z - координаты зрачка в мм, в системе, связанной
                     с головой (eye_z опционально)
      t_s          - отсчёты времени в секундах
      max_velocity - максимальная физиологическая скорость, мм за секунду
      eps, max_data - параметры области валидных значений, мм

    Возвращает:
      [x_, y_, z_] - список сглаженных координат длины N; крайние отсчёты
      берутся исходными. Это отличие от `star_eye_smoothing`, которая
      возвращает N-2 значения.
    """
    eye_x = np.asarray(eye_x, dtype=float)
    eye_y = np.asarray(eye_y, dtype=float)

    sx = star_eye_smoothing(eye_x, t_s, max_velocity, eps, max_data)
    sy = star_eye_smoothing(eye_y, t_s, max_velocity, eps, max_data)

    x_sm = [float(eye_x[0])] + sx + [float(eye_x[-1])]
    y_sm = [float(eye_y[0])] + sy + [float(eye_y[-1])]

    if eye_z is not None:
        eye_z = np.asarray(eye_z, dtype=float)
        sz = star_eye_smoothing(eye_z, t_s, max_velocity, eps, max_data)
        z_sm = [float(eye_z[0])] + sz + [float(eye_z[-1])]
    else:
        z_sm = None

    return [x_sm, y_sm, z_sm]

def find_central_position(eye_x, eye_y, eye_z=None):
    """
    Нейтральное положение зрачка: медиана по каждой координате.

    Допущение методики: медиана соответствует наиболее комфортному направлению
    взгляда. Возвращает `[x_c, y_c, z_c]`, где `z_c` равно None, если `z` не задан.

    **Про бинокулярные записи.** Нейтраль ищется для каждого глаза своя: у левого
    и правого она не совпадает, и общая на двоих внесла бы в углы постоянное
    смещение размером с половину межзрачкового расстояния.
    """
    x_c = float(np.nanmedian(np.asarray(eye_x, dtype=float)))
    y_c = float(np.nanmedian(np.asarray(eye_y, dtype=float)))
    z_c = float(np.nanmedian(np.asarray(eye_z, dtype=float))) if eye_z is not None else None
    return [x_c, y_c, z_c]

#: Радиус вращения зрачка вокруг центра глаза, мм. Пункт C-1 аудита:
#: решение автора методики — это **физиологическая** (антропометрическая)
#: константа, а не параметр прибора, поэтому она уместна как значение
#: по умолчанию. Переопределяется, если для испытуемого известно иное значение.
DEFAULT_EYE_RADIUS_MM: float = 12.0


def make_okulogram(eye_coord, neutral=None, r_eye: float = DEFAULT_EYE_RADIUS_MM):
    """
    Строит окулограмму: переводит линейное смещение зрачка в угол поворота глаза.

    Допущение методики: зрачок движется по сфере постоянного радиуса `r_eye`
    вокруг центра глаза, поэтому угол равен `atan2(x - x_c, r_eye)`.

    Параметры:
      eye_coord - смещение зрачка по одной оси, мм
      neutral   - нейтральное положение в мм; None = медиана самих данных
              (допущение методики: медиана соответствует наиболее комфортному
              направлению взгляда, поскольку глазодвигательные мышцы быстрые
              и длительное напряжение им несвойственно)
      r_eye - радиус вращения зрачка, мм; физиологическая константа,
              по умолчанию 12 мм

    Возвращает:
      массив углов в градусах, той же длины что `x`

    Пиксели сюда подавать нельзя: `r_eye` в миллиметрах, и размерность
    не сойдётся. Сначала переведите px в мм по известному анатомическому
    размеру на кадре либо считайте угол планарной геометрией
    (`mark_gaze_events_direction(geometry="plane")`, где `r` задаётся в тех же
    единицах, что координаты).

    **Про бинокулярные записи.** Для приборов вроде Vive Pro Eye, дающих
    положение каждого глаза отдельно, функция применяется к каждому глазу
    по отдельности, со своей нейтралью; см. `smooth_eye_position_mm`.
    """
    coord = np.asarray(eye_coord, dtype=float)
    if neutral is None:
        neutral = float(np.nanmedian(coord))
    return np.degrees(np.arctan2(coord - float(neutral), float(r_eye)))

###################################################
# данные в градусах
###################################################
