"""Разметка и анализ движений глаза и взора.

Различение из `docs/Sci_base.md`, определяющее всю терминологию: **движение глаза** —
относительно головы, его быстрое событие называется саккадой; **взор** —
ориентация относительно окружающего пространства, его быстрое событие —
**перевод взора**.

Взор всегда строится из положения головы и глаз относительно неподвижного
пространства: прибор даёт направление глаза в системе головы, поэтому взор —
это сумма ориентации головы и направления глаза. Иногда голову можно считать
неподвижной (подбородник, короткий интервал без движения головы) — тогда взор
совпадает с направлением глаза. Это допущение о записи: если голова двигалась,
а её движение не учтено, размечается движение глаза, а не взор.

Единицы во всём публичном API: секунды, градусы, герцы. Отсчёты времени
обязательны; ось в миллисекундах отвергается с явным сообщением.

Быстрый старт:

    import pandas as pd
    from eyetracking_analytics import mark_gaze_events_direction, enable_report_logging

    enable_report_logging()
    result = mark_gaze_events_direction(
        pd.read_csv("record.csv"),
        col_horizontal="gaze_yaw", col_vertical="gaze_pitch", col_time="T_s",
    )

Пороги по умолчанию выводятся из фактической частоты записи моделью главной
последовательности саккад, а не задаются константами.
"""
from __future__ import annotations

__version__ = "1.0.0"

# ── заданные константы: одно объявление на величину, см. `constants.py` ──
from . import constants as _constants

# ── разметка событий ──
# ── метрики ──
# ── моргания ──
from .blinks import (
    blink_mask,
    describe_blinks,
    find_blinks_by_gaps,
    find_blinks_by_threshold,
    refine_blink_bounds,
)
from .constants import (
    DEFAULT_BCEA_PROBABILITY,
    DEFAULT_FIXATION_VELOCITY_WINDOW_S,
    DEFAULT_MAX_AXIS_DT_S,
    DEFAULT_MIN_BLINK_GAP_S,
    DEFAULT_MIN_EVENT_DURATION_S,
    DEFAULT_REFINE_SEARCH_S,
    DEFAULT_SACCADE_MODEL_DT,
    DEFAULT_SACCADE_TEMPLATE_FS_HZ,
    DEFAULT_SIGMOID_P_HI,
    DEFAULT_SIGMOID_P_LO,
    DEFAULT_SIGMOID_R2_MIN,
    DEFAULT_SIGMOID_WINDOW_S,
    DEFAULT_TREND_ALPHA,
    EYE_RADIUS_MM,
    IDEAL_FIXATION_SPREAD_DEG,
    IDEAL_FIXATION_VELOCITY_DEG_S,
    LEGACY_GAUSSIAN_SIGMA_POINTS,
    LEGACY_MIN_PEAK_WIDTH_POINTS,
    LEGACY_SIGMOID_MIN_AMPLITUDE_DEG,
    LEGACY_SIGMOID_MIN_PEAK_VELOCITY_DEG_S,
    MAX_EYE_VELOCITY_DEG_S,
    MERGE_STAT_TOLERANCE,
    MIN_FIXATION_DURATION_S,
    N_SAMPLES_GAZE_SHIFT,
    N_SAMPLES_SACCADE,
    SPLIT_ALPHA,
    SPLIT_COV_WEIGHT,
    SPLIT_MIN_POINTS,
    SPLIT_SCORE_MIN_ALONE,
    SPLIT_SCORE_MIN_WITH_PVALUE,
)

# ── вход: ориентация головы и глаза -> направление взора ──
from .direction import (
    DEFAULT_QUATERNION_COLUMNS as DIRECTION_QUATERNION_COLUMNS,
)
from .direction import (
    DEFAULT_VECTOR_COLUMNS,
    FORWARD_XYZ,
    REPRESENTATIONS,
    GazeDirection,
    gaze_direction,
    orientation_to_direction,
)
from .eyetracking_funs import (
    BilinearFit,
    BilinearRegression,
    KsvResult,
    analyze_stable_intervals,
    find_std_fx,
    ksv,
    tr_length,
)

# ── сырые данные глаза ──
from .for_raw_data import (
    VelocityResult,
    angular_velocity_at_rate,
    find_central_position,
    make_okulogram,
    smooth_eye_position_mm,
    star_eye_smoothing,
)

# ── препроцессинг и статистика интервалов ──
from .gaze_mark import (
    axis_column_map,
    bcea_deg2,
    convex_hull_area_deg2,
    interval_stats_by_time,
    mark_eye_data_with_full_angle_from_angles,
    mark_eye_data_with_sigmoid,
    mark_eye_rest_from_angles,
    median_filter_dataframe,
    rename_axis_columns,
    resample_uniform_Ts_nan_event_ffill,
    stable_intervals_by_velocity,
)
from .markup import (
    DirectionMarkup,
    MarkupSummary,
    ScalarMarkup,
    describe_markup,
    mark_gaze_events_1d,
    mark_gaze_events_direction,
)

# ── модель порогов по частоте ──
from .models import (
    MinDetectableSaccade,
    PowerDurationModel,
    approx_params_by_fs,
    max_increment_deg,
    min_saccade_minima_by_sampling,
)

# ── ориентация ──
from .orientation import (
    angles_from_orientation,
    angular_distance_deg,
    parse_vector_column,
    quaternion_from_matrix_angles_deg,
    rotate_points,
)

# ── журнал вычислений ──
from .report import (
    Param,
    describe,
    enable_report_logging,
    get_logger,
    log_computation,
)

# ── модель траектории саккады ──
from .saccade_model import (
    SaccadeModel,
    match_saccade_templates,
    saccade_template,
    saccade_template_family,
    switch_times,
    trajectory,
)

# ── частота дискретизации ──
from .sampling import (
    SamplingInfo,
    decimate_to_rate,
    estimate_sampling_rate,
    require_seconds,
    resolve_target_rate,
)

__all__ = [
    "__version__",
    # заданные константы (модуль `constants`)
    "EYE_RADIUS_MM",
    "MAX_EYE_VELOCITY_DEG_S",
    "IDEAL_FIXATION_VELOCITY_DEG_S",
    "IDEAL_FIXATION_SPREAD_DEG",
    "MIN_FIXATION_DURATION_S",
    "DEFAULT_MIN_EVENT_DURATION_S",
    "DEFAULT_MAX_AXIS_DT_S",
    "DEFAULT_FIXATION_VELOCITY_WINDOW_S",
    "DEFAULT_TREND_ALPHA",
    "DEFAULT_BCEA_PROBABILITY",
    "DEFAULT_MIN_BLINK_GAP_S",
    "DEFAULT_REFINE_SEARCH_S",
    "SPLIT_ALPHA",
    "SPLIT_MIN_POINTS",
    "SPLIT_SCORE_MIN_WITH_PVALUE",
    "SPLIT_SCORE_MIN_ALONE",
    "SPLIT_COV_WEIGHT",
    "MERGE_STAT_TOLERANCE",
    "DEFAULT_SIGMOID_P_LO",
    "DEFAULT_SIGMOID_P_HI",
    "DEFAULT_SIGMOID_WINDOW_S",
    "DEFAULT_SIGMOID_R2_MIN",
    "LEGACY_SIGMOID_MIN_PEAK_VELOCITY_DEG_S",
    "LEGACY_SIGMOID_MIN_AMPLITUDE_DEG",
    "LEGACY_MIN_PEAK_WIDTH_POINTS",
    "LEGACY_GAUSSIAN_SIGMA_POINTS",
    "DEFAULT_SACCADE_MODEL_DT",
    "DEFAULT_SACCADE_TEMPLATE_FS_HZ",
    # разметка
    "mark_gaze_events_direction",
    "mark_gaze_events_1d",
    "DirectionMarkup",
    "ScalarMarkup",
    # вход: ориентация -> направление взора
    "gaze_direction",
    "orientation_to_direction",
    "GazeDirection",
    "REPRESENTATIONS",
    "FORWARD_XYZ",
    "DEFAULT_VECTOR_COLUMNS",
    "DIRECTION_QUATERNION_COLUMNS",
    "describe_markup",
    "MarkupSummary",
    # модель порогов
    "min_saccade_minima_by_sampling",
    "approx_params_by_fs",
    "max_increment_deg",
    "MinDetectableSaccade",
    "PowerDurationModel",
    "N_SAMPLES_SACCADE",
    "N_SAMPLES_GAZE_SHIFT",
    # модель траектории саккады
    "SaccadeModel",
    "switch_times",
    "trajectory",
    "saccade_template",
    "saccade_template_family",
    "match_saccade_templates",
    # частота
    "estimate_sampling_rate",
    "require_seconds",
    "resolve_target_rate",
    "decimate_to_rate",
    "SamplingInfo",
    # сырые данные глаза
    "make_okulogram",
    "find_central_position",
    "star_eye_smoothing",
    "smooth_eye_position_mm",
    "angular_velocity_at_rate",
    "VelocityResult",
    # ориентация
    "angles_from_orientation",
    "angular_distance_deg",
    "rotate_points",
    "quaternion_from_matrix_angles_deg",
    "parse_vector_column",
    # препроцессинг и статистика
    "median_filter_dataframe",
    "resample_uniform_Ts_nan_event_ffill",
    "interval_stats_by_time",
    "axis_column_map",
    "rename_axis_columns",
    "mark_eye_data_with_sigmoid",
    "mark_eye_data_with_full_angle_from_angles",
    "mark_eye_rest_from_angles",
    "stable_intervals_by_velocity",
    # моргания
    "find_blinks_by_gaps",
    "find_blinks_by_threshold",
    "refine_blink_bounds",
    "blink_mask",
    "describe_blinks",
    # метрики
    "ksv",
    "KsvResult",
    "tr_length",
    "find_std_fx",
    "analyze_stable_intervals",
    "bcea_deg2",
    "convex_hull_area_deg2",
    "BilinearRegression",
    "BilinearFit",
    # журнал
    "enable_report_logging",
    "describe",
    "get_logger",
    "log_computation",
    "Param",
]


def __getattr__(name: str):
    """Прежние имена констант объясняют замену, а не молча исчезают."""
    return _constants.renamed_constant(name)
