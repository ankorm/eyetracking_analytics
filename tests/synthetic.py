"""Детерминированные синтетические записи для regression-тестов.

Все записи строятся с фиксированным seed, поэтому воспроизводимы на любой машине.
Реальные записи испытуемых сюда не попадают: фикстуры должны лежать в репозитории,
а данные испытуемых — нет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Крутизна сигмоидального перехода, 1/с. Соответствует переходу ~40 мс.
_STEP_K = 120.0


def _sigmoid_steps(t: np.ndarray, times: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Сумма сигмоидальных ступенек: модель последовательности переводов взора."""
    out = np.zeros_like(t)
    for ts, amp in zip(times, amps):
        z = np.clip(-_STEP_K * (t - ts), -700.0, 700.0)
        out += amp / (1.0 + np.exp(z))
    return out


def gaze_2d(
    *,
    fs_hz: float = 250.0,
    duration_s: float = 20.0,
    n_steps: int = 30,
    noise_deg: float = 0.05,
    jitter: float = 0.0,
    seed: int = 20260817,
) -> pd.DataFrame:
    """Двумерная запись взора: yaw/pitch в градусах, время в секундах.

    jitter — относительный разброс шага метки времени (0 = равномерная сетка).
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs_hz * duration_s))
    t = np.arange(n, dtype=float) / fs_hz
    if jitter > 0:
        dt = (1.0 / fs_hz) * (1.0 + rng.normal(0.0, jitter, n - 1))
        dt = np.clip(dt, 1e-6, None)
        t = np.concatenate([[0.0], np.cumsum(dt)])

    times = np.sort(rng.uniform(0.5, t[-1] - 0.5, n_steps))
    yaw = _sigmoid_steps(t, times, rng.normal(0.0, 6.0, n_steps))
    pitch = _sigmoid_steps(t, times, rng.normal(0.0, 4.0, n_steps))
    yaw += rng.normal(0.0, noise_deg, n)
    pitch += rng.normal(0.0, noise_deg, n)
    return pd.DataFrame({"T_s": t, "gaze_yaw": yaw, "gaze_pitch": pitch})


def planar_xy(
    *,
    fs_hz: float = 100.0,
    duration_s: float = 20.0,
    n_steps: int = 20,
    seed: int = 4242,
) -> pd.DataFrame:
    """Точка падения взора на плоскости: смещения вокруг нейтрального положения.

    Единицы — те же, что у расстояния до плоскости (обычно мм).
    """
    rng = np.random.default_rng(seed)
    n = int(round(fs_hz * duration_s))
    t = np.arange(n, dtype=float) / fs_hz
    times = np.sort(rng.uniform(0.5, duration_s - 0.5, n_steps))
    x = 10.0 + _sigmoid_steps(t, times, rng.normal(0.0, 3.0, n_steps))
    y = 5.0 + _sigmoid_steps(t, times, rng.normal(0.0, 2.0, n_steps))
    x += rng.normal(0.0, 0.02, n)
    y += rng.normal(0.0, 0.02, n)
    return pd.DataFrame({"T_s": t, "X": x, "Y": y})


def oculogram_1d_mm(*, n: int = 600, seed: int = 77) -> np.ndarray:
    """Одномерный ряд координаты зрачка в миллиметрах, с выбросами и NaN."""
    rng = np.random.default_rng(seed)
    a = 12.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    a[::53] += rng.normal(0.0, 1.5, a[::53].size)   # выбросы
    a[rng.integers(0, n, 4)] = np.nan               # потеря трекинга
    return a


def gaze_with_gaps(*, fs_hz: float = 200.0, duration_s: float = 10.0, seed: int = 5) -> pd.DataFrame:
    """Запись с пропусками и дублями метки времени — для контроля качества данных."""
    df = gaze_2d(fs_hz=fs_hz, duration_s=duration_s, n_steps=12, seed=seed)
    rng = np.random.default_rng(seed)
    drop = rng.choice(len(df), size=len(df) // 20, replace=False)
    df = df.drop(index=drop).reset_index(drop=True)
    df.loc[10:14, "gaze_yaw"] = np.nan
    return df


def orientation_quaternions(*, n: int = 400, seed: int = 4242) -> pd.DataFrame:
    """Последовательность ориентаций как кватернионы `(x, y, z, w)`.

    Строится как плавный поворот вокруг медленно меняющейся оси — так выглядит
    ориентация головы в записи. Без scipy: ось и угол задаются явно.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    angle = np.radians(20.0 * np.sin(2.0 * np.pi * t) + 5.0 * rng.normal(0.0, 1.0, n).cumsum() / n)
    axis = np.column_stack([
        np.sin(0.7 * np.pi * t) + 0.3,
        np.cos(0.4 * np.pi * t),
        0.5 + 0.2 * t,
    ])
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = 0.5 * angle
    sin_half = np.sin(half)
    return pd.DataFrame({
        "q_x": axis[:, 0] * sin_half,
        "q_y": axis[:, 1] * sin_half,
        "q_z": axis[:, 2] * sin_half,
        "q_w": np.cos(half),
    })


#: Именованный набор записей, на котором фиксируется поведение методики.
RECORDS = {
    "gaze_2d_250hz": lambda: gaze_2d(fs_hz=250.0, duration_s=20.0, n_steps=30),
    "gaze_2d_90hz": lambda: gaze_2d(fs_hz=90.0, duration_s=20.0, n_steps=20, seed=901),
    "gaze_2d_200hz_jitter": lambda: gaze_2d(fs_hz=200.0, duration_s=15.0, n_steps=25,
                                            jitter=0.05, seed=2002),
    "planar_100hz": planar_xy,
    "gaze_with_gaps": gaze_with_gaps,
}
