"""Модель траектории саккады: форма, семейство и поиск похожих кривых.

Числа моментов переключения зафиксированы как эталон: формула переключения
перенесена из рабочего блокнота автора без упрощений, и любое её переписывание
обязано ломать этот тест.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyetracking_analytics import (
    SaccadeModel,
    match_saccade_templates,
    saccade_template,
    saccade_template_family,
    switch_times,
    trajectory,
)

# Значения, посчитанные исходной реализацией (saccad_model.ipynb) при параметрах
# по умолчанию: a1 = 2.8, a2 = 1.47, u = 1, lambda1 = -0.7, амплитуда 1.
REFERENCE_SWITCH = (2.3952438013, 3.6122006350, 3.9039136674)


def test_switch_times_match_reference() -> None:
    assert switch_times(SaccadeModel()) == pytest.approx(REFERENCE_SWITCH, abs=1e-9)


def test_trajectory_hits_the_target() -> None:
    """Движение приходит в заданную амплитуду с нулевой скоростью — это и есть
    условие, из которого находятся моменты переключения."""
    model = SaccadeModel()
    angle, speed, control = trajectory(model, dt=0.001)
    assert angle[0] == pytest.approx(0.0, abs=1e-9)
    assert angle[-1] == pytest.approx(model.amplitude, abs=1e-6)
    assert speed[-1] == pytest.approx(0.0, abs=1e-6)
    assert angle.size == speed.size == control.size
    # разгон, реверс, торможение: скорость сначала растёт, потом падает
    assert speed.max() > 0
    assert np.argmax(speed) < angle.size - 1


def test_template_respects_amplitude_and_duration() -> None:
    template = saccade_template(12.0, 0.04, fs_hz=1000.0)
    assert template["t_s"].iloc[0] == pytest.approx(0.0)
    assert template["t_s"].iloc[-1] == pytest.approx(0.04)
    assert template["angle_deg"].iloc[-1] == pytest.approx(12.0, rel=1e-6)
    assert template["speed_deg_s"].max() > 0


def test_peak_speed_grows_with_amplitude_and_falls_with_duration() -> None:
    """Поведение по смыслу: чем больше амплитуда и короче время, тем выше пик."""
    peak = lambda a, d: saccade_template(a, d, fs_hz=2000.0)["speed_deg_s"].max()  # noqa: E731
    assert peak(20.0, 0.04) > peak(10.0, 0.04)
    assert peak(10.0, 0.03) > peak(10.0, 0.06)


def test_family_is_one_shape_on_a_grid() -> None:
    family = saccade_template_family([5.0, 10.0], [0.03, 0.05], fs_hz=500.0)
    assert set(family["amplitude_deg"]) == {5.0, 10.0}
    assert set(family["duration_s"]) == {0.03, 0.05}
    # форма одна: нормированные кривые совпадают
    a = family[(family["amplitude_deg"] == 5.0) & (family["duration_s"] == 0.03)]
    b = family[(family["amplitude_deg"] == 10.0) & (family["duration_s"] == 0.03)]
    assert np.allclose(a["angle_deg"].to_numpy() / 5.0,
                       b["angle_deg"].to_numpy() / 10.0, atol=1e-9)


def test_match_recovers_amplitude_and_duration() -> None:
    """В записи с саккадой, построенной моделью, подбор находит её параметры."""
    fs, amplitude, duration, t_start = 500.0, 15.0, 0.045, 0.5
    t = np.arange(0.0, 2.0, 1.0 / fs)
    angle = np.zeros_like(t)
    template = saccade_template(amplitude, duration, fs_hz=fs)["angle_deg"].to_numpy()
    i = int(np.searchsorted(t, t_start))
    angle[i:i + template.size] = template
    angle[i + template.size:] = amplitude
    rng = np.random.default_rng(5)
    data = pd.DataFrame({"T_s": t, "angle": angle + rng.normal(0.0, 0.05, t.size)})
    events = pd.DataFrame({"t_start_s": [t_start], "t_end_s": [t_start + duration]})

    fit = match_saccade_templates(data, events, col_gaze_angle="angle",
                                  durations_s=np.arange(0.02, 0.08, 0.005))
    assert len(fit) == 1
    row = fit.iloc[0]
    assert row["amplitude_deg"] == pytest.approx(amplitude, rel=0.1)
    assert row["duration_s"] == pytest.approx(duration, abs=0.006)
    assert row["r2"] > 0.99


def test_match_returns_empty_frame_without_events() -> None:
    data = pd.DataFrame({"T_s": [0.0, 0.1], "angle": [0.0, 1.0]})
    empty = match_saccade_templates(data, pd.DataFrame(), col_gaze_angle="angle")
    assert empty.empty
    assert "amplitude_deg" in empty.columns
