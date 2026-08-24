"""Билинейная регрессия: расчёт без графики и видимые показатели.

Класс нужен для оценки показателя Хёрста: наклоны ломаной в осях
`ln(Δt)` -- `ln(f(Δt))` и есть искомые величины. Раньше `fit` рисовал график
как побочный эффект и безусловно импортировал matplotlib с seaborn, поэтому
в установке без extra `viz` -- а именно так ставит CI -- расчёт падал.
Эти тесты закрепляют: подгонка ничего не рисует, показатели видны наружу,
а числа не изменились.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from eyetracking_analytics import BilinearFit, BilinearRegression
from eyetracking_analytics import eyetracking_funs as ef


def _broken_line(n: int = 200, break_x: float = 0.4, seed: int = 0):
    """Ломаная с изломом и двумя выбросами, которые обязаны быть отброшены."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = np.where(x < break_x, 1.8 * x + 0.5, 0.3 * x + 1.1) + rng.normal(0, 0.05, n)
    y[5] = np.nan
    y[7] = np.inf
    return x, y


def test_fit_does_not_draw_and_needs_no_graphics_stack():
    """Главное свойство: расчёт не требует графического стека."""
    x, y = _broken_line()

    # сравниваем со снимком: графический стек мог подтянуть и кто-то другой
    before = set(sys.modules)
    BilinearRegression().fit(x, y)
    imported = set(sys.modules) - before

    assert not {m for m in imported if m.startswith(("matplotlib", "seaborn"))}


def test_fit_returns_visible_indicators():
    """Наружу видны наклоны, излом и меры качества, а не только пара чисел."""
    x, y = _broken_line()
    fit = BilinearRegression().fit(x, y)

    assert isinstance(fit, BilinearFit)
    assert fit.slopes == (fit.slope_left, fit.slope_right)
    assert fit.slope_left > fit.slope_right          # левый отрезок круче
    assert fit.n_valid == fit.n_left + fit.n_right
    assert fit.n_valid == len(x) - 3                 # NaN, inf и отброшенный первый
    assert min(x) < fit.break_x < max(x)
    assert 0.0 <= fit.r2 <= 1.0
    assert fit.r2 > fit.r2_single_line               # ломаная объясняет больше прямой
    assert fit.rmse > 0
    assert fit.params_used


def test_repeated_fit_is_idempotent():
    """Прежняя версия затирала threshold результатом, и второй fit расходился."""
    x, y = _broken_line()
    model = BilinearRegression()
    first = model.fit(x, y)
    second = model.fit(x, y)

    assert first.slopes == second.slopes
    assert first.break_x == second.break_x
    assert model.threshold == 0.5


def test_break_point_is_not_optimized():
    """Излом задаётся threshold и подгонкой не уточняется -- это надо видеть.

    Свойство прежней версии, сохранённое без изменений: minimize подбирает
    только (a_0, a_1, b_0). Тест зафиксирует момент, если излом начнут подбирать.
    """
    x, y = _broken_line()
    at_half = BilinearRegression(threshold=0.5).fit(x, y)
    at_third = BilinearRegression(threshold=0.75).fit(x, y)

    assert at_half.break_x != at_third.break_x
    # 0.5 ставит излом в середину размаха X, а не в истинный излом данных 0.4
    assert at_half.break_x == pytest.approx(min(x) + 0.5 * (max(x) - min(x)), abs=0.05)


def test_predict_matches_the_fitted_line():
    """Предсказание -- та же ломаная, по которой считается r2."""
    x, y = _broken_line()
    model = BilinearRegression()
    fit = model.fit(x, y)
    pred = model.predict()

    assert len(pred) == fit.n_valid
    # непрерывность в точке излома: два отрезка сходятся
    assert pred[fit.break_index - 1] == pytest.approx(pred[fit.break_index], abs=0.2)


def test_plot_requires_a_previous_fit():
    with pytest.raises(RuntimeError):
        BilinearRegression().plot_fit()


def test_old_name_still_works():
    """Класс вызывается из блокнотов автора: прежнее имя оставлено рабочим."""
    x, y = _broken_line()
    assert ef.biline_regression is BilinearRegression

    old_style = ef.biline_regression()
    old_style.fit(x, y)
    assert old_style.get_der() == BilinearRegression().fit(x, y).slopes
