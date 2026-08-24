"""Пункт N-1 аудита: numba-ветки должны быть ТОЧНЫМИ копиями python-веток.

Расхождение исправлено: ускоренные ветки отвечают только за арифметику, а
p-value и медиана считаются одним способом. Тесты теперь защищают от повторного
расхождения.

Эталоном методики принята python-ветка:
  * ols_slope_pvalue  — p-value по распределению Стьюдента (numba-ветка сама
    задокументирована автором как "approximation ... erf based");
  * star_eye_smoothing — рекуррентная медиана (единственная реализация,
    существовавшая до коммита c08eda7 "оптимизация").
"""
from __future__ import annotations

import numpy as np
import pytest

from eyetracking_analytics import for_raw_data as frd
from eyetracking_analytics import gaze_mark as gm

pytestmark = pytest.mark.skipif(not gm.HAS_NUMBA, reason="numba не установлена")


def _with_numba(module, flag: bool):
    """Контекст переключения ускоренной ветки."""
    class _Ctx:
        def __enter__(self):
            self.saved = module.HAS_NUMBA
            module.HAS_NUMBA = flag
        def __exit__(self, *exc):
            module.HAS_NUMBA = self.saved
    return _Ctx()


@pytest.mark.parametrize("n", [3, 4, 5, 7, 10, 30, 100, 500])
def test_ols_slope_pvalue_numba_matches_python(n: int) -> None:
    rng = np.random.default_rng(1000 + n)
    t = np.sort(rng.uniform(0.0, n / 250.0, n))
    y = rng.normal(0.0, 0.3, n) + 2.0 * t

    with _with_numba(gm, True):
        slope_nb, p_nb, n_nb = gm.ols_slope_pvalue(t, y)
    with _with_numba(gm, False):
        slope_py, p_py, n_py = gm.ols_slope_pvalue(t, y)

    assert n_nb == n_py
    assert slope_nb == pytest.approx(slope_py, rel=1e-12)
    assert p_nb == pytest.approx(p_py, rel=1e-9), (
        f"p-value расходится: numba={p_nb!r}, python={p_py!r}, "
        f"отношение={p_py / p_nb if p_nb else float('nan'):.3f}"
    )


_SEEDS = {"случайный": 2001, "ступеньки": 2002, "выбросы": 2003, "с_nan": 2004}


@pytest.mark.parametrize("kind", ["случайный", "ступеньки", "выбросы", "с_nan"])
def test_star_eye_smoothing_numba_matches_python(kind: str) -> None:
    # Не hash(kind): для строк он рандомизируется на каждый процесс, и тест
    # каждый запуск шёл на других данных — то есть был невоспроизводим.
    rng = np.random.default_rng(_SEEDS[kind])
    n = 300
    if kind == "случайный":
        a = 50.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    elif kind == "ступеньки":
        a = 50.0 + 10.0 * np.sign(np.sin(np.linspace(0.0, 20.0, n)))
    elif kind == "выбросы":
        a = 50.0 + rng.normal(0.0, 0.1, n)
        a[::17] = 90.0
    else:
        a = 50.0 + rng.normal(0.0, 1.0, n)
        a[rng.integers(0, n, 5)] = np.nan

    t = np.arange(n, dtype=float) / 120.0
    with _with_numba(frd, True):
        b_nb = np.asarray(frd.star_eye_smoothing(a, t, 60.0, 0.5), dtype=float)
    with _with_numba(frd, False):
        b_py = np.asarray(frd.star_eye_smoothing(a, t, 60.0, 0.5), dtype=float)

    assert b_nb.size == b_py.size
    assert np.array_equal(np.isnan(b_nb), np.isnan(b_py))
    m = ~np.isnan(b_py)
    assert np.array_equal(b_nb[m], b_py[m]), (
        f"различается {int(np.sum(b_nb[m] != b_py[m]))} из {int(m.sum())} точек, "
        f"max |разница| = {float(np.max(np.abs(b_nb[m] - b_py[m]))):.4f}"
    )


def test_logistic_numba_matches_expit() -> None:
    """_logistic_numba и _logistic должны совпадать (клиппинг ±700 против expit)."""
    t = np.linspace(-2.0, 2.0, 501)
    for L, k, t0, y0 in [(5.0, 100.0, 0.0, -2.0), (1.0, -400.0, 0.5, 3.0), (8.0, 1.0, 0.0, 0.0)]:
        a = gm._logistic_numba(t, L, k, t0, y0)
        b = gm._logistic(t, L, k, t0, y0)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 10, 50, 300])
def test_pvalue_matches_scipy_student(n: int) -> None:
    """Тождество betainc должно совпадать с 2 * stats.t.sf — источник p-value."""
    from scipy import stats

    rng = np.random.default_rng(500 + n)
    t = np.sort(rng.uniform(0.0, n / 250.0, n))
    y = rng.normal(0.0, 0.3, n) + 2.0 * t

    _, p, _ = gm.ols_slope_pvalue(t, y)
    _, t_stat, dof, _ = gm._ols_core_numba(t.astype(float), y.astype(float))
    assert p == pytest.approx(float(2.0 * stats.t.sf(abs(t_stat), df=dof)), rel=1e-11)


def test_smoothing_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="пустой"):
        frd.star_eye_smoothing(np.array([], dtype=float), np.array([], dtype=float), 1.0, 0.1)


def test_smoothing_requires_matching_time_length() -> None:
    with pytest.raises(ValueError, match="длины"):
        frd.star_eye_smoothing(np.zeros(10), np.zeros(5), 1.0, 0.1)


def test_smoothing_keeps_nan_instead_of_previous_value() -> None:
    """C-2: потеря трекинга не должна выглядеть как идеальная фиксация."""
    a = np.full(20, 50.0)
    a[10] = 1000.0                       # выброс вне области валидных значений
    t = np.arange(20, dtype=float) / 100.0
    out = np.asarray(
        frd.star_eye_smoothing(a, t, max_velocity=10.0, eps=0.1,
                               max_data=100.0, min_data=0.0),
        dtype=float,
    )
    assert np.isnan(out[10]), "невалидный отсчёт обязан дать NaN, а не повтор предыдущего"


def test_smoothing_threshold_scales_with_time_step() -> None:
    """Порог приращения учитывает фактический шаг времени."""
    # Одиночный выброс 2.0 на фоне нуля.
    a = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=float)
    dense = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    sparse = dense * 10.0                # шаг в 10 раз больше
    kw = dict(max_velocity=50.0, eps=1e-9, max_data=10.0, min_data=-10.0)
    out_dense = np.asarray(frd.star_eye_smoothing(a, dense, **kw), dtype=float)
    out_sparse = np.asarray(frd.star_eye_smoothing(a, sparse, **kw), dtype=float)
    # Частый шаг: приращение 2.0 превышает 50 * 0.01 = 0.5, значит это быстрое
    # движение и значение сохраняется как есть.
    assert out_dense[2] == pytest.approx(2.0)
    # Редкий шаг: 50 * 0.1 = 5.0, приращение допустимо, и выброс сглаживается.
    assert out_sparse[2] == pytest.approx(0.0)


# ────────── замкнутые формы статистических тестов против scipy.stats
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_welch_and_levene_match_scipy(seed: int) -> None:
    """Замкнутые формы должны совпадать со scipy: они заменили его в горячем пути."""
    import warnings

    from scipy import stats

    rng = np.random.default_rng(seed)
    worst_t = worst_l = 0.0
    for _ in range(400):
        n1 = int(rng.integers(2, 60))
        n2 = int(rng.integers(2, 60))
        scale = 10.0 ** rng.uniform(-3, 3)
        a = rng.normal(rng.uniform(-5, 5), scale, n1)
        b = rng.normal(rng.uniform(-5, 5), scale * rng.uniform(0.2, 5.0), n2)

        p_t = gm._safe_ttest_ind_pvalue(a, b)
        ref_t = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
        p_l = gm._safe_levene_pvalue(a, b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_l = float(stats.levene(a, b, center="median").pvalue)
        if np.isfinite(ref_t) and ref_t > 0:
            worst_t = max(worst_t, abs(p_t - ref_t) / ref_t)
        if np.isfinite(ref_l) and ref_l > 0:
            worst_l = max(worst_l, abs(p_l - ref_l) / ref_l)
    assert worst_t < 1e-10, f"Уэлч расходится со scipy: {worst_t:.3e}"
    assert worst_l < 1e-10, f"Левен расходится со scipy: {worst_l:.3e}"


def test_statistical_tests_handle_degenerate_samples() -> None:
    """Вырожденные случаи должны вести себя как прежде."""
    const_a = np.full(5, 3.0)
    const_b = np.full(7, 3.0)
    other = np.full(7, 4.0)
    varying = np.array([1.0, 2.0, 3.0, 4.0])

    assert gm._safe_ttest_ind_pvalue(const_a, const_b) == 1.0
    assert gm._safe_levene_pvalue(const_a, const_b) == 1.0
    assert gm._safe_ttest_ind_pvalue(const_a, other) == 0.0
    assert gm._safe_levene_pvalue(const_a, other) == 1.0
    assert np.isnan(gm._safe_ttest_ind_pvalue(np.array([1.0]), varying))
    # Одна выборка постоянна, другая нет: тесты считаются обычным образом.
    assert 0.0 < gm._safe_ttest_ind_pvalue(const_a, varying) < 1.0
    assert 0.0 < gm._safe_levene_pvalue(const_a, varying) < 1.0


def test_split_criterion_no_longer_depends_on_scipy_stats() -> None:
    """N-2: раньше без scipy.stats критерий деградировал до эвристики score > 2.5."""
    import inspect

    source = inspect.getsource(gm._best_homogeneity_split)
    assert "stats is not None" not in source
    assert not hasattr(gm, "stats"), "мягкий импорт scipy.stats больше не нужен"
