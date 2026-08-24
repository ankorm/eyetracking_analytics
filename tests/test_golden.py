"""Regression-тесты: результаты методики не должны меняться незаметно.

Если тест упал — это значит одно из двух:
  * в код внесено изменение, которое меняет численные результаты (тогда изменение
    должно быть осознанным, описанным в CHANGELOG, а фикстуры перегенерированы
    командой `python -m tests.generate_golden <случай>`);
  * расчёт пошёл не по эталонной ветке (см. `golden.reference_branches`).

Уровни сравнения — два, потому что бит-в-бит воспроизводимость достижима только
внутри одного окружения:

* **всегда**: схема, число строк, позиции NaN, целочисленные столбцы и столбцы
  решений методики (границы событий, длительности, счётчики) — точно;
  непрерывные величины — с допуском `golden.RTOL`;
* **в окружении, где снят эталон**: всё до последнего бита
  (`test_case_matches_golden_bit_exact`).

Причина второго уровня: numpy 2.x считает тригонометрию другими SIMD-ядрами,
чем 1.24, поэтому на заявленном минимуме зависимостей эксцентриситет и скорость
отличаются в последних битах (замерено: до 1.3e-11 при значениях в сотни
град/с). Требовать бит-в-бит по всей матрице поддержки нельзя, а требовать
неизменности разметки — можно и нужно.
"""
from __future__ import annotations

import json

import pytest

from eyetracking_analytics import for_raw_data as frd
from eyetracking_analytics import gaze_mark as gm

from . import golden


def _meta() -> dict:
    meta_path = golden.DATA_DIR / "meta.json"
    assert meta_path.exists(), "нет tests/data/meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", sorted(golden.CASES))
def test_case_matches_golden(case: str) -> None:
    frames = golden.run_case(case)
    for name, actual in frames.items():
        path = golden.path_for(case, name)
        assert path.exists(), (
            f"нет фикстуры {path.name}; сгенерируйте: python -m tests.generate_golden {case}"
        )
        expected = golden.load(case, name)
        problems = golden.compare(actual, expected, tol=golden.tolerance_for(case))
        assert not problems, f"{case}/{name}:\n  " + "\n  ".join(problems)


def _library_versions_match() -> bool:
    recorded = _meta()["окружение"]
    current = golden.environment()
    return all(recorded.get(k) == current.get(k) for k in ("numpy", "pandas", "scipy"))


@pytest.mark.skipif(
    not _library_versions_match(),
    reason="версии numpy/pandas/scipy отличаются от тех, в которых снят эталон: "
           "бит-в-бит совпадение недостижимо, работает сравнение с допуском",
)
@pytest.mark.parametrize("case", sorted(golden.CASES))
def test_case_matches_golden_bit_exact(case: str) -> None:
    """В окружении эталона расхождений быть не должно вообще — до последнего бита."""
    frames = golden.run_case(case)
    for name, actual in frames.items():
        problems = golden.compare(actual, golden.load(case, name), exact=True)
        assert not problems, f"{case}/{name}:\n  " + "\n  ".join(problems)


def test_fixtures_generated_on_reference_branch() -> None:
    """Эталон обязан быть снят на python-ветке, иначе он зависит от наличия numba."""
    recorded = _meta()["окружение"].get("эталонная_ветка")
    assert recorded == "python", (
        f"фикстуры сняты на ветке {recorded!r}, а не на эталонной python-ветке: "
        "перегенерируйте их командой `python -m tests.generate_golden`"
    )


# ────────────────────── ускоренная ветка против эталона, на уровне конвейера
#
# Пункт N-1 требует не бит-в-бит совпадения (оно недостижимо: numba суммирует
# циклом, numpy — попарно), а того, чтобы ускоритель не менял **решений**
# методики. Здесь это проверяется на полном конвейере, а не на отдельных
# функциях, как в test_numba_parity.py.

_PIPELINE_CASES = ("spherical_90hz", "sigmoid_90hz", "direction_angles_250hz", "eye_layer")


@pytest.mark.skipif(not gm.HAS_NUMBA, reason="numba не установлена")
@pytest.mark.parametrize("case", _PIPELINE_CASES)
def test_numba_does_not_change_markup(case: str) -> None:
    reference = golden.run_case(case)

    saved = (gm.HAS_NUMBA, frd.HAS_NUMBA)
    gm.HAS_NUMBA = True
    frd.HAS_NUMBA = True
    try:
        accelerated = golden.CASES[case]()
    finally:
        gm.HAS_NUMBA, frd.HAS_NUMBA = saved

    assert set(accelerated) == set(reference), "ускоренная ветка вернула другой набор таблиц"
    for name, ref in reference.items():
        problems = golden.compare(accelerated[name], ref, tol=golden.tolerance_for(case))
        assert not problems, (
            f"numba изменила результат в {case}/{name} больше, чем на округление:\n  "
            + "\n  ".join(problems)
        )
