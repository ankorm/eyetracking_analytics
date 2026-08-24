"""Генератор эталонных фикстур.

Запуск из корня репозитория:

    python -m tests.generate_golden           # перегенерировать всё
    python -m tests.generate_golden models    # только отдельные случаи

Фикстуры фиксируют ТЕКУЩЕЕ поведение методики. Перегенерировать их допустимо
только вместе с осознанным изменением методики и записью в CHANGELOG.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

from . import golden


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "неизвестно"


def main(argv: list[str]) -> int:
    selected = argv or sorted(golden.CASES)
    unknown = [c for c in selected if c not in golden.CASES]
    if unknown:
        print(f"неизвестные случаи: {unknown}", file=sys.stderr)
        print(f"доступные: {sorted(golden.CASES)}", file=sys.stderr)
        return 2

    written = []
    for case in selected:
        frames = golden.run_case(case)   # всегда python-ветка: см. golden.reference_branches
        for name, frame in frames.items():
            path = golden.dump(case, name, frame)
            written.append(f"{path.name}  ({len(frame)} строк, {len(frame.columns)} колонок)")
            print(f"  записано {path.name}: {len(frame)} строк, {len(frame.columns)} колонок")

    meta_path = golden.DATA_DIR / "meta.json"
    meta = {
        "сгенерировано": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "окружение": golden.environment(),
        "случаи": selected,
        "комментарий": (
            "Эталон текущего поведения методики, снятый на python-ветках "
            "(golden.reference_branches). От наличия numba не зависит: согласие "
            "ускоренных ветвей проверяют test_numba_parity.py и "
            "test_golden.py::test_numba_does_not_change_markup."
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nвсего файлов: {len(written)}; метаданные: {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
