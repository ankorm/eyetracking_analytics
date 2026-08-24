###################################################
# Журнал вычислений: какие параметры использованы и откуда они взяты
###################################################
"""Человекочитаемый отчёт о том, при каких значениях получен результат.

Зачем. Число без условий его получения непроверяемо. Порог скорости по модели
меняется на порядок в диапазоне 60…500 Гц, окно сглаживания зависит от частоты,
часть порогов задаётся пользователем вручную — и по одному значению KSV или по
числу саккад нельзя понять, что именно было применено. Журнал делает это видимым.

Библиотека **не печатает в stdout**: сообщения идут в логгер
`eyetracking_analytics`. Для работы в блокноте достаточно один раз вызвать
`enable_report_logging()`.

Каждый параметр несёт **источник**: из данных, из модели, задан пользователем
или взят по умолчанию. Без источника строка «порог 20.99 град/с» бесполезна.

Тот же набор значений возвращается в машинном виде и попадает в поле
`params_used` результатов — чтобы сохраняться рядом с таблицами и позволять
перепроверку.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LOGGER_NAME = "eyetracking_analytics"

#: Источники значений параметров.
FROM_DATA = "из данных"
FROM_MODEL = "модель"
FROM_USER = "пользователь"
FROM_DEFAULT = "по умолчанию"
COMPUTED = "вычислено"

_NAME_WIDTH = 34


def get_logger() -> logging.Logger:
    """Логгер библиотеки. По умолчанию сообщения никуда не выводятся."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(logging.NullHandler())
    return logger


def enable_report_logging(level: int = logging.INFO, stream: Any = None) -> logging.Handler:
    """Включает вывод журнала (удобно в блокноте).

    Возвращает добавленный обработчик, чтобы его можно было убрать
    через `logging.getLogger(LOGGER_NAME).removeHandler(...)`.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(min(level, logger.level or level))
    return handler


@dataclass(frozen=True)
class Param:
    """Значение параметра вместе с единицами, источником и пояснением."""

    value: Any
    unit: str = ""
    source: str = ""
    note: str = ""

    def format_value(self) -> str:
        v = self.value
        if isinstance(v, bool):
            text = "да" if v else "нет"
        elif isinstance(v, int):
            text = f"{v}"
        elif isinstance(v, float):
            if v != v:                       # NaN
                text = "нет данных"
            elif v == 0.0:
                text = "0"
            elif abs(v) >= 1e5 or abs(v) < 1e-3:
                text = f"{v:.4g}"
            else:
                text = f"{v:.4f}".rstrip("0").rstrip(".")
        else:
            text = str(v)
        return f"{text} {self.unit}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "значение": self.value,
            "единицы": self.unit,
            "источник": self.source,
            "пояснение": self.note,
        }


def format_report(title: str, params: Mapping[str, Param]) -> str:
    """Собирает текст отчёта: имя, значение с единицами, источник, пояснение."""
    lines = [f"[{LOGGER_NAME}] {title}"]
    for name, param in params.items():
        dots = "." * max(3, _NAME_WIDTH - len(name))
        tail = ""
        if param.source or param.note:
            inside = param.source
            if param.note:
                inside = f"{inside}: {param.note}" if inside else param.note
            tail = f"   ({inside})"
        lines.append(f"  {name} {dots} {param.format_value()}{tail}")
    return "\n".join(lines)


def log_computation(
        title: str,
        params: Mapping[str, Param],
        *,
        warnings_text: list[str] | None = None,
        logger: logging.Logger | None = None,
) -> dict[str, dict[str, Any]]:
    """Пишет отчёт в журнал и возвращает его машинное представление.

    Предупреждения (расхождение заданного и модельного значения, потеря
    наблюдений, приведение частоты) выводятся отдельными строками уровня WARNING,
    чтобы их было видно и без включённого INFO.
    """
    log = logger or logging.getLogger(LOGGER_NAME)
    if log.isEnabledFor(logging.INFO):
        log.info(format_report(title, params))
    for text in warnings_text or []:
        log.warning(f"[{LOGGER_NAME}] {title}: {text}")
    return {name: param.as_dict() for name, param in params.items()}


def describe(params_used: Mapping[str, Mapping[str, Any]], title: str = "параметры расчёта") -> str:
    """Восстанавливает текст отчёта из сохранённого `params_used`."""
    restored = {
        name: Param(
            value=item.get("значение"),
            unit=item.get("единицы", ""),
            source=item.get("источник", ""),
            note=item.get("пояснение", ""),
        )
        for name, item in params_used.items()
    }
    return format_report(title, restored)
