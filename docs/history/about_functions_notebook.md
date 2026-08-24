# Извлечено из about_functions.ipynb

Ноутбук удалён из ветки `public` (библиотека не должна содержать блокноты).
Содержательный текст о входных данных и допущениях методики сохранён здесь.

```python
'''
для подключения библиотеки:
git submodule add git@github.com:ankorm/eyetracking_analytics.git './eyetracking_analytics'
git submodule update --init --recursive
'''

import os
import sys

submodule_path = os.path.abspath('./eyetracking_analytics')
if submodule_path not in sys.path:
    sys.path.append(submodule_path)

## import eyetracking_analytics.eyetracking_funs as eyes_bib
```

---

```python
import pkgutil
import inspect

# Перебираем модули и выводим функции
print("Список всех функций в подмодуле eyetracking_analytics:\n")
for module_info in pkgutil.iter_modules([submodule_path]):
    module_name = module_info.name
    try:
        # Импортируем модуль
        module = __import__(f"eyetracking_analytics.{module_name}", fromlist=[''])
        # Ищем функции в модуле
        functions = inspect.getmembers(module, inspect.isfunction)
        if functions:
            print(f"Модуль: {module_name}")
            for func_name, _ in functions:
                print(f"  - {func_name}")
            print()
    except Exception as e:
        print(f"Ошибка при обработке модуля {module_name}: {e}\n")
```

---

*** Входные данные ***

По итогу работы айтрекера выходными данными могут быть как данные в пикселях, так и метрические данные о положении значка.
Например в системе ViveProEye (плагин записи SR-pal) данные выдеются в положении глаза в метрических единицах.
К таким данным есть несколько подходов:
- фитинг модели формы глаза, при этом надо учитывать, что зрачок находится не на поверхности склеры;
- Если считать, что зрачок движется по некоторой сфере, с постоянным радиусом, то имея данные о перемещении зрачка возможно произвести расчет углового положения глаза.

Здесь используется второй метод.

Важно учитывать, что все вычисления базируются на
- калибровке (ее важно делать в соответсвиями с требованиями окулографа)
- определении нейтрального положения: "глаз смотрит прямо"

Пользуясь тем, что глазодвинательные мышцы - это быстрые мышцы, для которых несвойсвенно длительное напряжение, то медианное значение будет соответсвовать наиболее комфортному для глаза направлению. Это направление мы будем считать нулевым и отсчитывать отклонения от него.
