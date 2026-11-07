# VACV QSAR: обучение только на официальном train

Прогнозируем `SI VACV` по SMILES. Метрика Kaggle — MSE на исходной шкале SI. Официальный train содержит 320 строк, 83 нуля, 312 канонических структур и максимум SI 36 066,667. В test — 597 строк.

## Подтверждённые модели

- Лучший результат: **886 520,98398**, 70% CPU + 30% однослойного GCN. Точный отправленный файл: `versions/top1/submission.csv`; полная модель: `versions/top1/blend_model.joblib`.
- Восстановленный второй результат: **898 313,50898**, двухслойный GCN в смеси 70/30. CSV и веса точно воспроизведены: `versions/candidates/historical_two_layer_70_30/`.
- Старый защищённый `versions/top2/` содержит прежний результат **936 601,51958**. Эту папку сохранили без изменений. Актуальное соответствие мест и файлов находится в `versions/ranking.json`.
- Старый 35% кандидат остаётся неподтверждённым: `versions/candidates/legacy_35_pending/`. Новый исследовательский цикл не рекомендует его как доказанное улучшение.

Новые модели не заменяют лидера на основании одной локальной метрики. Автоматической отправки на Kaggle нет.

## Результат проведённого исследования

Реально обучены Tanimoto Kernel Ridge, Gaussian Process, CatBoost, ExtraTrees-log, GCN с несколькими инициализациями, GCN с весами связей, GCN со смешанной функцией потерь, residual MLP и компактный D-MPNN. Использованы canonical grouped CV, seeds 42/43, scaffold CV, Optuna с SQLite и вложенная CV 5 внешних × 3 внутренних фолда.

Сначала воспроизведён старый baseline с 250 деревьями в CV. Затем отдельно исправлено расхождение с итоговой моделью: `reference350` и nested tree350 используют 350 деревьев. На этой сопоставимой проверке средний OOF MSE baseline около **4 074 346**, а nested-подбора около **4 115 965**. Убедительного нового улучшения нет; рекомендован сохранённый top1.

Примерно 95% общей OOF квадратичной ошибки приходится на единственную молекулу с максимальным SI. Поэтому уменьшение MSE на ней может скрыть ухудшение большинства прогнозов. Все строки и фолды включены в основную метрику; исключение выброса приводится только как диагностика.

Ранее использованный подбор процента по максимумам test-прогнозов исключён из нового протокола. Test не используется для выбора архитектуры, весов, признаков, ограничений SI или препроцессинга.

## Установка и окружение

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Проверенное окружение записано в `experiments/research_20261107/environment.txt`. Эксперименты работают на CPU. Исходный код до изменений сохранён в `experiments/research_20261107/code_snapshot/`, хэши лидеров — в `initial_hashes.json`. Git-репозитория изначально не было; публикация на GitHub не выполнялась.

## Инференс лучшей модели

```powershell
.\.venv\Scripts\python.exe main.py predict --smiles "CCO"
.\.venv\Scripts\python.exe main.py similar --smiles "CCO" --top 10
.\.venv\Scripts\python.exe main.py batch --input molecules.csv --output predictions.csv
.\.venv\Scripts\python.exe main.py validate-submission --file versions/top1/submission.csv
```

Для batch требуется колонка `SMILES`. Similar показывает ближайшие официальные train-молекулы, Tanimoto, их SI и название. Predict, batch и submit по умолчанию используют top1, а не устаревший reference.

## Формирование submission

```powershell
.\.venv\Scripts\python.exe main.py make-submission --output results/new_submission/submission.csv
.\.venv\Scripts\python.exe main.py validate-submission --file results/new_submission/submission.csv
```

Ровно `index,SI VACV`, 597 строк, официальный порядок индексов, конечные неотрицательные числа, SHA256 и повторное получение прогнозов из модели. Существующий выходной файл не перезаписывается. `submit` — совместимый псевдоним `make-submission`; он создаёт файл локально, не загружая его на Kaggle.

## Обучение нового локального checkpoint

```powershell
.\.venv\Scripts\python.exe main.py train --device cpu --out results/new_training
.\.venv\Scripts\python.exe main.py train --device cpu --config configs/kernel_ridge.json --out results/krr_training
```

Обучение не читает test и не объявляет полученную модель улучшением. Стандартный train обучает фиксированный reference-рецепт с 350 деревьями; это новый локальный fit, а не обещание побайтного воспроизведения исторического top1. Для точного инференса top1 применяйте его сохранённые веса.

## Оценка и исследования

```powershell
.\.venv\Scripts\python.exe main.py evaluate --cv canonical --seeds 42,43
.\.venv\Scripts\python.exe main.py evaluate --cv scaffold --seeds 42
.\.venv\Scripts\python.exe main.py evaluate --cv canonical --nested
.\.venv\Scripts\python.exe main.py evaluate --cv canonical --models all
.\.venv\Scripts\python.exe main.py optimize --trials 8
.\.venv\Scripts\python.exe main.py research --phase neural
.\.venv\Scripts\python.exe main.py research --phase reports
```

Кэшированные результаты возобновляются, а не переобучаются без необходимости. Optuna выполняет указанное число дополнительных trials, но ограничен 180 секундами за запуск; SQLite сохраняет историю. Его tuning-score не является независимой оценкой. Full-data Optuna не включается в nested-оценку.

Историческую CV с 250 деревьями можно воспроизвести командой `python -m src.research reference`. Новые `main.py evaluate` используют согласованный вариант с 350 деревьями. Сырые химические признаки кэшируются по каноническому SMILES и версии RDKit; scaler, imputer и удаление констант обучаются только внутри training-фолда.

## Запреты и происхождение данных

- Обучение разрешено только на SHA256-проверенном официальном `train.csv`. Его SHA256: `d9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8`.
- Единственные входы конкурсной модели — признаки из SMILES. Целевая переменная — официальный `SI VACV`.
- IC50/CC50 исключаются до разбора значений CSV. Их нельзя использовать для вычисления или исправления обучающих SI.
- Test и внешние активности не попадают в fit, CV или подбор. Внешние предобученные веса не используются.
- Top1/top2 защищены от записи. Все прежние эксперименты сохраняются; сохранённые файлы test-кандидатов архивируются перед новым обучением старым скриптом.
- `experiments/research_20261107/official_training_used.csv` содержит только фактически разрешённые SMILES и SI; исследовательская база с ним не смешивается.

## Отдельная научная база

```powershell
.\.venv\Scripts\python.exe -m research_data_collection collect
.\.venv\Scripts\python.exe -m research_data_collection export
```

Используется публичный ChEMBL API с запросами по VACV-ассаям, не по тестовым молекулам. Сохраняются исходные ответы, DOI, URL, единицы, ограничения измерений и происхождение SI. Из уже скачанных реальных данных экспортированы **59 кандидатных пар**, но **0 пар прошли первичную научную верификацию**. Это явно указано в CSV; вымышленных или автоматически одобренных измерений нет. `training_allowed=False` для всех внешних записей.

## Проверки и отчёты

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты проверяют официальные данные, запрет внешних меток, инвариантность к IC50/CC50, корректность SMILES, fingerprints, групп, OOF, tensor inputs, CPU, загрузки checkpoints в новом процессе, submission, manifests, хэшей и защиты лидеров.

Основные результаты:

- `reports/data_and_model_audit.md` — аудит данных и старого кода.
- `reports/research_audit.md` — протокол и ограничения исследования.
- `reports/model_comparison.md` — сопоставимые результаты и nested CV.
- `reports/experiment_leaderboard.csv` — OOF и nested метрики раздельно.
- `reports/final_recommendation.md` — итоговое решение и точный submission.
- `reports/submission_leaderboard.csv` — только известные Kaggle scores; неизвестные остаются пустыми.
- `experiments/research_20261107/` — конфигурации, фолды, checkpoints, OOF, Optuna, хэши и тестовые результаты.

## Дополнительная проверка больших SI

В `src/tail_research.py` реализованы и обучены 8 вариантов: исходный log-контроль, raw MSE, корневая шкала, усиленные веса высоких SI, смешанные log/raw потери, два выхода и high/low mixture. В `src/tail_calibration.py` проверены ещё 4 схемы калибровки по внутренним OOF. Значения SI и фолды не изменялись; test не использовался.

Ни один вариант не улучшил полный MSE на обоих seed'ах. Текущий эталон сохранён. Когда максимальная молекула находится во внешнем validation, в training нет значений выше 5 000; этот недостаток сигнала не устраняется одной функцией потерь.

Отчёт: `reports/tail_handling_report.md`. Подробные сравнения: `reports/tail_experiment_leaderboard.csv`, `reports/tail_extreme_predictions.csv`. Все веса и кривые обучения: `experiments/tail_20261107/`. 12 новых тестов прошли; загрузка checkpoint после перезапуска проверена.

```powershell
.\.venv\Scripts\python.exe -m src.tail_research
.\.venv\Scripts\python.exe -m src.tail_calibration
.\.venv\Scripts\python.exe -m src.tail_report
```

## Попарное обучение изменений SI

В `src/pair_research.py` обучены 7 конфигураций на тех же 5 canonical-фолдах и seed 42/43: аналоги без поправки, Ridge log/raw, парное Tanimoto-ядро log/raw и контекстная попарная сеть log/raw. Пары и известные аналоги берутся только из training-фолда. Это производные от 320 официальных измерений, а не новые независимые экспериментальные метки.

Лучший простой blend: 90% эталона + 10% попарного log-Ridge. Его средний OOF MSE **4 040 093**, против **4 074 346** у эталона. Условная nested-проверка с выбором веса только по inner OOF дала **4 105 226**, поэтому вариант не заменяет top1 и не рекомендован к Kaggle-отправке. Семейство Ridge выбрано после screening; nested-проверка честно помечена как условная, а не независимая оценка всего поиска семейств.

Отчёт: `reports/pair_learning_report.md`. Веса, пары, аналоги и OOF сохранены в `experiments/pair_20261107/`. 13 проверок, включая отсутствие утечек через концы пар и аналоги, прошли.

```powershell
.\.venv\Scripts\python.exe -m src.pair_research
.\.venv\Scripts\python.exe -m src.pair_nested
.\.venv\Scripts\python.exe -m src.pair_report
```
# Isotonic GCN candidate, 73.57% CPU / 26.43% GCN

Готовый submission: `results/isotonic_gcn_7357_2643/submission.csv`. Это отдельный кандидат; подтверждённые `versions/top1` и `versions/top2` не меняются. Изотонический калибратор обучен на OOF-прогнозах GCN из официального `train.csv`; финальные GCN обучены на всём `train.csv`. `test.csv` читается только при создании предсказаний.

Чтобы повторно обучить модель и создать новый файл в отдельной папке:

```powershell
.\.venv\Scripts\python.exe -m src.isotonic_gcn --out results/isotonic_gcn_7357_2643_rerun
```

Скрипт по умолчанию переиспользует сохранённые train-only OOF-чекпойнты калибровки. Для полного повторного обучения OOF-моделей добавьте `--retrain-oof`. Результаты валидации сохранённого кандидата:

```powershell
.\.venv\Scripts\python.exe main.py validate-submission --file results/isotonic_gcn_7357_2643/submission.csv
```

Проверка текущих вложенных OOF-фолдов для этой фиксированной пользователем пропорции дала средний MSE 4,103,932.60; это хуже эталона 4,074,346.07. Внешний Kaggle-score неизвестен до загрузки файла.
