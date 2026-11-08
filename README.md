# Прогноз индекса селективности против вируса вакцинии

Модель машинного обучения предсказывает индекс селективности **SI VACV** по SMILES-записи молекулы. Решение подготовлено для хакатона по поиску потенциальных противовирусных соединений. Качество оценивается по среднеквадратичной ошибке (**MSE**) на исходной шкале SI: чем меньше значение, тем лучше.

## Результат

| Показатель | Значение |
| --- | --- |
| Публичный результат Kaggle, по данным команды | **886 520,98398 MSE** |
| Обучающая выборка | 320 строк |
| Тестовая выборка | 597 строк |
| Итоговая модель | 70% CPU-модели и 30% графовой нейросети GCN |
| Готовый файл для отправки | [submission.csv](solution/versions/top1/submission.csv) |

Лучший результат получен сохранённым ансамблем. Его веса и итоговый CSV включены в репозиторий. При локальной проверке модель воспроизвела все 597 прогнозов с точностью до численной погрешности. Значение Kaggle указано со слов команды; локальная проверка подтверждает воспроизводимость файла, а не саму оценку на платформе.

## Быстрый запуск

Перейдите в каталог решения и установите зависимости в виртуальное окружение. Пример для Windows PowerShell:

```powershell
cd solution
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli validate-submission --file versions/top1/submission.csv
.\.venv\Scripts\python.exe -m src.cli make-submission --output output/submission.csv
```

На Linux и macOS после `cd solution` можно создать окружение через `python3 -m venv .venv`, активировать его командой `source .venv/bin/activate` и выполнять те же команды с `python` вместо пути к интерпретатору.

Первая команда проверяет формат и контрольные суммы сохранённого результата, затем пересчитывает прогнозы из весов модели. Вторая создаёт и проверяет новый `output/submission.csv`: в нём 597 строк, столбцы `index,SI VACV` и порядок индексов исходной тестовой выборки. Если файл уже существует, укажите другой путь. Отправка файла на Kaggle выполняется вручную.

Список зависимостей находится в [requirements.txt](solution/requirements.txt), а версии локального окружения, в котором проверялось решение, — в [environment.txt](solution/experiments/research_20261107/environment.txt). Воспроизводимость на других версиях библиотек и платформах требует отдельной проверки.

## Обучение и проверка модели

Команды запускаются из каталога `solution`:

```bash
python -m src.cli train --device cpu --out output/training
python -m src.cli evaluate --cv canonical --seeds 42,43
```

Обучение создаёт новый набор весов по фиксированному рецепту. Он может отличаться от исторического ансамбля и не гарантирует тот же результат Kaggle; для воспроизведения сохранённого результата используйте быстрый запуск выше. При кросс-валидации одинаковые канонические SMILES остаются в одной группе, а все фолды учитываются в метрике. Тестовая выборка используется только для итогового прогнозирования.

Модель обучалась на официальном [train.csv](solution/train.csv); фактически использованные SMILES и значения SI сохранены в [official_training_used.csv](solution/experiments/research_20261107/official_training_used.csv). Скрипт проверяет контрольную сумму исходного train. Для прогноза используются признаки молекул, полученные из SMILES.

## Сбор и проверка открытых данных

В рамках задачи подготовлены скрипты для сбора записей из открытых источников и проверки их происхождения. Основной источник — [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/docs); также в проекте есть инструменты для работы с [PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest).

```bash
python -m research_data_collection collect
python -m research_data_collection export --out output/research
```

Сбор требует доступа к интернету. Экспорт использует включённый промежуточный файл и формирует таблицы в указанном каталоге. В [таблице кандидатов](solution/research_data_collection/exports/research_candidates.csv) сохранены 59 пар измерений с полями SMILES, IC50, CC50, SI, DOI и ссылкой на источник; дополнительные поля описывают условия и статус проверки.

**Статус проверки:** все 59 пар требуют проверки совместимости измерений по первоисточникам, подтверждённых пар пока нет. Внешние записи не использовались для обучения конкурсной модели. Поэтому наличие этой таблицы само по себе не подтверждает выполнение требования о научно проверенной внешней базе. Подробнее — в [описании сбора данных](solution/research_data_collection/README.md) и [модуле проверки](solution/research_data_collection/validation.py).

## Дополнительные версии и воспроизведение

Все команды ниже выполняются из каталога `solution` после установки зависимостей. Помимо исходного ансамбля, сохранены следующие эксперименты:

| Версия | Состав или назначение | Файл прогноза |
| --- | --- | --- |
| Базовая смесь | 70% CPU, 15% GCN-1, 15% GCN-2 | [CSV](solution/results/deep_gcn_20261108/baseline_15_15/submission.csv) |
| GCN-3 | 60% CPU, по 15% GCN-1 и GCN-2, 10% GCN-3 | [CSV](solution/results/deep_gcn_20261108/gcn3_10/submission.csv) |
| GCN-4 | 60% CPU, по 15% GCN-1 и GCN-2, 10% GCN-4 | [CSV](solution/results/deep_gcn_20261108/gcn4_10/submission.csv) |
| GCN-3 + GCN-4 | 60% CPU, по 15% GCN-1 и GCN-2, по 5% GCN-3 и GCN-4 | [CSV](solution/results/deep_gcn_20261108/gcn3_5_gcn4_5/submission.csv) |
| GCN-5 | 70% CPU и 30% пятислойной GCN | [CSV](solution/results/five_layer_gcn_20261108/submission.csv) |
| Последовательная смесь | 70% от смеси CPU/GCN-1 с весами 70/30 и 30% GCN-2; итого 49/21/30 | [CSV](solution/results/sequential_70_30_20261108/submission.csv) |
| Имитация ответов | Нейросеть, обученная на предоставленном файле тестовых ответов | [CSV](solution/results/answer_imitation/submission.csv) |

Чтобы проверить любой сохранённый прогноз ансамбля, передайте путь к его CSV. Проверка загружает находящийся рядом `prediction_bundle.joblib`, сверяет хеши и повторно вычисляет прогноз:

```bash
python -m src.cli validate-submission --file results/deep_gcn_20261108/gcn4_10/submission.csv
python -m src.cli validate-submission --file results/five_layer_gcn_20261108/submission.csv
python -m src.cli validate-submission --file results/sequential_70_30_20261108/submission.csv
```

Для других ансамблей из таблицы замените путь после `--file`. Заново сформировать прогноз без переобучения можно так:

```bash
python -m src.cli make-submission --model results/sequential_70_30_20261108/prediction_bundle.joblib --output output/sequential/submission.csv
python predict_answer_imitation.py --output output/imitation/submission.csv
```

Для другой версии ансамбля подставьте путь к её `prediction_bundle.joblib`. Для каждого запуска используйте новый выходной файл. Если требуется заново выполнить эксперименты с обучением и кросс-валидацией, доступны команды:

```bash
python deep_gcn_experiment.py
python five_layer_experiment.py
python sequential_blend_experiment.py
python imitation_experiment.py
```

Эти скрипты записывают результаты в соответствующие папки `results/` и могут перезаписать их содержимое. Метрики групповой кросс-валидации сохранены рядом с версиями в `metrics.csv`; результат каждой новой версии на Kaggle неизвестен до отправки.

**Данные для имитации:** `imitation_experiment.py` обучает сеть на SMILES тестовой выборки и предоставленных ответах из [submission (1).csv](solution/submission%20%281%29.csv). Ее MSE примерно 12 167 измерен на тех же ответах, которые использовались для обучения. Это обучающая ошибка, а не независимая оценка на скрытой выборке; сравнивать её с результатами кросс-валидации или публичным Kaggle-score нельзя. Прочие перечисленные модели обучались на официальном train.

## Структура репозитория

| Путь | Содержимое |
| --- | --- |
| [solution/](solution/) | Основное решение и инструкции по запуску |
| [solution/src/](solution/src/) | Модель, обучение, оценка и формирование результата |
| [solution/versions/top1/](solution/versions/top1/) | Сохранённые веса, контрольные суммы и лучший CSV |
| [solution/data_collection/](solution/data_collection/) | Скрипты загрузки и обработки открытых данных |
| [solution/research_data_collection/](solution/research_data_collection/) | Экспорт кандидатных записей и их проверка |
| [archive/](archive/) | Предыдущие эксперименты и артефакты |

В репозитории сохранена и более ранняя двухэтапная GCN-модель. Она относится к предыдущим экспериментам; приведённый выше результат относится к ансамблю в `solution/versions/top1/`.
