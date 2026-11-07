# Внешние данные и границы использования

## Обязательное правило соревнования

Организаторы ответили «нет» на использование IC50/CC50 из test-молекул, на использование IC50/CC50 из train как дополнительных данных, и на обучение на объединении train и test. Единственная целевая метка модели — предоставленный `SI VACV` из `train.csv`. Общий `competition_data.py` не загружает колонки IC50/CC50 из competition CSV; CV, preprocessing, подбор и обучение используют только train. Test используется отдельно для финального предсказания. ChEMBL-значения не поступают в обучение, CV, модельный выбор или ручную правку submission. Внешние выгрузки и карантин сохранены только для аудита исследовательских данных.

## ChEMBL

Источник — [официальный ChEMBL REST API](https://www.ebi.ac.uk/chembl/api/data/docs). Полная текущая проверенная выгрузка находится в `raw/chembl_batch/`; конкретные URL и полнота зафиксированы в её `manifest.json`. `raw/chembl_all/` и `raw/chembl_full/` — незавершённые загрузки, не использовать. Raw ответы сохраняются для аудита; там могут встречаться тестовые структуры и значения, которые запрещено использовать.

`processed/chembl_candidates.csv` содержит предварительные пары вне теста. `processed/chembl_quarantine.csv` содержит строки, прошедшие первичные машинные фильтры; `review_status` остаётся `pending_primary_source_review`. Эти файлы не являются входом модели. `validated.csv` оставлен пустым: даже после ручной сверки ChEMBL IC50/CC50 нельзя подключать к обучению по ответу организаторов.

Предыдущий эксперимент `paper_provisional` и его CV-метрики удалены: набор пересекался с тестом и не является допустимой оценкой или доказательством качества. Его результаты нельзя цитировать как валидные.

## Воспроизводимый порядок

```powershell
.\.venv\Scripts\python.exe data_collection/candidates.py --raw data/external/raw/chembl_batch --output data/external/processed/chembl_candidates.csv
.\.venv\Scripts\python.exe data_collection/build_quarantine.py --candidates data/external/processed/chembl_candidates.csv --out data/external/processed
.\.venv\Scripts\python.exe data_collection/validate.py --input data/external/processed/chembl_quarantine.csv --test test.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Официальная спецификация позволяет рассчитывать молекулярные признаки по SMILES; это не заменяет лабораторную метку активности. CV выполняется только на исходном `train.csv`, с canonical-группировкой дубликатов и без внешних меток, совпадающих со структурой теста или валидационного фолда. Эталонный Kaggle submission хранится отдельно в `versions/reference/` и не перезаписывается этими экспериментами.
