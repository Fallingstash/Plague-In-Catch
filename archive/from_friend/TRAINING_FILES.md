# Где лежит эталон и что нужно для обучения

## Эталонные варианты

- Подтверждённый лучший Kaggle-вариант: `versions/top1/submission.csv` — score **886520.98398**.
- Полная модель этого варианта: `versions/top1/blend_model.joblib`.
- CPU-модель, которую использует экспериментальный isotonic-кандидат как базовую часть: `versions/reference/model.joblib`.
- Готовый isotonic-кандидат 73.57% CPU / 26.43% GCN: `results/isotonic_gcn_7357_2643/submission.csv`.
- Его полный checkpoint для повторных предсказаний: `results/isotonic_gcn_7357_2643/prediction_bundle.joblib`.

`versions/top1` — эталон для сравнения и не перезаписывается. `versions/reference/model.joblib` — отдельная CPU-база, используемая скриптом isotonic-кандидата.

## Основной скрипт обучения

`src/isotonic_gcn.py` обучает GCN, строит train-only OOF-прогнозы для калибратора, обучает финальные GCN на всём официальном train и записывает submission. Импортируемые Python-файлы проекта:

- `competition_data.py`
- `vacv_cpu_improved.py`
- `versions/test/training.py`
- `src/validation.py`
- `src/features.py`
- `src/models.py`
- `src/research.py`
- `src/tail_research.py`
- `src/tail_calibration.py`
- `src/inference.py`

Нужны также `requirements.txt`, официальный `train.csv`, `test.csv`, `sample_submit.csv` и модель `versions/reference/model.joblib`. `train.csv` должен оставаться исходным официальным файлом: загрузчик проверяет его SHA256 и использует только `SMILES` и `SI VACV`. `test.csv` применяется только для итогового предсказания.

По умолчанию для ускорения переиспользуются уже рассчитанные OOF-модели из `experiments/tail_20261107/42_control_log/` и `43_control_log/`, а также протокол `experiments/tail_20261107/protocol.json`. Для повторного обучения этих OOF-моделей передайте `--retrain-oof`. Защита эталонных файлов также сверяется с `experiments/research_20261107/initial_hashes.json`.

## Запуск в PowerShell

Из корневой папки проекта:

```powershell
.\.venv\Scripts\python.exe -m src.isotonic_gcn --out results/isotonic_gcn_rerun
```

Для полного повторного обучения OOF-моделей:

```powershell
.\.venv\Scripts\python.exe -m src.isotonic_gcn --retrain-oof --out results/isotonic_gcn_full_retrain
```

Проверить submission и его воспроизводимость из checkpoint:

```powershell
.\.venv\Scripts\python.exe main.py validate-submission --file results/isotonic_gcn_7357_2643/submission.csv
```

Результат Kaggle для isotonic-кандидата пока неизвестен; локальный OOF MSE не заменяет публичную оценку соревнования.
