# Последний train-only замер и Kaggle результаты

Подтверждённый reference — 95% CPU + 5% GCN; пользователь сообщил Kaggle MSE **1 076 308,29676**, что на 8,64% ниже предыдущего эталона 1 178 057,61138. Его зафиксированный файл и composite bundle лежат в `versions/reference/`.

| Модель/бленд | CV MSE seed 42 | seed 43 | OOF average |
|---|---:|---:|---:|
| CPU baseline | 4 190 501 | 4 190 971 | 4 190 736 |
| GCN + Morgan | 4 006 895 | 3 747 659 | 3 877 277 |
| 95% CPU + 5% GCN | 4 173 902 | 4 166 594 | 4 170 248 |
| 90% CPU + 10% GCN | ~4 158 084 | ~4 142 450 | ~4 150 267 |

После неудачи чистого GCN submission (Kaggle MSE 3 183 221,94995) доля графовой модели увеличивается постепенно. Текущий test-кандидат `versions/test/submission.csv` — 90% CPU reference baseline + 10% GCN. Его score ещё не известен; CV-улучшение небольшое, поэтому он остаётся экспериментом. Не заменять подтверждённый reference до нового Kaggle результата.

CV, preprocessing и обучение используют только 320 строк train и `SI VACV`. Колонки IC50/CC50 не загружаются. Test появляется только после обучения для инференса. Все фолды и строки включены; высокий SI не удаляется. Проверенные строки: reference MSE 1 076 308,29676, GCN raw 3 183 221,94995, текущий 10% blend — ожидает Kaggle score.

Пересоздать текущий test-кандидат:

```powershell
.\.venv\Scripts\python.exe versions/test/training.py --out versions/test --folds 5 --split-seeds 42,43 --epochs 80 --graph-weight 0.10
```