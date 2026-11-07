"""
prep.py — препроцессинг для стратегии C (двухступенчатая).

Вход:  train.csv, test.csv
Выход:
  - train_clean.csv   — все 318 строк, с is_active и log10_SI
  - train_active.csv  — только 237 активных (для регрессора)
  - test_clean.csv    — 597 строк для предсказания

Колонки strain и Culture СОХРАНЯЮТСЯ — их разрешено использовать как признаки.

Запуск: python prep.py
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"

CENSOR_LOG10_VALUE = 4.0   # log10(10000) — консервативно, для fallback-B


def load_and_clean_train(train_path=TRAIN_PATH, v2=False):
    print("=" * 60)
    print("1. Загрузка и очистка TRAIN")
    print("=" * 60)

    df = pd.read_csv(train_path)
    print(f"  Исходно: {len(df)} строк")

    if not v2:
        # Preserve the legacy cleanup for the original Kaggle training file.
        bad = df["SI VACV"] == 0
        print(f"  Удаляем SI=0 (баг): {bad.sum()} строк")
        df = df[~bad].copy()

    # Censoring is defined by a missing IC50 measurement.
    df["is_active"] = df["IC50_uM_VACV"].notna().astype(int)
    df["is_censored"] = 1 - df["is_active"]

    if v2:
        si = pd.to_numeric(df["SI VACV"], errors="coerce")
        positive_active = (df["is_active"] == 1) & (si > 0)
        df["log10_SI"] = np.nan
        df.loc[positive_active, "log10_SI"] = np.log10(si.loc[positive_active])
        df.loc[df["is_active"] == 0, "log10_SI"] = 0.0
    else:
        df["log10_SI"] = np.where(
            df["is_active"] == 1,
            np.log10(df["SI VACV"]),
            CENSOR_LOG10_VALUE,
        )

    print(f"  После очистки: {len(df)} строк")
    print(f"  Активных (IC50 задан):        {(df['is_active']==1).sum()}")
    print(f"  Цензурированных (IC50 пуст):  {(df['is_active']==0).sum()}")
    print(f"  Уникальных Culture:          {df['Culture'].nunique()}")
    print(f"  Уникальных strain:           {df['strain'].nunique()}")
    return df


def load_test():
    print("\n" + "=" * 60)
    print("2. Загрузка TEST")
    print("=" * 60)
    df = pd.read_csv(TEST_PATH)
    print(f"  Строк: {len(df)}")
    print(f"  Уникальных Culture: {df['Culture'].nunique()}")
    print(f"  Уникальных strain:  {df['strain'].nunique()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Prepare train_v2.csv into train_clean_v2.csv without discarding zero SI rows.",
    )
    args = parser.parse_args()
    if args.v2:
        train = load_and_clean_train(ROOT / "train_v2.csv", v2=True)
        train.to_csv(ROOT / "train_clean_v2.csv", index=False)
        train[train["is_active"] == 1].to_csv(ROOT / "train_active_v2.csv", index=False)
        print(f"  train_clean_v2.csv: {len(train)} строк")
        print(f"  train_active_v2.csv: {(train['is_active'] == 1).sum()} строк")
        return

    train = load_and_clean_train()
    test = load_test()

    train.to_csv("train_clean.csv", index=False)
    test.to_csv("test_clean.csv", index=False)
    train[train["is_active"] == 1].to_csv("train_active.csv", index=False)

    print("\n" + "=" * 60)
    print("3. СОХРАНЕНО")
    print("=" * 60)
    print(f"  train_clean.csv   : {len(train)} строк")
    print(f"  train_active.csv  : {(train['is_active']==1).sum()} строк")
    print(f"  test_clean.csv    : {len(test)} строк")


if __name__ == "__main__":
    main()