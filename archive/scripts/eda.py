"""
EDA для хакатона VACV. Запускать: python eda.py
Все блоки печатают в консоль — можно читать по ходу выполнения.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem

# ---------- загрузка ----------
TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("test.csv")

def load_data():
    print("=" * 70)
    print("БЛОК 1: РАЗМЕРЫ И СТРУКТУРА ДАННЫХ")
    print("=" * 70)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train: {train.shape[0]} строк, {train.shape[1]} колонок")
    print(f"Test:  {test.shape[0]} строк, {test.shape[1]} колонок")
    print(f"\nКолонки train: {list(train.columns)}")
    print(f"Колонки test:  {list(test.columns)}")
    print(f"\nРазница колонок: в test нет -> {set(train.columns) - set(test.columns)}")
    print(f"\nПервые 3 строки train:")
    print(train.head(3).to_string())
    return train, test


def check_missing(train):
    print("\n" + "=" * 70)
    print("БЛОК 2: ПРОПУСКИ")
    print("=" * 70)
    missing = train.isna().sum().sort_values(ascending=False)
    pct = (missing / len(train) * 100).round(1)
    for col in train.columns:
        print(f"  {col:20s}: {missing[col]:4d} пропусков ({pct[col]:5.1f}%)")


def check_si_distribution(train):
    print("\n" + "=" * 70)
    print("БЛОК 3: РАСПРЕДЕЛЕНИЕ SI (целевая переменная)")
    print("=" * 70)

    si = train["SI VACV"]
    print(f"Всего значений: {si.notna().sum()} из {len(si)}")
    print(f"min={si.min():.6f}, max={si.max():.2f}, median={si.median():.2f}, mean={si.mean():.2f}")
    print(f"Стд. отклонение: {si.std():.2f}")

    # Ключевые «подозрительные» значения
    print(f"\n--- Подозрительные значения ---")
    print(f"  SI == 10000 (заглушка 'не активно'): {(si == 10000).sum()} ({((si==10000).sum()/len(si)*100):.1f}%)")
    print(f"  SI == 0:                              {(si == 0).sum()}")
    print(f"  SI < 0 (невозможно):                  {(si < 0).sum()}")
    print(f"  SI < 0.01:                            {(si < 0.01).sum()}")
    print(f"  SI > 10000 (шире заглушки):           {(si > 10000).sum()}")

    # Квантили
    print(f"\nКвантили (без 10000-заглушек):")
    clean = si[si != 10000]
    print(f"  1%={clean.quantile(0.01):.4f}, 25%={clean.quantile(0.25):.4f}, "
          f"50%={clean.quantile(0.5):.4f}, 75%={clean.quantile(0.75):.2f}, "
          f"99%={clean.quantile(0.99):.2f}")

    # Логарифмический масштаб — критичен для MSE
    print(f"\n--- log10(SI) ---")
    log_si = np.log10(si[si > 0].replace(10000, np.nan).dropna())
    print(f"  N={len(log_si)}, mean={log_si.mean():.3f}, std={log_si.std():.3f}")
    print(f"  min={log_si.min():.2f}, max={log_si.max():.2f}")


def check_censoring(train):
    print("\n" + "=" * 70)
    print("БЛОК 4: ЦЕНЗУРИРОВАНИЕ (IC50 пустой vs реальный)")
    print("=" * 70)

    ic50_missing = train["IC50_uM_VACV"].isna()
    si_10000 = train["SI VACV"] == 10000

    print(f"IC50 пустой: {ic50_missing.sum()} ({ic50_missing.mean()*100:.1f}%)")
    print(f"SI = 10000: {si_10000.sum()} ({si_10000.mean()*100:.1f}%)")

    cross = pd.crosstab(
        ic50_missing.rename("IC50 пустой"),
        si_10000.rename("SI==10000")
    )
    print("\nКросс-таблица:")
    print(cross)

    # Проверка: IC50 есть, но SI = 10000 — это может быть настоящий высокий SI
    has_ic50_and_10000 = (~ic50_missing) & si_10000
    print(f"\nСтрок, где IC50 есть, но SI всё равно 10000: {has_ic50_and_10000.sum()}")
    if has_ic50_and_10000.sum() > 0:
        print(train.loc[has_ic50_and_10000, ["Title", "IC50_uM_VACV", "CC50_uM", "SI VACV"]].head())

    # Дополнительно: SI==0 при IC50 есть — это баг
    zero_si = (train["SI VACV"] == 0)
    if zero_si.sum() > 0:
        print(f"\nSI=0 (баг в данных): {zero_si.sum()} строк")
        print(train.loc[zero_si, ["Title", "IC50_uM_VACV", "CC50_uM", "SI VACV"]].head())


def check_smiles(train, test):
    print("\n" + "=" * 70)
    print("БЛОК 5: SMILES — ВАЛИДНОСТЬ И ДУБЛИКАТЫ")
    print("=" * 70)

    def validate_smiles(smi):
        try:
            return Chem.MolFromSmiles(smi) is not None
        except Exception:
            return False

    for name, df in [("train", train), ("test", test)]:
        invalid = ~df["SMILES"].apply(validate_smiles)
        print(f"\n{name}: невалидных SMILES = {invalid.sum()} ({invalid.mean()*100:.1f}%)")
        if invalid.sum() > 0:
            print(df.loc[invalid, ["Title", "SMILES"]].head(10).to_string())

    # Дубликаты в train
    print("\n--- Дубликаты в train ---")
    dup = train["SMILES"].duplicated(keep=False)
    print(f"Строк с дублирующимся SMILES: {dup.sum()}")
    print(f"Уникальных SMILES: {train['SMILES'].nunique()}")

    # Показать примеры дубликатов с разными SI (важно!)
    dup_groups = train[train["SMILES"].duplicated(keep=False)].groupby("SMILES")
    inconsistent = []
    for smi, grp in dup_groups:
        si_vals = grp["SI VACV"].dropna().unique()
        if len(si_vals) > 1:
            inconsistent.append((smi, si_vals))
    print(f"SMILES-дубликатов с РАЗНЫМ SI: {len(inconsistent)}")
    if inconsistent:
        print("  Примеры:")
        for smi, vals in inconsistent[:5]:
            print(f"    {smi[:60]}... -> SI: {vals}")


def check_overlap(train, test):
    print("\n" + "=" * 70)
    print("БЛОК 6: ПЕРЕСЕЧЕНИЕ SMILES МЕЖДУ TRAIN И TEST (data leakage!)")
    print("=" * 70)

    train_smi = set(train["SMILES"].dropna())
    test_smi = set(test["SMILES"].dropna())

    overlap = train_smi & test_smi
    print(f"Уникальных SMILES в train: {len(train_smi)}")
    print(f"Уникальных SMILES в test:  {len(test_smi)}")
    print(f"Пересечение:               {len(overlap)} ({len(overlap)/len(test_smi)*100:.1f}% от test)")

    if len(overlap) > 0:
        print("\n⚠️  ВНИМАНИЕ: если тест содержит молекулы из train, это может быть")
        print("   умышленная проверка 'запомнил ли ты данные'. Или же это НОРМАЛЬНО —")
        print("   тест 'повторяет' часть train, чтобы проверить стабильность модели.")
        print("   Надо посмотреть, что говорит Kaggle на этот счёт.")
        print("\n  Примеры общих SMILES:")
        for smi in list(overlap)[:5]:
            print(f"    {smi[:70]}...")


def check_distributions(train):
    print("\n" + "=" * 70)
    print("БЛОК 7: РАСПРЕДЕЛЕНИЯ (визуально — сохраним в PNG)")
    print("=" * 70)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Сырые SI
    si = train["SI VACV"].dropna()
    axes[0, 0].hist(si, bins=50)
    axes[0, 0].set_title("Сырой SI (плохо — зашкаливает 10000)")
    axes[0, 0].set_xlabel("SI")

    # 2. log10(SI) без 10000
    clean = si[(si > 0) & (si != 10000)]
    axes[0, 1].hist(np.log10(clean), bins=50, color="green")
    axes[0, 1].set_title(f"log10(SI), без 10000 (N={len(clean)})")
    axes[0, 1].set_xlabel("log10(SI)")

    # 3. Лог-нормальность через Q-Q plot (или просто гистограмма по log)
    log_all = np.log10(si[si > 0].replace(10000, np.nan).dropna())
    axes[1, 0].hist(log_all, bins=50, color="orange")
    axes[1, 0].set_title("log10(SI), включая очень маленькие значения")
    axes[1, 0].set_xlabel("log10(SI)")

    # 4. IC50 vs CC50 scatter (логи)
    ic50 = train["IC50_uM_VACV"]
    cc50 = train["CC50_uM"]
    mask = (ic50 > 0) & (cc50 > 0)
    axes[1, 1].scatter(np.log10(cc50[mask]), np.log10(ic50[mask]), alpha=0.4, s=10)
    axes[1, 1].set_xlabel("log10(CC50)")
    axes[1, 1].set_ylabel("log10(IC50)")
    axes[1, 1].set_title(f"CC50 vs IC50 (N={mask.sum()})")

    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=100)
    print("  Сохранено в eda_distributions.png")


if __name__ == "__main__":
    train, test = load_data()
    check_missing(train)
    check_si_distribution(train)
    check_censoring(train)
    check_smiles(train, test)
    check_overlap(train, test)
    check_distributions(train)

    print("\n" + "=" * 70)
    print("ГОТОВО. Теперь смотрим на цифры и делаем выводы.")
    print("=" * 70)