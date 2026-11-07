import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
N_BITS = 2048
SEED = 42
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
EXPANDED_TRAIN_COLUMNS = [
    "Title",
    "CC50_uM",
    "IC50_uM_VACV",
    "SI VACV",
    "SMILES",
    "strain",
    "Culture",
    "is_active",
    "is_censored",
    "log10_SI",
    "DOI",
    "source",
]
COLLECTED_COLUMNS = ["SMILES", "IC50uM", "CC50uM", "SI", "DOI", "source"]


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def load_collected_sources():
    chembl_path = DATA_DIR / "collected_raw.csv"
    articles_path = DATA_DIR / "collectedfromarticles.csv"
    for path in (chembl_path, articles_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source data is missing: {path}")

    chembl = pd.read_csv(chembl_path)
    articles = pd.read_csv(articles_path)
    required_chembl = {"SMILES", "IC50uM", "source"}
    if not required_chembl.issubset(chembl.columns):
        raise ValueError(f"{chembl_path.name} is missing {required_chembl - set(chembl.columns)}")
    if not set(COLLECTED_COLUMNS).issubset(articles.columns):
        raise ValueError(f"{articles_path.name} does not match the article data schema")

    chembl_rows = pd.DataFrame(
        {
            "SMILES": chembl["SMILES"],
            "IC50uM": pd.to_numeric(chembl["IC50uM"], errors="coerce"),
            "CC50uM": np.nan,
            "SI": np.nan,
            "DOI": chembl.get("DOI", pd.Series([None] * len(chembl))),
            "source": "ChEMBL",
        }
    )
    article_rows = articles[COLLECTED_COLUMNS].copy()
    combined = pd.concat([chembl_rows, article_rows], ignore_index=True)
    if len(chembl_rows) != 454 or len(article_rows) != 31:
        raise ValueError(
            f"Expected 454 ChEMBL and 31 article rows; found {len(chembl_rows)} and {len(article_rows)}"
        )
    combined.to_csv(DATA_DIR / "collected.csv", index=False)
    return chembl, articles, combined


def make_expanded_training_set(train, articles):
    article_si = pd.to_numeric(articles["SI"], errors="coerce")
    article_ic50 = pd.to_numeric(articles["IC50uM"], errors="coerce")
    article_cc50 = pd.to_numeric(articles["CC50uM"], errors="coerce")
    appended = pd.DataFrame(
        {
            "Title": articles["source"].astype(str),
            "CC50_uM": article_cc50,
            "IC50_uM_VACV": article_ic50,
            "SI VACV": article_si,
            "SMILES": articles["SMILES"],
            "strain": "Figshare",
            "Culture": "Figshare",
            "is_active": article_ic50.notna().astype(int),
            "is_censored": (article_si >= 10000).fillna(False).astype(int),
            "log10_SI": np.where(
                article_ic50.notna() & article_si.gt(0),
                np.log10(article_si.where(article_si.gt(0), 1.0)),
                4.0,
            ),
            "DOI": articles["DOI"],
            "source": articles["source"],
        }
    )
    original = train.copy()
    original["DOI"] = None
    original["source"] = "Kaggle"
    expanded = pd.concat([original[EXPANDED_TRAIN_COLUMNS], appended], ignore_index=True)
    expanded["is_active"] = pd.to_numeric(expanded["IC50_uM_VACV"], errors="coerce").notna().astype(int)
    expanded["is_censored"] = pd.to_numeric(expanded["SI VACV"], errors="coerce").ge(10000).astype(int)
    si = pd.to_numeric(expanded["SI VACV"], errors="coerce")
    expanded["log10_SI"] = np.where(
        expanded["is_active"].eq(1) & si.gt(0),
        np.log10(si.where(si.gt(0), 1.0)),
        4.0,
    )
    return expanded


def featurize(smiles_values, descriptor_names, warning_path):
    descriptor_rows = []
    fingerprint_rows = []
    success_count = 0
    with warning_path.open("a", encoding="utf-8") as log:
        for row_index, smiles in enumerate(smiles_values):
            molecule = None
            try:
                molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            except Exception as error:
                log.write(f"row={row_index}\tsmiles={smiles!r}\terror={error}\n")
            if molecule is None:
                log.write(f"row={row_index}\tsmiles={smiles!r}\terror=invalid SMILES\n")
                descriptor_rows.append([np.nan] * len(descriptor_names))
                fingerprint_rows.append(np.zeros(N_BITS, dtype=np.uint8))
                continue
            success_count += 1
            try:
                descriptors = [
                    Descriptors.descList[index][1](molecule)
                    for index in range(len(descriptor_names))
                ]
            except Exception as error:
                log.write(f"row={row_index}\tsmiles={smiles!r}\tdescriptor_error={error}\n")
                descriptors = [np.nan] * len(descriptor_names)
            try:
                fingerprint = GENERATOR.GetFingerprintAsNumPy(molecule).astype(np.uint8)
            except Exception as error:
                log.write(f"row={row_index}\tsmiles={smiles!r}\tfingerprint_error={error}\n")
                fingerprint = np.zeros(N_BITS, dtype=np.uint8)
            descriptor_rows.append(descriptors)
            fingerprint_rows.append(fingerprint)
    return (
        pd.DataFrame(descriptor_rows, columns=descriptor_names),
        np.asarray(fingerprint_rows, dtype=np.uint8),
        success_count,
    )


def build_fold_assignments(original_folds, base_smiles, external_smiles, all_classifier_smiles):
    fold_count = len(original_folds)
    base_assignments = np.full(len(base_smiles), -1, dtype=int)
    for fold in original_folds:
        fold_id = int(fold["fold"])
        base_assignments[np.asarray(fold["validation_indices"], dtype=int)] = fold_id
    if (base_assignments < 0).any():
        raise ValueError("Saved folds do not assign every original training row")

    base_canonical_members = defaultdict(list)
    for index, smiles in enumerate(base_smiles):
        canonical = canonicalize(smiles)
        if canonical is not None:
            base_canonical_members[canonical].append(int(base_assignments[index]))
    base_canonical_folds = {}
    conflicting_base_canonicals = set()
    for canonical, folds in base_canonical_members.items():
        fold_counts = np.bincount(folds, minlength=fold_count)
        base_canonical_folds[canonical] = int(np.flatnonzero(fold_counts == fold_counts.max())[0])
        if len(set(folds)) > 1:
            conflicting_base_canonicals.add(canonical)

    rng = np.random.default_rng(SEED)
    counts = np.bincount(base_assignments, minlength=fold_count).astype(int)
    external_canonical = [canonicalize(smiles) for smiles in external_smiles]
    classifier_canonical = [canonicalize(smiles) for smiles in all_classifier_smiles]
    external_groups = defaultdict(list)
    for offset, canonical in enumerate(external_canonical):
        external_groups[canonical if canonical is not None else f"invalid_ext_{offset}"].append(offset)

    group_items = list(external_groups.items())
    tie_breakers = rng.permutation(len(group_items))
    tie_lookup = {int(group_index): order for order, group_index in enumerate(tie_breakers)}
    group_items = [
        item
        for _, item in sorted(
            enumerate(group_items),
            key=lambda pair: (-len(pair[1][1]), tie_lookup[pair[0]]),
        )
    ]
    external_assignments = np.full(len(external_smiles), -1, dtype=int)
    for canonical, offsets in group_items:
        if canonical in base_canonical_folds:
            fold_id = base_canonical_folds[canonical]
        else:
            fold_id = int(np.flatnonzero(counts == counts.min())[0])
        external_assignments[offsets] = fold_id
        counts[fold_id] += len(offsets)

    classifier_assignments = np.full(len(classifier_canonical), -1, dtype=int)
    classifier_assignments[: len(base_assignments)] = base_assignments
    classifier_assignments[
        len(base_assignments) : len(base_assignments) + len(external_assignments)
    ] = external_assignments
    base_and_article_count = len(base_assignments) + len(external_assignments)
    classifier_groups = defaultdict(list)
    for row_index, canonical in enumerate(classifier_canonical):
        classifier_groups[canonical if canonical is not None else f"invalid_cls_{row_index}"].append(row_index)
    for canonical, indices in classifier_groups.items():
        known_folds = {
            int(classifier_assignments[index])
            for index in indices
            if classifier_assignments[index] >= 0
        }
        if len(known_folds) > 1:
            if canonical not in conflicting_base_canonicals:
                raise ValueError(f"Duplicate canonical SMILES already assigned across folds: {canonical}")
            fold_id = base_canonical_folds[canonical]
            for index in indices:
                if index >= len(base_assignments) and index < base_and_article_count:
                    classifier_assignments[index] = fold_id
                elif index >= base_and_article_count:
                    classifier_assignments[index] = fold_id
            counts[fold_id] += sum(index >= base_and_article_count for index in indices)
            continue
        if known_folds:
            fold_id = next(iter(known_folds))
        else:
            fold_id = int(np.flatnonzero(counts == counts.min())[0])
        for index in indices:
            classifier_assignments[index] = fold_id
        counts[fold_id] += sum(index >= base_and_article_count for index in indices)

    if (classifier_assignments < 0).any():
        raise ValueError("Expanded classifier folds are incomplete")
    return (
        np.concatenate([base_assignments, external_assignments]),
        external_assignments,
        classifier_assignments,
        len(conflicting_base_canonicals),
    )


def folds_from_assignments(assignments):
    folds = []
    for fold_id in range(5):
        validation_indices = np.flatnonzero(assignments == fold_id)
        train_indices = np.flatnonzero(assignments != fold_id)
        if len(validation_indices) == 0:
            raise ValueError(f"Expanded fold {fold_id} has no validation rows")
        folds.append(
            {
                "fold": fold_id,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
            }
        )
    return folds


def main():
    RDLogger.DisableLog("rdApp.error")
    train = pd.read_csv(ROOT / "train_clean.csv")
    chembl, articles, combined_sources = load_collected_sources()
    expanded = make_expanded_training_set(train, articles)
    if len(expanded) != 349 or expanded["is_active"].sum() != 268:
        raise ValueError(
            f"Unexpected expanded train size/activity: {len(expanded)}, "
            f"{expanded['is_active'].sum()}"
        )
    expanded.to_csv(ROOT / "traincleanexpanded.csv", index=False)
    expanded.loc[expanded["is_active"] == 1].to_csv(
        ROOT / "trainactiveexpanded.csv", index=False
    )

    descriptor_names = [name for name, _ in Descriptors.descList]
    warning_path = DATA_DIR / "expanded_warnings.log"
    descriptor_base, fingerprint_base, parsed_base = featurize(
        expanded["SMILES"], descriptor_names, warning_path
    )
    chembl_descriptor, chembl_fingerprint, parsed_chembl = featurize(
        chembl["SMILES"], descriptor_names, warning_path
    )
    descriptor_base.to_csv(ROOT / "featurestraindesc_expanded.csv", index=False)
    np.savez_compressed(
        ROOT / "featurestrainfp_expanded.npz", fingerprints=fingerprint_base
    )
    chembl_descriptor.to_csv(
        ROOT / "featurestrainclassifierdesc_expanded.csv", index=False
    )
    np.savez_compressed(
        ROOT / "featurestrainclassifierfp_expanded.npz",
        fingerprints=chembl_fingerprint,
    )
    meta_base = expanded[
        ["SMILES", "is_active", "log10_SI", "SI VACV", "source"]
    ].rename(columns={"is_active": "isactive", "log10_SI": "log10SI"})
    meta_base.to_csv(ROOT / "featurestrainmeta_expanded.csv", index=False)

    original_fold_data = json.loads(
        (RESULTS_DIR / "folds.json").read_text(encoding="utf-8")
    )
    original_folds = original_fold_data["folds"]
    all_classifier_smiles = pd.concat(
        [expanded["SMILES"], chembl["SMILES"]], ignore_index=True
    ).tolist()
    (
        base_assignments,
        article_assignments,
        classifier_assignments,
        conflicting_base_canonical_count,
    ) = build_fold_assignments(
        original_folds,
        train["SMILES"].tolist(),
        articles["SMILES"].tolist(),
        all_classifier_smiles,
    )
    if not np.array_equal(base_assignments, np.concatenate(
        [
            np.asarray(
                [
                    next(
                        fold["fold"]
                        for fold in original_folds
                        if index in fold["validation_indices"]
                    )
                    for index in range(len(train))
                ],
                dtype=int,
            ),
            base_assignments[len(train):],
        ]
    )):
        raise RuntimeError("Original fold assignments were not preserved")
    if len(classifier_assignments) != len(expanded) + len(chembl):
        raise ValueError("Classifier fold assignment count does not match rows")

    fold_artifact = {
        "n_splits": 5,
        "seed_for_new_rows": SEED,
        "source_folds": "experiments/results/folds.json",
        "note": (
            "Original Kaggle rows retain their existing validation fold. "
            "Added molecules are assigned deterministically and canonical-SMILES "
            "duplicates stay in the same fold where possible. Five pre-existing "
            "canonical groups cross historical folds and must be purged from each "
            "validation fold's training indices."
        ),
        "training_349_fold_ids": base_assignments.tolist(),
        "classifier_803_fold_ids": classifier_assignments.tolist(),
        "training_349_folds": folds_from_assignments(base_assignments),
        "classifier_803_folds": folds_from_assignments(classifier_assignments),
        "fold_counts_349": np.bincount(base_assignments, minlength=5).tolist(),
        "fold_counts_803": np.bincount(classifier_assignments, minlength=5).tolist(),
        "preexisting_canonical_groups_crossing_original_folds": conflicting_base_canonical_count,
    }
    (RESULTS_DIR / "folds_expanded.json").write_text(
        json.dumps(fold_artifact), encoding="utf-8"
    )
    summary = {
        "collected_rows": len(combined_sources),
        "chembl_rows": len(chembl),
        "figshare_rows": len(articles),
        "expanded_train_rows": len(expanded),
        "expanded_active_rows": int(expanded["is_active"].sum()),
        "classifier_rows": len(expanded) + len(chembl),
        "classifier_active_rows": int(
            expanded["is_active"].sum() + chembl["IC50uM"].notna().sum()
        ),
        "parsed_expanded_train_smiles": parsed_base,
        "parsed_chembl_smiles": parsed_chembl,
        "feature_shapes": {
            "expanded_train_descriptors": list(descriptor_base.shape),
            "expanded_train_fingerprints": list(fingerprint_base.shape),
            "classifier_chembl_descriptors": list(chembl_descriptor.shape),
            "classifier_chembl_fingerprints": list(chembl_fingerprint.shape),
        },
        "fold_counts_349": fold_artifact["fold_counts_349"],
        "fold_counts_803": fold_artifact["fold_counts_803"],
        "preexisting_canonical_groups_crossing_original_folds": conflicting_base_canonical_count,
    }
    (DATA_DIR / "expanded_data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Deliverable collected.csv: {len(combined_sources)} rows")
    print(
        f"Expanded regression data: {len(expanded)} rows "
        f"({int(expanded['is_active'].sum())} active)"
    )
    print(
        f"Expanded classifier data: {len(expanded) + len(chembl)} rows "
        f"({summary['classifier_active_rows']} active)"
    )
    print(
        f"Parsed structures: expanded {parsed_base}/{len(expanded)}, "
        f"ChEMBL {parsed_chembl}/{len(chembl)}"
    )
    print(f"Fold counts 349: {fold_artifact['fold_counts_349']}")
    print(f"Fold counts 803: {fold_artifact['fold_counts_803']}")


if __name__ == "__main__":
    main()
