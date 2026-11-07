import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "final_collected_dataset.csv"
COLUMNS = [
    "canonical_smiles",
    "original_smiles",
    "IC50",
    "IC50_unit",
    "CC50",
    "CC50_unit",
    "SI",
    "organism",
    "virus_strain",
    "cell_line",
    "assay_id",
    "source",
    "DOI",
    "measurement_relation",
    "standard_type",
]


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def number(value):
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if pd.notna(result) and np.isfinite(result) else np.nan


def extract_cell_line(text):
    patterns = (
        r"\bVero(?:\s*\d+)?\b",
        r"\bHFF\b",
        r"\bHEL(?:\s*cells?)?\b",
        r"\bHeLa(?:S3)?\b",
        r"\bA549\b",
        r"\bCV-?1\b",
        r"\bBSC-?\d+\b",
        r"\bLLC-MK2\b",
    )
    match = next((re.search(pattern, text, re.IGNORECASE) for pattern in patterns
                  if re.search(pattern, text, re.IGNORECASE)), None)
    return match.group(0) if match else np.nan


def extract_strain(text):
    match = re.search(
        r"Vaccinia virus\s+([A-Za-z0-9][A-Za-z0-9./_-]*?)(?=\s+(?:infected|assessed|measured|in\b))",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" ,;") if match else np.nan


def source_rows():
    rows = []

    chembl = pd.read_csv(DATA_DIR / "collected_raw.csv")
    for row in chembl.to_dict(orient="records"):
        virus = str(row.get("virus_source", "Vaccinia")).strip()
        organism = {
            "vaccinia": "Vaccinia virus",
            "cowpox": "Cowpox virus",
            "monkeypox": "Monkeypox virus",
            "variola": "Variola virus",
        }.get(virus.lower(), virus if virus else "Vaccinia virus")
        rows.append(
            {
                "original_smiles": row.get("SMILES"),
                "IC50": number(row.get("IC50uM")),
                "IC50_unit": "uM",
                "CC50": number(row.get("CC50_uM")),
                "CC50_unit": "uM",
                "organism": organism,
                "virus_strain": np.nan,
                "cell_line": np.nan,
                "assay_id": np.nan,
                "source": "ChEMBL",
                "DOI": row.get("DOI"),
                "measurement_relation": "=",
                "standard_type": "IC50",
            }
        )

    figshare = pd.read_csv(DATA_DIR / "collectedfromarticles.csv")
    for row in figshare.to_dict(orient="records"):
        ic50 = number(row.get("IC50uM"))
        cc50 = number(row.get("CC50uM"))
        rows.append(
            {
                "original_smiles": row.get("SMILES"),
                "IC50": ic50,
                "IC50_unit": "uM",
                "CC50": cc50,
                "CC50_unit": "uM",
                "SI": cc50 / ic50 if np.isfinite(ic50) and ic50 > 0 and np.isfinite(cc50) and cc50 > 0 else np.nan,
                "organism": "Vaccinia virus",
                "virus_strain": np.nan,
                "cell_line": np.nan,
                "assay_id": "Figshare",
                "source": "Figshare",
                "DOI": row.get("DOI"),
                "measurement_relation": "=",
                "standard_type": "IC50/CC50",
            }
        )

    pubchem_path = DATA_DIR / "pubchem_collected.csv"
    if pubchem_path.is_file():
        pubchem = pd.read_csv(pubchem_path)
        for row in pubchem.to_dict(orient="records"):
            ic50 = number(row.get("IC50uM"))
            cc50 = number(row.get("CC50uM"))
            ic_aid = str(row.get("IC50_AID", "")).strip()
            cc_aid = str(row.get("CC50_AID", "")).strip()
            same_assay = bool(ic_aid and cc_aid and ic_aid == cc_aid)
            assay_ids = "|".join(sorted({aid for aid in (ic_aid, cc_aid) if aid}))
            context = str(row.get("assay_context", ""))
            rows.append(
                {
                    "original_smiles": row.get("SMILES"),
                    "IC50": ic50,
                    "IC50_unit": "uM",
                    "CC50": cc50,
                    "CC50_unit": "uM",
                    "SI": cc50 / ic50 if same_assay and ic50 > 0 and cc50 > 0 else np.nan,
                    "organism": "Vaccinia virus",
                    "virus_strain": extract_strain(context),
                    "cell_line": extract_cell_line(context),
                    "assay_id": assay_ids or np.nan,
                    "source": "PubChem",
                    "DOI": row.get("DOI"),
                    "measurement_relation": "=",
                    "standard_type": "IC50/CC50",
                }
            )

    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    rows = source_rows()
    valid_rows = []
    invalid_smiles = []

    for row in rows:
        original = row.get("original_smiles")
        canonical = canonicalize(original)
        if canonical is None:
            invalid_smiles.append(original)
            continue
        row["canonical_smiles"] = canonical
        if pd.isna(row.get("IC50")) and pd.isna(row.get("CC50")):
            continue
        if "SI" not in row:
            same_assay_pair = (
                row["source"] == "Figshare"
                and np.isfinite(row["IC50"])
                and row["IC50"] > 0
                and np.isfinite(row["CC50"])
                and row["CC50"] > 0
            )
            row["SI"] = row["CC50"] / row["IC50"] if same_assay_pair else np.nan
        valid_rows.append(row)

    dataset = pd.DataFrame(valid_rows)
    for column in COLUMNS:
        if column not in dataset:
            dataset[column] = np.nan
    dataset = dataset[COLUMNS]

    # Rows without assay identifiers cannot safely be deduplicated by assay.
    identified = dataset["assay_id"].notna() & dataset["assay_id"].astype(str).str.strip().ne("")
    identified_rows = dataset.loc[identified].drop_duplicates(
        subset=["canonical_smiles", "assay_id"], keep="first"
    )
    dataset = pd.concat([dataset.loc[~identified], identified_rows], ignore_index=True)
    dataset = dataset[COLUMNS]
    dataset.to_csv(OUTPUT_PATH, index=False)

    relation = dataset["measurement_relation"].fillna("")
    stats = {
        "total_rows": len(dataset),
        "rows_with_calculated_SI": int(dataset["SI"].notna().sum()),
        "unique_canonical_smiles": int(dataset["canonical_smiles"].nunique()),
        "censored_rows": int((relation != "=").sum()),
        "invalid_smiles_dropped": len(invalid_smiles),
        "source_rows": dataset["source"].value_counts().to_dict(),
    }
    print("Saved:", OUTPUT_PATH)
    print("Invalid SMILES dropped:", len(invalid_smiles))
    for smiles in invalid_smiles:
        print("Invalid SMILES:", smiles)
    print("First 5 rows:")
    print(dataset.head(5).to_string(index=False))
    print("Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
