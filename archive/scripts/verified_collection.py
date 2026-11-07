import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "final_collected_dataset.csv"
USER_AGENT = "PlagueIn-verified-literature-collection/1.0"
TIMEOUT = 30
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

COMPOUNDS = [
    {
        "name": "Tecovirimat (ST-246)",
        "pubchem_name": "Tecovirimat",
        "IC50": 0.05,
        "CC50": 100.0,
        "measurement_relation": ">",
        "virus_strain": "VACV (strain unspecified)",
        "cell_line": np.nan,
        "assay_id": "PMC1797744_T1",
        "source": "PMC1797744",
        "DOI": "10.1128/AAC.00879-06",
        "standard_type": "EC50",
    },
    {
        "name": "Marboran",
        "pubchem_name": "Methisazone",
        "IC50": 3.3,
        "CC50": 427.0,
        "measurement_relation": ">",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "PMC4264105_T1",
        "source": "PMC4264105",
        "DOI": "10.2174/1568005053174627",
        "standard_type": "EC50",
    },
    {
        "name": "Cidofovir",
        "pubchem_name": "Cidofovir",
        "IC50": 31.0,
        "CC50": 317.0,
        "measurement_relation": ">",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "PMC4264105_T1",
        "source": "PMC4264105",
        "DOI": "10.2174/1568005053174627",
        "standard_type": "EC50",
    },
    {
        "name": "Idoxuridine",
        "pubchem_name": "Idoxuridine",
        "IC50": 6.0,
        "CC50": 260.0,
        "measurement_relation": ">",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "PMC4264105_T1",
        "source": "PMC4264105",
        "DOI": "10.2174/1568005053174627",
        "standard_type": "EC50",
    },
    {
        "name": "Bufalin",
        "pubchem_name": "Bufalin",
        "IC50": 0.640,
        "CC50": 65.9,
        "measurement_relation": "=",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "core_2024_CG",
        "source": "core.ac.uk",
        "DOI": np.nan,
        "standard_type": "IC50",
    },
    {
        "name": "Digoxin",
        "pubchem_name": "Digoxin",
        "IC50": 4.64,
        "CC50": 65.9,
        "measurement_relation": "=",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "core_2024_CG",
        "source": "core.ac.uk",
        "DOI": np.nan,
        "standard_type": "IC50",
    },
    {
        "name": "Nigericin",
        "pubchem_name": "Nigericin",
        "IC50": 0.0079,
        "CC50": 8.2,
        "measurement_relation": "=",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "SD_2010_Nigericin",
        "source": "ScienceDirect",
        "DOI": np.nan,
        "standard_type": "IC50",
    },
    {
        "name": "Monensin",
        "pubchem_name": "Monensin",
        "IC50": 0.0216,
        "CC50": 23.0,
        "measurement_relation": "=",
        "virus_strain": "VACV",
        "cell_line": np.nan,
        "assay_id": "SD_2010_Nigericin",
        "source": "ScienceDirect",
        "DOI": np.nan,
        "standard_type": "IC50",
    },
]


def pubchem_smiles(name):
    compound = urllib.parse.quote(name, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{compound}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    properties = payload["PropertyTable"]["Properties"]
    if len(properties) != 1:
        raise ValueError(f"Expected one PubChem result for {name!r}, got {len(properties)}")
    return properties[0]["SMILES"]


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES for source structure: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def build_rows():
    rows = []
    for compound in COMPOUNDS:
        smiles = pubchem_smiles(compound["pubchem_name"])
        canonical = canonicalize(smiles)
        ic50 = float(compound["IC50"])
        cc50 = float(compound["CC50"])
        relation = compound["measurement_relation"]
        rows.append(
            {
                "canonical_smiles": canonical,
                "original_smiles": smiles,
                "IC50": ic50,
                "IC50_unit": "uM",
                "CC50": cc50,
                "CC50_unit": "uM",
                "SI": cc50 / ic50 if relation == "=" else np.nan,
                "organism": "Vaccinia virus",
                "virus_strain": compound["virus_strain"],
                "cell_line": compound["cell_line"],
                "assay_id": compound["assay_id"],
                "source": compound["source"],
                "DOI": compound["DOI"],
                "measurement_relation": relation,
                "standard_type": compound["standard_type"],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    RDLogger.DisableLog("rdApp.error")

    existing = pd.read_csv(DATASET_PATH)
    if existing.columns.tolist() != COLUMNS:
        raise ValueError(f"Unexpected dataset schema: {existing.columns.tolist()}")

    # These rows were added in the prior expansion and are explicitly deferred here.
    deferred = existing["source"].isin(
        {"PMC3910790", "PMC6335999", "PMC7651861", "PMC12846358"}
    )
    removed_deferred_rows = int(deferred.sum())
    base = existing.loc[~deferred].copy()

    additions = build_rows()
    for row in additions.to_dict(orient="records"):
        if canonicalize(row["original_smiles"]) != row["canonical_smiles"]:
            raise ValueError(f"Canonical SMILES mismatch for {row['source']}")
        if row["measurement_relation"] == "=":
            expected_si = row["CC50"] / row["IC50"]
            if not np.isclose(row["SI"], expected_si):
                raise ValueError(f"SI formula check failed for {row['source']}")
        elif not pd.isna(row["SI"]):
            raise ValueError("Censored measurements must have SI=NaN")

    existing_keys = {
        (row.canonical_smiles, str(row.assay_id))
        for row in base.loc[
            base["assay_id"].notna(), ["canonical_smiles", "assay_id"]
        ].itertuples(index=False)
    }
    new_rows = []
    for row in additions.to_dict(orient="records"):
        key = (row["canonical_smiles"], str(row["assay_id"]))
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)

    new_frame = pd.DataFrame(new_rows, columns=COLUMNS)
    updated = pd.concat([base, new_frame], ignore_index=True)
    identified = updated["assay_id"].notna()
    identified_rows = updated.loc[identified].drop_duplicates(
        subset=["canonical_smiles", "assay_id"], keep="first"
    )
    updated = pd.concat(
        [updated.loc[~identified], identified_rows], ignore_index=True
    )[COLUMNS]

    for smiles in updated["canonical_smiles"].dropna():
        if canonicalize(smiles) != smiles:
            raise ValueError(f"Noncanonical or invalid stored SMILES: {smiles}")

    updated.to_csv(DATASET_PATH, index=False)
    new_unique_smiles = len(
        set(new_frame["canonical_smiles"])
        - set(base["canonical_smiles"].dropna())
    )
    stats = {
        "removed_deferred_rows": removed_deferred_rows,
        "rows_added_by_source": (
            new_frame.groupby("source").size().astype(int).to_dict()
            if not new_frame.empty
            else {}
        ),
        "rows_added": len(new_frame),
        "new_unique_smiles": new_unique_smiles,
        "total_rows": len(updated),
        "total_unique_smiles": int(updated["canonical_smiles"].nunique()),
        "total_rows_with_calculated_SI": int(updated["SI"].notna().sum()),
        "total_censored_rows": int(
            (updated["measurement_relation"].fillna("") != "=").sum()
        ),
    }

    print("Updated:", DATASET_PATH)
    print("Rows removed from deferred sources:", removed_deferred_rows)
    print("Rows added by source:", stats["rows_added_by_source"])
    print("Rows added:", stats["rows_added"])
    print("New unique SMILES:", new_unique_smiles)
    print("Total rows:", stats["total_rows"])
    print("Total unique SMILES:", stats["total_unique_smiles"])
    print("Total rows with calculated SI:", stats["total_rows_with_calculated_SI"])
    print("Total censored rows:", stats["total_censored_rows"])
    print("First 5 rows:")
    print(updated.head(5).to_string(index=False))
    print("Added records:")
    print(new_frame.to_string(index=False))


if __name__ == "__main__":
    main()
