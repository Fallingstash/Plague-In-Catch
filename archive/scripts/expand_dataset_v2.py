import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "final_collected_dataset.csv"
USER_AGENT = "PlagueIn-literature-dataset/1.0"
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

# Table 2 reports EC50 for antiviral activity and CC50 in HEL cells.
PMC3910790_COMPOUNDS = {
    "Cidofovir": {
        "cc50": 397.0,
        "cc50_relation": "=",
        "ec50": {"VACV-Cop": 6.9, "VACV-Lis": 9.1, "VACV-WR": 8.2},
        "pubchem_name": "Cidofovir",
    },
    "CMX001": {
        "cc50": 2.4,
        "cc50_relation": ">=",
        "ec50": {"VACV-Cop": 0.005, "VACV-Lis": 0.023, "VACV-WR": 0.013},
        "pubchem_name": "CMX001",
    },
    "KAY-2-41": {
        "cc50": 14.0,
        "cc50_relation": "=",
        "ec50": {"VACV-Cop": 0.48, "VACV-Lis": 0.41, "VACV-WR": 0.80},
        "systematic_name": (
            "1-(2-deoxy-1-methyl-4-thio-beta-D-ribofuranosyl)thymine"
        ),
    },
}

SKIPPED_SOURCES = {
    "PMC7651861": "The article page was blocked during retrieval; no verified compound SMILES were available.",
    "PMC6335999": "Table values are indexed by compound number; full structures were not reliably resolved to SMILES.",
    "PMC4067006": "The table gives R substituents, not a complete molecular structure/SMILES.",
    "core.ac.uk cardiac glycosides": "No specific paper/table or compound-level measurements were identified; no values were inferred.",
}


def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def pubchem_smiles(name):
    encoded_name = urllib.parse.quote(name, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded_name}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    payload = fetch_json(url)
    properties = payload["PropertyTable"]["Properties"]
    if len(properties) != 1:
        raise ValueError(f"Expected one PubChem structure for {name!r}, got {len(properties)}")
    return properties[0]["SMILES"]


def opsin_smiles(systematic_name):
    encoded_name = urllib.parse.quote(systematic_name, safe="")
    payload = fetch_json(f"https://opsin.ch.cam.ac.uk/opsin/{encoded_name}.json")
    if payload.get("status") != "SUCCESS" or not payload.get("smiles"):
        raise ValueError(f"OPSIN could not resolve {systematic_name!r}")
    return payload["smiles"]


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES returned for structure: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def literature_rows():
    rows = []
    for compound, measurements in PMC3910790_COMPOUNDS.items():
        if "pubchem_name" in measurements:
            original_smiles = pubchem_smiles(measurements["pubchem_name"])
            smiles_origin = "PubChem"
        else:
            # PubChem has no entry under this name; use the article's full systematic name.
            original_smiles = opsin_smiles(measurements["systematic_name"])
            smiles_origin = "OPSIN from article systematic name"

        canonical = canonicalize(original_smiles)
        for strain, ec50 in measurements["ec50"].items():
            rows.append(
                {
                    "canonical_smiles": canonical,
                    "original_smiles": original_smiles,
                    "IC50": float(ec50),
                    "IC50_unit": "uM",
                    "CC50": float(measurements["cc50"]),
                    "CC50_unit": "uM",
                    "SI": float(measurements["cc50"]) / float(ec50),
                    "organism": "Vaccinia virus",
                    "virus_strain": strain,
                    "cell_line": "HEL",
                    "assay_id": f"PMC3910790_T2_{strain}",
                    "source": "PMC3910790",
                    "DOI": "10.1128/AAC.01601-13",
                    "measurement_relation": measurements["cc50_relation"],
                    "standard_type": "EC50",
                    "_compound": compound,
                    "_smiles_origin": smiles_origin,
                }
            )
    return rows


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Base dataset not found: {DATASET_PATH}")
    RDLogger.DisableLog("rdApp.error")

    base = pd.read_csv(DATASET_PATH)
    if base.columns.tolist() != COLUMNS:
        raise ValueError(f"Unexpected base dataset schema: {base.columns.tolist()}")
    additions = pd.DataFrame(literature_rows())

    for row in additions.to_dict(orient="records"):
        if canonicalize(row["original_smiles"]) != row["canonical_smiles"]:
            raise ValueError(f"Canonical SMILES mismatch for {row['_compound']}")
        if not (np.isfinite(row["IC50"]) and row["IC50"] > 0):
            raise ValueError(f"Invalid EC50 for {row['_compound']}")
        if not (np.isfinite(row["CC50"]) and row["CC50"] > 0):
            raise ValueError(f"Invalid CC50 for {row['_compound']}")
        expected_si = row["CC50"] / row["IC50"]
        if not np.isclose(row["SI"], expected_si):
            raise ValueError(f"Invalid SI for {row['_compound']} / {row['virus_strain']}")

    known_keys = {
        (row.canonical_smiles, str(row.assay_id))
        for row in base.loc[base["assay_id"].notna(), ["canonical_smiles", "assay_id"]].itertuples(index=False)
    }
    new_rows = []
    for row in additions.to_dict(orient="records"):
        key = (row["canonical_smiles"], row["assay_id"])
        if key in known_keys:
            continue
        known_keys.add(key)
        new_rows.append(row)

    added_records = pd.DataFrame(new_rows)
    new_frame = added_records[COLUMNS] if not added_records.empty else pd.DataFrame(columns=COLUMNS)
    updated = pd.concat([base, new_frame], ignore_index=True)

    identified = updated["assay_id"].notna()
    identified_rows = updated.loc[identified].drop_duplicates(
        subset=["canonical_smiles", "assay_id"], keep="first"
    )
    updated = pd.concat(
        [updated.loc[~identified], identified_rows], ignore_index=True
    )[COLUMNS]
    updated.to_csv(DATASET_PATH, index=False)

    source_rows = updated.loc[updated["source"] == "PMC3910790"]
    preexisting_smiles = set(
        updated.loc[updated["source"] != "PMC3910790", "canonical_smiles"].dropna()
    )
    stats = {
        "rows_added_by_source": source_rows.groupby("source").size().astype(int).to_dict(),
        "rows_appended_this_run": len(new_frame),
        "new_unique_smiles": len(
            set(source_rows["canonical_smiles"].dropna()) - preexisting_smiles
        ),
        "new_rows_with_calculated_SI": int(source_rows["SI"].notna().sum()),
        "total_rows": len(updated),
        "total_unique_smiles": int(updated["canonical_smiles"].nunique()),
        "total_rows_with_calculated_SI": int(updated["SI"].notna().sum()),
        "total_censored_rows": int(
            (updated["measurement_relation"].fillna("") != "=").sum()
        ),
        "skipped_sources": SKIPPED_SOURCES,
        "smiles_resolution": (
            added_records.groupby("_smiles_origin").size().astype(int).to_dict()
            if not added_records.empty
            else {}
        ),
    }

    print("Updated:", DATASET_PATH)
    print("New source rows:", stats["rows_added_by_source"])
    print("New unique SMILES:", stats["new_unique_smiles"])
    print("New rows with calculated SI:", stats["new_rows_with_calculated_SI"])
    print("Total rows:", stats["total_rows"])
    print("Total unique SMILES:", stats["total_unique_smiles"])
    print("Total rows with calculated SI:", stats["total_rows_with_calculated_SI"])
    print("Total censored rows:", stats["total_censored_rows"])
    print("First 5 rows:")
    print(updated.head(5).to_string(index=False))
    print("Added rows:")
    print(updated[updated["source"] == "PMC3910790"].to_string(index=False))
    print("Skipped sources:")
    print(json.dumps(SKIPPED_SOURCES, indent=2))


if __name__ == "__main__":
    main()
