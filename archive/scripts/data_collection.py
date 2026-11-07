import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WARNINGS_PATH = ROOT / "warnings.log"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/"
USER_AGENT = "PlagueIn-hackathon-data-collection/1.0"
REQUEST_TIMEOUT = 45
REQUEST_DELAY_SECONDS = 0.15
MAX_RETRIES = 4
MIN_VACV_PAIRED_MOLECULES_FOR_BONUS = 20
ACTIVITY_FIELDS = (
    "molecule_chembl_id,standard_type,standard_value,standard_units,"
    "standard_relation,assay_chembl_id,document_chembl_id"
)

UNIT_TO_UM = {
    "m": 1_000_000.0,
    "mol/l": 1_000_000.0,
    "molar": 1_000_000.0,
    "mm": 1_000.0,
    "mmol/l": 1_000.0,
    "millimolar": 1_000.0,
    "um": 1.0,
    "μm": 1.0,
    "µm": 1.0,
    "umol/l": 1.0,
    "micromolar": 1.0,
    "nm": 0.001,
    "nmol/l": 0.001,
    "nanomolar": 0.001,
    "pm": 0.000001,
    "pmol/l": 0.000001,
    "picomolar": 0.000001,
}
MASS_CONCENTRATION_UNITS = {
    "ug.ml-1",
    "ug/ml",
    "μg.ml-1",
    "μg/ml",
    "µg.ml-1",
    "µg/ml",
    "mg.l-1",
    "mg/l",
}


class ChEMBLClient:
    def __init__(self):
        self.molecule_cache = {}
        self.document_cache = {}
        self.last_request_at = 0.0

    def get_json(self, url):
        if url.startswith("/"):
            url = "https://www.ebi.ac.uk" + url
        if not url.startswith("https://www.ebi.ac.uk/chembl/api/data/"):
            raise ValueError(f"Unexpected ChEMBL URL: {url}")

        for attempt in range(MAX_RETRIES):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < REQUEST_DELAY_SECONDS:
                time.sleep(REQUEST_DELAY_SECONDS - elapsed)
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                    self.last_request_at = time.monotonic()
                    return json.load(response)
            except urllib.error.HTTPError as error:
                self.last_request_at = time.monotonic()
                if error.code == 404:
                    raise
                if error.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES - 1:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2**attempt
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError):
                self.last_request_at = time.monotonic()
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"ChEMBL request failed after retries: {url}")

    def molecule(self, chembl_id):
        if chembl_id not in self.molecule_cache:
            path = urllib.parse.quote(chembl_id, safe="")
            self.molecule_cache[chembl_id] = self.get_json(
                f"{CHEMBL_API}molecule/{path}.json"
            )
        return self.molecule_cache[chembl_id]

    def document(self, document_id):
        if not document_id:
            return None
        if document_id not in self.document_cache:
            path = urllib.parse.quote(document_id, safe="")
            self.document_cache[document_id] = self.get_json(
                f"{CHEMBL_API}document/{path}.json"
            )
        return self.document_cache[document_id]


def warn(message):
    with WARNINGS_PATH.open("a", encoding="utf-8") as log:
        log.write(message.rstrip() + "\n")


def normalize_unit(unit):
    normalized = str(unit or "").strip().lower()
    normalized = normalized.replace("−", "-").replace(" ", "")
    normalized = normalized.replace("µmol/l", "umol/l")
    normalized = normalized.replace("μmol/l", "umol/l")
    return UNIT_TO_UM.get(normalized)


def as_micromolar(value, unit, molecular_weight=None):
    normalized_unit = str(unit or "").strip().lower().replace("−", "-").replace(" ", "")
    if normalized_unit in MASS_CONCENTRATION_UNITS:
        try:
            number = float(value)
            weight = float(molecular_weight)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(number)
            or number <= 0
            or not math.isfinite(weight)
            or weight <= 0
        ):
            return None
        converted = number * 1000.0 / weight
        return converted if math.isfinite(converted) and converted > 0 else None
    factor = normalize_unit(unit)
    if factor is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    converted = number * factor
    return converted if math.isfinite(converted) and converted > 0 else None


def search_targets(client, query, required_term):
    query_string = urllib.parse.urlencode({"q": query, "limit": 1000})
    payload = client.get_json(f"{CHEMBL_API}target/search.json?{query_string}")
    found = {}
    for target in payload.get("targets", []):
        searchable = " ".join(
            (target.get("pref_name") or "", target.get("organism") or "")
        ).casefold()
        if required_term.casefold() not in searchable:
            continue
        chembl_id = target.get("target_chembl_id")
        if chembl_id:
            found[chembl_id] = {
                "target_chembl_id": chembl_id,
                "pref_name": target.get("pref_name"),
                "organism": target.get("organism"),
            }
    return list(found.values())


def fetch_target_activities(client, target_id):
    params = urllib.parse.urlencode(
        {
            "target_chembl_id": target_id,
            "limit": 1000,
            "only": ACTIVITY_FIELDS,
        }
    )
    next_url = f"{CHEMBL_API}activity.json?{params}"
    activities = []
    seen_urls = set()
    while next_url:
        if next_url in seen_urls:
            raise RuntimeError(f"ChEMBL activity pagination loop for target {target_id}")
        seen_urls.add(next_url)
        page = client.get_json(next_url)
        activities.extend(page.get("activities", []))
        next_url = page.get("page_meta", {}).get("next")
        print(
            f"  {target_id}: fetched {len(activities)} activities",
            flush=True,
        )
    return activities


def raw_activity_value(activity):
    relation = (activity.get("standard_relation") or "").strip()
    if relation and relation != "=":
        return None
    try:
        value = float(activity.get("standard_value"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def molecule_records(client, activities, source_label):
    grouped = {}
    for activity in activities:
        molecule_id = activity.get("molecule_chembl_id")
        activity_type = (activity.get("standard_type") or "").strip().upper()
        if not molecule_id or activity_type not in {"IC50", "CC50"}:
            continue
        value = raw_activity_value(activity)
        if value is None:
            continue
        record = grouped.setdefault(
            molecule_id,
            {"IC50uM": [], "CC50_uM": [], "documents": set()},
        )
        record["IC50uM" if activity_type == "IC50" else "CC50_uM"].append(
            (value, activity.get("standard_units"))
        )
        document_id = activity.get("document_chembl_id")
        if document_id:
            record["documents"].add(document_id)

    rows = []
    for molecule_id, measurements in grouped.items():
        if not measurements["IC50uM"]:
            continue
        try:
            molecule = client.molecule(molecule_id)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            warn(f"ChEMBL molecule missing: {molecule_id} (HTTP 404)")
            continue
        structures = molecule.get("molecule_structures") or molecule.get(
            "moleculestructures"
        ) or {}
        smiles = structures.get("canonical_smiles") or structures.get(
            "canonicalsmiles"
        )
        molecular_weight = None
        try:
            molecule_properties = molecule.get("molecule_properties") or {}
            molecular_weight = float(molecule_properties.get("full_mwt"))
        except (TypeError, ValueError):
            if smiles:
                parsed_molecule = Chem.MolFromSmiles(smiles)
                if parsed_molecule is not None:
                    molecular_weight = Descriptors.MolWt(parsed_molecule)

        def converted_measurements(measurements):
            values = [
                as_micromolar(value, unit, molecular_weight)
                for value, unit in measurements
            ]
            return [value for value in values if value is not None]

        ic50_values = converted_measurements(measurements["IC50uM"])
        cc50_values = converted_measurements(measurements["CC50_uM"])
        if not ic50_values:
            continue
        doi_values = set()
        for document_id in sorted(measurements["documents"]):
            try:
                document = client.document(document_id)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                warn(f"ChEMBL document missing: {document_id} (HTTP 404)")
                continue
            if document and document.get("doi"):
                doi_values.add(str(document["doi"]))
        ic50 = float(np.median(ic50_values))
        cc50 = float(np.median(cc50_values)) if cc50_values else np.nan
        si = cc50 / ic50 if math.isfinite(cc50) else np.nan
        rows.append(
            {
                "chemblid": molecule_id,
                "SMILES": smiles,
                "IC50uM": ic50,
                "CC50_uM": cc50,
                "SI": si,
                "DOI": "; ".join(sorted(doi_values)) or None,
                "source": "ChEMBL",
                "virus_source": source_label,
            }
        )
    return rows


def save_source_rows(rows, path):
    columns = [
        "chemblid",
        "SMILES",
        "IC50uM",
        "CC50_uM",
        "SI",
        "DOI",
        "source",
        "virus_source",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def collect_chembl_campaign(client, query, required_term, source_label):
    targets = search_targets(client, query, required_term)
    print(f"{source_label}: matched {len(targets)} targets")
    activity_rows = []
    for target in targets:
        print(
            f"  Target {target['target_chembl_id']}: "
            f"{target.get('pref_name')} ({target.get('organism')})",
            flush=True,
        )
        activity_rows.extend(
            fetch_target_activities(client, target["target_chembl_id"])
        )
    records = molecule_records(client, activity_rows, source_label)
    return targets, activity_rows, records


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        warn(f"Missing SMILES: {smiles!r}")
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as error:
        warn(f"RDKit parse error for SMILES {smiles!r}: {error}")
        return None
    if molecule is None:
        warn(f"Invalid SMILES: {smiles!r}")
        return None
    try:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception as error:
        warn(f"RDKit canonicalization error for SMILES {smiles!r}: {error}")
        return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = ChEMBLClient()
    campaign_summary = []

    vaccinia_targets, vaccinia_activities, vaccinia_rows = collect_chembl_campaign(
        client, "vaccinia", "vaccinia", "Vaccinia"
    )
    save_source_rows(vaccinia_rows, DATA_DIR / "chembl_vaccinia_raw.csv")
    chembl_rows = list(vaccinia_rows)
    save_source_rows(chembl_rows, DATA_DIR / "chembl_raw.csv")
    campaign_summary.append(
        {
            "query": "vaccinia",
            "targets": vaccinia_targets,
            "activity_count": len(vaccinia_activities),
            "candidate_molecules_with_ic50": len(vaccinia_rows),
        }
    )
    print(
        f"Saved Vaccinia intermediate: {len(vaccinia_rows)} molecules "
        "with measured IC50",
        flush=True,
    )

    paired_vaccinia = sum(
        math.isfinite(float(row["CC50_uM"])) for row in vaccinia_rows
    )
    if paired_vaccinia < MIN_VACV_PAIRED_MOLECULES_FOR_BONUS:
        for query, required_term in (
            ("cowpox", "cowpox"),
            ("monkeypox", "monkeypox"),
            ("variola", "variola"),
        ):
            targets, activities, rows = collect_chembl_campaign(
                client, query, required_term, query.title()
            )
            save_source_rows(rows, DATA_DIR / f"chembl_{query}_raw.csv")
            chembl_rows.extend(rows)
            save_source_rows(chembl_rows, DATA_DIR / "chembl_raw.csv")
            campaign_summary.append(
                {
                    "query": query,
                    "targets": targets,
                    "activity_count": len(activities),
                    "candidate_molecules_with_ic50": len(rows),
                }
            )
            print(
                f"Saved {query} intermediate: {len(rows)} molecules "
                "with measured IC50",
                flush=True,
            )
    else:
        print(
            f"Vaccinia has {paired_vaccinia} IC50/CC50 pairs; "
            "skipping related-virus bonus searches",
            flush=True,
        )

    raw_columns = [
        "chemblid",
        "SMILES",
        "IC50uM",
        "CC50_uM",
        "SI",
        "DOI",
        "source",
        "virus_source",
    ]
    raw = pd.DataFrame(chembl_rows, columns=raw_columns)
    raw.to_csv(DATA_DIR / "chembl_raw.csv", index=False)
    raw.to_csv(DATA_DIR / "collected_raw.csv", index=False)

    train = pd.read_csv(ROOT / "train.csv")
    train_canonical = {
        canonical
        for canonical in (canonicalize(smiles) for smiles in train["SMILES"])
        if canonical is not None
    }

    valid_rows = []
    valid_smiles_count = 0
    invalid_smiles = 0
    train_duplicates = 0
    incomplete_pair_count = 0
    canonical_counts = {}
    for row in raw.to_dict(orient="records"):
        canonical = canonicalize(row.get("SMILES"))
        if canonical is None:
            invalid_smiles += 1
            continue
        valid_smiles_count += 1
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
        if canonical in train_canonical:
            train_duplicates += 1
            continue
        try:
            ic50 = float(row["IC50uM"])
            cc50 = float(row["CC50_uM"])
        except (TypeError, ValueError):
            incomplete_pair_count += 1
            continue
        if not math.isfinite(ic50) or not math.isfinite(cc50) or ic50 <= 0 or cc50 <= 0:
            incomplete_pair_count += 1
            continue
        si = cc50 / ic50
        if not math.isfinite(si) or si <= 0:
            continue
        valid_rows.append(
            {
                "canonical_smiles": canonical,
                "SMILES": canonical,
                "IC50uM": ic50,
                "CC50uM": cc50,
                "SI": si,
                "is_censored": 0,
                "DOI": row.get("DOI"),
                "source": f"ChEMBL ({row.get('virus_source')})",
            }
        )

    collected = pd.DataFrame(
        valid_rows,
        columns=[
            "canonical_smiles",
            "SMILES",
            "IC50uM",
            "CC50uM",
            "SI",
            "is_censored",
            "DOI",
            "source",
        ],
    )
    duplicate_count = (
        int(collected["canonical_smiles"].duplicated(keep="first").sum())
        if not collected.empty
        else 0
    )
    if duplicate_count:
        collected = collected.drop_duplicates("canonical_smiles", keep="first").copy()
    collected = collected.drop(columns=["canonical_smiles"])
    collected.to_csv(DATA_DIR / "collected.csv", index=False)

    all_source_labels = [
        "ChEMBL (Vaccinia)",
        "ChEMBL (Cowpox)",
        "ChEMBL (Monkeypox)",
        "ChEMBL (Variola)",
    ]
    source_row_counts = {label: 0 for label in all_source_labels}
    if not collected.empty:
        source_row_counts.update(collected["source"].value_counts().to_dict())
    raw_source_counts = (
        raw["virus_source"].value_counts().to_dict() if not raw.empty else {}
    )
    summary = {
        "chembl_campaigns": campaign_summary,
        "bindingdb": {
            "status": "skipped",
            "reason": "The provided bulk archive is about 500 MB and the API route requires a target sequence; no suitable VACV sequence endpoint was provided.",
        },
        "raw_chembl_rows": len(raw),
        "raw_candidate_counts_by_virus": raw_source_counts,
        "source_row_counts": source_row_counts,
        "valid_smiles_before_deduplication": valid_smiles_count,
        "invalid_smiles": invalid_smiles,
        "duplicates_with_train": train_duplicates,
        "duplicates_within_collected": duplicate_count,
        "duplicate_raw_structures_across_sources": sum(
            count - 1 for count in canonical_counts.values() if count > 1
        ),
        "incomplete_ic50_cc50_pairs_dropped": incomplete_pair_count,
        "final_rows": len(collected),
        "is_censored_counts": {
            "0": int((collected["is_censored"] == 0).sum()),
            "1": int((collected["is_censored"] == 1).sum()),
        },
        "si_distribution": (
            collected["SI"].describe().to_dict()
            if not collected.empty
            else {}
        ),
        "note": "Incomplete IC50/CC50 pairs were excluded because SI cannot be computed reliably; collected records therefore have is_censored=0.",
    }
    (DATA_DIR / "collection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCollection statistics")
    print(f"Raw ChEMBL candidate rows: {len(raw)}")
    print(f"Valid SMILES before deduplication: {valid_smiles_count}")
    print(f"Invalid/missing SMILES: {invalid_smiles}")
    print(f"Duplicates with train: {train_duplicates}")
    print(f"Duplicates within collected: {duplicate_count}")
    print(f"Incomplete IC50/CC50 pairs dropped: {incomplete_pair_count}")
    print(f"Final rows: {len(collected)}")
    print(f"Raw candidate rows by virus: {raw_source_counts}")
    print(f"Rows by source: {summary['source_row_counts']}")
    print(f"is_censored counts: {summary['is_censored_counts']}")
    if not collected.empty:
        print("SI distribution:")
        print(collected["SI"].describe().to_string())
    print(f"BindingDB: {summary['bindingdb']['status']} ({summary['bindingdb']['reason']})")
    print(f"Final data: {DATA_DIR / 'collected.csv'}")
    print(f"Summary: {DATA_DIR / 'collection_summary.json'}")


if __name__ == "__main__":
    main()
