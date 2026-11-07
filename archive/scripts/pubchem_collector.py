import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUG_CSV = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/csv"
SEARCH_TERMS = (
    "vaccinia[All Fields]",
    "VACV[All Fields]",
    "orthopox[All Fields]",
)
RELATED_ASSAY_IDS = ("2221754",)
RETMAX = 500
REQUEST_TIMEOUT = 35
REQUEST_DELAY_SECONDS = 0.34
USER_AGENT = "VACV-research-dataset-collector/1.0"
VIRUS_TERMS = ("vaccinia", "vacv", "orthopox")
ACTIVITY_TERMS = (
    "antiviral",
    "anti-viral",
    "cytotoxic",
    "cytotoxicity",
    "selectivity index",
    "virus-induced",
    "viral replication",
    "plaque formation",
)


def request_json(url, params=None):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def batched(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def search_assay_ids():
    ids = []
    search_counts = {}
    for term in SEARCH_TERMS:
        payload = request_json(
            EUTILS + "esearch.fcgi",
            {
                "db": "pcassay",
                "term": term,
                "retmax": RETMAX,
                "retmode": "json",
            },
        )
        result = payload["esearchresult"]
        found = [str(value) for value in result["idlist"]]
        ids.extend(found)
        search_counts[term] = {
            "matching_records": int(result["count"]),
            "fetched": len(found),
        }
        print(
            f"PubChem search {term}: {result['count']} matches, "
            f"{len(found)} retrieved",
            flush=True,
        )
        time.sleep(REQUEST_DELAY_SECONDS)
    ids.extend(RELATED_ASSAY_IDS)
    return list(dict.fromkeys(ids)), search_counts


def fetch_assay_summaries(aids):
    summaries = []
    for chunk in batched(aids, 100):
        payload = request_json(
            EUTILS + "esummary.fcgi",
            {
                "db": "pcassay",
                "id": ",".join(chunk),
                "retmode": "json",
            },
        )
        result = payload["result"]
        for aid in result.get("uids", []):
            summary = result.get(str(aid), {})
            name = str(summary.get("assayname", ""))
            description = str(summary.get("assaydescription", ""))
            name_text = name.lower()
            is_related_cytotoxicity = (
                str(aid) in RELATED_ASSAY_IDS and "cytotoxic" in name_text
            )
            if not is_related_cytotoxicity and (
                not any(term in name_text for term in VIRUS_TERMS)
                or not any(term in name_text for term in ACTIVITY_TERMS)
            ):
                continue
            summaries.append(
                {
                    "aid": str(aid),
                    "assayname": name,
                    "assaydescription": description,
                    "source": str(summary.get("source", "")),
                    "target": str(summary.get("target", "")),
                }
            )
        time.sleep(REQUEST_DELAY_SECONDS)
    return summaries


def fetch_assay_csv(aid):
    url = PUG_CSV.format(aid=urllib.parse.quote(str(aid), safe=""))
    return request_text(url)


def parse_assay_csv(aid, assay):
    csv_text = fetch_assay_csv(aid)
    table = pd.read_csv(io.StringIO(csv_text), skiprows=[1, 2, 3], low_memory=False)
    table.columns = [str(column).strip() for column in table.columns]
    required = {
        "PUBCHEM_RESULT_TAG",
        "PUBCHEM_CID",
        "PUBCHEM_EXT_DATASOURCE_SMILES",
        "Standard Type",
    }
    if not required.issubset(table.columns):
        return [], {
            "aid": aid,
            "status": "missing_required_columns",
            "columns": list(table.columns),
            "downloaded_rows": len(table),
        }

    value_column = (
        "PubChem Standard Value"
        if "PubChem Standard Value" in table.columns
        else "Standard Value"
    )
    records = []
    data_rows = table[pd.to_numeric(table["PUBCHEM_RESULT_TAG"], errors="coerce").notna()]
    for _, row in data_rows.iterrows():
        measurement_type = str(row.get("Standard Type", "")).strip().upper()
        if measurement_type not in {"IC50", "CC50"}:
            continue
        value = pd.to_numeric(row.get(value_column), errors="coerce")
        cid = pd.to_numeric(row.get("PUBCHEM_CID"), errors="coerce")
        smiles = row.get("PUBCHEM_EXT_DATASOURCE_SMILES")
        relation = str(row.get("Standard Relation", "=")).strip()
        if pd.isna(value) or float(value) <= 0 or pd.isna(cid):
            continue
        if relation not in {"", "=", "=="}:
            continue
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            continue
        records.append(
            {
                "aid": str(aid),
                "assayname": assay["assayname"],
                "source": assay.get("source", ""),
                "cid": str(int(cid)),
                "sid": str(row.get("PUBCHEM_SID", "")),
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
                "smiles": Chem.MolToSmiles(molecule, canonical=True),
                "measurement_type": measurement_type,
                "value_uM": float(value),
                "relation": relation,
            }
        )
    return records, {
        "aid": aid,
        "assayname": assay["assayname"],
        "downloaded_rows": len(table),
        "usable_concentration_measurements": len(records),
        "status": "ok",
    }


def cell_context(assay_name):
    normalized = str(assay_name).lower()
    patterns = (
        r"\bvero(?:\s*\d+)?\b",
        r"\bhff\b",
        r"\bhel\s*cells?\b",
        r"\bhela\b",
        r"\ba549\b",
        r"\bcv-?1\b",
        r"\bbsc-?\d+\b",
        r"\bllc-mk2\b",
        r"\bht-1080\b",
    )
    matches = [re.search(pattern, normalized) for pattern in patterns]
    return next((match.group(0) for match in matches if match), "")


def pair_measurements(records):
    grouped = {}
    for record in records:
        key = (record["aid"], record["canonical_smiles"], record["measurement_type"])
        grouped.setdefault(key, []).append(record)
    assay_measurements = {}
    for (aid, canonical_smiles, measurement_type), group in grouped.items():
        first = group[0]
        assay_measurements.setdefault((aid, canonical_smiles), {}).update(
            {
                measurement_type: float(np.median([item["value_uM"] for item in group])),
                "smiles": first["smiles"],
                "cid": first["cid"],
                "sid": first["sid"],
                "assayname": first["assayname"],
                "source": first["source"],
            }
        )
    pairs = []
    by_structure = {}
    for (_, canonical_smiles), measurement in assay_measurements.items():
        by_structure.setdefault(canonical_smiles, []).append(measurement)

    for canonical_smiles, measurements in by_structure.items():
        same_assay_pairs = []
        for measurement in measurements:
            if "IC50" in measurement and "CC50" in measurement:
                same_assay_pairs.append(
                    {
                        "ic": measurement["IC50"],
                        "cc": measurement["CC50"],
                        "ic_name": measurement["assayname"],
                        "cc_name": measurement["assayname"],
                        "ic_aid": None,
                        "cc_aid": None,
                        "method": "same_assay",
                        **measurement,
                    }
                )
        pairs.extend(
            {
                "SMILES": canonical_smiles,
                "IC50uM": pair["ic"],
                "CC50uM": pair["cc"],
                "SI": pair["cc"] / pair["ic"],
                "DOI": "",
                "source": "PubChem BioAssay",
                "PubChem_CID": pair.get("cid", ""),
                "IC50_AID": pair.get("ic_aid") or "",
                "CC50_AID": pair.get("cc_aid") or "",
                "assay_context": pair["ic_name"],
                "pairing_method": pair["method"],
            }
            for pair in same_assay_pairs
        )

        antiviral = [
            item
            for item in measurements
            if "IC50" in item
            and any(term in item["assayname"].lower() for term in ("antiviral", "virus-induced", "viral replication", "plaque formation"))
        ]
        cytotoxic = [
            item
            for item in measurements
            if "CC50" in item
            and any(term in item["assayname"].lower() for term in ("cytotoxic", "cytotoxicity"))
        ]
        for ic_measurement in antiviral:
            for cc_measurement in cytotoxic:
                ic_cell = cell_context(ic_measurement["assayname"])
                cc_cell = cell_context(cc_measurement["assayname"])
                if not ic_cell or not cc_cell or ic_cell != cc_cell:
                    continue
                pairs.append(
                    {
                        "SMILES": canonical_smiles,
                        "IC50uM": ic_measurement["IC50"],
                        "CC50uM": cc_measurement["CC50"],
                        "SI": cc_measurement["CC50"] / ic_measurement["IC50"],
                        "DOI": "",
                        "source": "PubChem BioAssay",
                        "PubChem_CID": ic_measurement.get("cid", ""),
                        "IC50_AID": next(
                            (
                                record["aid"]
                                for record in records
                                if record["canonical_smiles"] == canonical_smiles
                                and record["measurement_type"] == "IC50"
                                and record["assayname"] == ic_measurement["assayname"]
                            ),
                            "",
                        ),
                        "CC50_AID": next(
                            (
                                record["aid"]
                                for record in records
                                if record["canonical_smiles"] == canonical_smiles
                                and record["measurement_type"] == "CC50"
                                and record["assayname"] == cc_measurement["assayname"]
                            ),
                            "",
                        ),
                        "assay_context": f"{ic_measurement['assayname']} | {cc_measurement['assayname']}",
                        "pairing_method": "matched_cell_line",
                    }
                )

    output = pd.DataFrame(pairs)
    if output.empty:
        return output
    output = output[
        np.isfinite(pd.to_numeric(output["IC50uM"], errors="coerce"))
        & np.isfinite(pd.to_numeric(output["CC50uM"], errors="coerce"))
        & (pd.to_numeric(output["IC50uM"], errors="coerce") > 0)
        & (pd.to_numeric(output["CC50uM"], errors="coerce") > 0)
    ].copy()
    output["SI"] = output["CC50uM"] / output["IC50uM"]
    output = output.drop_duplicates(
        subset=["SMILES", "IC50uM", "CC50uM", "IC50_AID", "CC50_AID"]
    )
    return output.reset_index(drop=True)


def load_known_structures():
    known = set()
    RDLogger.DisableLog("rdApp.error")
    paths = [
        ROOT / "train_v2.csv",
        DATA_DIR / "collected.csv",
        DATA_DIR / "collectedfromarticles.csv",
    ]
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "SMILES" not in frame.columns:
            continue
        for smiles in frame["SMILES"].dropna().astype(str):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                known.add(Chem.MolToSmiles(molecule, canonical=True))
    return known


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aids, search_counts = search_assay_ids()
    summaries = fetch_assay_summaries(aids)
    manifest_path = DATA_DIR / "pubchem_assay_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "search_terms": search_counts,
                "unique_aids_retrieved": len(aids),
                "candidate_assays": summaries,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Candidate VACV-related concentration assays: {len(summaries)}", flush=True)

    records = []
    assay_status = []
    for index, assay in enumerate(summaries, start=1):
        try:
            assay_records, status = parse_assay_csv(assay["aid"], assay)
            records.extend(assay_records)
            assay_status.append(status)
            print(
                f"AID {assay['aid']} ({index}/{len(summaries)}): "
                f"{len(assay_records)} concentration records",
                flush=True,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            pd.errors.ParserError,
        ) as error:
            assay_status.append(
                {
                    "aid": assay["aid"],
                    "assayname": assay["assayname"],
                    "status": "request_or_parse_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"AID {assay['aid']} skipped: {error}", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    raw_path = DATA_DIR / "pubchem_assay_measurements_raw.csv"
    raw_frame = pd.DataFrame(records)
    raw_frame.to_csv(raw_path, index=False)
    paired = pair_measurements(records)
    known = load_known_structures()
    if not paired.empty:
        paired["already_in_known_data"] = paired["SMILES"].isin(known)
        paired["new_vs_known_data"] = ~paired["already_in_known_data"]
        output_columns = [
            "SMILES",
            "IC50uM",
            "CC50uM",
            "SI",
            "DOI",
            "source",
            "PubChem_CID",
            "IC50_AID",
            "CC50_AID",
            "assay_context",
            "pairing_method",
            "already_in_known_data",
            "new_vs_known_data",
        ]
        paired = paired[output_columns].sort_values(
            ["new_vs_known_data", "PubChem_CID"], ascending=[False, True]
        )
    output_path = DATA_DIR / "pubchem_collected.csv"
    paired.to_csv(output_path, index=False)

    novel = (
        paired.loc[paired["new_vs_known_data"]]
        if not paired.empty
        else paired
    )
    summary = {
        "search_terms": search_counts,
        "unique_aids_retrieved": len(aids),
        "candidate_assays": len(summaries),
        "assays_downloaded": sum(item.get("status") == "ok" for item in assay_status),
        "usable_ic50_cc50_measurements": len(records),
        "paired_rows": len(paired),
        "unique_paired_canonical_smiles": int(paired["SMILES"].nunique()) if not paired.empty else 0,
        "novel_pair_rows_vs_known_data": len(novel),
        "unique_novel_structures_vs_known_data": int(novel["SMILES"].nunique()) if not novel.empty else 0,
        "pairs_already_in_train_or_collections": int(paired["already_in_known_data"].sum()) if not paired.empty else 0,
        "assay_status": assay_status,
        "outputs": {
            "paired_csv": str(output_path),
            "raw_measurements_csv": str(raw_path),
            "assay_manifest": str(manifest_path),
        },
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS_DIR / "pubchem_collection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "PubChem collection summary: "
        + json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"assay_status", "outputs"}
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved paired measurements: {output_path}", flush=True)


if __name__ == "__main__":
    main()
