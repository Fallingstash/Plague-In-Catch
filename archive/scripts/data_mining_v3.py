import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
USER_AGENT = "PlagueIn-data-mining-v3/1.0"
TIMEOUT = 25
REQUEST_DELAY = 0.18
ACTIVITY_FIELDS = (
    "molecule_chembl_id,standard_type,standard_value,standard_units,"
    "standard_relation,assay_chembl_id,document_chembl_id"
)
RESULT_COLUMNS = [
    "SMILES",
    "IC50",
    "CC50",
    "SI",
    "assay_id",
    "DOI",
    "source",
    "compound_id",
    "virus",
    "IC50_relation",
    "CC50_relation",
]
ORTHOPOX_QUERIES = {
    "Vaccinia": "vaccinia",
    "Cowpox": "cowpox",
    "Monkeypox": "monkeypox",
    "Variola": "variola",
}
UNIT_TO_UM = {
    "m": 1_000_000.0,
    "mol/l": 1_000_000.0,
    "mm": 1_000.0,
    "mmol/l": 1_000.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "umol/l": 1.0,
    "nm": 0.001,
    "nmol/l": 0.001,
    "pm": 0.000001,
    "pmol/l": 0.000001,
}


def get_json(url, attempts=3, timeout=TIMEOUT):
    last_error = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    raise last_error


def get_text(url):
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/xml,text/xml", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pmcid = urllib.parse.urlparse(url).path.strip("/").split("/")[-2]
        fallback_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            + urllib.parse.urlencode({"db": "pmc", "id": pmcid})
        )
        fallback = urllib.request.Request(
            fallback_url,
            headers={"Accept": "application/xml,text/xml", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(fallback, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def normalized_unit(unit):
    value = str(unit or "").strip().lower().replace("μ", "µ")
    value = value.replace("µm", "um")
    return UNIT_TO_UM.get(value)


def to_micromolar(value, unit):
    factor = normalized_unit(unit)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if factor is None or not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric * factor


def find_targets(query):
    targets = {}
    next_url = CHEMBL_API + "target/search.json?" + urllib.parse.urlencode(
        {"q": query, "limit": 1000}
    )
    seen = set()
    while next_url:
        if next_url in seen:
            raise RuntimeError(f"ChEMBL target pagination loop: {query}")
        seen.add(next_url)
        payload = get_json(next_url)
        for target in payload.get("targets", []):
            name = str(target.get("pref_name") or "")
            organism = str(target.get("organism") or "")
            if query.casefold() not in f"{name} {organism}".casefold():
                continue
            chembl_id = target.get("target_chembl_id")
            if chembl_id:
                targets[chembl_id] = {
                    "target_chembl_id": chembl_id,
                    "pref_name": name,
                    "organism": organism,
                }
        next_url = payload.get("page_meta", {}).get("next")
        if next_url:
            next_url = urllib.parse.urljoin("https://www.ebi.ac.uk", next_url)
        time.sleep(REQUEST_DELAY)
    return list(targets.values())


def fetch_activities(target_id):
    fields = urllib.parse.urlencode(
        {
            "target_chembl_id": target_id,
            "limit": 1000,
            "only": ACTIVITY_FIELDS,
        }
    )
    next_url = CHEMBL_API + "activity.json?" + fields
    activities = []
    seen = set()
    while next_url:
        if next_url in seen:
            raise RuntimeError(f"ChEMBL activity pagination loop: {target_id}")
        seen.add(next_url)
        payload = get_json(next_url)
        activities.extend(payload.get("activities", []))
        next_url = payload.get("page_meta", {}).get("next")
        if next_url:
            next_url = urllib.parse.urljoin("https://www.ebi.ac.uk", next_url)
        time.sleep(REQUEST_DELAY)
    return activities


def fetch_molecule_smiles(molecule_id):
    url = CHEMBL_API + f"molecule/{urllib.parse.quote(molecule_id, safe='')}.json"
    payload = get_json(url, attempts=1, timeout=8)
    structures = payload.get("molecule_structures") or {}
    return structures.get("canonical_smiles") or structures.get("canonicalsmiles")


def fetch_document_doi(document_id):
    if not document_id:
        return ""
    url = CHEMBL_API + f"document/{urllib.parse.quote(document_id, safe='')}.json"
    try:
        return str(get_json(url).get("doi") or "")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"  ChEMBL DOI lookup failed for {document_id}: {error}", flush=True)
        return ""


def pair_chembl_activities(activities, virus):
    grouped = {}
    for activity in activities:
        activity_type = str(activity.get("standard_type") or "").strip().upper()
        if activity_type not in {"IC50", "CC50"}:
            continue
        relation = str(activity.get("standard_relation") or "=").strip()
        if relation not in {"=", ">", "<", ">=", "<="}:
            continue
        assay_id = activity.get("assay_chembl_id")
        molecule_id = activity.get("molecule_chembl_id")
        value = to_micromolar(
            activity.get("standard_value"), activity.get("standard_units")
        )
        if not assay_id or not molecule_id or value is None:
            continue
        key = (str(molecule_id), str(assay_id))
        assay_record = grouped.setdefault(
            key,
            {"IC50": [], "CC50": [], "documents": set()},
        )
        assay_record[activity_type].append((value, relation))
        if activity.get("document_chembl_id"):
            assay_record["documents"].add(activity["document_chembl_id"])

    rows = []
    smiles_cache = {}
    document_cache = {}
    for (molecule_id, assay_id), measurements in grouped.items():
        if not measurements["IC50"] or not measurements["CC50"]:
            continue
        if molecule_id not in smiles_cache:
            try:
                smiles_cache[molecule_id] = fetch_molecule_smiles(molecule_id)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
                smiles_cache[molecule_id] = None
                print(f"  ChEMBL structure lookup failed for {molecule_id}: {error}", flush=True)
            time.sleep(REQUEST_DELAY)
        smiles = smiles_cache[molecule_id]
        canonical = canonicalize(smiles)
        if canonical is None:
            continue
        ic50_items = measurements["IC50"]
        cc50_items = measurements["CC50"]
        ic50 = float(np.median([value for value, _ in ic50_items]))
        cc50 = float(np.median([value for value, _ in cc50_items]))
        if not ic50 > 0 or not cc50 > 0:
            continue
        ic50_relations = {relation for _, relation in ic50_items}
        cc50_relations = {relation for _, relation in cc50_items}
        ic50_relation = next(iter(ic50_relations)) if len(ic50_relations) == 1 else "mixed"
        cc50_relation = next(iter(cc50_relations)) if len(cc50_relations) == 1 else "mixed"
        exact_pair = ic50_relation == "=" and cc50_relation == "="
        doi = ""
        if measurements["documents"]:
            document_id = sorted(measurements["documents"])[0]
            if document_id not in document_cache:
                document_cache[document_id] = fetch_document_doi(document_id)
                time.sleep(REQUEST_DELAY)
            doi = document_cache[document_id]
        rows.append(
            {
                "SMILES": canonical,
                "IC50": ic50,
                "CC50": cc50,
                "SI": cc50 / ic50 if exact_pair else np.nan,
                "assay_id": assay_id,
                "DOI": doi,
                "source": "ChEMBL_v3",
                "compound_id": molecule_id,
                "virus": virus,
                "IC50_relation": ic50_relation,
                "CC50_relation": cc50_relation,
            }
        )
    return rows


def run_chembl_campaign(virus):
    query = ORTHOPOX_QUERIES[virus]
    targets = find_targets(query)
    print(f"ChEMBL {virus}: found {len(targets)} targets", flush=True)
    all_rows = []
    failures = []
    target_counts = []
    for target in targets:
        target_id = target["target_chembl_id"]
        try:
            activities = fetch_activities(target_id)
            paired = pair_chembl_activities(activities, virus)
            all_rows.extend(paired)
            target_counts.append(
                {
                    **target,
                    "activity_count": len(activities),
                    "paired_count": len(paired),
                }
            )
            print(
                f"  {target_id}: {len(activities)} activities, "
                f"{len(paired)} same-assay IC50/CC50 pairs",
                flush=True,
            )
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            RuntimeError,
        ) as error:
            failures.append(
                {
                    "target_chembl_id": target_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  {target_id}: failed: {error}", flush=True)

    frame = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(["SMILES", "assay_id"], keep="first")
        valid = frame["SMILES"].map(canonicalize).notna()
        frame = frame.loc[valid].copy()
    return frame, {
        "targets": target_counts,
        "target_search_count": len(targets),
        "failures": failures,
    }


def parse_concentration(text):
    normalized = str(text).replace(",", "")
    match = re.search(
        r"(?P<relation>>=|<=|>|<|=)?\s*(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        normalized,
    )
    if not match:
        return None, ""
    value = float(match.group("value"))
    return value, match.group("relation") or "="


def element_text(element):
    return " ".join("".join(element.itertext()).split())


def pubchem_smiles(name, cache):
    key = name.strip()
    if key in cache:
        return cache[key]
    encoded = urllib.parse.quote(key, safe="")
    url = PUBCHEM + encoded + "/property/CanonicalSMILES,IsomericSMILES/JSON"
    try:
        payload = get_json(url, attempts=1, timeout=8)
        records = payload.get("PropertyTable", {}).get("Properties", [])
        smiles = records[0].get("SMILES") if len(records) == 1 else None
        cache[key] = smiles
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        cache[key] = None
    time.sleep(REQUEST_DELAY)
    return cache[key]


def get_unit(text):
    lowered = text.lower().replace("μ", "µ")
    for token, unit in (
        ("µm", "uM"),
        ("um", "uM"),
        ("microm", "uM"),
        ("nm", "nM"),
        ("nanom", "nM"),
        ("mm", "mM"),
        ("millim", "mM"),
    ):
        if token in lowered:
            return unit
    return None


def table_rows(table):
    rows = []
    for tr in table.iter():
        if tr.tag.rsplit("}", 1)[-1].lower() != "tr":
            continue
        cells = [
            element_text(cell)
            for cell in list(tr)
            if cell.tag.rsplit("}", 1)[-1].lower() in {"td", "th"}
        ]
        if cells:
            rows.append(cells)
    return rows


def cell_line_from_text(text):
    for pattern in (r"\bVero(?:\s*\d+)?\b", r"\bHFF\b", r"\bA549\b", r"\bHeLa\b", r"\bHEL\b", r"\bBSC-1\b"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def parse_europepmc_article(result, pubchem_cache):
    pmcid = result.get("pmcid")
    if not pmcid:
        return [], {"pmcid": "", "status": "no_pmcid"}
    identifier = urllib.parse.quote(str(pmcid), safe="")
    url = EUROPE_PMC + identifier + "/fullTextXML"
    try:
        root = ET.fromstring(get_text(url))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ET.ParseError,
        ValueError,
    ) as error:
        return [], {
            "pmcid": pmcid,
            "status": "fulltext_error",
            "error": f"{type(error).__name__}: {error}",
        }

    title = ""
    title_element = root.find(".//article-title")
    if title_element is not None:
        title = element_text(title_element)
    doi = ""
    for article_id in root.findall(".//article-id"):
        if article_id.get("pub-id-type") == "doi":
            doi = element_text(article_id)
            break
    article_text = element_text(root)
    cell_line = cell_line_from_text(article_text)
    rows_out = []
    unresolved_names = set()
    tables_seen = 0

    for table_index, table in enumerate(root.findall(".//table-wrap"), start=1):
        rows = table_rows(table)
        if not rows:
            continue
        tables_seen += 1
        headers = None
        ic_index = cc_index = None
        for row_index, cells in enumerate(rows):
            lowered = [cell.lower().replace(" ", "") for cell in cells]
            candidates_ic = [
                i for i, cell in enumerate(lowered)
                if "ic50" in cell or "ec50" in cell
            ]
            candidates_cc = [i for i, cell in enumerate(lowered) if "cc50" in cell]
            if candidates_ic and candidates_cc:
                headers = cells
                ic_index = candidates_ic[0]
                cc_index = candidates_cc[0]
                data_start = row_index + 1
                break
        if headers is None:
            continue
        unit = get_unit(" ".join(headers))
        if unit is None:
            continue
        factor = normalized_unit(unit)
        assay_id = f"{pmcid}_T{table_index}"
        for cells in rows[data_start:]:
            if max(ic_index, cc_index) >= len(cells):
                continue
            compound_name = cells[0].strip()
            if not compound_name or re.search(r"\b(?:IC50|CC50|EC50)\b", compound_name, re.I):
                continue
            ic_raw, ic_relation = parse_concentration(cells[ic_index])
            cc_raw, cc_relation = parse_concentration(cells[cc_index])
            if ic_raw is None or cc_raw is None:
                continue
            ic50 = ic_raw * factor
            cc50 = cc_raw * factor
            smiles = pubchem_smiles(compound_name, pubchem_cache)
            canonical = canonicalize(smiles)
            if canonical is None:
                unresolved_names.add(compound_name)
                continue
            exact = ic_relation == "=" and cc_relation == "="
            rows_out.append(
                {
                    "SMILES": canonical,
                    "IC50": ic50,
                    "CC50": cc50,
                    "SI": cc50 / ic50 if exact and ic50 > 0 else np.nan,
                    "assay_id": assay_id,
                    "DOI": doi or result.get("doi", ""),
                    "source": "EuropePMC",
                    "compound_id": compound_name,
                    "virus": "Vaccinia virus",
                    "IC50_relation": ic_relation,
                    "CC50_relation": cc_relation,
                }
            )
        time.sleep(REQUEST_DELAY)

    return rows_out, {
        "pmcid": pmcid,
        "title": title,
        "doi": doi or result.get("doi", ""),
        "status": "ok",
        "tables_seen": tables_seen,
        "extracted_rows": len(rows_out),
        "unresolved_compound_names": sorted(unresolved_names),
    }


def run_europepmc():
    queries = (
        "(Vaccinia OR VACV) AND IC50 AND CC50 AND OPEN_ACCESS:Y",
        "TITLE:(Vaccinia OR VACV) AND IC50 AND CC50 AND OPEN_ACCESS:Y",
    )
    payload = None
    query = queries[0]
    search_failures = []
    for candidate_query in queries:
        url = EUROPE_PMC + "search?" + urllib.parse.urlencode(
            {"query": candidate_query, "format": "json", "pageSize": 100, "page": 1}
        )
        try:
            payload = get_json(url)
            query = candidate_query
            break
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
        ) as error:
            search_failures.append(
                {
                    "query": candidate_query,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if payload is None:
        raise RuntimeError(f"EuropePMC search failed: {search_failures}")
    results = payload.get("resultList", {}).get("result", [])
    print(
        f"EuropePMC: {payload.get('hitCount', 0)} matches; "
        f"checking {len(results)} open-access results",
        flush=True,
    )
    rows = []
    statuses = []
    pubchem_cache = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(parse_europepmc_article, result, pubchem_cache): result
            for result in results
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = futures[future]
            article_rows, status = future.result()
            rows.extend(article_rows)
            statuses.append(status)
            if index % 10 == 0 or article_rows:
                print(
                    f"  article {index}/{len(results)}: "
                    f"{result.get('pmcid') or 'no PMCID'}, "
                    f"{len(article_rows)} structure-resolved rows",
                    flush=True,
                )
            if index % 10 == 0:
                save_and_stats(
                    pd.DataFrame(rows, columns=RESULT_COLUMNS),
                    DATA_DIR / "europepmc_v3.csv",
                )
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(["SMILES", "assay_id"], keep="first")
        frame = frame.loc[frame["SMILES"].map(canonicalize).notna()].copy()
    return frame, {
        "query": query,
        "search_failures": search_failures,
        "hit_count": payload.get("hitCount", 0),
        "results_checked": len(results),
        "articles": statuses,
    }


def save_and_stats(frame, path):
    frame = pd.DataFrame(frame, columns=RESULT_COLUMNS)
    if not frame.empty:
        frame["canonical"] = frame["SMILES"].map(canonicalize)
        frame = frame.loc[frame["canonical"].notna()].copy()
        frame["SMILES"] = frame["canonical"]
        frame = frame.drop(columns=["canonical"])
        frame = frame.drop_duplicates(["SMILES", "assay_id"], keep="first")
    frame.to_csv(path, index=False)
    si_count = int(frame["SI"].notna().sum()) if not frame.empty else 0
    return {
        "file": str(path),
        "rows": len(frame),
        "unique_smiles": int(frame["SMILES"].nunique()) if not frame.empty else 0,
        "rows_with_si": si_count,
        "invalid_smiles_removed": 0,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")

    report = {}
    try:
        vacv_frame, vacv_meta = run_chembl_campaign("Vaccinia")
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        RuntimeError,
    ) as error:
        vacv_frame = pd.DataFrame(columns=RESULT_COLUMNS)
        vacv_meta = {
            "targets": [],
            "target_search_count": 0,
            "failures": [{"stage": "vaccinia", "error": f"{type(error).__name__}: {error}"}],
        }
        print(f"ChEMBL Vaccinia stage failed: {error}", flush=True)
    report["chembl_vaccinia"] = vacv_meta
    report["chembl_vaccinia_stats"] = save_and_stats(
        vacv_frame, DATA_DIR / "chembl_expanded_v3.csv"
    )
    print("ChEMBL Vaccinia pairs:", report["chembl_vaccinia_stats"], flush=True)

    try:
        europe_frame, europe_meta = run_europepmc()
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
    ) as error:
        europe_frame = pd.DataFrame(columns=RESULT_COLUMNS)
        europe_meta = {
            "query": "(Vaccinia OR VACV) AND (IC50) AND (CC50) AND OPEN_ACCESS:Y",
            "hit_count": 0,
            "results_checked": 0,
            "articles": [],
            "search_error": f"{type(error).__name__}: {error}",
        }
        print(f"EuropePMC stage failed: {error}", flush=True)
    report["europepmc"] = europe_meta
    report["europepmc_stats"] = save_and_stats(
        europe_frame, DATA_DIR / "europepmc_v3.csv"
    )
    print("EuropePMC rows:", report["europepmc_stats"], flush=True)

    other_frames = []
    other_meta = {}
    for virus in ("Cowpox", "Monkeypox", "Variola"):
        try:
            frame, metadata = run_chembl_campaign(virus)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            RuntimeError,
        ) as error:
            frame = pd.DataFrame(columns=RESULT_COLUMNS)
            metadata = {
                "targets": [],
                "target_search_count": 0,
                "failures": [{"stage": virus, "error": f"{type(error).__name__}: {error}"}],
            }
            print(f"ChEMBL {virus} stage failed: {error}", flush=True)
        other_frames.append(frame)
        other_meta[virus] = metadata
        print(f"ChEMBL {virus} stats:", {
            "rows": len(frame),
            "with_si": int(frame["SI"].notna().sum()) if not frame.empty else 0,
        }, flush=True)
    other_frame = (
        pd.concat(other_frames, ignore_index=True)
        if other_frames
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    report["other_orthopox"] = other_meta
    report["other_orthopox_stats"] = save_and_stats(
        other_frame, DATA_DIR / "other_orthopox_v3.csv"
    )

    report_path = RESULTS_DIR / "data_mining_v3.json"
    report["outputs"] = {
        "chembl_vaccinia": report["chembl_vaccinia_stats"],
        "europepmc": report["europepmc_stats"],
        "other_orthopox": report["other_orthopox_stats"],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Other orthopox pairs:", report["other_orthopox_stats"], flush=True)
    print("Report:", report_path, flush=True)


if __name__ == "__main__":
    main()
