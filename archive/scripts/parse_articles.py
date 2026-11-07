import csv
from html.parser import HTMLParser
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WARNINGS_PATH = ROOT / "warnings.log"
USER_AGENT = "PlagueIn-article-data-collector/1.0"
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3
OUTPUT_COLUMNS = ["SMILES", "IC50uM", "CC50uM", "SI", "DOI", "source"]
SOURCE_LABELS = {
    "figshare": "Figshare 30655366 (VACV)",
    "pmc6335999": "PMC6335999",
    "pmc7651861": "PMC7651861",
    "sciencedirect": "ScienceDirect S0166354210007424",
}
SOURCE_COUNT_LABELS = {
    "figshare": SOURCE_LABELS["figshare"],
    "pmc6335999": SOURCE_LABELS["pmc6335999"],
    "pmc7651861": SOURCE_LABELS["pmc7651861"],
    "sciencedirect": SOURCE_LABELS["sciencedirect"],
}
PUBCHEM_NAMES = [
    "Cidofovir",
    "ST-246",
    "Nigericin",
    "Monensin",
    "Concanamycin A",
    "Concanamycin B",
    "Concanamycin C",
]


class TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.table_depth = 0
        self.current_table = None
        self.current_row = None
        self.current_cell = None
        self.tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.table_depth == 0:
                self.current_table = []
            self.table_depth += 1
        elif self.table_depth and tag == "tr" and self.table_depth == 1:
            self.current_row = []
        elif self.table_depth and tag in {"td", "th"} and self.table_depth == 1:
            self.current_cell = []

    def handle_data(self, data):
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.current_cell is not None:
            self.current_row.append(" ".join(" ".join(self.current_cell).split()))
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if any(self.current_row):
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0 and self.current_table:
                self.tables.append(self.current_table)
                self.current_table = None


def fetch_bytes(url, accept="*/*"):
    last_error = None
    for attempt in range(MAX_RETRIES):
        request = urllib.request.Request(
            url,
            headers={"Accept": accept, "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2**attempt)
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"Request failed: {url}") from last_error


def warn(message):
    with WARNINGS_PATH.open("a", encoding="utf-8") as log:
        log.write(message.rstrip() + "\n")


def parse_measurement(value):
    text = str(value or "").strip().replace(",", "")
    if not text or text.casefold() in {"na", "n/a", "nd", "not determined", "-", "—"}:
        return None, False
    censored = text.startswith((">", "<", "≥", "≤"))
    number_text = re.sub(r"^[><≥≤]\s*", "", text)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", number_text)
    if not match:
        return None, censored
    try:
        number = float(match.group(0))
    except ValueError:
        return None, censored
    if not math.isfinite(number) or number <= 0:
        return None, censored
    return number, censored


def canonicalize(smiles, source):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as error:
        warn(f"{source}: RDKit parse error for SMILES {smiles!r}: {error}")
        return None
    if molecule is None:
        warn(f"{source}: invalid SMILES {smiles!r}")
        return None
    try:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception as error:
        warn(f"{source}: RDKit canonicalization error for SMILES {smiles!r}: {error}")
        return None


def pubchem_smiles(name, cache):
    if name in cache:
        return cache[name]
    quoted_name = urllib.parse.quote(name, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quoted_name}/property/CanonicalSMILES,IsomericSMILES/JSON"
    )
    try:
        payload = json.loads(fetch_bytes(url, accept="application/json"))
        properties = payload["PropertyTable"]["Properties"][0]
        smiles = (
            properties.get("ConnectivitySMILES")
            or properties.get("CanonicalSMILES")
            or properties.get("SMILES")
            or properties.get("IsomericSMILES")
        )
        cache[name] = smiles
    except (urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError) as error:
        cache[name] = None
        warn(f"PubChem lookup failed for {name!r}: {error}")
    return cache[name]


def save_source_records(records, filename):
    path = DATA_DIR / filename
    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(path, index=False)
    return path


def find_column(headers, required_terms, excluded_terms=()):
    for index, header in enumerate(headers):
        normalized = re.sub(r"[^a-z0-9]+", " ", str(header).lower())
        if all(term in normalized for term in required_terms) and not any(
            term in normalized for term in excluded_terms
        ):
            return index
    return None


def fetch_figshare():
    article = json.loads(
        fetch_bytes(
            "https://api.figshare.com/v2/articles/30655366",
            accept="application/json",
        )
    )
    doi = article.get("doi")
    file_record = next(
        (
            file_record
            for file_record in article.get("files", [])
            if file_record.get("name", "").lower().endswith(".csv")
        ),
        None,
    )
    if not file_record:
        raise ValueError("Figshare article has no CSV supplementary file")
    content = fetch_bytes(file_record["download_url"], accept="text/csv")
    decoded = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UnicodeError("Could not decode Figshare supplementary CSV")

    reader = csv.reader(decoded.splitlines())
    headers = next(reader, [])
    compound_column = find_column(headers, ["compound", "id"])
    smiles_column = find_column(headers, ["smiles"])
    cc50_column = find_column(headers, ["cc50"])
    ec50_column = find_column(headers, ["ec50", "vacv"])
    if None in (compound_column, smiles_column, cc50_column, ec50_column):
        raise ValueError(f"Unexpected Figshare CSV columns: {headers!r}")

    records = []
    skipped_censored_ec50 = 0
    censored_cc50_count = 0
    for values in reader:
        if max(compound_column, smiles_column, cc50_column, ec50_column) >= len(values):
            continue
        compound_id = values[compound_column].strip()
        smiles = values[smiles_column].strip()
        cc50_raw = values[cc50_column].strip()
        ec50_raw = values[ec50_column].strip()
        cc50_value, cc50_censored = parse_measurement(cc50_raw)
        ec50_value, ec50_censored = parse_measurement(ec50_raw)
        if not smiles or cc50_value is None or ec50_value is None:
            continue
        if ec50_censored and not cc50_censored:
            skipped_censored_ec50 += 1
            continue

        # This supplementary table reports CC50 and VACV EC50 in mM.
        cc50_um = cc50_value * 1000.0
        ec50_um = ec50_value * 1000.0
        censored = cc50_censored and cc50_raw.startswith(">")
        si = 10000.0 if censored else cc50_um / ec50_um
        if censored:
            censored_cc50_count += 1
        records.append(
            {
                "SMILES": smiles,
                "IC50uM": ec50_um,
                "CC50uM": cc50_um,
                "SI": si,
                "DOI": doi,
                "source": SOURCE_LABELS["figshare"],
                "_censored": censored,
                "_compound": compound_id,
            }
        )
    return records, {
        "status": "ok",
        "doi": doi,
        "supplementary_file": file_record.get("name"),
        "rows_parsed": len(records),
        "censored_cc50_rows": censored_cc50_count,
        "censored_ec50_rows_skipped": skipped_censored_ec50,
    }


def eutils_pmc_xml(pmcid):
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        + urllib.parse.urlencode({"db": "pmc", "id": pmcid, "retmode": "xml"})
    )
    content = fetch_bytes(url, accept="application/xml,text/xml")
    return ET.fromstring(content)


def xml_article_doi(root):
    for node in root.findall(".//article-id"):
        if node.attrib.get("pub-id-type", "").casefold() == "doi":
            return (node.text or "").strip() or None
    return None


def table_rows(root):
    extracted = []
    for table in root.findall(".//table-wrap"):
        rows = []
        for row in table.findall(".//tr"):
            cells = [
                " ".join("".join(cell.itertext()).split())
                for cell in list(row.findall("./th")) + list(row.findall("./td"))
            ]
            if cells:
                rows.append(cells)
        if rows:
            extracted.append(rows)
    return extracted


def html_table_rows(content):
    parser = TableHTMLParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    return parser.tables


def parse_pmc_source(pmcid, allowed_compounds, pubchem_cache):
    root = eutils_pmc_xml(pmcid)
    doi = xml_article_doi(root)
    records = []
    parsed_table_rows = 0
    tables = table_rows(root)
    fallback_note = None
    if not tables:
        table_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/table/tab1/"
        try:
            table_content = fetch_bytes(table_url, accept="text/html")
            tables = html_table_rows(table_content)
            if not tables and b"recaptcha" in table_content.lower():
                fallback_note = "Table page was blocked by the site's browser challenge."
        except Exception as error:
            fallback_note = f"Table page fetch failed: {error}"
    for rows in tables:
        headers = [header.casefold() for header in rows[0]]
        compound_column = find_column(headers, ["compound"])
        cc50_column = find_column(headers, ["cc50"])
        ic50_column = find_column(headers, ["ic50"])
        si_column = find_column(headers, ["si"])
        if None in (compound_column, cc50_column, ic50_column):
            continue
        for values in rows[1:]:
            if max(compound_column, cc50_column, ic50_column) >= len(values):
                continue
            compound = values[compound_column].strip()
            compound_key = re.sub(r"\s+", " ", compound).casefold()
            matched_name = next(
                (name for name in allowed_compounds if name.casefold() == compound_key),
                None,
            )
            if not matched_name:
                continue
            cc50, cc50_censored = parse_measurement(values[cc50_column])
            ic50, ic50_censored = parse_measurement(values[ic50_column])
            if cc50 is None or ic50 is None or cc50_censored or ic50_censored:
                continue
            smiles = pubchem_smiles(matched_name, pubchem_cache)
            if not smiles:
                continue
            supplied_si = None
            if si_column is not None and si_column < len(values):
                supplied_si, si_censored = parse_measurement(values[si_column])
                if si_censored:
                    supplied_si = None
            records.append(
                {
                    "SMILES": smiles,
                    "IC50uM": ic50,
                    "CC50uM": cc50,
                    "SI": supplied_si if supplied_si is not None else cc50 / ic50,
                    "DOI": doi or pmcid,
                    "source": SOURCE_LABELS[
                        "pmc6335999" if pmcid == "PMC6335999" else "pmc7651861"
                    ],
                    "_censored": False,
                    "_compound": matched_name,
                }
            )
            parsed_table_rows += 1
    return records, {
        "status": "ok" if parsed_table_rows else "metadata_only_no_numeric_table",
        "doi": doi,
        "table_rows_with_resolvable_smiles": parsed_table_rows,
        "allowed_compounds": allowed_compounds,
        "note": fallback_note or (
            "No numeric table was available in the returned full-text XML/table page."
            if not parsed_table_rows
            else None
        ),
    }


def fetch_sciencedirect_xml():
    url = (
        "https://api.elsevier.com/content/article/PII:S0166354210007424"
        "?httpAccept=text/xml"
    )
    content = fetch_bytes(url, accept="application/xml,text/xml")
    return ET.fromstring(content)


def parse_sciencedirect(pubchem_cache):
    try:
        root = fetch_sciencedirect_xml()
    except Exception as xml_error:
        html_url = "https://www.sciencedirect.com/science/article/pii/S0166354210007424"
        try:
            root = None
            html_content = fetch_bytes(html_url, accept="text/html")
            tables = html_table_rows(html_content)
        except Exception as html_error:
            raise RuntimeError(
                f"Elsevier XML unavailable ({xml_error}); ScienceDirect HTML unavailable "
                f"({html_error})"
            ) from html_error
    else:
        tables = []
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] != "table":
                continue
            rows = []
            for row in node.iter():
                if row.tag.rsplit("}", 1)[-1] not in {"row", "tr"}:
                    continue
                cells = [
                    " ".join("".join(cell.itertext()).split())
                    for cell in row
                    if cell.tag.rsplit("}", 1)[-1] in {"entry", "td", "th"}
                ]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
    doi = None
    if root is not None:
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "doi" and element.text:
                doi = element.text.strip()
                break
    records = []
    table_count = 0
    for rows in tables:
        if len(rows) < 2:
            continue
        headers = [header.casefold() for header in rows[0]]
        compound_column = find_column(headers, ["compound"])
        cc50_column = find_column(headers, ["cc50"])
        ic50_column = find_column(headers, ["ic50"])
        si_column = find_column(headers, ["si"])
        if None in (compound_column, cc50_column, ic50_column):
            continue
        table_count += 1
        for values in rows[1:]:
            if max(compound_column, cc50_column, ic50_column) >= len(values):
                continue
            compound = values[compound_column].strip()
            cc50, cc50_censored = parse_measurement(values[cc50_column])
            ic50, ic50_censored = parse_measurement(values[ic50_column])
            if cc50 is None or ic50 is None or cc50_censored or ic50_censored:
                continue
            if compound.casefold() in {
                "nigericin",
                "concanamycin a",
                "concanamycin b",
                "concanamycin c",
            }:
                ic50 /= 1000.0
            smiles = pubchem_smiles(compound, pubchem_cache)
            if not smiles:
                continue
            supplied_si = None
            if si_column is not None and si_column < len(values):
                supplied_si, si_censored = parse_measurement(values[si_column])
                if si_censored:
                    supplied_si = None
            records.append(
                {
                    "SMILES": smiles,
                    "IC50uM": ic50,
                    "CC50uM": cc50,
                    "SI": supplied_si if supplied_si is not None else cc50 / ic50,
                    "DOI": doi or "10.1016/j.antiviral.2010.10.001",
                    "source": SOURCE_LABELS["sciencedirect"],
                    "_censored": False,
                    "_compound": compound,
                }
            )
    return records, {
        "status": "ok" if table_count else "metadata_only_no_numeric_table",
        "doi": doi,
        "numeric_tables_found": table_count,
        "note": "The Elsevier/ScienceDirect response did not expose a numeric table."
        if not table_count
        else None,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_status = {}
    source_records = {}
    pubchem_cache = {}

    try:
        records, source_status["figshare"] = fetch_figshare()
    except Exception as error:
        records = []
        source_status["figshare"] = {"status": "failed", "error": str(error)}
    source_records["figshare"] = records
    save_source_records(records, "articles_figshare_raw.csv")
    print(f"Figshare: {len(records)} candidate rows")

    pmc_sources = (
        (
            "PMC6335999",
            ["Cidofovir"],
            "articles_pmc6335999_raw.csv",
        ),
        (
            "PMC7651861",
            ["Cidofovir", "ST-246"],
            "articles_pmc7651861_raw.csv",
        ),
    )
    for pmcid, compounds, filename in pmc_sources:
        key = "pmc6335999" if pmcid == "PMC6335999" else "pmc7651861"
        try:
            records, source_status[key] = parse_pmc_source(
                pmcid, compounds, pubchem_cache
            )
        except Exception as error:
            records = []
            source_status[key] = {"status": "failed", "error": str(error)}
        source_records[key] = records
        save_source_records(records, filename)
        print(f"{pmcid}: {len(records)} candidate rows ({source_status[key]['status']})")

    for name in PUBCHEM_NAMES:
        pubchem_smiles(name, pubchem_cache)
    source_status["pubchem_names"] = {
        name: "resolved" if pubchem_cache.get(name) else "not_found"
        for name in PUBCHEM_NAMES
    }

    try:
        records, source_status["sciencedirect"] = parse_sciencedirect(pubchem_cache)
    except Exception as error:
        records = []
        source_status["sciencedirect"] = {"status": "failed", "error": str(error)}
    source_records["sciencedirect"] = records
    save_source_records(records, "articles_sciencedirect_raw.csv")
    print(
        f"ScienceDirect: {len(records)} candidate rows "
        f"({source_status['sciencedirect']['status']})"
    )

    all_records = [
        row
        for key in ("figshare", "pmc6335999", "pmc7651861", "sciencedirect")
        for row in source_records.get(key, [])
    ]
    raw_path = DATA_DIR / "collectedfromarticles_raw.csv"
    save_source_records(all_records, raw_path.name)

    train = pd.read_csv(ROOT / "train.csv")
    train_canonical = {
        canonical
        for canonical in (canonicalize(smiles, "train.csv") for smiles in train["SMILES"])
        if canonical is not None
    }
    valid_records = []
    invalid_smiles = 0
    duplicates_with_train = 0
    canonical_seen = set()
    duplicates_within_articles = 0
    for row in all_records:
        canonical = canonicalize(row.get("SMILES"), row.get("source", "article"))
        if canonical is None:
            invalid_smiles += 1
            continue
        if canonical in train_canonical:
            duplicates_with_train += 1
            continue
        if canonical in canonical_seen:
            duplicates_within_articles += 1
            continue
        canonical_seen.add(canonical)
        ic50, _ = parse_measurement(row.get("IC50uM"))
        cc50, _ = parse_measurement(row.get("CC50uM"))
        if ic50 is None or cc50 is None:
            continue
        si = row.get("SI")
        try:
            si = float(si)
        except (TypeError, ValueError):
            si = cc50 / ic50
        if not math.isfinite(si) or si <= 0:
            si = cc50 / ic50
        valid_records.append(
            {
                "SMILES": canonical,
                "IC50uM": ic50,
                "CC50uM": cc50,
                "SI": si,
                "DOI": row.get("DOI"),
                "source": row.get("source"),
                "_censored": bool(row.get("_censored", False)),
            }
        )

    collected = pd.DataFrame(valid_records)
    if collected.empty:
        collected = pd.DataFrame(columns=OUTPUT_COLUMNS + ["_censored"])
    source_counts = {
        label: int((collected["source"] == label).sum())
        for label in SOURCE_COUNT_LABELS.values()
    }
    censored_count = int(collected["_censored"].sum()) if len(collected) else 0
    final = collected[OUTPUT_COLUMNS]
    final_path = DATA_DIR / "collectedfromarticles.csv"
    final.to_csv(final_path, index=False)

    si_stats = (
        {
            "min": float(final["SI"].min()),
            "max": float(final["SI"].max()),
            "median": float(final["SI"].median()),
        }
        if len(final)
        else {}
    )
    summary = {
        "source_status": source_status,
        "candidate_rows_by_source": {
            SOURCE_LABELS[key]: len(source_records.get(key, []))
            for key in ("figshare", "pmc6335999", "pmc7651861", "sciencedirect")
        },
        "final_rows_by_source": source_counts,
        "candidate_rows": len(all_records),
        "valid_smiles_before_deduplication": len(all_records) - invalid_smiles,
        "invalid_smiles": invalid_smiles,
        "duplicates_with_train": duplicates_with_train,
        "duplicates_within_articles": duplicates_within_articles,
        "censored_rows": censored_count,
        "final_rows": len(final),
        "si_distribution": si_stats,
        "note": (
            "Figshare censored CC50 measurements are represented by SI=10000; "
            "the requested final schema has no is_censored column."
        ),
    }
    (DATA_DIR / "articles_collection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nArticle collection statistics")
    for label, count in source_counts.items():
        print(f"{label}: {count} final rows")
    print(f"Candidate rows: {len(all_records)}")
    print(f"Valid SMILES before deduplication: {len(all_records) - invalid_smiles}")
    print(f"Invalid SMILES: {invalid_smiles}")
    print(f"Duplicates with train: {duplicates_with_train}")
    print(f"Duplicates within articles: {duplicates_within_articles}")
    print(f"Censored rows (SI=10000): {censored_count}")
    print(f"Final rows: {len(final)}")
    if si_stats:
        print(
            f"SI distribution: min={si_stats['min']:.6g}, "
            f"max={si_stats['max']:.6g}, median={si_stats['median']:.6g}"
        )
    else:
        print("SI distribution: no rows")
    for source, status in source_status.items():
        if isinstance(status, dict):
            print(f"{source}: {status.get('status', 'complete')}")
            if status.get("error"):
                print(f"  error: {status['error']}")
            if status.get("note"):
                print(f"  note: {status['note']}")
    print(f"Final data: {final_path}")
    print(f"Summary: {DATA_DIR / 'articles_collection_summary.json'}")


if __name__ == "__main__":
    main()
