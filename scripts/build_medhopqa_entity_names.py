#!/usr/bin/env python3
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
PAT_DB = re.compile(r"^DB\d{5}$")
PAT_UP = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$")
PAT_ANY = re.compile(r"\b(?:DB\d{5}|[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?)\b")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_ids(raw_dir: Path) -> List[str]:
    ids = set()
    for split_name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        path = raw_dir / split_name
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            for item in row.get("choices", []):
                token = str(item).strip()
                if PAT_DB.match(token) or PAT_UP.match(token):
                    ids.add(token)
            for item in row.get("answer", []):
                token = str(item).strip()
                if PAT_DB.match(token) or PAT_UP.match(token):
                    ids.add(token)
            text = f"{row.get('question', '')} {row.get('context', '')}"
            for token in PAT_ANY.findall(text):
                ids.add(token)
    return sorted(ids)


def _request_json(url: str, *, timeout: int = 30, tries: int = 3) -> Optional[dict]:
    last_exc: Optional[BaseException] = None
    for attempt in range(tries):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "gnn-trm-medhop-resolver/1.0"})
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < tries:
                time.sleep(1.0 + attempt)
    if last_exc:
        raise last_exc
    return None


def _resolve_drugbank(drugbank_id: str) -> Optional[str]:
    url = (
        f"https://mychem.info/v1/query?q=drugbank.id:{drugbank_id}"
        "&fields=drugbank.name,unii.preferred_term,chebi.name,drugbank.synonyms"
        "&size=5"
    )
    payload = _request_json(url)
    if not isinstance(payload, dict):
        return None
    for hit in payload.get("hits", []):
        drugbank = hit.get("drugbank", {})
        if isinstance(drugbank, dict):
            name = drugbank.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
            synonyms = drugbank.get("synonyms")
            if isinstance(synonyms, list) and synonyms:
                syn = str(synonyms[0]).strip()
                if syn:
                    return syn
        unii = hit.get("unii", {})
        if isinstance(unii, dict):
            name = unii.get("preferred_term")
            if isinstance(name, str) and name.strip():
                return name.strip()
        chebi = hit.get("chebi", {})
        if isinstance(chebi, dict):
            name = chebi.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _resolve_uniprot(uniprot_id: str) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    payload = _request_json(url)
    if not isinstance(payload, dict):
        return None

    protein_desc = payload.get("proteinDescription", {})
    name = None
    if isinstance(protein_desc, dict):
        rec = protein_desc.get("recommendedName", {})
        if isinstance(rec, dict):
            full = rec.get("fullName", {})
            if isinstance(full, dict):
                value = full.get("value")
                if isinstance(value, str) and value.strip():
                    name = value.strip()
        if not name:
            subs = protein_desc.get("submissionNames", [])
            if isinstance(subs, list) and subs:
                first = subs[0]
                if isinstance(first, dict):
                    full = first.get("fullName", {})
                    if isinstance(full, dict):
                        value = full.get("value")
                        if isinstance(value, str) and value.strip():
                            name = value.strip()

    gene = None
    genes = payload.get("genes", [])
    if isinstance(genes, list) and genes:
        gene_name = genes[0].get("geneName", {})
        if isinstance(gene_name, dict):
            value = gene_name.get("value")
            if isinstance(value, str) and value.strip():
                gene = value.strip()

    if name and gene and gene.lower() not in name.lower():
        return f"{name} ({gene})"
    return name or gene


def _resolve_one(identifier: str) -> Tuple[str, Optional[str], str]:
    try:
        if PAT_DB.match(identifier):
            return identifier, _resolve_drugbank(identifier), "drugbank"
        if PAT_UP.match(identifier):
            return identifier, _resolve_uniprot(identifier), "uniprot"
    except Exception:
        return identifier, None, "error"
    return identifier, None, "unknown"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resolve MedHopQA DrugBank/UniProt IDs into natural-language names")
    ap.add_argument("--raw-dir", default="data/medhopqa_raw")
    ap.add_argument("--out-json", default="data/medhopqa/entity_names.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    raw_dir = (REPO_ROOT / args.raw_dir).resolve() if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)
    out_json = (REPO_ROOT / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ids = _extract_ids(raw_dir)
    if args.limit > 0:
        ids = ids[: args.limit]

    name_map: Dict[str, str] = {}
    stats = {"drugbank": 0, "uniprot": 0, "resolved": 0, "unresolved": 0}

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {ex.submit(_resolve_one, identifier): identifier for identifier in ids}
        for fut in as_completed(futures):
            identifier, name, source = fut.result()
            if source in stats:
                stats[source] += 1
            if name:
                name_map[f"medhop_ent::{identifier}"] = f"{name} [{identifier}]"
                stats["resolved"] += 1
            else:
                name_map[f"medhop_ent::{identifier}"] = identifier
                stats["unresolved"] += 1

    out_json.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "raw_dir": str(raw_dir),
        "out_json": str(out_json),
        "num_ids": len(ids),
        "stats": stats,
    }
    (out_json.parent / "entity_names_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
