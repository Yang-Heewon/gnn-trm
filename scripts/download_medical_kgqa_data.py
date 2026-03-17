#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = REPO_ROOT / "data" / ".downloads"

PRIMEKGQA_FILES = {
    "train_v2_call_bioLLM.json": "https://zenodo.org/api/records/14725324/files/train_v2_call_bioLLM.json/content",
    "val_v2_call_bioLLM.json": "https://zenodo.org/api/records/14725324/files/val_v2_call_bioLLM.json/content",
    "test_v2_call_bioLLM.json": "https://zenodo.org/api/records/14725324/files/test_v2_call_bioLLM.json/content",
}
MEDHOP_SERVER = "https://datasets-server.huggingface.co"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _download_primekgqa(force: bool) -> Dict[str, object]:
    raw_dir = REPO_ROOT / "data" / "primekgqa_raw"
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: Dict[str, str] = {}
    warnings: List[str] = []
    for filename, url in PRIMEKGQA_FILES.items():
        out_path = raw_dir / filename
        if force or not out_path.exists():
            print(f"[download] PrimeKGQA {filename} -> {out_path}")
            _download_file(url, out_path)
        else:
            print(f"[skip] PrimeKGQA file already exists: {out_path}")
        try:
            with out_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            if filename.startswith("train_"):
                warning = f"{filename} validation failed: {exc}"
                print(f"[warn] {warning}")
                warnings.append(warning)
            else:
                raise
        downloaded_files[filename] = str(out_path)
    meta = {
        "dataset": "primekgqa",
        "sources": PRIMEKGQA_FILES,
        "raw_dir": str(raw_dir),
        "downloaded_files": downloaded_files,
        "warnings": warnings,
    }
    return meta


def _fetch_medhop_rows(split_name: str, chunk_size: int = 100) -> List[Dict]:
    out: List[Dict] = []
    offset = 0
    while True:
        url = (
            f"{MEDHOP_SERVER}/rows?dataset=bigbio/medhop&config=medhop_bigbio_qa"
            f"&split={split_name}&offset={offset}&length={chunk_size}"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        if not rows:
            break
        out.extend(item.get("row", {}) for item in rows)
        if len(rows) < chunk_size:
            break
        offset += len(rows)
    return out


def _fetch_medhop_parquet_urls() -> Dict[str, str]:
    url = f"{MEDHOP_SERVER}/parquet?dataset=bigbio/medhop&config=medhop_bigbio_qa"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    parquet_files = payload.get("parquet_files", [])
    return {item["split"]: item["url"] for item in parquet_files if item.get("split") and item.get("url")}


def _download_medhopqa(force: bool) -> Dict[str, object]:
    raw_dir = REPO_ROOT / "data" / "medhopqa_raw"
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] MedHopQA via Hugging Face parquet export")
    parquet_urls = _fetch_medhop_parquet_urls()
    split_names = ("train", "validation")
    split_counts: Dict[str, int] = {}
    split_files: Dict[str, str] = {}
    parquet_files: Dict[str, str] = {}
    for split_name in split_names:
        parquet_url = parquet_urls[split_name]
        parquet_path = raw_dir / f"{split_name}.parquet"
        if force or not parquet_path.exists():
            print(f"[download] MedHopQA {split_name}.parquet -> {parquet_path}")
            _download_file(parquet_url, parquet_path)
        else:
            print(f"[skip] MedHopQA parquet already exists: {parquet_path}")
        parquet_files[split_name] = str(parquet_path)

        rows = pd.read_parquet(parquet_path).to_dict(orient="records")
        out_path = raw_dir / f"{split_name}.jsonl"
        split_counts[split_name] = _write_jsonl(out_path, rows)
        split_files[split_name] = str(out_path)

    meta = {
        "dataset": "medhopqa",
        "source": f"{MEDHOP_SERVER}/parquet",
        "config": "medhop_bigbio_qa",
        "raw_dir": str(raw_dir),
        "parquet_files": parquet_files,
        "split_counts": split_counts,
        "split_files": split_files,
    }
    return meta


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download raw PrimeKGQA / MedHopQA data into the repo")
    ap.add_argument("--dataset", choices=["primekgqa", "medhopqa", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    if args.dataset in {"primekgqa", "all"}:
        results.append(_download_primekgqa(args.force))
    if args.dataset in {"medhopqa", "all"}:
        results.append(_download_medhopqa(args.force))

    meta_path = DOWNLOAD_ROOT / "medical_kgqa_download_meta.json"
    meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote metadata: {meta_path}")
    for item in results:
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
