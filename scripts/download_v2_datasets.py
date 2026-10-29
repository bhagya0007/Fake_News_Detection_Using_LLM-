"""V2 Raw Dataset Acquisition Script.

Downloads verified, official raw dataset artifacts for:
- AVeriTeC (train.json, dev.json from official GitHub repo)
- MultiFC (MultiFC_Codalab.zip from CopeNLU publication page)
- FEVER (shared_task_dev.jsonl from fever.ai)

Computes SHA-256 hashes and generates data/v2_raw/checksums.json.
Does NOT modify data/raw/ (frozen V1 baseline).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_v2_datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_V2_DIR = PROJECT_ROOT / "data" / "v2_raw"
CHECKSUMS_FILE = RAW_V2_DIR / "checksums.json"

AVERITEC_BASE_URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/"
AVERITEC_FILES = ["train.json", "dev.json"]

MULTIFC_ZIP_URL = "https://copenlu.github.io/publication/2019_emnlp_augenstein/MultiFC_Codalab.zip"

FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, timeout: int = 60) -> None:
    """Download a file with user agent and progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"File already exists: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return

    logger.info(f"Downloading {url} -> {dest_path.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    logger.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")


def main() -> None:
    RAW_V2_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    download_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. AVeriTeC
    averitec_dir = RAW_V2_DIR / "averitec"
    averitec_dir.mkdir(parents=True, exist_ok=True)
    for fname in AVERITEC_FILES:
        dest = averitec_dir / fname
        url = AVERITEC_BASE_URL + fname
        download_file(url, dest)
        sha = compute_sha256(dest)
        rel_path = str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")
        manifest[rel_path] = {
            "relative_path": rel_path,
            "dataset": "averitec",
            "size_bytes": dest.stat().st_size,
            "sha256": sha,
            "source_url": url,
            "download_date": download_date,
            "license": "CC BY-NC 4.0",
        }

    # 2. MultiFC
    multifc_dir = RAW_V2_DIR / "multifc"
    multifc_dir.mkdir(parents=True, exist_ok=True)
    multifc_zip = multifc_dir / "MultiFC_Codalab.zip"
    download_file(MULTIFC_ZIP_URL, multifc_zip, timeout=120)
    sha_zip = compute_sha256(multifc_zip)
    rel_zip = str(multifc_zip.relative_to(PROJECT_ROOT)).replace("\\", "/")
    manifest[rel_zip] = {
        "relative_path": rel_zip,
        "dataset": "multifc",
        "size_bytes": multifc_zip.stat().st_size,
        "sha256": sha_zip,
        "source_url": MULTIFC_ZIP_URL,
        "download_date": download_date,
        "license": "Unresolved / Unknown (see v2-dataset-audit.md)",
    }

    # Extract MultiFC nested archives
    logger.info("Extracting MultiFC_Codalab.zip...")
    import io
    with zipfile.ZipFile(multifc_zip, "r") as zf:
        # Extract public_data.zip
        if "public_data.zip" in zf.namelist():
            pdata = zf.read("public_data.zip")
            with zipfile.ZipFile(io.BytesIO(pdata), "r") as pzf:
                pzf.extractall(multifc_dir)
                for item in pzf.namelist():
                    extracted_path = multifc_dir / item
                    if extracted_path.is_file():
                        sha = compute_sha256(extracted_path)
                        rel = str(extracted_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                        manifest[rel] = {
                            "relative_path": rel,
                            "dataset": "multifc",
                            "size_bytes": extracted_path.stat().st_size,
                            "sha256": sha,
                            "source_url": MULTIFC_ZIP_URL + "#public_data.zip",
                            "download_date": download_date,
                            "license": "Unresolved / Unknown",
                        }
                        logger.info(f"Extracted public_data: {item} ({extracted_path.stat().st_size} bytes)")

        # Extract reference_multi_fc.zip
        if "reference_multi_fc.zip" in zf.namelist():
            rdata = zf.read("reference_multi_fc.zip")
            with zipfile.ZipFile(io.BytesIO(rdata), "r") as rzf:
                rzf.extractall(multifc_dir)
                for item in rzf.namelist():
                    extracted_path = multifc_dir / item
                    if extracted_path.is_file():
                        sha = compute_sha256(extracted_path)
                        rel = str(extracted_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                        manifest[rel] = {
                            "relative_path": rel,
                            "dataset": "multifc",
                            "size_bytes": extracted_path.stat().st_size,
                            "sha256": sha,
                            "source_url": MULTIFC_ZIP_URL + "#reference_multi_fc.zip",
                            "download_date": download_date,
                            "license": "Unresolved / Unknown",
                        }
                        logger.info(f"Extracted reference: {item} ({extracted_path.stat().st_size} bytes)")

    # 3. FEVER
    fever_dir = RAW_V2_DIR / "fever"
    fever_dir.mkdir(parents=True, exist_ok=True)
    fever_dev = fever_dir / "shared_task_dev.jsonl"
    download_file(FEVER_DEV_URL, fever_dev, timeout=60)
    sha_fever = compute_sha256(fever_dev)
    rel_fever = str(fever_dev.relative_to(PROJECT_ROOT)).replace("\\", "/")
    manifest[rel_fever] = {
        "relative_path": rel_fever,
        "dataset": "fever",
        "size_bytes": fever_dev.stat().st_size,
        "sha256": sha_fever,
        "source_url": FEVER_DEV_URL,
        "download_date": download_date,
        "license": "CC BY-SA 3.0 / Wikipedia Copyright",
    }

    # Write checksums manifest
    with open(CHECKSUMS_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote checksum manifest to {CHECKSUMS_FILE} ({len(manifest)} files recorded).")


if __name__ == "__main__":
    main()
