#!/usr/bin/env python3
"""Verify packaged inputs and restore their exact original bytes.

Use ``materialize_inputs(repository_root)`` before the analysis. Large inputs
may be stored as ordered gzip parts to keep individual GitHub files small.
This utility performs no scientific cleaning, imputation or model fitting.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


OUTPUT_NAMES = {
    "methylation_clean": "data_methylation_hm27_hm450_merged_clean.txt",
    "methylation_raw": "data_methylation_hm27_hm450_merged.txt",
    "clinical_patient": "data_clinical_patient.txt",
    "clinical_sample": "data_clinical_sample.txt",
    "expression": "data_mrna_seq_v2_rsem.txt",
}


def _root(root: str | Path | None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Packaged path leaves the repository: {relative}")
    return path


def _load(root: Path) -> dict:
    manifest = json.loads((root / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data provenance schema")
    keys = [item["key"] for item in manifest["inputs"]]
    if len(keys) != len(set(keys)) or set(keys) != set(OUTPUT_NAMES):
        raise ValueError("The manifest must contain each of the five analysis inputs exactly once")
    return manifest


def verify_package(root: str | Path | None = None, *, include_references: bool = True) -> dict[str, int]:
    """Check stored SHA-256 hashes and decompressed input hashes without writing."""
    base = _root(root)
    manifest = _load(base)
    checked = 0
    decompressed_bytes = 0
    for item in manifest["inputs"]:
        digest = hashlib.sha256()
        size = 0
        for part in item["parts"]:
            path = _package_path(base, part["path"])
            if path.stat().st_size != part["bytes"] or _sha256(path) != part["sha256"]:
                raise ValueError(f"Stored input failed verification: {part['path']}")
            with gzip.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            checked += 1
        if size != item["uncompressed_bytes"] or digest.hexdigest() != item["uncompressed_sha256"]:
            raise ValueError(f"Decompressed input failed verification: {item['key']}")
        decompressed_bytes += size
    if include_references:
        for item in manifest["reference_summaries"] + manifest["frozen_files"]:
            path = _package_path(base, item["path"])
            if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
                raise ValueError(f"Reference file failed verification: {item['path']}")
            checked += 1
    return {"stored_files_checked": checked, "input_bytes_verified": decompressed_bytes}


def materialize_inputs(
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    verify: bool = True,
) -> dict[str, Path]:
    """Return the five original input paths, restoring missing or changed files.

    By default files are restored into ``data/prepared``. Existing valid files
    are reused. Each new file is assembled in a temporary file and checked
    before replacement, so partial decompression cannot become an input.
    """
    base = _root(root)
    manifest = _load(base)
    target_dir = Path(output_dir).resolve() if output_dir is not None else base / "data" / "prepared"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for item in manifest["inputs"]:
        destination = target_dir / OUTPUT_NAMES[item["key"]]
        paths[item["key"]] = destination
        if destination.exists() and destination.stat().st_size == item["uncompressed_bytes"]:
            if not verify or _sha256(destination) == item["uncompressed_sha256"]:
                continue
        temporary = None
        try:
            digest = hashlib.sha256()
            size = 0
            with tempfile.NamedTemporaryFile(dir=target_dir, prefix=destination.name + ".", suffix=".tmp", delete=False) as writer:
                temporary = Path(writer.name)
                for part in item["parts"]:
                    path = _package_path(base, part["path"])
                    if verify and (path.stat().st_size != part["bytes"] or _sha256(path) != part["sha256"]):
                        raise ValueError(f"Stored input failed verification: {part['path']}")
                    with gzip.open(path, "rb") as reader:
                        for block in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(block)
                            digest.update(block)
                            size += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            if size != item["uncompressed_bytes"] or digest.hexdigest() != item["uncompressed_sha256"]:
                raise ValueError(f"Restored input failed verification: {item['key']}")
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root; defaults to the parent of tools")
    parser.add_argument("--output-dir", type=Path, help="Destination for restored inputs")
    parser.add_argument("--verify-only", action="store_true", help="Check the package without restoring inputs")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_package(args.root), indent=2))
    else:
        print(json.dumps({key: str(path) for key, path in materialize_inputs(args.root, args.output_dir).items()}, indent=2))


if __name__ == "__main__":
    main()
