#!/usr/bin/env python3
"""Download the UCI Heart Disease archive and extract the Cleveland source file.

The script deliberately extracts only the member consumed by this project.  It is
safe to run repeatedly: existing downloaded and extracted files are reused unless
``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DEFAULT_MEMBER = "processed.cleveland.data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an acquisition run."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the UCI Heart Disease archive and extract the processed "
            "Cleveland data file used by this project."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to keep the archive and extracted raw data.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="URL of the UCI ZIP archive.",
    )
    parser.add_argument(
        "--member",
        default=DEFAULT_MEMBER,
        help="Exact ZIP member to extract.",
    )
    parser.add_argument(
        "--archive-name",
        default="heart_disease_uci.zip",
        help="Filename to use for the downloaded ZIP archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive and replace the extracted member if present.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Network timeout in seconds.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path, timeout: float) -> None:
    """Download *url* atomically to *destination*."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mlops-heart-disease-assignment/1.0"},
    )
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=destination.parent, suffix=".part"
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary_path.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=CHUNK_SIZE)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("Downloaded archive is empty.")
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_member(archive_path: Path, member: str, destination: Path) -> None:
    """Extract exactly one known member from a ZIP archive atomically.

    Extracting an individual member avoids path-traversal risks associated with
    bulk extraction and keeps the raw-data directory focused on this project's
    source file.
    """

    with zipfile.ZipFile(archive_path) as archive:
        available_members = archive.namelist()
        if member not in available_members:
            available = ", ".join(sorted(available_members))
            raise ValueError(
                f"Archive does not contain {member!r}. Available members: {available}"
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent, suffix=".part"
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            with archive.open(member) as source, temporary_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=CHUNK_SIZE)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Extracted member {member!r} is empty.")
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def main() -> int:
    """Run data acquisition and print reproducibility-friendly file metadata."""

    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if Path(args.member).name != args.member:
        raise ValueError("--member must name a file at the archive root, not a path.")
    if Path(args.archive_name).name != args.archive_name:
        raise ValueError("--archive-name must be a filename, not a path.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / args.archive_name
    extracted_path = output_dir / args.member

    if args.force or not archive_path.exists():
        logging.info("Downloading UCI archive from %s", args.url)
        download_archive(args.url, archive_path, args.timeout)
    else:
        logging.info("Reusing existing archive: %s", archive_path)

    if args.force or not extracted_path.exists():
        logging.info("Extracting %s from %s", args.member, archive_path)
        extract_member(archive_path, args.member, extracted_path)
    else:
        logging.info("Reusing existing extracted data: %s", extracted_path)

    print(f"archive={archive_path}")
    print(f"archive_sha256={sha256(archive_path)}")
    print(f"raw_data={extracted_path}")
    print(f"raw_data_sha256={sha256(extracted_path)}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError, ValueError, zipfile.BadZipFile) as error:
        logging.error("Data acquisition failed: %s", error)
        raise SystemExit(1) from error
