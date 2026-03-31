#!/usr/bin/env python
"""
Download benchmark datasets for CISSN experiments.

Usage
-----
    # Download all missing datasets
    python scripts/download_datasets.py

    # Download specific datasets only
    python scripts/download_datasets.py --datasets ETTh2,ETTm1,ETTm2

    # Use a different data root
    python scripts/download_datasets.py --data-root ./data

Dataset sources
---------------
  ETTh1, ETTh2, ETTm1, ETTm2
      github.com/zhouhaoyi/ETDataset  (direct GitHub raw download, no auth)

  weather, exchange_rate, electricity (ECL), traffic, national_illness (ILI)
      Autoformer Google Drive folder shared by thuml/Autoformer
      https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
      Requires:  pip install gdown   (script will prompt to install)

  solar (Solar-Energy, Alabama 137 stations)
      github.com/laiguokun/multivariate-time-series-data  (direct GitHub raw)

Requirements
------------
  Built-in only for ETT + Solar.
  ``gdown`` for the Autoformer Drive datasets (Weather, Exchange, ECL, Traffic, ILI).
  Install via:  uv pip install gdown   or   pip install gdown
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

# ── Source registry ──────────────────────────────────────────────────────────

_ETT_RAW = "https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small"
_SOLAR_RAW = (
    "https://github.com/laiguokun/multivariate-time-series-data"
    "/raw/master/solar-energy/solar_AL.txt.gz"
)

# Autoformer Google Drive folder (public, shared by thuml/Autoformer)
_AUTOFORMER_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy"
)

# Maps dataset key → (method, source, dest_relative_to_data_root)
# method: "github_raw" | "gdrive_folder" | "github_raw_gz"
REGISTRY: dict = {
    "ETTh1": (
        "github_raw",
        f"{_ETT_RAW}/ETTh1.csv",
        "ETT/ETTh1.csv",
    ),
    "ETTh2": (
        "github_raw",
        f"{_ETT_RAW}/ETTh2.csv",
        "ETT/ETTh2.csv",
    ),
    "ETTm1": (
        "github_raw",
        f"{_ETT_RAW}/ETTm1.csv",
        "ETT/ETTm1.csv",
    ),
    "ETTm2": (
        "github_raw",
        f"{_ETT_RAW}/ETTm2.csv",
        "ETT/ETTm2.csv",
    ),
    # ── Autoformer Drive datasets ─────────────────────────────────────────
    # These are downloaded as a folder; the script picks out the right files.
    "weather":       ("gdrive_folder", "weather.csv",           "weather.csv"),
    "exchange_rate": ("gdrive_folder", "exchange_rate.csv",     "exchange_rate.csv"),
    "ECL":           ("gdrive_folder", "electricity.csv",       "electricity.csv"),
    "traffic":       ("gdrive_folder", "traffic.csv",           "traffic.csv"),
    "ILI":           ("gdrive_folder", "national_illness.csv",  "national_illness.csv"),
    # ── Solar Energy (LSTNet repo, gzip-compressed) ───────────────────────
    "solar": (
        "github_raw_gz",
        _SOLAR_RAW,
        "solar_AL.txt",
    ),
}

ALL_DATASETS = list(REGISTRY)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()
    else:
        mb = downloaded / (1024 ** 2)
        sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)


def _ensure_gdown() -> bool:
    """Return True if gdown is importable; offer to install otherwise."""
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        pass
    ans = input(
        "\n  gdown is not installed. Install it now? [y/N] "
    ).strip().lower()
    if ans == "y":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "--quiet"]
        )
        try:
            import gdown  # noqa: F401
            return True
        except ImportError:
            pass
    print(
        "  gdown not available. Install manually:\n"
        "    uv pip install gdown\n"
        "  Then re-run this script."
    )
    return False


def _download_gdrive_folder(dest_root: Path, requested_files: list[str]) -> list[str]:
    """
    Download the Autoformer Google Drive folder into a temp sub-directory,
    then move only the requested files to dest_root.
    Returns the list of successfully moved filenames.
    """
    import gdown

    tmp = dest_root / "_autoformer_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading Autoformer dataset folder from Google Drive …")
    print(f"  This may take several minutes (total ~2 GB for all files).")
    gdown.download_folder(
        url=_AUTOFORMER_FOLDER_URL,
        output=str(tmp),
        quiet=False,
        use_cookies=False,
    )

    moved: list[str] = []
    for fname in requested_files:
        # gdown may nest the folder; search recursively
        found = list(tmp.rglob(fname))
        if found:
            target = dest_root / fname
            shutil.move(str(found[0]), str(target))
            print(f"  ✓ {fname} → {target}")
            moved.append(fname)
        else:
            print(f"  ✗ {fname} not found in downloaded folder")

    # Clean up temp directory
    try:
        shutil.rmtree(tmp)
    except OSError:
        pass

    return moved


def _download_gz(url: str, dest: Path) -> None:
    import gzip
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    _download_url(url, gz_path)
    print(f"  Decompressing {gz_path.name} …")
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


# ── Main logic ────────────────────────────────────────────────────────────────

def download(datasets: list[str], data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    # Separate datasets by method
    github_raw_datasets = []
    github_raw_gz_datasets = []
    gdrive_datasets = []

    for name in datasets:
        method, src, dest_rel = REGISTRY[name]
        dest = data_root / dest_rel

        if dest.exists():
            print(f"  [skip]  {name} already present at {dest}")
            continue

        if method == "github_raw":
            github_raw_datasets.append((name, src, dest))
        elif method == "github_raw_gz":
            github_raw_gz_datasets.append((name, src, dest))
        elif method == "gdrive_folder":
            gdrive_datasets.append((name, src, dest))

    # ── Direct GitHub downloads ───────────────────────────────────────────
    for name, src, dest in github_raw_datasets:
        print(f"\n[{name}]")
        try:
            _download_url(src, dest)
            print(f"  ✓ saved to {dest}")
        except Exception as e:
            print(f"  ✗ failed: {e}")

    # ── Direct GitHub gzip downloads ──────────────────────────────────────
    for name, src, dest in github_raw_gz_datasets:
        print(f"\n[{name}]")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download_gz(src, dest)
            print(f"  ✓ saved to {dest}")
        except Exception as e:
            print(f"  ✗ failed: {e}")

    # ── Google Drive Folder downloads ─────────────────────────────────────
    if gdrive_datasets:
        print(f"\n[Autoformer Drive datasets: {', '.join(n for n, _, _ in gdrive_datasets)}]")
        if not _ensure_gdown():
            print_manual_instructions([n for n, _, _ in gdrive_datasets])
            return

        requested_files = [src for _, src, _ in gdrive_datasets]
        moved = _download_gdrive_folder(data_root, requested_files)

        missing = set(requested_files) - set(moved)
        if missing:
            print(
                "\n  Some files were not found in the downloaded folder.\n"
                "  Download them manually from:\n"
                f"  {_AUTOFORMER_FOLDER_URL}\n"
                "  and place them in:  {data_root}/\n"
            )


def print_manual_instructions(names: list[str]) -> None:
    print("\n  Manual download instructions:")
    print(f"  1. Open: {_AUTOFORMER_FOLDER_URL}")
    print("  2. Download the following files and place them in ./data/:")
    for name in names:
        _, src_fname, dest_rel = REGISTRY[name]
        print(f"       {src_fname}  →  data/{dest_rel}")
    print()


def print_usage_table(data_root: Path) -> None:
    """Print a quick summary of which datasets are present and ready."""
    print("\nDataset readiness:")
    print(f"  {'Dataset':<16} {'Status':<10} {'Path'}")
    print(f"  {'-'*16} {'-'*10} {'-'*40}")
    for name, (_, _, dest_rel) in REGISTRY.items():
        dest = data_root / dest_rel
        status = "ready" if dest.exists() else "missing"
        print(f"  {name:<16} {status:<10} {dest}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CISSN benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(ALL_DATASETS),
        help=(
            "Comma-separated list of datasets to download. "
            f"Choices: {', '.join(ALL_DATASETS)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory where datasets will be stored (default: ./data).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the readiness status of all datasets and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root.resolve()

    if args.status:
        print_usage_table(data_root)
        return

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in requested if d not in REGISTRY]
    if unknown:
        print(f"Unknown datasets: {unknown}. Available: {ALL_DATASETS}")
        sys.exit(1)

    print(f"Data root : {data_root}")
    print(f"Datasets  : {', '.join(requested)}\n")

    download(requested, data_root)
    print_usage_table(data_root)


if __name__ == "__main__":
    main()
