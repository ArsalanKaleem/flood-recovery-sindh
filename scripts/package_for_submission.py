"""
Package the full research deliverable into a single zip for archiving,
sharing with collaborators, or supplementing journal submission.

Structure of the output zip:
  flood-recovery-sindh-package-YYYY-MM-DD/
    README.md
    docs/                   — all docs + reproduction manifest + results summary
    results/
      figures/              — every PNG in results/figures/
      tables/               — every CSV in results/tables/
    src/                    — all analysis Python modules
    gee_scripts/            — GEE pipeline scripts
    config/                 — AOI geojson
    environment.yml         — if present
    MANIFEST.txt            — generated listing of files + sizes + total

Does NOT include:
  data/raw/     (download instructions in reproduction_manifest.md)
  data/processed/ (regenerable from pipeline; too large for packaging)
  .git/         (use `git bundle` separately if full history is needed)
  __pycache__/, .ipynb_checkpoints/

Usage:
  python scripts/package_for_submission.py

Writes to:  dist/flood-recovery-sindh-package-YYYY-MM-DD.zip
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"

# Folders to include in full.
INCLUDE_DIRS = [
    "docs",
    "results/figures",
    "results/tables",
    "src",
    "gee_scripts",
    "config",
]

# Individual files to include if present.
INCLUDE_FILES = [
    "README.md",
    "environment.yml",
    ".gitignore",
    "requirements.txt",
]

# Patterns to ALWAYS exclude from any directory walk.
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".DS_Store",
    "Thumbs.db",
    "_tmp_",  # any tmp files
}


def is_excluded(path: Path) -> bool:
    """Check if a path should be skipped."""
    parts = path.parts
    for excl in EXCLUDE_PATTERNS:
        if any(excl in part for part in parts):
            return True
    return False


def collect_files() -> list[Path]:
    """Enumerate all files that will go into the package."""
    files: list[Path] = []

    # Directory contents.
    for d in INCLUDE_DIRS:
        dir_path = REPO / d
        if not dir_path.exists():
            print(f"  SKIP (missing): {d}/")
            continue
        for f in dir_path.rglob("*"):
            if f.is_file() and not is_excluded(f):
                files.append(f)

    # Individual files.
    for f in INCLUDE_FILES:
        file_path = REPO / f
        if file_path.exists():
            files.append(file_path)
        else:
            print(f"  SKIP (missing): {f}")

    return sorted(set(files))


def file_sha256(path: Path, block_size: int = 65536) -> str:
    """SHA-256 of a file, used for the manifest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"


def build_manifest(files: list[Path], package_name: str) -> str:
    """Produce a text manifest listing every packaged file with size and hash."""
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Flood Recovery Sindh — Package Manifest",
        f"=" * 70,
        f"Package: {package_name}",
        f"Built:   {today}",
        f"Files:   {len(files)}",
        f"",
        f"{'relative path':<70} {'size':>10}  sha256",
        "-" * 110,
    ]
    total_bytes = 0
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        size = f.stat().st_size
        total_bytes += size
        digest = file_sha256(f)[:12]  # short hash
        lines.append(f"{rel:<70} {human_size(size):>10}  {digest}...")

    lines += [
        "-" * 110,
        f"{'TOTAL':<70} {human_size(total_bytes):>10}",
        "",
        "Notes:",
        "  - This package does not include data/raw/ or data/processed/.",
        "    See docs/reproduction_manifest.md to regenerate them.",
        "  - SHA-256 hashes are abbreviated (first 12 chars) for compactness.",
    ]
    return "\n".join(lines)


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)

    date_str = dt.date.today().isoformat()
    package_name = f"flood-recovery-sindh-package-{date_str}"
    out_zip = DIST / f"{package_name}.zip"

    print(f"\nBuilding package: {package_name}\n")
    print("Collecting files...")
    files = collect_files()
    print(f"  Found {len(files)} files to include.\n")

    print("Computing manifest...")
    manifest = build_manifest(files, package_name)

    # Preview counts per directory.
    counts: dict[str, int] = {}
    for f in files:
        top = f.relative_to(REPO).parts[0]
        counts[top] = counts.get(top, 0) + 1
    print("  By top-level directory:")
    for k, v in sorted(counts.items()):
        print(f"    {k:20s} {v:>4} files")
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\n  Total uncompressed size: {human_size(total_bytes)}")

    print(f"\nWriting zip → {out_zip}")
    with zipfile.ZipFile(
        out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for f in files:
            arcname = Path(package_name) / f.relative_to(REPO)
            zf.write(f, arcname.as_posix())
        zf.writestr(f"{package_name}/MANIFEST.txt", manifest)

    out_size = out_zip.stat().st_size
    ratio = (1 - out_size / total_bytes) * 100 if total_bytes > 0 else 0
    print(f"\nDone.")
    print(f"  Package: {out_zip}")
    print(f"  Size: {human_size(out_size)} (compression saved {ratio:.1f}%)")


if __name__ == "__main__":
    main()
