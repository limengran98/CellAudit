#!/usr/bin/env python3
"""Verify source integrity and screen the public repository for private artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
TEXT_SUFFIXES = {"", ".json", ".md", ".py", ".sh", ".svg", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_PARTS = {
    ".env",
    ".pytest_cache",
    "__pycache__",
    "paper",
    "papers",
    "outputs",
    "runs",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".gz", ".h5", ".h5ad", ".npy", ".npz", ".pt", ".pth", ".tar", ".zip"}
MAX_FILE_BYTES = 1_000_000
PRIVATE_PATTERNS = (
    re.compile(r"/(?:data|home|root)/[^\s\"']+"),
    re.compile(r"(?:sk|ghp|github_pat)-[A-Za-z0-9_-]{12,}"),
    re.compile(r"github\.com/[A-Za-z0-9_.-]+/CellAudit", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or ".git" in relative.parts or path == MANIFEST:
            continue
        files.append(path)
    return sorted(files)


def verify_manifest(files: list[Path]) -> None:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ./", 1)
        expected[relative] = digest
    observed = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(name for name in expected.keys() & observed.keys() if expected[name] != observed[name])
        raise SystemExit(
            f"manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )


def verify_scope(files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS or part.startswith("paper_") for part in relative.parts):
            raise SystemExit(f"release-excluded path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"data or checkpoint artifact: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"oversized repository artifact: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(text)
            if match:
                raise SystemExit(f"private text pattern in {relative}: {pattern.pattern}")


def main() -> None:
    files = tracked_source_files()
    verify_scope(files)
    verify_manifest(files)
    print(f"OK: {len(files)} source files match the manifest; no private artifacts detected")


if __name__ == "__main__":
    main()
