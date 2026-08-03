"""Freeze a corpus build: record the consumed file list, or verify a copy against it.

The 5.40B build's freeze did not write down which files it consumed, and the provenance ledger had
to be reconstructed afterwards from mtimes against the tokenization log — the paper's
limitations section calls that out and prescribes this script: at freeze time, write the
authoritative manifest (relative path, byte size and SHA-256 of every document); from
then on, any copy of the corpus — on the crawler box, the training box, or restored
years later — is checked against the manifest, not against hope.

Generate (run where the corpus lives, e.g. the crawler VPS):

    python scripts/corpus_freeze.py generate --corpus ~/wieszcz-xix/data/clean \
        --out ~/wieszcz-xix/data/freeze_2026-08-03_manifest.sha256

Verify (run on any copy; exits non-zero on any mismatch):

    python scripts/corpus_freeze.py verify --corpus data/clean \
        --manifest data/freeze_v2_manifest.sha256

The manifest is `sha256sum` format (hash, two spaces, ./relative/path), sorted by path,
so plain `cd corpus && sha256sum -c manifest` works too; verify here adds the two checks
sha256sum -c skips: files present on disk but absent from the manifest, and a total
count/byte summary suitable for the freeze record.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def iter_manifest(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        yield rel, digest


def generate(corpus: Path, out: Path) -> None:
    files = sorted(p for p in corpus.rglob("*.txt") if p.is_file())
    total = 0
    with out.open("w", encoding="utf-8") as sink:
        for p in files:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            total += p.stat().st_size
            sink.write(f"{h.hexdigest()}  ./{p.relative_to(corpus)}\n")
    print(f"{len(files):,} files, {total:,} bytes -> {out}")


def verify(corpus: Path, manifest: Path) -> None:
    expected = dict(iter_manifest(manifest))
    on_disk = {f"./{p.relative_to(corpus)}"
               for p in corpus.rglob("*.txt") if p.is_file()}
    missing = sorted(set(expected) - on_disk)
    extra = sorted(on_disk - set(expected))
    bad: list[str] = []
    total = 0
    for rel, digest in expected.items():
        p = corpus / rel
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        total += p.stat().st_size
        if h.hexdigest() != digest:
            bad.append(rel)

    print(f"manifest {len(expected):,} files | on disk {len(on_disk):,} | "
          f"{total:,} bytes hashed")
    for label, items in (("MISSING", missing), ("EXTRA", extra), ("CORRUPT", bad)):
        if items:
            print(f"{label}: {len(items)}")
            for r in items[:10]:
                print(f"  {r}")
    if missing or extra or bad:
        sys.exit(1)
    print("copy matches the freeze manifest exactly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "verify"])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, help="verify: manifest to check against")
    ap.add_argument("--out", type=Path, help="generate: manifest destination")
    args = ap.parse_args()

    corpus = args.corpus.expanduser()
    if args.mode == "generate":
        if not args.out:
            sys.exit("generate needs --out")
        generate(corpus, args.out.expanduser())
    else:
        if not args.manifest:
            sys.exit("verify needs --manifest")
        verify(corpus, args.manifest.expanduser())


if __name__ == "__main__":
    main()
