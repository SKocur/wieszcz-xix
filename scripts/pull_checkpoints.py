"""Mirror the ladder's checkpoints from the network volume to local disk, verified.

The training loop prints `ckpt <name> @ step N sha256 <16 hex>` at every checkpoint
write, and the launch logs live in metrics/. Each downloaded file is therefore verified
against the digest recorded at write time, not merely against a byte count. Downloads
stream through a running SHA-256, survive the endpoint's spurious 403s, resume partial
files after an interruption, and stop cleanly if local free space runs low. Protocol
checkpoints (final, stable_branch, predecay, epoch boundaries) come first, so the
irreplaceable files are safe before the step series starts.

Re-running skips everything already present and verified, so the script is idempotent.

    python scripts/pull_checkpoints.py                    # all three runs -> checkpoints/
    python scripts/pull_checkpoints.py --runs 349m        # substring filter
    python scripts/pull_checkpoints.py --protocol-only    # skip the step*.pt series
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volume_s3 import VOLUME_ID, client, human  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PREFIX = "wieszcz-xix/checkpoints/"
RUNS = [
    ("wieszcz_47m_6b7_2026-08-05_s1337", "train_wieszcz_47m_6b7.out"),
    ("wieszcz_107m_6b7_2026-08-06_s1337", "train_wieszcz_107m_6b7.out"),
    ("wieszcz_349m_6b7_2026-08-07_s1337", "train_wieszcz_349m_6b7.out"),
]
PROTOCOL = ("final.pt", "stable_branch.pt", "predecay.pt", "epoch1.pt", "epoch2.pt")
CKPT_LINE = re.compile(r"ckpt (\S+) @ step \d+ sha256 ([0-9a-f]{16})")
MIN_FREE = 15 << 30          # stop rather than fill the disk
RETRIES = 8


def expected_shas(log_name: str) -> dict[str, str]:
    """name -> 16-hex prefix, keeping the LAST write (rotated step files reuse names)."""
    out: dict[str, str] = {}
    for m in CKPT_LINE.finditer((REPO_ROOT / "metrics" / log_name).read_text(errors="ignore")):
        out[m.group(1)] = m.group(2)
    return out


def hash_file(path: Path) -> hashlib._hashlib.HASH:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h


def pull(s3, key: str, dest: Path, size: int, want: str | None) -> str:
    """Download key -> dest with resume + running sha256; return the full digest."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(RETRIES):
        try:
            done = part.stat().st_size if part.exists() else 0
            if done > size:
                part.unlink()
                done = 0
            h = hash_file(part) if done else hashlib.sha256()
            if done < size:
                kwargs = {"Bucket": VOLUME_ID, "Key": key}
                if done:
                    kwargs["Range"] = f"bytes={done}-"
                body = s3.get_object(**kwargs)["Body"]
                t0, base = time.time(), done
                with open(part, "ab") as fh:
                    while chunk := body.read(8 << 20):
                        fh.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                rate = (done - base) / max(time.time() - t0, 1e-6)
                print(f"    {human(done)} at {human(int(rate))}/s", flush=True)
            digest = h.hexdigest()
            if part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size:,} != remote {size:,}")
            if want and not digest.startswith(want):
                part.unlink()   # corrupt in flight; restart from zero
                raise IOError(f"sha {digest[:16]} != logged {want}")
            part.rename(dest)
            return digest
        except Exception as exc:  # noqa: BLE001 — endpoint 403s, resets, sha mismatch
            if attempt == RETRIES - 1:
                raise
            wait = min(60, 5 * (attempt + 1))
            print(f"    retry {attempt + 1}/{RETRIES} in {wait}s: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dest", default=str(REPO_ROOT / "checkpoints"))
    p.add_argument("--runs", default="", help="substring filter on run names")
    p.add_argument("--protocol-only", action="store_true", help="skip step*.pt files")
    args = p.parse_args()

    s3 = client()
    jobs: list[tuple[str, str, int, str | None, Path]] = []   # (run, name, size, sha, dest)
    for run, log in RUNS:
        if args.runs and args.runs not in run:
            continue
        shas = expected_shas(log)
        listed = s3.get_paginator("list_objects_v2").paginate(
            Bucket=VOLUME_ID, Prefix=f"{CKPT_PREFIX}{run}/")
        for page in listed:
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[1]
                if args.protocol_only and name not in PROTOCOL:
                    continue
                jobs.append((run, name, obj["Size"],
                             shas.get(name), Path(args.dest) / run / name))

    # Irreplaceable files first, then the step series, small runs before large.
    order = {n: i for i, n in enumerate(PROTOCOL)}
    runpos = {run: i for i, (run, _) in enumerate(RUNS)}
    jobs.sort(key=lambda j: (j[1] not in order, order.get(j[1], 0), runpos[j[0]], j[1]))

    total = sum(j[2] for j in jobs)
    print(f"{len(jobs)} files, {human(total)} total", flush=True)

    pulled = skipped = 0
    for run, name, size, want, dest in jobs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        sums = dest.parent / "SHA256SUMS"
        if dest.exists() and dest.stat().st_size == size:
            digest = hash_file(dest).hexdigest()
            if want is None or digest.startswith(want):
                skipped += 1
                print(f"[skip] {run}/{name} (verified)", flush=True)
                continue
            print(f"[redo] {run}/{name}: local sha {digest[:16]} != logged {want}", flush=True)
            dest.unlink()
        if shutil.disk_usage(dest.parent).free < MIN_FREE + size:
            sys.exit(f"STOPPING: under {human(MIN_FREE)} free would remain; "
                     f"{pulled} pulled, {skipped} skipped so far")
        print(f"[pull] {run}/{name} ({human(size)}, logged sha "
              f"{want or 'none — recording ours'})", flush=True)
        digest = pull(s3, f"{CKPT_PREFIX}{run}/{name}", dest, size, want)
        lines = [l for l in (sums.read_text().splitlines() if sums.exists() else [])
                 if not l.endswith(f"  {name}")]
        lines.append(f"{digest}  {name}")
        sums.write_text("\n".join(sorted(lines)) + "\n")
        pulled += 1

    print(f"done: {pulled} pulled, {skipped} already present, "
          f"{human(shutil.disk_usage(args.dest).free)} free", flush=True)


if __name__ == "__main__":
    main()
