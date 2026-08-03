"""Read the RunPod network volume over its S3-compatible endpoint, without a pod.

`GetObject` accepts a byte range, so a 108 MB slice of a 10.8 GB token file costs one
ranged request instead of the pod lifecycle `pull_val.sh` needs.

Credentials are NOT the RunPod API key. They are a separate S3 access key/secret pair
created under Settings -> S3 API Keys, read from .env as RUNPOD_S3_ACCESS_KEY /
RUNPOD_S3_SECRET_KEY.

The repo lives under `wieszcz-xix/` on the volume and `data/` sits beside it at the root,
so checkpoint keys carry the repo prefix and the run directory. The flat `checkpoints/*.pt`
layout in older notes was the 349M's and no longer exists.

    python scripts/volume_s3.py ls
    python scripts/volume_s3.py ls wieszcz-xix/checkpoints/
    python scripts/volume_s3.py stat data/clean/tokens.bin
    python scripts/volume_s3.py get data/clean/tokens.bin out.bin --tail 108048582
    python scripts/volume_s3.py get wieszcz-xix/checkpoints/<run>/final.pt ckpt.pt

The endpoint returns a spurious 403 now and then; the same call succeeds on a retry, so
treat a single failure as noise rather than as a credential problem.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VOLUME_ID = "7d8e8x9azm"          # the "wieszcz-xix" volume; also the bucket name
DATACENTER = "eu-ro-1"
ENDPOINT = f"https://s3api-{DATACENTER}.runpod.io"


def load_env() -> None:
    """Read .env without a dependency, and without ever printing a value."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def client():
    import boto3

    load_env()
    access = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret = os.environ.get("RUNPOD_S3_SECRET_KEY")
    if not access or not secret:
        sys.exit(
            "RUNPOD_S3_ACCESS_KEY / RUNPOD_S3_SECRET_KEY not set.\n"
            "Create them in the RunPod console: Settings -> S3 API Keys (these are NOT\n"
            "the RUNPOD_API_KEY), then add both to .env."
        )
    return boto3.client(
        "s3", region_name=DATACENTER, endpoint_url=ENDPOINT,
        aws_access_key_id=access, aws_secret_access_key=secret,
    )


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n/1:,.1f} {unit}"
        n /= 1024
    return str(n)


def cmd_ls(args) -> None:
    """List one level by default: the volume holds hundreds of thousands of small .txt
    files, and a recursive listing paginates for minutes."""
    s3 = client()
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": VOLUME_ID, "Prefix": args.prefix}
    if not args.recursive:
        kwargs["Delimiter"] = "/"

    total = count = 0
    for page in paginator.paginate(**kwargs):
        for pref in page.get("CommonPrefixes", []):
            print(f"{'<dir>':>14}  {'':16}  {pref['Prefix']}")
        for obj in page.get("Contents", []):
            print(f"{obj['Size']:>14,}  {obj['LastModified']:%Y-%m-%d %H:%M}  {obj['Key']}")
            total += obj["Size"]
            count += 1
            if args.limit and count >= args.limit:
                print(f"\n(stopped at --limit {args.limit})")
                return
    print(f"\n{count} objects here, {human(total)}")


def cmd_stat(args) -> None:
    s3 = client()
    head = s3.head_object(Bucket=VOLUME_ID, Key=args.key)
    size = head["ContentLength"]
    print(f"key   : {args.key}")
    print(f"bytes : {size:,}  ({human(size)})")
    print(f"tokens: {size // 2:,}  (if uint16)")
    print(f"mtime : {head['LastModified']:%Y-%m-%d %H:%M:%S}")
    print(f"etag  : {head['ETag']}")


def cmd_get(args) -> None:
    s3 = client()
    kwargs = {"Bucket": VOLUME_ID, "Key": args.key}

    # Explicit window rather than a suffix range, so the logged range is auditable.
    if args.tail or args.offset or args.length:
        size = s3.head_object(Bucket=VOLUME_ID, Key=args.key)["ContentLength"]
        if args.tail:
            start, end = size - args.tail, size - 1
        else:
            start = args.offset or 0
            end = start + args.length - 1 if args.length else size - 1
        if start < 0:
            sys.exit(f"range starts before the object (size {size:,})")
        kwargs["Range"] = f"bytes={start}-{end}"
        print(f"range: bytes {start:,}-{end:,} of {size:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = s3.get_object(**kwargs)["Body"]
    written = 0
    with open(out, "wb") as fh:
        while chunk := body.read(8 << 20):
            fh.write(chunk)
            written += len(chunk)
            print(f"\r  {human(written)}", end="", flush=True)
    print(f"\nwrote {out} ({written:,} bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list objects on the volume (one level by default)")
    p_ls.add_argument("prefix", nargs="?", default="")
    p_ls.add_argument("--recursive", action="store_true", help="descend into prefixes")
    p_ls.add_argument("--limit", type=int, help="stop after N objects")
    p_ls.set_defaults(func=cmd_ls)

    p_stat = sub.add_parser("stat", help="size and mtime of one object")
    p_stat.add_argument("key")
    p_stat.set_defaults(func=cmd_stat)

    p_get = sub.add_parser("get", help="download an object, optionally a byte range")
    p_get.add_argument("key")
    p_get.add_argument("out")
    p_get.add_argument("--tail", type=int, help="download only the last N bytes")
    p_get.add_argument("--offset", type=int, help="start byte of a range read")
    p_get.add_argument("--length", type=int, help="number of bytes to read from --offset")
    p_get.set_defaults(func=cmd_get)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
