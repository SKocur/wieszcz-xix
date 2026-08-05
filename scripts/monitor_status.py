"""One-shot status sheet for a training run: pod state, metrics tail, throughput, ETA.

Every fact here outlives the pod: pod inventory comes from the RunPod REST API, run
metrics and the manifest from the network volume's S3 endpoint (the training loop
writes its CSVs straight onto the volume). No SSH involved, so the same command works
from any machine holding .env and answers the questions that matter between
checkpoints: is a GPU billing, is the loss moving, when does the run end.

    python scripts/monitor_status.py                  # newest run on the volume
    python scripts/monitor_status.py --run wieszcz_47m_2026-08-06_s1337
    python scripts/monitor_status.py --log 4000       # also tail the launch log

Exit code 1 marks states that need attention — a stalled run, a finished run with a
pod still billing, an interrupted run with no pod — so a cron line can alert on the
exit code alone.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

from volume_s3 import VOLUME_ID, client, load_env

REST = "https://rest.runpod.io/v1"
REPO_PREFIX = "wieszcz-xix/"
STALL_MINUTES = 25
IDLE_HOURS = 24


def list_pods() -> list[dict] | None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        return None
    req = urllib.request.Request(
        REST + "/pods", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if isinstance(data, dict):
        data = data.get("pods") or data.get("data") or []
    return data


def s3_list(s3, prefix: str) -> list[dict]:
    out = []
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=VOLUME_ID, Prefix=prefix, Delimiter="/")
    for page in pages:
        out.extend(page.get("Contents", []))
    return out


def s3_text(s3, key: str) -> str:
    body = s3.get_object(Bucket=VOLUME_ID, Key=key)["Body"].read()
    return body.decode("utf-8", errors="replace")


def local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run lineage; default: newest *_train.csv on the volume")
    ap.add_argument("--log", type=int, metavar="BYTES",
                    help="also tail this many bytes of the newest train_*.out")
    args = ap.parse_args()

    load_env()
    problems: list[str] = []

    print("== pods ==")
    pods = list_pods()
    running: list[dict] = []
    if pods is None:
        print("RUNPOD_API_KEY not set; skipping")
    elif not pods:
        print("none")
    else:
        for p in pods:
            gpu = ((p.get("machine") or {}).get("gpuTypeId")
                   or ",".join(p.get("gpuTypeIds") or []) or "?")
            cost = p.get("costPerHr")
            cost_s = f"${float(cost):.2f}/h" if cost is not None else "?/h"
            print(f"{p.get('id', '?')}  {p.get('name', '?')}  "
                  f"{p.get('desiredStatus', '?')}  {gpu}  {cost_s}")
            if p.get("desiredStatus") == "RUNNING":
                running.append(p)

    s3 = client()
    metrics = s3_list(s3, REPO_PREFIX + "metrics/")
    trains = [o for o in metrics if o["Key"].endswith("_train.csv")]
    if not trains:
        print("\nno runs on the volume yet")
        if running:
            print("\n== attention ==\npod RUNNING with no run metrics at all")
            sys.exit(1)
        return

    if args.run:
        want = f"{REPO_PREFIX}metrics/{args.run}_train.csv"
        matches = [o for o in trains if o["Key"] == want]
        if not matches:
            sys.exit(f"no {want} on the volume")
        train_obj = matches[0]
    else:
        train_obj = max(trains, key=lambda o: o["LastModified"])
    run = train_obj["Key"].rsplit("/", 1)[1][: -len("_train.csv")]

    rows = list(csv.DictReader(io.StringIO(s3_text(s3, train_obj["Key"]))))
    if not rows:
        sys.exit(f"{run}_train.csv is empty")
    last = rows[-1]
    now = time.time()
    step = int(last["step"])
    age_min = (now - float(last["unix_ts"])) / 60

    max_steps = n_params = device = started = None
    try:
        man = json.loads(s3_text(s3, f"{REPO_PREFIX}metrics/{run}_manifest.json"))
        max_steps = man["config_values"]["max_steps"]
        n_params = man.get("n_params_millions")
        device = (man.get("env") or {}).get("device_name")
        started = man.get("started_unix")
    except Exception as e:
        print(f"(manifest unreadable: {e})")

    val_s = ""
    try:
        vrows = list(csv.DictReader(
            io.StringIO(s3_text(s3, f"{REPO_PREFIX}metrics/{run}_val.csv"))))
        if vrows:
            val_s = f" | val {float(vrows[-1]['val_loss']):.4f} @ step {int(vrows[-1]['step']):,}"
    except Exception:
        pass

    k = min(20, len(rows) - 1)
    sps = 0.0
    if k >= 1:
        dt = float(last["unix_ts"]) - float(rows[-1 - k]["unix_ts"])
        ds = step - int(rows[-1 - k]["step"])
        sps = ds / dt if dt > 0 else 0.0

    print(f"\n== run {run} ==")
    meta_bits = []
    if n_params:
        meta_bits.append(f"params {n_params:.1f}M")
    if device:
        meta_bits.append(device)
    if started:
        meta_bits.append(f"started {local(started)}")
    if meta_bits:
        print(" | ".join(meta_bits))
    prog = f" / {max_steps:,} ({step / (max_steps - 1) * 100:.1f}%)" if max_steps else ""
    print(f"step   {step:,}{prog}")
    print(f"loss   {float(last['loss']):.4f} | grad_norm {last['grad_norm']}"
          f" | lr x{last['lr_mult']}{val_s}")
    print(f"speed  {float(last['tok_s']) / 1e3:,.0f}k tok/s | {sps:.2f} steps/s"
          f" | last row {age_min:.1f} min ago")
    if max_steps and sps > 0 and step < max_steps - 1:
        eta_s = (max_steps - 1 - step) / sps
        line = f"eta    {eta_s / 3600:.1f} h -> {local(now + eta_s)}"
        if running and running[0].get("costPerHr") is not None:
            line += f" | ~${eta_s / 3600 * float(running[0]['costPerHr']):.0f} left"
        print(line)

    ckpts = s3_list(s3, f"{REPO_PREFIX}checkpoints/{run}/")
    done = any(o["Key"].endswith("/final.pt") for o in ckpts)
    print(f"\n== checkpoints ({run}) ==")
    if ckpts:
        for o in ckpts:
            print(f"{o['Size']:>14,}  {o['LastModified']:%Y-%m-%d %H:%M}  "
                  f"{o['Key'].rsplit('/', 1)[1]}")
    else:
        print("none yet")

    if age_min > IDLE_HOURS * 60:
        print(f"\n(idle: newest metrics are {age_min / 1440:.1f} days old — no active run)")
        if running:
            problems.append("a pod is RUNNING but the newest metrics are stale for "
                            "days — likely billing for nothing")
    else:
        if done and running:
            problems.append("final.pt present but a pod is RUNNING — if no next rung "
                            "is training, terminate it (runpod_train.sh --down)")
        if not done and running and age_min > STALL_MINUTES:
            problems.append(f"no new metrics for {age_min:.0f} min with a RUNNING pod "
                            "— inspect the pod (pgrep / nvidia-smi / tail train_*.out)")
        if not done and not running and age_min > STALL_MINUTES:
            problems.append("run incomplete and no pod is up — training interrupted; "
                            "escalate, never rent a GPU autonomously")

    if args.log:
        outs = [o for o in s3_list(s3, REPO_PREFIX)
                if o["Key"].rsplit("/", 1)[1].startswith("train_")
                and o["Key"].endswith(".out")]
        if outs:
            newest = max(outs, key=lambda o: o["LastModified"])
            start = max(0, newest["Size"] - args.log)
            body = s3.get_object(Bucket=VOLUME_ID, Key=newest["Key"],
                                 Range=f"bytes={start}-")["Body"].read()
            print(f"\n== tail {newest['Key']} ==")
            print(body.decode("utf-8", errors="replace"))
        else:
            print("\n(no train_*.out on the volume)")

    if problems:
        print("\n== attention ==")
        for p in problems:
            print(f"- {p}")
        sys.exit(1)
    print("\nhealthy")


if __name__ == "__main__":
    main()
