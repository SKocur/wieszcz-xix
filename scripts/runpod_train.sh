#!/usr/bin/env bash
# Create a GPU pod on the network volume, push the local tree, launch train.py detached.
# Does not tear the pod down: training runs for days. Use --down when final.pt is safe.
#
#   ./scripts/runpod_train.sh                          # print the plan, create nothing
#   ./scripts/runpod_train.sh --go                     # create the pod, push code, train
#   ./scripts/runpod_train.sh --config configs/x.json --gpu "NVIDIA ..." --seed 1337 --go
#   ./scripts/runpod_train.sh --gpus                   # GPU types + live availability/price
#   ./scripts/runpod_train.sh --down                   # terminate the pod (volume persists)
#
# Pod choice, resume scoping and the detached-launch pattern: docs/training.md.
set -euo pipefail

# ----------------------------- config ----------------------------------------------
API="https://rest.runpod.io/v1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATACENTER="EU-RO-1"           # network volumes are region-locked; must match the volume
VOLUME_NAME="wieszcz-xix"      # created by runpod_prep.sh, looked up by name below

GPU_TYPE="NVIDIA RTX PRO 4500 Blackwell"   # 32 GB, sm_120. One card type for the whole
GPU_COUNT=1                                # ladder; override with --gpu, prices via --gpus.
# cu1281, not cu1290: RunPod gates a cu12.9 image off a 12.8-driver host.
TRAIN_IMAGE="runpod/pytorch:1.1.0-cu1281-torch291-ubuntu2204"
CONTAINER_DISK_GB=40           # CUDA image + pip + torch.compile cache
MOUNT="/workspace"
CONFIG="configs/wieszcz_47m_6b7.json"
DATA_BIN="data/tokens_frozen_6.69B.bin"    # at the volume root, beside the repo copy
VAL_BIN="data/val_2026-08-03.bin"
SEED=1337                      # train.py's default; named here so it reaches the resume glob
FORCE=0                        # launch even if another train.py is already on the pod

# ----------------------------- helpers ---------------------------------------------
MODE="dry"
while [ $# -gt 0 ]; do
  case "$1" in
    --go)     MODE="go";;
    --down)   MODE="down";;
    --gpus)   MODE="gpus";;
    --config) CONFIG="$2"; shift;;
    --gpu)    GPU_TYPE="$2"; shift;;
    --seed)   SEED="$2"; shift;;
    --force)  FORCE=1;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
  shift
done

[ -f "$REPO_ROOT/$CONFIG" ] || { echo "ERROR: no such config: $CONFIG" >&2; exit 2; }
STEM="$(basename "$CONFIG" .json)"

die(){ echo "ERROR: $*" >&2; exit 1; }
is2xx(){ case "$HTTP" in 2??) return 0;; *) return 1;; esac; }
jget(){ python3 -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

api(){  # api METHOD PATH [json-body] -> prints body, sets $HTTP
  local method="$1" path="$2" body="${3:-}" out
  out="$(mktemp)"
  if [ -n "$body" ]; then
    HTTP="$(curl -s -o "$out" -w '%{http_code}' -X "$method" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
      -d "$body" "$API$path")"
  else
    HTTP="$(curl -s -o "$out" -w '%{http_code}' -X "$method" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" "$API$path")"
  fi
  cat "$out"; rm -f "$out"
}

# pod id, empty if none
find_train_pod(){
  api GET /pods | python3 -c "
import sys,json
d=json.load(sys.stdin); pods=d if isinstance(d,list) else d.get('pods',[])
for p in pods:
    if p.get('name')=='wieszcz-train': print(p['id']); break
"
}

# ----------------------------- 0. preflight ----------------------------------------
set -a; source "$REPO_ROOT/.env" 2>/dev/null || die "no .env at $REPO_ROOT"; set +a
[ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY not set in .env"
api GET /pods >/dev/null; [ "$HTTP" = "200" ] || die "auth/API check failed (HTTP $HTTP)"
echo "API reachable, auth OK."

# ----------------------------- --gpus: list GPU types + availability ---------------
if [ "$MODE" = "gpus" ]; then
  echo "GPU types via GraphQL (id | displayName | VRAM | secure price in $DATACENTER):"
  curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d "{\"query\":\"query{ gpuTypes{ id displayName memoryInGb secureCloud lowestPrice(input:{gpuCount:1,secureCloud:true,dataCenterId:\\\"$DATACENTER\\\"}){ uninterruptablePrice minimumBidPrice } } }\"}" \
    https://api.runpod.io/graphql | python3 -c "
import sys,json
d=json.load(sys.stdin)
gs=(d.get('data') or {}).get('gpuTypes') or []
for g in gs:
    lp=g.get('lowestPrice') or {}
    price=lp.get('uninterruptablePrice')
    avail='AVAIL @\$%s/hr'%price if price else 'none in $DATACENTER'
    print(f\"{g.get('memoryInGb'):>3}GB  {g.get('id'):32} {avail}\")
" 2>/dev/null || echo "(parse failed, check raw GraphQL output)"
  exit 0
fi

# ----------------------------- --down: terminate + exit ----------------------------
if [ "$MODE" = "down" ]; then
  pid="$(find_train_pod)"
  [ -n "$pid" ] || { echo "no wieszcz-train pod running."; exit 0; }
  api DELETE "/pods/$pid" >/dev/null; echo "terminated $pid (volume + checkpoints persist)."
  exit 0
fi

VOLUME_ID="$(api GET /networkvolumes | python3 -c "
import sys,json
for v in json.load(sys.stdin):
    if v.get('name')=='$VOLUME_NAME': print(v['id']); break
")"
[ -n "$VOLUME_ID" ] || die "volume '$VOLUME_NAME' not found; run runpod_prep.sh first"

cat <<PLANEOF

PLAN
  volume  : $VOLUME_NAME ($VOLUME_ID) @ $DATACENTER
  pod     : GPU ${GPU_COUNT}x $GPU_TYPE, image $TRAIN_IMAGE, mount $MOUNT
  code    : rsync local working tree -> pod (data/ + checkpoints/ excluded)
  train   : python src/train.py --config $CONFIG --seed $SEED
  resume  : newest step*.pt under checkpoints/${STEM}_20*_s${SEED}/  (scoped to THIS variant)
  output  : $MOUNT/wieszcz-xix/checkpoints/${STEM}_<date>_s${SEED}/  (persists on the volume)
PLANEOF

if [ "$MODE" != "go" ]; then
  echo; echo "(dry run: pass --go to create the $GPU_TYPE pod and launch training)"; exit 0
fi

# ----------------------------- 1. reuse-or-create the training pod ------------------
echo; echo "== training pod =="
POD_ID="$(find_train_pod)"
if [ -n "$POD_ID" ]; then
  echo "reusing existing wieszcz-train pod $POD_ID (will resume training on it)"
else
  pod_body="$(VOLUME_ID="$VOLUME_ID" GPU_TYPE="$GPU_TYPE" GPU_COUNT="$GPU_COUNT" \
             TRAIN_IMAGE="$TRAIN_IMAGE" DATACENTER="$DATACENTER" MOUNT="$MOUNT" \
             CONTAINER_DISK_GB="$CONTAINER_DISK_GB" python3 <<'PY'
import json, os
print(json.dumps({
  'name': 'wieszcz-train',
  'computeType': 'GPU',
  'cloudType': 'SECURE',
  'gpuTypeIds': [os.environ['GPU_TYPE']],
  'gpuCount': int(os.environ['GPU_COUNT']),
  'imageName': os.environ['TRAIN_IMAGE'],
  'dataCenterIds': [os.environ['DATACENTER']],
  'networkVolumeId': os.environ['VOLUME_ID'],
  'volumeMountPath': os.environ['MOUNT'],
  'containerDiskInGb': int(os.environ['CONTAINER_DISK_GB']),
  'ports': ['22/tcp'],
  'supportPublicIp': True,
}))
PY
)"
  resp="$(api POST /pods "$pod_body")"
  is2xx || die "pod create failed (HTTP $HTTP): $resp"
  POD_ID="$(printf '%s' "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)"
  [ -n "$POD_ID" ] || die "pod create returned no id (HTTP $HTTP). Response: $resp"
  echo "created pod $POD_ID"
fi

# ----------------------------- 2. wait for RUNNING + SSH ----------------------------
echo "waiting for RUNNING + SSH..."
SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 60); do
  sleep 10
  d="$(api GET "/pods/$POD_ID")"
  [ "$i" = 1 ] && { echo "--- pod object (first poll) ---"; printf '%s\n' "$d" | python3 -m json.tool 2>/dev/null | head -45; echo "---"; }
  read -r STATUS SSH_HOST SSH_PORT < <(printf '%s' "$d" | python3 -c "
import sys,json
d=json.load(sys.stdin)
status=d.get('desiredStatus') or d.get('lastStatus') or d.get('status') or ''
host=port=''
ports=d.get('portMappings') or (d.get('runtime') or {}).get('ports') or []
if isinstance(ports,dict):
    for k,v in ports.items():
        if k.startswith('22'): port=str(v)
    host=d.get('publicIp') or ''
else:
    for p in ports:
        if str(p.get('privatePort'))=='22' or str(p.get('internalPort'))=='22':
            host=p.get('ip') or p.get('publicIp') or ''; port=str(p.get('publicPort') or p.get('externalPort') or '')
print(status, host, port)
")
  echo "  [$i] status=$STATUS ssh=${SSH_HOST:-?}:${SSH_PORT:-?}"
  [ "$STATUS" = "RUNNING" ] && [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && break
done
[ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] || die "pod did not expose SSH in time"

# Without ServerAlive*, a dropped connection hangs the client on a half-open socket forever.
SSHP=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
      -p "$SSH_PORT" "root@$SSH_HOST")
echo "SSH: root@$SSH_HOST:$SSH_PORT"

echo "waiting for sshd to accept connections..."
ssh_up=0
for j in $(seq 1 30); do
  if "${SSHP[@]}" -o ConnectTimeout=8 -o BatchMode=yes true 2>/dev/null; then ssh_up=1; echo "  sshd up"; break; fi
  sleep 10
done
[ "$ssh_up" = 1 ] || die "sshd never came up"

# Account-level keys only cover the machine that launched the pod; the VPS monitor
# needs its own door for stall diagnosis (pgrep / nvidia-smi / tail train_*.out).
if [ -n "${MONITOR_PUBKEY:-}" ]; then
  "${SSHP[@]}" "mkdir -p /root/.ssh && grep -qF '$MONITOR_PUBKEY' /root/.ssh/authorized_keys 2>/dev/null || printf '%s\n' '$MONITOR_PUBKEY' >> /root/.ssh/authorized_keys"
  echo "monitor pubkey authorized on the pod"
fi

# ----------------------------- 3. push local code onto the pod ---------------------
# The volume's repo copy came from the VPS and predates local changes.
echo; echo "== syncing local code -> pod =="
# --no-owner --no-group: MooseFS rejects chown, and plain `rsync -a` then exits 23.
rsync -az --no-owner --no-group --exclude data --exclude .venv --exclude checkpoints --exclude '*.pt' \
  --exclude models --exclude output --exclude .git --exclude __pycache__ --exclude metrics \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  "$REPO_ROOT/" "root@$SSH_HOST:$MOUNT/wieszcz-xix/"

# ----------------------------- 4. launch training detached -------------------------
echo; echo "== launching training =="
"${SSHP[@]}" bash -s <<REMOTE
set -euo pipefail
cd $MOUNT/wieszcz-xix
mkdir -p checkpoints
# torch + CUDA come from the image; the rest is not in it. --find-links takes them off the
# volume when they are there and still falls back to PyPI when they are not, so a launch does
# not depend on PyPI being reachable at the moment the GPU has already started billing.
# Detached: run inline this prints nothing for minutes, and a dropped SSH takes it with it.
# The marker file carries the exit code, so the poll can tell started from succeeded.
rm -f /workspace/pip.done
setsid bash -c "pip install --find-links /workspace/wheels tokenizers numpy tqdm > /workspace/pip.log 2>&1; echo \\\$? > /workspace/pip.done" < /dev/null > /dev/null 2>&1 &
echo "installing deps (detached)..."
for i in \$(seq 1 60); do
  [ -f /workspace/pip.done ] && break
  sleep 10
done
[ -f /workspace/pip.done ] || { echo "FATAL: pip still not finished after 10 min"; tail -5 /workspace/pip.log; exit 1; }
PIP_RC=\$(cat /workspace/pip.done)
[ "\$PIP_RC" = "0" ] || { echo "FATAL: pip failed (rc=\$PIP_RC)"; tail -20 /workspace/pip.log; exit 1; }
python -c "import tokenizers, numpy, tqdm" || { echo "FATAL: deps import failed after install"; exit 1; }
echo "deps OK"

# check before the GPU starts billing
test -s $MOUNT/$DATA_BIN || { echo "FATAL: $MOUNT/$DATA_BIN missing"; exit 1; }
test -s $MOUNT/$VAL_BIN  || { echo "FATAL: $MOUNT/$VAL_BIN missing"; exit 1; }
echo "train bin: \$(ls -lh $MOUNT/$DATA_BIN | awk '{print \$5}')  val bin: \$(ls -lh $MOUNT/$VAL_BIN | awk '{print \$5}')"

# Pods are reused, so a second launch would put two runs on one GPU with interleaved checkpoints.
if pgrep -f 'src/train.py' >/dev/null 2>&1; then
  if [ "$FORCE" = 1 ]; then
    echo "WARNING: training already running on this pod; --force given, launching anyway"
  else
    echo "FATAL: src/train.py is already running on this pod:"; pgrep -af 'src/train.py'
    echo "Wait for it, or pass --force if you really mean to run two at once."
    exit 1
  fi
fi

# Newest step*.pt of this variant and seed only. A flat glob would hand a 349M
# checkpoint to a 47M run, failing on a shape mismatch after the pod is paid for.
export STEM="$STEM" SEED="$SEED"
LATEST=\$(python - <<'PY'
import glob, os, re
stem, seed = os.environ['STEM'], os.environ['SEED']
best, bn = '', -1
for c in glob.glob(f'checkpoints/{stem}_20*_s{seed}/step*.pt'):
    m = re.search(r'step(\d+)\.pt', c)
    if m and int(m.group(1)) > bn: bn, best = int(m.group(1)), c
print(best)
PY
)
if [ -n "\$LATEST" ]; then echo "resuming from \$LATEST"; RESUME="--resume \$LATEST"; else echo "fresh run (no checkpoints for \$STEM seed \$SEED)"; fi
if [ -z "\$LATEST" ]; then RESUME=""; else RESUME="--resume \$LATEST"; fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Hardware trace for the paper: utilization/VRAM/power once a minute, appended on
# the volume so pod restarts continue the same file.
pgrep -f 'nvidia-smi.*-l 60' >/dev/null 2>&1 || \
  setsid bash -c "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 60 >> gpu_${STEM}.csv 2>/dev/null" < /dev/null > /dev/null 2>&1 &
# setsid: training must outlive this SSH session.
# expandable_segments:True reclaims fragmented reserved memory; fp32 inductor buffers OOM'd without it.
# PYTHONUNBUFFERED=1: block-buffered stdout leaves train.out empty for ages under `tail -f`.
setsid bash -c "PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py --config $CONFIG --seed $SEED --data $MOUNT/$DATA_BIN --val-data $MOUNT/$VAL_BIN \$RESUME > train_${STEM}.out 2>&1" < /dev/null &
sleep 8
echo "--- first lines of train_${STEM}.out ---"; head -20 train_${STEM}.out || true
echo "--- training PID(s) ---"; pgrep -af 'src/train.py' || true
REMOTE

# ----------------------------- 5. hand back monitoring commands --------------------
cat <<DONE

== training launched on pod $POD_ID ==
watch it:     ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST 'tail -f /workspace/wieszcz-xix/train_${STEM}.out'
checkpoints:  ls on  /workspace/wieszcz-xix/checkpoints/  (persist on volume $VOLUME_ID)
pull final:   scp -P $SSH_PORT root@$SSH_HOST:/workspace/wieszcz-xix/checkpoints/${STEM}_*_s${SEED}/final.pt .
              (or: python scripts/volume_s3.py get wieszcz-xix/checkpoints/<run>/final.pt out.pt)
STOP paying:  ./scripts/runpod_train.sh --down     (terminate the pod when final.pt is safe)

$GPU_TYPE bills by the hour while this pod exists. Run --down when training is done.
Check throughput in the first minutes (metrics/${STEM}_*_s${SEED}_train.csv): if tokens/s per
dollar is worse than the 5090's 60.7k tok/s @ \$0.99/hr at 349M, move the run before it costs anything.
DONE
