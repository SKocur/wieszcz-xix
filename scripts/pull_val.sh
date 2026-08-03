#!/usr/bin/env bash
# Pull the held-out validation split off the network volume, so src/eval_val.py can run
# locally. train.py holds out the last 1% of tokens.bin, and the 5.40B-token file the 350M
# trained on lives only on the volume; the local copy is an older, smaller build.
#
# A network volume is only reachable from inside a pod, so the cheapest CPU pod streams
# the tail out and is then terminated. Superseded by scripts/volume_s3.py.
#
#   ./scripts/pull_val.sh          # plan only, creates nothing
#   ./scripts/pull_val.sh --go     # create pod, copy the split down, terminate pod
set -euo pipefail

API="https://rest.runpod.io/v1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATACENTER="EU-RO-1"
VOLUME_NAME="wieszcz-xix"
POD_NAME="wieszcz-val"
CPU_FLAVOR="cpu5c"
CPU_VCPU=2               # one sequential read, so the minimum
CONTAINER_DISK_GB=10
MOUNT="/workspace"
PREP_IMAGE="python:3.11-slim"

REMOTE_TOKENS="$MOUNT/data/clean/tokens.bin"
OUT="$REPO_ROOT/data/clean/val_tail.bin"
MANIFEST="$REPO_ROOT/data/clean/val_tail.json"

VAL_FRACTION="0.01"      # must match train.py: n_val = max(block+1, int(len(data)*0.01))

GO=0; [ "${1:-}" = "--go" ] && GO=1
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

set +u; . "$REPO_ROOT/.env"; set -u
[ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY not set in .env"
echo "RUNPOD_API_KEY present (len=${#RUNPOD_API_KEY})"
api GET /pods >/dev/null; [ "$HTTP" = "200" ] || die "auth/API check failed (HTTP $HTTP)"

VOLUME_ID="$(api GET /networkvolumes | python3 -c "
import sys,json
for v in json.load(sys.stdin):
    if v.get('name')=='$VOLUME_NAME': print(v['id']); break
")"
[ -n "$VOLUME_ID" ] || die "network volume '$VOLUME_NAME' not found"
echo "volume: $VOLUME_ID ($VOLUME_NAME @ $DATACENTER)"

if [ "$GO" != 1 ]; then
  cat <<EOF

plan:
  pod    : CPU $CPU_FLAVOR x${CPU_VCPU} vCPU, $PREP_IMAGE, volume $VOLUME_ID at $MOUNT
  source : $REMOTE_TOKENS  (last ${VAL_FRACTION} of tokens = the training-time val split)
  output : $OUT (+ $MANIFEST)
  after  : pod terminated, volume untouched (read-only operation)

(dry run: pass --go to actually do it)
EOF
  exit 0
fi

# A forgotten CPU pod bills quietly forever, so kill leftovers and trap this one.
for pid in $(api GET /pods | python3 -c "
import sys,json
for p in json.load(sys.stdin):
    if p.get('name')=='$POD_NAME': print(p['id'])
"); do echo "terminating stale pod $pid"; api DELETE "/pods/$pid" >/dev/null; done

POD_ID=""
cleanup(){ [ -n "$POD_ID" ] && { echo "terminating pod $POD_ID"; api DELETE "/pods/$POD_ID" >/dev/null || true; }; }
trap cleanup EXIT

export CPU_FLAVOR CPU_VCPU PREP_IMAGE DATACENTER VOLUME_ID MOUNT CONTAINER_DISK_GB POD_NAME
pod_body="$(python3 <<'PY'
import json, os
# python:3.11-slim ships no sshd; install one and seed authorized_keys from PUBLIC_KEY.
start = (
    "set -e; apt-get update -qq; "
    "DEBIAN_FRONTEND=noninteractive apt-get install -yqq openssh-server >/dev/null; "
    "mkdir -p /run/sshd /root/.ssh; "
    "printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; "
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
    "exec /usr/sbin/sshd -D -e"
)
print(json.dumps({
  'name': os.environ['POD_NAME'],
  'computeType': 'CPU',
  'cloudType': 'SECURE',
  'cpuFlavorIds': [os.environ['CPU_FLAVOR']],
  'vcpuCount': int(os.environ['CPU_VCPU']),
  'imageName': os.environ['PREP_IMAGE'],
  'dataCenterIds': [os.environ['DATACENTER']],
  'networkVolumeId': os.environ['VOLUME_ID'],
  'volumeMountPath': os.environ['MOUNT'],
  'containerDiskInGb': int(os.environ['CONTAINER_DISK_GB']),
  'ports': ['22/tcp'],
  'supportPublicIp': True,
  'dockerStartCmd': ['bash','-lc', start],
}))
PY
)"
resp="$(api POST /pods "$pod_body")"
is2xx || die "pod create failed (HTTP $HTTP): $resp"
POD_ID="$(printf '%s' "$resp" | jget "['id']")"
echo "created pod $POD_ID, waiting for RUNNING + SSH..."

SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 60); do
  sleep 10
  read -r STATUS SSH_HOST SSH_PORT < <(api GET "/pods/$POD_ID" | python3 -c "
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

SSHP=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@$SSH_HOST")
echo "waiting for sshd (installed by the start command, so it lags the port mapping)..."
ok=0
for j in $(seq 1 30); do
  "${SSHP[@]}" -o ConnectTimeout=8 -o BatchMode=yes true 2>/dev/null && { ok=1; break; }
  sleep 10
done
[ "$ok" = 1 ] || die "sshd never came up"

# Recomputed from the file, so a corpus that grew since the run cannot shift the window.
read -r TOTAL_BYTES VAL_BYTES VAL_TOKENS TOTAL_TOKENS < <("${SSHP[@]}" bash -lc "'
b=\$(stat -c %s $REMOTE_TOKENS)
python3 - \$b <<PY
import sys
b=int(sys.argv[1]); t=b//2
n=max(1025, int(t*$VAL_FRACTION))
print(b, n*2, n, t)
PY
'")
echo "tokens.bin: $TOTAL_BYTES B = $TOTAL_TOKENS tokens | val split = $VAL_TOKENS tokens ($VAL_BYTES B)"

mkdir -p "$(dirname "$OUT")"
echo "streaming the split down..."
"${SSHP[@]}" "tail -c $VAL_BYTES $REMOTE_TOKENS" > "$OUT"

LOCAL_BYTES="$(stat -f %z "$OUT")"
[ "$LOCAL_BYTES" = "$VAL_BYTES" ] || die "size mismatch: got $LOCAL_BYTES, expected $VAL_BYTES"
echo "verifying checksum end to end..."
REMOTE_SHA="$("${SSHP[@]}" "tail -c $VAL_BYTES $REMOTE_TOKENS | sha256sum | cut -d' ' -f1")"
LOCAL_SHA="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
[ "$REMOTE_SHA" = "$LOCAL_SHA" ] || die "sha256 mismatch (remote $REMOTE_SHA / local $LOCAL_SHA)"
echo "sha256 ok: $LOCAL_SHA"

python3 - "$MANIFEST" "$TOTAL_BYTES" "$TOTAL_TOKENS" "$VAL_BYTES" "$VAL_TOKENS" "$LOCAL_SHA" <<'PY'
import json, sys
p, tb, tt, vb, vt, sha = sys.argv[1:]
json.dump({
    "source": "runpod network volume wieszcz-xix:/workspace/data/clean/tokens.bin",
    "source_bytes": int(tb), "source_tokens": int(tt),
    "val_fraction": 0.01, "val_bytes": int(vb), "val_tokens": int(vt),
    "sha256": sha,
    "note": "last 1% of tokens.bin, the held-out split train.py used for the 350M/100M runs",
}, open(p, "w"), indent=2)
PY
echo "wrote $OUT ($LOCAL_BYTES B) + $MANIFEST"
