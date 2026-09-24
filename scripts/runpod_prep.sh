#!/usr/bin/env bash
# CPU stage: a throwaway pod pulls the clean corpus from the Hetzner VPS, tokenizes it and
# writes data/clean/tokens.bin onto the network volume, then terminates. The volume persists
# for scripts/runpod_train.sh.
#
#   ./scripts/runpod_prep.sh            # validate + print the plan, create nothing
#   ./scripts/runpod_prep.sh --go       # create the volume + pod and run prep
#
# Prerequisites:
#   - RUNPOD_API_KEY in .env (sourced here; the value is never printed).
#   - Workstation SSH public key registered on the RunPod account, so it reaches the pod.
#   - The Hetzner VPS reachable at $VPS with key $VPS_KEY.
#
# Why a CPU pod, and why the corpus is streamed rather than copied: docs/data-preparation.md.
set -euo pipefail

# ----------------------------- config ----------------------------------------------
API="https://rest.runpod.io/v1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATACENTER="EU-RO-1"            # region-locked volume; keep in sync with runpod_train.sh
VOLUME_NAME="wieszcz-xix"
VOLUME_SIZE_GB=100             # ~17 GB corpus + ~10 GB tokens.bin + checkpoints

CPU_FLAVOR="cpu5c"             # Compute-Optimized gen5
CPU_VCPU=16                    # encode_batch parallelizes across cores (Rayon), ~10x over 2 vCPU
CONTAINER_DISK_GB=30
MOUNT="/workspace"             # volume mount point inside the pod
PREP_IMAGE="python:3.11-slim"

# The box holding the corpus and the repo. Set these for your own host; the one this ran
# against is decommissioned. VPS_KEY is copied to the rented pod so it can pull the corpus
# over a direct link, so use a key issued for that purpose rather than a general one.
VPS="${VPS:?set VPS, e.g. user@host}"
VPS_KEY="${VPS_KEY:?set VPS_KEY, path to the key used for this transfer only}"
VPS_CLEAN="${VPS_CLEAN:-/srv/wieszcz/data/clean}"   # 202k *.txt (excludes tokens.bin)
VPS_REPO="${VPS_REPO:-/srv/wieszcz/wieszcz-xix}"

# ----------------------------- helpers ---------------------------------------------
GO=0; [ "${1:-}" = "--go" ] && GO=1

die(){ echo "ERROR: $*" >&2; exit 1; }

# Create returns 200 or 204 depending on the endpoint, both with a body.
is2xx(){ case "$HTTP" in 2??) return 0;; *) return 1;; esac; }

# JSON extraction without a jq dependency.
jget(){ python3 -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

api(){  # api METHOD PATH [json-body]  -> prints body, sets $HTTP
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

# ----------------------------- 0. preflight ----------------------------------------
set -a; source "$REPO_ROOT/.env" 2>/dev/null || die "no .env at $REPO_ROOT"; set +a
[ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY not set in .env"
echo "RUNPOD_API_KEY present (len=${#RUNPOD_API_KEY})"

api GET /pods >/dev/null; [ "$HTTP" = "200" ] || die "auth/API check failed (HTTP $HTTP)"
echo "API reachable, auth OK."

if [ "${1:-}" = "--down" ]; then
  pid="$(api GET /pods | python3 -c "
import sys,json
d=json.load(sys.stdin); pods=d if isinstance(d,list) else d.get('pods',[])
for p in pods:
    if p.get('name')=='wieszcz-prep': print(p['id']); break
")"
  [ -n "$pid" ] || { echo "no wieszcz-prep pod running."; exit 0; }
  api DELETE "/pods/$pid" >/dev/null; echo "terminated $pid (volume + tokens.bin persist)."
  exit 0
fi

cat <<PLANEOF

PLAN
  volume : $VOLUME_NAME  ${VOLUME_SIZE_GB}GB  @ $DATACENTER
  pod    : CPU $CPU_FLAVOR x${CPU_VCPU} vCPU, image $PREP_IMAGE, mount $MOUNT
  source : $VPS:$VPS_CLEAN  (+ repo $VPS_REPO)
  output : $MOUNT/data/clean/tokens.bin  (persists on the volume)
PLANEOF

if [ "$GO" -ne 1 ]; then
  echo; echo "(dry run: pass --go to create the volume + pod and run tokenization)"; exit 0
fi

# ----------------------------- 1. network volume (idempotent) ----------------------
echo; echo "== network volume =="
vols="$(api GET /networkvolumes)"
VOLUME_ID="$(printf '%s' "$vols" | python3 -c "
import sys,json
for v in json.load(sys.stdin):
    if v.get('name')=='$VOLUME_NAME': print(v['id']); break
")"
if [ -n "$VOLUME_ID" ]; then
  echo "reusing existing volume $VOLUME_ID"
else
  body="$(python3 -c "import json;print(json.dumps({'name':'$VOLUME_NAME','size':$VOLUME_SIZE_GB,'dataCenterId':'$DATACENTER'}))")"
  resp="$(api POST /networkvolumes "$body")"
  is2xx || die "volume create failed (HTTP $HTTP): $resp"
  VOLUME_ID="$(printf '%s' "$resp" | jget "['id']")"
  echo "created volume $VOLUME_ID"
fi

# ----------------------------- 1b. clean up any prior prep pod ----------------------
# Volumes are reused by name, but pod create always makes a new pod.
echo; echo "== cleaning up any prior wieszcz-prep pod =="
old_ids="$(api GET /pods | python3 -c "
import sys,json
d=json.load(sys.stdin)
pods=d if isinstance(d,list) else d.get('pods',[])
for p in pods:
    if p.get('name')=='wieszcz-prep': print(p['id'])
")"
for pid in $old_ids; do echo "terminating old pod $pid"; api DELETE "/pods/$pid" >/dev/null; done

# ----------------------------- 2. CPU prep pod -------------------------------------
echo; echo "== CPU prep pod =="
# python:3.11-slim ships no sshd; the start command installs one and seeds authorized_keys
# from the PUBLIC_KEY RunPod injects. sshd -D also keeps the container alive.
# Quoted heredoc: $PUBLIC_KEY must survive unexpanded into the container's runtime env.
export CPU_FLAVOR CPU_VCPU PREP_IMAGE DATACENTER VOLUME_ID MOUNT CONTAINER_DISK_GB
pod_body="$(python3 <<'PY'
import json, os
start = (
    "set -e; "
    "apt-get update -qq; "
    "DEBIAN_FRONTEND=noninteractive apt-get install -yqq openssh-server >/dev/null; "
    "mkdir -p /run/sshd /root/.ssh; "
    "printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; "
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
    "exec /usr/sbin/sshd -D -e"
)
print(json.dumps({
  'name': 'wieszcz-prep',
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

# ----------------------------- 3. wait for RUNNING ---------------------------------
SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 60); do
  sleep 10
  d="$(api GET "/pods/$POD_ID")"
  # Status/port field names vary by pod shape; dump the first poll so the schema is visible.
  [ "$i" = 1 ] && { echo "--- pod object (first poll) ---"; printf '%s\n' "$d" | python3 -m json.tool 2>/dev/null | head -40; echo "---"; }
  read -r STATUS SSH_HOST SSH_PORT < <(printf '%s' "$d" | python3 -c "
import sys,json
d=json.load(sys.stdin)
status=d.get('desiredStatus') or d.get('lastStatus') or d.get('status') or ''
host=port=''
ports=d.get('portMappings') or (d.get('runtime') or {}).get('ports') or []
if isinstance(ports,dict):  # {'22/tcp': 12345}
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
[ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] || die "pod did not expose SSH in time (check the dumped pod object for the right field names)"

SSHP=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@$SSH_HOST")
echo "SSH: root@$SSH_HOST:$SSH_PORT"

# The port maps as soon as the container starts, but sshd only exists once apt finishes.
echo "waiting for sshd to accept connections..."
ssh_up=0
for j in $(seq 1 30); do
  if "${SSHP[@]}" -o ConnectTimeout=8 -o BatchMode=yes true 2>/dev/null; then ssh_up=1; echo "  sshd up (after ${j}0s-ish)"; break; fi
  sleep 10
done
[ "$ssh_up" = 1 ] || die "sshd never came up (check the pod's start command / PUBLIC_KEY injection)"

# ----------------------------- 4. drive prep on the pod ----------------------------
# The pod gets the VPS key so it can pull the corpus over the direct EU link. It is
# terminated at the end, so exposure is bounded.
#
# The corpus is 202k tiny *.txt. Landing them as files on the MooseFS volume runs at
# ~1.6 MB/s on metadata latency alone, so they are never written: the VPS tars them into
# one sorted stream and the pod pipes it through the tokenizer into a single tokens.bin.
echo; echo "== staging key + tokenizer on the pod =="
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$SSH_PORT" \
    "$VPS_KEY" "root@$SSH_HOST:/root/.ssh/vps_key"

# Mirrors train.build_token_cache (same tokenizer files, same <|endoftext|> append, uint16,
# sorted doc order) but reads a tar stream instead of globbing files.
"${SSHP[@]}" "cat > /workspace/stream_tokenize.py" <<'PY'
import sys, os, tarfile
import numpy as np
from tokenizers import ByteLevelBPETokenizer

TOKD = "/workspace/wieszcz-xix/tokenizer"
OUT = sys.argv[1]; TMP = OUT + ".tmp"
BATCH = 2000
CHUNK = 64 * 1024 * 1024                        # per-document writes stall the loop on MooseFS
                                                # latency (~0.25M tok/s); 64MB blocks hit ~1GB/s
tok = ByteLevelBPETokenizer(TOKD + "/vocab.json", TOKD + "/merges.txt")
EOT = tok.token_to_id("<|endoftext|>")

docs = toks = 0
texts = []
buf = bytearray()

def encode_flush(f):
    global docs, toks
    if not texts:
        return
    for enc in tok.encode_batch(texts):         # parallel across cores (Rayon)
        ids = enc.ids + ([EOT] if EOT is not None else [])
        buf.extend(np.asarray(ids, dtype=np.uint16).tobytes())
        docs += 1; toks += len(ids)
    texts.clear()
    if len(buf) >= CHUNK:
        f.write(buf); buf.clear()

with open(TMP, "wb") as f:
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:   # streaming, no seek
        for m in tar:
            if not m.isfile() or not m.name.endswith(".txt"):
                continue
            fh = tar.extractfile(m)
            if fh is None:
                continue
            texts.append(fh.read().decode("utf-8"))
            if len(texts) >= BATCH:
                encode_flush(f)
                if docs % 20000 < BATCH:
                    print(f"{docs} docs, {toks/1e6:.1f}M tokens", flush=True)
    encode_flush(f)
    if buf:
        f.write(buf); buf.clear()
os.replace(TMP, OUT)
print(f"DONE {docs} docs, {toks} tokens -> {OUT}", flush=True)
PY

# Driver: the VPS streams the sorted corpus tar, the pod tokenizes it.
"${SSHP[@]}" "cat > /workspace/stream_prep.sh" <<'DRV'
#!/usr/bin/env bash
set -uo pipefail
VPS="$1"; VPS_CLEAN="$2"; OUT="$3"; KEY=/root/.ssh/vps_key
log(){ echo "[stream $(date -u +%H:%M:%S)] $*"; }
log "streaming $VPS:$VPS_CLEAN -> $OUT (parallel encode_batch)"
# No LC_ALL=C: glibc collation is what fixed document order in the frozen corpus, and it
# differs from Python's sorted() in build_token_cache. See docs/data-preparation.md.
ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$VPS" \
  "cd '$VPS_CLEAN' && find . -maxdepth 1 -name '*.txt' ! -name '.*' | sort | tar cf - -T -" \
  | python3 /workspace/stream_tokenize.py "$OUT"
rc=${PIPESTATUS[1]}
log "pipeline done rc=$rc tokens.bin=$(ls -lh "$OUT" 2>/dev/null | awk '{print $5}')"
DRV

"${SSHP[@]}" bash -s <<REMOTE
set -euo pipefail
chmod 600 /root/.ssh/vps_key
echo '[pod] installing tools (tokenizers + numpy; no torch needed for streaming) ...'
apt-get update -qq && apt-get install -yqq openssh-client rsync ca-certificates >/dev/null
pip install -q --find-links $MOUNT/wheels tokenizers numpy
mkdir -p $MOUNT/data/clean $MOUNT/wieszcz-xix
ln -sfn $MOUNT/data $MOUNT/wieszcz-xix/data   # so the GPU stage resolves data/clean/tokens.bin
echo '[pod] pulling repo (code + tokenizer, no data) from VPS ...'
rsync -az --exclude data --exclude .venv --exclude '*.pt' \
  -e "ssh -i /root/.ssh/vps_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  $VPS:$VPS_REPO/ $MOUNT/wieszcz-xix/
chmod +x /workspace/stream_prep.sh
echo '[pod] launching parallel streaming tokenize (detached, survives disconnect) ...'
setsid bash /workspace/stream_prep.sh "$VPS" "$VPS_CLEAN" "$MOUNT/data/clean/tokens.bin" </dev/null >/workspace/stream.log 2>&1 &
sleep 4; echo '[pod] launched; stream.log so far:'; cat /workspace/stream.log || true
REMOTE

# ----------------------------- 5. wait for tokens.bin -------------------------------
# tokens.bin is written under a .tmp name and os.replace'd, so its presence means done.
# The job is detached, so killing this poll loop does not stop tokenization.
echo; echo "== tokenizing (streamed + parallel), polling ... =="
done=0
for i in $(seq 1 80); do   # up to ~40 min
  sleep 30
  line="$("${SSHP[@]}" -o ConnectTimeout=10 'test -s /workspace/data/clean/tokens.bin && echo TOKENS_PRESENT; tail -1 /workspace/stream.log 2>/dev/null' 2>/dev/null || true)"
  echo "  [$i] $line"
  printf '%s' "$line" | grep -q TOKENS_PRESENT && { done=1; break; }
  printf '%s' "$line" | grep -q 'rc=[1-9]' && die "tokenizer failed; see /workspace/stream.log on the pod (left running)"
done
[ "$done" = 1 ] || die "tokenization not done in time; the job may still run, check /workspace/stream.log"

echo; echo "== verify =="
"${SSHP[@]}" "ls -lh $MOUNT/data/clean/tokens.bin"

echo; echo "== terminating prep pod (volume + tokens persist) =="
api DELETE "/pods/$POD_ID" >/dev/null; echo "pod $POD_ID terminated. volume $VOLUME_ID ready."
echo "next: ./scripts/runpod_train.sh --go   (mounts $VOLUME_ID, trains the 350M)"
