"""Local inference playground for the wieszcz-xix base model.

A stdlib HTTP server plus an embedded HTML page. Generation goes through
train.GPT.generate and sample.load_model, so decoding matches src/sample.py.

The checkpoint is a base model, not instruction-tuned: it continues text rather than
answering questions.

    .venv/bin/python app.py                          # -> http://127.0.0.1:8000
    .venv/bin/python app.py --ckpt models/wieszcz-349m-2026-07-25/model.pt --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# src/ modules import each other flat, so it has to be on the path before they load.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
from tokenizers import ByteLevelBPETokenizer  # noqa: E402

from train import GPT, get_device  # noqa: E402
from sample import load_model  # noqa: E402

TOKENIZER_DIR = Path(__file__).parent / "tokenizer"
EOT = "<|endoftext|>"

# One model shared across requests; MPS's single command stream is not reentrant, so
# _LOCK serialises generation.
_MODEL: GPT | None = None
_TOK: ByteLevelBPETokenizer | None = None
_DEVICE = "cpu"
_EOT_ID: int | None = None
_BLOCK: int = 1024
_LOCK = threading.Lock()

MAX_NEW_CAP = 500


def boot(ckpt: str) -> None:
    global _MODEL, _TOK, _DEVICE, _EOT_ID, _BLOCK
    _DEVICE = get_device()
    print(f"[boot] device={_DEVICE}  ckpt={ckpt}", flush=True)
    _MODEL = load_model(ckpt, _DEVICE)
    _BLOCK = int(_MODEL.cfg["block_size"])
    n_params = sum(p.numel() for p in _MODEL.parameters())
    _TOK = ByteLevelBPETokenizer(str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))
    _EOT_ID = _TOK.token_to_id(EOT)
    print(f"[boot] ready: {n_params/1e6:.1f}M params, block_size={_BLOCK}", flush=True)


def generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> dict:
    assert _MODEL is not None and _TOK is not None
    ids = _TOK.encode(prompt).ids or [_EOT_ID]
    # Prompt + new tokens must fit the RoPE window.
    if len(ids) >= _BLOCK:
        ids = ids[-(_BLOCK - 1):]
    room = _BLOCK - len(ids)
    n_new = max(1, min(max_new_tokens, room, MAX_NEW_CAP))

    idx = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
    t0 = time.time()
    with _LOCK:
        out = _MODEL.generate(idx, n_new, temperature=temperature, top_k=top_k, top_p=top_p, eot_id=_EOT_ID)
    dt = time.time() - t0

    new_ids = out[0].tolist()[len(ids):]
    completion = _TOK.decode(new_ids)
    return {
        "prompt": _TOK.decode(ids),
        "completion": completion,
        "n_new": len(new_ids),
        "ms": round(dt * 1000),
        "tok_per_s": round(len(new_ids) / dt, 1) if dt > 0 else None,
    }


def stream_events(prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int):
    """Yield SSE payloads: one {"delta": text} per completed piece, then {"done": True}."""
    assert _MODEL is not None and _TOK is not None
    ids = _TOK.encode(prompt).ids or [_EOT_ID]
    if len(ids) >= _BLOCK:
        ids = ids[-(_BLOCK - 1):]
    room = _BLOCK - len(ids)
    n_new = max(1, min(max_new_tokens, room, MAX_NEW_CAP))
    idx = torch.tensor([ids], dtype=torch.long, device=_DEVICE)

    produced: list[int] = []
    emitted = 0
    t0 = time.time()
    with _LOCK:
        for tok_id in _MODEL.stream(idx, n_new, temperature=temperature, top_k=top_k, top_p=top_p, eot_id=_EOT_ID):
            if _EOT_ID is not None and tok_id == _EOT_ID:
                break
            produced.append(tok_id)
            # Byte-level BPE splits multibyte chars across tokens, which decode() renders
            # as U+FFFD. Hold back the replacement chars until the next token settles the
            # bytes; diffing by length instead emits U+FFFD and swallows the real char.
            stable = _TOK.decode(produced).rstrip("�")
            if len(stable) > emitted:
                yield {"delta": stable[emitted:]}
                emitted = len(stable)
    # If generation stopped mid-character, the tail is genuinely incomplete.
    tail = _TOK.decode(produced)[emitted:]
    if tail:
        yield {"delta": tail}
    dt = time.time() - t0
    yield {"done": True, "n_new": len(produced), "ms": round(dt * 1000),
           "tok_per_s": round(len(produced) / dt, 1) if dt > 0 else None}


PAGE = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wieszcz XIX — pracownia</title>
<style>
  :root {
    --ink: #2b2118; --ink-soft: #6b5d4a; --paper: #f4ecd8; --paper-edge: #e7dabb;
    --rule: #cbb98f; --accent: #7c3a24; --accent-soft: #a9683f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #ded0ac; color: var(--ink);
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    line-height: 1.6; display: flex; justify-content: center; padding: 2rem 1rem;
  }
  .sheet {
    width: 100%; max-width: 780px; background: var(--paper);
    border: 1px solid var(--paper-edge);
    box-shadow: 0 1px 0 #fff8e7 inset, 0 18px 40px rgba(43,33,24,.22);
    padding: 2.2rem 2.4rem 2.6rem;
  }
  header { text-align: center; border-bottom: 2px double var(--rule); padding-bottom: 1rem; margin-bottom: 1.4rem; }
  h1 { margin: 0; font-size: 2.1rem; letter-spacing: .06em; font-weight: 600; }
  .sub { margin: .35rem 0 0; color: var(--ink-soft); font-size: .92rem; font-style: italic; }
  .badge { display:inline-block; margin-top:.6rem; font-size:.72rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--accent); border:1px solid var(--accent-soft);
    padding:.15rem .55rem; border-radius:2px; font-style:normal; }
  label { display: block; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 .35rem; }
  textarea {
    width: 100%; min-height: 120px; resize: vertical; background: #fbf6e6;
    border: 1px solid var(--rule); color: var(--ink); padding: .8rem .9rem;
    font-family: inherit; font-size: 1.05rem; line-height: 1.55; border-radius: 3px;
  }
  textarea:focus { outline: 2px solid var(--accent-soft); outline-offset: 1px; }
  .controls { display: flex; flex-wrap: wrap; gap: 1.1rem 1.6rem; margin: 1.1rem 0; align-items: end; }
  .ctl { flex: 1 1 130px; min-width: 120px; }
  .ctl output { font-variant-numeric: tabular-nums; color: var(--ink); font-size:.9rem; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .row { display: flex; justify-content: space-between; align-items: baseline; gap:.5rem; }
  button {
    background: var(--accent); color: #f8efd8; border: none; padding: .7rem 1.5rem;
    font-family: inherit; font-size: 1rem; letter-spacing: .05em; cursor: pointer;
    border-radius: 3px; transition: background .15s;
  }
  button:hover { background: #8f4529; }
  button:disabled { background: #b7a685; cursor: default; }
  .hint { font-size: .78rem; color: var(--ink-soft); }
  .out { margin-top: 1.6rem; border-top: 2px double var(--rule); padding-top: 1.3rem; }
  .out h2 { font-size: .8rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-soft); margin:0 0 .6rem; }
  .manuscript { white-space: pre-wrap; font-size: 1.12rem; line-height: 1.7;
    background: #fbf6e6; border: 1px solid var(--rule); padding: 1.1rem 1.2rem; border-radius: 3px; min-height: 3rem; }
  .manuscript .given { color: var(--ink-soft); }
  .manuscript .cont { color: var(--ink); }
  .meta { margin-top: .6rem; font-size: .78rem; color: var(--ink-soft); font-variant-numeric: tabular-nums; }
  .err { color: var(--accent); font-style: italic; }
  .blink::after { content: "▍"; animation: b 1s steps(2) infinite; color: var(--accent-soft); }
  @keyframes b { 50% { opacity: 0; } }
</style>
</head>
<body>
  <main class="sheet">
    <header>
      <h1>Wieszcz&nbsp;XIX</h1>
      <p class="sub">pracownia promptowania — model uczony na polszczyźnie 1800–1918</p>
      <span class="badge">model bazowy · dopisuje ciąg dalszy</span>
    </header>

    <label for="prompt">Zacznij zdanie, a wieszcz je poprowadzi</label>
    <textarea id="prompt" placeholder="Rankiem, gdy słońce wzeszło nad Wisłą,">Rankiem, gdy </textarea>

    <div class="controls">
      <div class="ctl">
        <div class="row"><label for="temp">Temperatura</label><output id="temp_v">0,9</output></div>
        <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.9">
      </div>
      <div class="ctl">
        <div class="row"><label for="topp">Top-p</label><output id="topp_v">0,9</output></div>
        <input type="range" id="topp" min="0.1" max="1.0" step="0.05" value="0.9">
      </div>
      <div class="ctl">
        <div class="row"><label for="topk">Top-k</label><output id="topk_v">50</output></div>
        <input type="range" id="topk" min="0" max="200" step="10" value="50">
      </div>
      <div class="ctl">
        <div class="row"><label for="maxn">Długość</label><output id="maxn_v">200</output></div>
        <input type="range" id="maxn" min="20" max="500" step="20" value="200">
      </div>
    </div>

    <div class="row">
      <button id="go">Poprowadź pióro</button>
      <span class="hint">Ctrl/⌘ + Enter</span>
    </div>

    <section class="out">
      <h2>Rękopis</h2>
      <div id="manuscript" class="manuscript"><span class="given"></span></div>
      <div id="meta" class="meta"></div>
    </section>
  </main>

<script>
  const $ = (id) => document.getElementById(id);
  const pl = (v) => String(v).replace(".", ",");
  for (const [s, o, f] of [["temp","temp_v",pl],["topp","topp_v",pl],["topk","topk_v",String],["maxn","maxn_v",String]]) {
    $(s).addEventListener("input", () => $(o).textContent = f($(s).value));
  }

  async function run() {
    const prompt = $("prompt").value;
    const body = {
      prompt,
      temperature: parseFloat($("temp").value),
      top_p: parseFloat($("topp").value),
      top_k: parseInt($("topk").value, 10),
      max_new_tokens: parseInt($("maxn").value, 10),
    };
    $("go").disabled = true;
    const m = $("manuscript");
    m.innerHTML = '<span class="given">' + escapeHtml(prompt) + '</span><span id="cont" class="cont blink"></span>';
    const contEl = $("cont");
    $("meta").textContent = "wieszcz duma…";
    try {
      const r = await fetch("/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
      if (!r.ok) { const d = await r.json().catch(()=>({})); throw new Error(d.error || ("HTTP " + r.status)); }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "", meta = null, errored = null;
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        const parts = buf.split("\\n\\n");
        buf = parts.pop();
        for (const part of parts) {
          const line = part.replace(/^data: ?/, "").trim();
          if (!line) continue;
          const d = JSON.parse(line);
          if (d.delta !== undefined) contEl.textContent += d.delta;
          else if (d.done) meta = d;
          else if (d.error) errored = d.error;
        }
      }
      contEl.classList.remove("blink");
      if (errored) throw new Error(errored);
      if (meta) $("meta").textContent = `${meta.n_new} tokenów · ${meta.ms} ms` + (meta.tok_per_s ? ` · ${pl(meta.tok_per_s)} tok/s` : "");
      else $("meta").textContent = "";
    } catch (e) {
      const c = $("cont"); if (c) c.classList.remove("blink");
      $("meta").innerHTML = '<span class="err">Błąd: ' + escapeHtml(e.message) + '</span>';
    } finally {
      $("go").disabled = false;
    }
  }
  function escapeHtml(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
  $("go").addEventListener("click", run);
  $("prompt").addEventListener("keydown", (e) => { if ((e.metaKey||e.ctrlKey) && e.key === "Enter") run(); });
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path == "/stream":
            self._do_stream()
            return
        if self.path != "/generate":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            result = generate(
                prompt=str(req.get("prompt", "")),
                max_new_tokens=int(req.get("max_new_tokens", 200)),
                temperature=float(req.get("temperature", 0.9)),
                top_p=float(req.get("top_p", 0.9)),
                top_k=int(req.get("top_k", 50)),
            )
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"), "application/json; charset=utf-8")

    def _do_stream(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode("utf-8"), "application/json; charset=utf-8")
            return
        # HTTP/1.0: no Content-Length, so connection close marks the end of the stream.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for ev in stream_events(
                prompt=str(req.get("prompt", "")),
                max_new_tokens=int(req.get("max_new_tokens", 200)),
                temperature=float(req.get("temperature", 0.9)),
                top_p=float(req.get("top_p", 0.9)),
                top_k=int(req.get("top_k", 50)),
            ):
                self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client closed the tab mid-stream
        except Exception as e:
            try:
                self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models/wieszcz-349m-2026-07-25/model.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    boot(args.ckpt)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}  (Ctrl-C aby zakończyć)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] zamykam", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
