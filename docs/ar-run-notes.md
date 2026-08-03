# Wieszcz-AR — notatki z uruchomienia treningu

## Stan na start (2026-07-23, noc)

- **Model:** 106.9M params, klasyczny autoregresyjny (decoder-only, next-token),
  GQA 12q/4kv, config `configs/wieszcz_ar_100m.json`.
- **Korpus:** 2 441 095 312 tokenów (`data/clean/tokens.bin`, 4,88 GB, uint16), z 71 668
  plików `.txt`. Retokenizacja po dociągnięciu korpusu z VPS — stary cache dyfuzyjny
  (1,85 mld) skasowany przed startem.
- **Trening:** na SPEED (RTX 3080) jako Scheduled Task `wieszcz-train-ar`
  (`C:\Users\skocu\wieszcz-train-ar.cmd` → `wsl.exe -u skocur -- ... python -u src/train.py
  --config configs/wieszcz_ar_100m.json > data/train_ar.log 2>&1`).
- **Tempo:** ~3,0 s/krok → 18 437 kroków ≈ ~13–15 h.

## Dlaczego 18 437 kroków = 1 epoka

- `tokeny_na_krok = batch 8 × block 1024 × grad_accum 16 = 131 072`.
- `train.py` odcina 1% na walidację: `n_val = 24 410 953` → `train_tokens = 2 416 684 359`.
- `max_steps = 2 416 684 359 / 131 072 = 18 437`.
- Świadomie 1 epoka na start — kolejne dokłada się przez `--resume` z podbitym `max_steps`
  (WSD ma płaską fazę, więc wydłużenie runu nie wymaga restartu). Do ~4 epok powtórzenia
  „za darmo" (Muennighoff 2023). 2,44 mld tokenów na 100M to ~24 tok/param — powyżej
  Chinchilli (optimum ~2 mld), model data-rich, świadomie niedowymiarowany pod inferencję.

## Checkpointy (gęste + rotacja)

- `ckpt_every = 500` → `checkpoints/step{N}.pt` co ~18 min (crash kosztuje ≤ ~18 min).
- `keep_last = 4` + `keep_every = 5000` — na dysku żyją 4 ostatnie (okno ~72 min) plus
  co-5000 kamień milowy, który przeżywa czystkę (ślad jakości / decay-eval). Rotacja w
  `prune_ckpts()` — dysk ograniczony do ~7 GB niezależnie od długości runu.
- `final.pt` na końcu. Każdy checkpoint = wagi + Muon + AdamW + step + cfg → wznawialny.
- Termometr jakości bez checkpointu: `val loss` w logu co `eval_every=250` (1% korpusu).

## Zdrowy start (potwierdzony)

```
Corpus: 2,441,095,312 tokens (24,410,953 val) | device: cuda | bf16 + FlashAttention
Model: 106.9M params | GQA 12q/4kv | Muon+AdamW
step 0  | loss 9.1727 | grad_norm 8.109 | lr x0.001
step 20 | loss 8.9678 | grad_norm 7.698 | lr x0.011
step 40 | loss 8.4854 | grad_norm 5.055 | lr x0.021
VRAM: 6.3 GB used / 6.5 GB reserved
```

- VRAM 6,3/6,5 GB (nvidia-smi ~7,7 GB z kontekstem CUDA) — mieści się z zapasem ~2,5 GB.
- GPU 92–99%, loss spada od 9,17, grad_norm maleje, `compile:true` działa, zero OOM/NaN.

## Sprawdzanie jakości w trakcie

- Sampler: `python src/sample.py --ckpt checkpoints/step<N>.pt --prompt "Rankiem, gdy"`.
- **Nie na GPU treningowym** (zostaje ~2,5 GB zapasu → ryzyko OOM). Bezpiecznie: ściągnąć
  checkpoint na Maca (`.venv`) albo sampling na CPU. Ściąganie: `scripts/pull_ckpt.sh` da
  się przerobić na `step<N>.pt` (dyfuzyjny wzorzec).
- Niuans WSD: checkpointy z fazy stabilnej są przy wysokim LR → surowy sample/loss lekko
  zaniża prawdziwą jakość. Dokładny odczyt = odgałęzienie z decayem (`--resume` + krótki zjazd).

## Trwałość (WAŻNE)

- Task `/sc once` — **nie wznowi się po restarcie**. SPEED musi zostać **zalogowany**
  (`Win+L`, nigdy *wylogowanie*), bez restartu — inaczej WSL VM padnie i trening z nim.
- Jak padnie: `--resume checkpoints/step<N>.pt` z ostatniego checkpointu.
- Reguły WSL do długich jobów + haczyk `pgrep -f` (samo-match monitora): notatka pamięci
  `speed-pc-ssh-access`.
