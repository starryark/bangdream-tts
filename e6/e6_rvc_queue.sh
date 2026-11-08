#!/usr/bin/env bash
# E6 (local, Applio venv): RVC converters for the four roles Base can't reach (E6_PLAN.md §3.6).
#   bash e6_rvc_queue.sh mewtype_arale mewtype_nonoka millsage_hotaru ikka_yomogi
# E4/E5 recipe (e4/e4_rvc_train.sh): 32 kHz, preprocess 3.0 s chunks, rmvpe + contentvec, 200 epochs,
# batch 8, HiFi-GAN, index. Training clips = the row's E6a Japanese training rows (bundle/ja_rows.jsonl,
# source ja_on, split train; at most 300 by construction), copied from data_on/wav24.
# A row is done once its epoch-200 weights exist; failures leave e6/failed_<row>. While e6/PAUSE
# exists the queue waits before its next row (frees the GPU for other local jobs).
set -uo pipefail
RVC=E:/bd_rvc
PY=$RVC/rvcenv/Scripts/python.exe
BD="E:/STEINS;GATEVoice/bangdream"
E6="$BD/e6"
cd "$RVC/Applio"
for ROW in "$@"; do
  M=bd_$ROW
  if ls "logs/$M/${M}_200e_"*s.pth >/dev/null 2>&1; then echo "[e6_rvc] $ROW done earlier"; continue; fi
  while [ -f "$E6/PAUSE" ]; do sleep 60; done
  echo "[e6_rvc] $ROW start $(date +%T)"
  rm -rf "$RVC/data/$ROW" && mkdir -p "$RVC/data/$ROW"
  "$PY" - "$BD" "$ROW" "$RVC/data/$ROW" <<'EOF'
import json, shutil, sys
from pathlib import Path
bd, row, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
n = 0
for l in (bd / "e6" / "bundle" / "ja_rows.jsonl").open(encoding="utf-8"):
    r = json.loads(l)
    if r["speaker"] == row and r["source"] == "ja_on" and r["split"] == "train":
        src = bd / r["audio"].replace("/workspace/bd/", "")
        shutil.copy2(src, dst / src.name)
        n += 1
print(f"[e6_rvc] {row}: {n} training clips copied")
EOF
  if "$PY" core.py preprocess --model-name "$M" --dataset-path "$RVC/data/$ROW" --sample-rate 32000 \
       --cpu-cores 8 --cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode none \
     && "$PY" core.py extract --model-name "$M" --f0-method rmvpe --cpu-cores 8 --gpu 0 --sample-rate 32000 \
       --embedder-model contentvec --include-mutes 2 \
     && "$PY" core.py train --model-name "$M" --save-every-epoch 25 --save-only-latest --sample-rate 32000 \
       --total-epoch 200 --batch-size 8 --gpu 0 --vocoder HiFi-GAN --index-algorithm Auto \
     && ls "logs/$M/${M}_200e_"*s.pth >/dev/null 2>&1; then   # Applio can exit 0 after a CUDA failure (Hotaru, 2026-11-07)
    echo "[e6_rvc] $ROW done $(date +%T): $(ls logs/$M/*.pth | tr '\n' ' ')"
  else
    echo "[e6_rvc] $ROW FAILED $(date +%T)"; touch "$E6/failed_$ROW"
  fi
done
echo "[e6_rvc] queue empty $(date +%T)"
