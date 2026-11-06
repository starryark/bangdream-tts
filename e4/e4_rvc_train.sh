#!/usr/bin/env bash
# E4 step 2 (local, Applio venv): per-voice RVC converters, E4_PLAN.md "Converter".
#
#   bash e4_rvc_train.sh data  ppp_kasumi pp_eve     # copy train clips + preprocess (CPU only)
#   bash e4_rvc_train.sh train ppp_kasumi pp_eve     # extract (rmvpe, contentvec) + train 200 ep + index
#
# Applio and its venv live in E:\bd_rvc because torch's DLL loading breaks on the ';' in
# E:\STEINS;GATEVoice. Models are logs/bd_<row>/bd_<row>_<epoch>e_<step>s.pth (every 50 epochs).
set -euo pipefail
STAGE=$1; shift
RVC=E:/bd_rvc
PY=$RVC/rvcenv/Scripts/python.exe
BD="E:/STEINS;GATEVoice/bangdream"
cd "$RVC/Applio"
for ROW in "$@"; do
  M=bd_$ROW
  if [ "$STAGE" = data ]; then
    mkdir -p "$RVC/data/$ROW"
    "$PY" - "$BD" "$ROW" "$RVC/data/$ROW" <<'EOF'
import json, shutil, sys
from pathlib import Path
bd, row, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
train = {json.loads(l)["id"] for l in (bd / "data" / "split.jsonl").open(encoding="utf-8")
         if json.loads(l)["row"] == row and json.loads(l)["split"] == "train"}
n = 0
for i in sorted(train):
    shutil.copy2(bd / "data" / "wav24" / row / f"{i}.wav", dst / f"{i}.wav")
    n += 1
print(f"[e4_rvc] {row}: {n} train clips copied")
EOF
    "$PY" core.py preprocess --model-name "$M" --dataset-path "$RVC/data/$ROW" --sample-rate 32000 \
      --cpu-cores 8 --cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode none
    echo "[e4_rvc] $ROW preprocessed: $(ls logs/$M/sliced_audios | wc -l) slices"
  elif [ "$STAGE" = train ]; then
    "$PY" core.py extract --model-name "$M" --f0-method rmvpe --cpu-cores 8 --gpu 0 --sample-rate 32000 \
      --embedder-model contentvec --include-mutes 2
    "$PY" core.py train --model-name "$M" --save-every-epoch 50 --save-only-latest --sample-rate 32000 \
      --total-epoch 200 --batch-size 8 --gpu 0 --vocoder HiFi-GAN --index-algorithm Auto
    ls logs/$M/*.pth logs/$M/*.index
  fi
done
