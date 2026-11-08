#!/usr/bin/env bash
# E6 (pod side): train RVC converters with the E5 recipe (Applio 59c1131) for the rows given as arguments.
# Started by e6/rvc_pod.sh upload, from /workspace/rvc. The user chose a rented pod (2026-11-07) after three
# local GPU resets. The recipe is the one in e6_rvc_queue.sh: 32 kHz, preprocess 3.0 s chunks, rmvpe +
# contentvec, 200 epochs, batch 8, HiFi-GAN. Torch 2.11.0+cu128, as in the local Applio venv. The rows train
# side by side on one GPU.
# Markers in /workspace/rvc/out: done_<row> | failed_<row> | INSTALL_FAILED, then RVC_DONE.
set -uo pipefail
R=/workspace/rvc
A=$R/applio
D=$R/data
O=$R/out
mkdir -p "$O" "$A/logs"
cd "$A"
echo "[rvc_pod] install $(date +%T)"
pip install -q --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
  || { echo "[rvc_pod] torch install failed"; touch "$O/INSTALL_FAILED" "$O/RVC_DONE"; exit 1; }
pip uninstall -y -q torchvision || true     # the image's torchvision is built for its torch 2.8; Applio doesn't use it
# everything else in requirements-engine.txt except torch/torchaudio (above) and the Windows-only lines
# (pip's markers skip those anyway; onnxruntime's line says != "win32" and must stay)
grep -vE '^(torch|torchaudio)==|^#|^[[:space:]]*$|sys_platform == "win32"' requirements-engine.txt > /tmp/req.txt
pip install -q --no-cache-dir -r /tmp/req.txt \
  || { echo "[rvc_pod] requirements install failed"; touch "$O/INSTALL_FAILED" "$O/RVC_DONE"; exit 1; }
python -c "import torch, torchaudio, faiss, librosa, transformers, onnxruntime, soundfile; print('[rvc_pod] torch', torch.__version__, torchaudio.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'onnxruntime', onnxruntime.__version__)" \
  || { touch "$O/INSTALL_FAILED" "$O/RVC_DONE"; exit 1; }
N=$(( $(nproc) / $# )); [ "$N" -gt 8 ] && N=8; [ "$N" -lt 2 ] && N=2
train_row() {
  local ROW=$1 M=bd_$1
  echo "[rvc_pod] $ROW start $(date +%T): $(ls "$D/$ROW" | wc -l) clips, $N cpu cores"
  if python core.py preprocess --model-name "$M" --dataset-path "$D/$ROW" --sample-rate 32000 \
       --cpu-cores "$N" --cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode none \
     && python core.py extract --model-name "$M" --f0-method rmvpe --cpu-cores "$N" --gpu 0 --sample-rate 32000 \
       --embedder-model contentvec --include-mutes 2 \
     && python core.py train --model-name "$M" --save-every-epoch 50 --save-only-latest --sample-rate 32000 \
       --total-epoch 200 --batch-size 8 --gpu 0 --vocoder HiFi-GAN --index-algorithm Auto \
     && ls "logs/$M/${M}_200e_"*s.pth > /dev/null 2>&1; then
    echo "[rvc_pod] $ROW done $(date +%T)"; touch "$O/done_$ROW"
  else
    echo "[rvc_pod] $ROW FAILED $(date +%T)"; touch "$O/failed_$ROW"
  fi
}
for ROW in "$@"; do
  train_row "$ROW" > "$O/$ROW.log" 2>&1 &
  sleep 20      # stagger the two preprocess/extract steps
done
wait
echo "[rvc_pod] all finished $(date +%T)"
touch "$O/RVC_DONE"
