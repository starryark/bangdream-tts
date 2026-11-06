#!/usr/bin/env bash
# E0 pre-flight, full run: ASR -> split/vectors/probe -> floor renders -> scoring.
# Stops at the first failing step; each step resumes or skips finished work.
set -euo pipefail
cd "$(dirname "$0")"
PY="E:/STEINS;GATEVoice/kurisu_qwen3tts/finetune/.venv-tts/Scripts/python.exe"
LOG="../logs"
mkdir -p "$LOG"
unset BD_DATA BD_OUT
for step in b02_asr b03_vectors b04_floor b05_score; do
  echo "[run_e0] $(date +%H:%M:%S) start $step"
  "$PY" -I "$step.py" > "$LOG/$step.log" 2>&1 || { echo "[run_e0] $(date +%H:%M:%S) FAILED $step (see logs/$step.log)"; exit 1; }
  echo "[run_e0] $(date +%H:%M:%S) done $step"
done
echo "[run_e0] $(date +%H:%M:%S) ALL DONE"
