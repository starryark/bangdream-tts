#!/usr/bin/env bash
# E5 step 3 (local): sources -> conversion -> controls -> scoring -> report + listening sets, for a batch
# of voices whose epoch-200 converters exist (E5_PLAN.md §3-4).
#
#   bash e5_rows.sh <tag> <row>[,<row>...]        e.g.  bash e5_rows.sh kasumi_eve ppp_kasumi,pp_eve
#
# Shares the GPU with the converter queue: creates e5/PAUSE (the queue waits before its next voice),
# waits until no Applio extract/train process is running, does its work, then removes PAUSE.
set -euo pipefail
TAG=$1; ROWS=$2
E5="E:/STEINS;GATEVoice/bangdream/e5"
E4="E:/STEINS;GATEVoice/bangdream/e4"
TTS="E:/STEINS;GATEVoice/kurisu_qwen3tts/finetune/.venv-tts/Scripts/python.exe"
RVC=E:/bd_rvc/rvcenv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 E4_VC="$E5/vc" E4_ROWS="$ROWS" E5_ROWS="$ROWS"
touch "$E5/PAUSE"
trap 'rm -f "$E5/PAUSE"' EXIT
busy() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'core.py (preprocess|extract|train)' }).Count"; }
idle() {   # the queue is between voices (its last line is not a "start") and no Applio job is running
  local last; last=$(grep "\[e5_queue\]" "$E5/e5_rvc_queue.log" | tail -1)
  [[ "$last" != *" start "* ]] && [ "$(busy | tr -d '\r')" = "0" ]
}
[ "${E5_NOWAIT:-0}" = 1 ] || until idle; do sleep 60; done   # E5_NOWAIT=1: share the GPU with a running converter
echo "[e5_rows] $TAG start $(date +%T)"
cd "$E5" && "$TTS" -I e5_sources.py
cd E:/bd_rvc/Applio
"$RVC" "$E4/e4_convert.py" 200 2>&1 | grep "\[e4_convert\]"
E4_NOSHIFT=1 "$RVC" "$E4/e4_convert.py" 200 2>&1 | grep "\[e4_convert\]"
"$RVC" "$E4/e4_shift.py"
cd "$E5" && E4_TAG="_$TAG" "$TTS" -I "$E4/e4_score.py" 200 shift 200_noshift source > "$E5/score_$TAG.log" 2>&1
grep -c "e4_score" "$E5/score_$TAG.log"
"$TTS" -I e5_collect.py listen
echo "[e5_rows] $TAG done $(date +%T)"
