#!/usr/bin/env bash
# E5 step 2 (local): train the converters listed in queue.txt one at a time with E4's recipe
# (../e4/e4_rvc_train.sh data + train). The list is re-read before each voice, so voices can be
# appended while it runs. A voice is done once its epoch-200 weights exist. Before each voice the
# queue waits while e5/PAUSE exists, which frees the GPU for other local jobs between converters.
set -uo pipefail
E5="E:/STEINS;GATEVoice/bangdream/e5"
E4="E:/STEINS;GATEVoice/bangdream/e4"
while true; do
  ROW=""
  for r in $(grep -v '^#' "$E5/queue.txt" | tr -d '\r'); do
    if ! ls "E:/bd_rvc/Applio/logs/bd_$r/bd_${r}_200e_"*s.pth >/dev/null 2>&1 && [ ! -f "$E5/failed_$r" ]; then ROW=$r; break; fi
  done
  [ -z "$ROW" ] && { echo "[e5_queue] queue empty $(date +%T)"; break; }
  while [ -f "$E5/PAUSE" ]; do sleep 60; done
  echo "[e5_queue] $ROW start $(date +%T)"
  if bash "$E4/e4_rvc_train.sh" data "$ROW" && bash "$E4/e4_rvc_train.sh" train "$ROW"; then
    echo "[e5_queue] $ROW done $(date +%T)"
  else
    echo "[e5_queue] $ROW FAILED $(date +%T)"; touch "$E5/failed_$ROW"
  fi
done
