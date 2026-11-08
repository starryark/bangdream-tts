#!/usr/bin/env bash
# E6 local chain (Git Bash): every local step up to the listening sets, one after another.
#   bash e6/local_chain.sh > e6/logs/local_chain.log 2>&1
# Steps skip their own finished outputs, so a rerun resumes.
set -euo pipefail
BD="E:/STEINS;GATEVoice/bangdream"
E6="$BD/e6"
TTS="E:/STEINS;GATEVoice/kurisu_qwen3tts/finetune/.venv-tts/Scripts/python.exe"
RVC=E:/bd_rvc/rvcenv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
ROLES="mewtype_arale mewtype_nonoka millsage_hotaru ikka_yomogi"
cd "$E6"
step() { echo "=== $1 $(date +%T)"; }
# FROM=<step> skips the steps before it (e.g. FROM=rvc after a crash, so the bundle and probe are not rebuilt).
STEPS="onbuild asr rows probe vc_texts vc_src rvc"
skip=1; [ -z "${FROM:-}" ] && skip=0
want() { [ "$skip" = 0 ] && return 0; [ "$1" = "$FROM" ] && { skip=0; return 0; }; echo "=== $1 skipped (FROM=$FROM)"; return 1; }
want onbuild  && { step onbuild;  "$TTS" -I -X utf8 e6_onbuild.py --workers 8 | tail -n 32; }
want asr      && { step asr;      "$TTS" -I -X utf8 e6_asr.py | grep -v "^\[qwen-asr\]\|^\[anime-whisper\]" | tail -n 3; }
want rows     && { step rows;     "$TTS" -I -X utf8 e6_rows.py; }
want probe    && { step probe;    "$TTS" -I -X utf8 e6_probe.py; }
want vc_texts && { step vc_texts; [ -s vc/texts.json ] || "$TTS" -I -X utf8 e6_vc.py texts; }
want vc_src   && { step vc_src;   [ -s vc/sources.jsonl ] || "$TTS" -I -X utf8 e6_vc.py sources; }
want rvc      || { echo "unknown FROM=$FROM (steps: $STEPS)"; exit 1; }
step rvc;      bash e6_rvc_queue.sh $ROLES
for r in $ROLES; do [ ! -f "failed_$r" ] || { echo "converter failed: $r"; exit 1; }; done
step convert;  (cd E:/bd_rvc/Applio && E4_VC="$E6/vc" "$RVC" "$BD/e4/e4_convert.py" 200 2>&1 | grep "\[e4_convert\]")
step vc_score; "$TTS" -I -X utf8 e6_vc.py score
step listen;   "$TTS" -I -X utf8 e6_vc.py listen
echo "=== LOCAL_CHAIN_DONE $(date +%T)"
