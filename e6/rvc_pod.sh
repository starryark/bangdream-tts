#!/usr/bin/env bash
# E6 (local, Git Bash): train the hotaru and yomogi RVC converters on a rented pod and bring them back. The user
# chose this on 2026-11-07, after the local GPU reset three times during RVC training.
#   bash e6/rvc_pod.sh all > e6/logs/rvc_pod.log 2>&1    # launch -> upload -> wait -> download -> stop
#   bash e6/rvc_pod.sh launch|upload|status|download|stop
# The pod gets only Applio's code, the four model files the recipe uses (hifi-gan f0G32k/f0D32k, rmvpe,
# contentvec) and the two rows' training clips (the same copy e6_rvc_queue.sh makes from bundle/ja_rows.jsonl).
# Back come the converters: weights, config and f0 folders (e4_convert's pitch target), without G_/D_
# optimizer states or audio. They go to E:/bd_rvc/Applio/logs/bd_<row>, and any local partial folder there is
# moved aside first. The converters stay local and are never published. The pod is deleted at the end.
set -euo pipefail
BD="E:/STEINS;GATEVoice/bangdream"
W="$BD/e6/work"
PJ="$W/pod_rvc.json"
APP=E:/bd_rvc/Applio
DATA=E:/bd_rvc/data
ROWS="millsage_hotaru ikka_yomogi"
KEY=~/.ssh/runpod_ed25519
# ip/port are read first, so ssh keeps the caller's stdin (tar streams); Windows python ends its lines with \r\n.
ssh_pod() { local ip port; read ip port < <(python -c "import json;d=json.load(open(r'$(cygpath -w "$PJ")'));print(d['ip'], d['port'])" | tr -d '\r'); ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$ip" "$@"; }

launch() {
  local id="" gpu resp
  rm -f "$PJ"
  for gpu in "NVIDIA GeForce RTX 4090" "NVIDIA L40S" "NVIDIA RTX A6000" "NVIDIA H100 80GB HBM3"; do
    body=$(GPU="$gpu" python - <<EOF
import json, os
print(json.dumps({"name": "bd-e6-rvc", "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "gpuTypeIds": [os.environ["GPU"]], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 60,
  "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": open(r"$(cygpath -w $KEY.pub)").read().strip()}}))
EOF
)
    resp=$(curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$body")
    id=$(echo "$resp" | python -c "import json,sys
try: print(json.load(sys.stdin).get('id') or '')
except Exception: print('')")
    [ -n "$id" ] && { echo "[rvc_pod] created $id on $gpu $(date +%T)"; break; }
    echo "[rvc_pod] $gpu: not created: ${resp:0:300}"
  done
  [ -n "$id" ] || { echo "[rvc_pod] launch failed on every GPU type"; return 1; }
  echo "{\"id\": \"$id\", \"gpu\": \"$gpu\"}" > "$PJ"
  for i in $(seq 1 60); do
    info=$(curl -s https://rest.runpod.io/v1/pods/$id -H "Authorization: Bearer $RUNPOD_API_KEY")
    ipport=$(echo "$info" | python -c "import json,sys; d=json.load(sys.stdin); ip=d.get('publicIp') or ''; pm=d.get('portMappings') or {}; print(ip, pm.get('22') or pm.get(22) or '')")
    set -- $ipport
    if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
      echo "{\"id\": \"$id\", \"gpu\": \"$gpu\", \"ip\": \"$1\", \"port\": $2}" > "$PJ"; echo "[rvc_pod] ssh root@$1 -p $2"
      for j in $(seq 1 120); do   # a fresh host may first pull the ~10 GB image
        err=$(ssh_pod true 2>&1) && { echo "[rvc_pod] ssh up after $((j * 10)) s"; return 0; }
        [ $((j % 12)) -eq 0 ] && echo "[rvc_pod] $(date +%T) waiting for ssh: ${err:0:200}"
        sleep 10
      done
      echo "[rvc_pod] ssh not answering after 20 min"; return 1
    fi
    sleep 10
  done
  echo "[rvc_pod] no ssh endpoint after 10 min: ${info:0:300}"; return 1
}

upload() {
  ssh_pod "mkdir -p /workspace/rvc/applio /workspace/rvc/data /workspace/rvc/out" || return 1
  local t0=$(date +%s)
  (cd "$APP" && tar -cf - --exclude='__pycache__' --exclude='rvc/models/pretraineds/refinegan' \
      --exclude='rvc/models/pretraineds/hifi-gan/f0?40k.pth' --exclude='rvc/models/pretraineds/hifi-gan/f0?48k.pth' \
      --exclude='rvc/models/predictors/fcpe.pt' --exclude='assets/zluda' --exclude='assets/audios' \
      core.py requirements-engine.txt rvc assets) | ssh_pod "tar --no-same-owner -xf - -C /workspace/rvc/applio" || return 1
  (cd "$DATA" && tar -cf - $ROWS) | ssh_pod "tar --no-same-owner -xf - -C /workspace/rvc/data" || return 1
  (cd "$DATA" && for r in $ROWS; do sha256sum "$r"/*; done) > "$W/rvc_data.sha256"   # checked on the pod below
  (cd "$W" && tar -cf - rvc_data.sha256) | ssh_pod "tar --no-same-owner -xf - -C /workspace/rvc/data" || return 1
  (cd "$BD/e6" && tar -cf - rvc_pod_run.sh) | ssh_pod "tar --no-same-owner -xf - -C /workspace/rvc" || return 1
  ssh_pod "cd /workspace/rvc && for r in $ROWS; do echo \$r \$(ls data/\$r | wc -l); done; (cd data && sha256sum -c --quiet rvc_data.sha256) && echo '[rvc_pod] clip checksums OK'; ls -la applio/rvc/models/pretraineds/hifi-gan applio/rvc/models/predictors applio/rvc/models/embedders/contentvec | grep -v '^total'" < /dev/null || return 1
  ssh_pod "cd /workspace/rvc/data && sha256sum -c --quiet rvc_data.sha256" < /dev/null || { echo "[rvc_pod] clip checksums differ on the pod"; return 1; }
  # fully detached (setsid in a subshell), so this ssh returns at once and a dropped connection can't stop the run
  ssh_pod "cd /workspace/rvc && (setsid nohup bash rvc_pod_run.sh $ROWS > out/main.log 2>&1 < /dev/null &); sleep 3; test -f out/main.log" < /dev/null || return 1
  echo "[rvc_pod] uploaded in $(( ($(date +%s) - t0) / 60 )) min; training started $(date +%T)"
}

status() {
  ssh_pod "cd /workspace/rvc/out && cat main.log; ls; for r in $ROWS; do [ -f \$r.log ] && echo \"\$r: \$(grep -o 'epoch=[0-9]* | step=[0-9]* | time=[0-9:]* | training_speed=[0-9:]*' \$r.log | tail -1)\$(grep -m1 -E 'Traceback|Error' \$r.log)\"; done; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"
}

wait_done() {
  local t0=$(date +%s) n=0
  while ! ssh_pod "test -f /workspace/rvc/out/RVC_DONE" < /dev/null 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -lt $(( 5 * 3600 )) ] || { echo "[rvc_pod] not done after 5 h"; return 1; }
    n=$((n + 1)); [ $((n % 5)) -eq 1 ] && { echo "--- $(date +%T)"; status 2>/dev/null || echo "[rvc_pod] status failed"; }
    sleep 180
  done
  echo "--- finished $(date +%T)"; status
}

download() {
  local stamp=$(date +%Y%m%d_%H%M) ok=0
  for ROW in $ROWS; do
    local M=bd_$ROW
    ssh_pod "cd /workspace/rvc/out && cat $ROW.log" > "$BD/e6/logs/rvc_pod_$ROW.log" 2>/dev/null || true
    if ! ssh_pod "test -f /workspace/rvc/out/done_$ROW" < /dev/null; then echo "[rvc_pod] $ROW not done on the pod"; continue; fi
    [ -d "$APP/logs/$M" ] && mv "$APP/logs/$M" "$APP/logs/${M}_local_partial_$stamp"
    mkdir -p "$APP/logs/$M"
    ssh_pod "cd /workspace/rvc/applio/logs/$M && tar -cf - --exclude='G_*.pth' --exclude='D_*.pth' --exclude=sliced_audios --exclude=extracted --exclude=eval ." \
      | tar --no-same-owner -xf - -C "$APP/logs/$M"
    if ls "$APP/logs/$M/${M}_200e_"*s.pth > /dev/null 2>&1 && ls "$APP/logs/$M/f0_voiced/"*.npy > /dev/null 2>&1; then
      echo "[rvc_pod] $ROW downloaded: $(ls "$APP/logs/$M" | tr '\n' ' ')"; ok=$((ok + 1))
    else
      echo "[rvc_pod] $ROW download incomplete"; DL_PROBLEM=1
    fi
  done
  ssh_pod "cat /workspace/rvc/out/main.log" > "$BD/e6/logs/rvc_pod_main.log" 2>/dev/null || true
  [ "$ok" -eq "$(echo $ROWS | wc -w)" ]
}

stop() {
  id=$(python -c "import json; print(json.load(open(r'$(cygpath -w "$PJ")'))['id'])")
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$id" -H "Authorization: Bearer $RUNPOD_API_KEY"; echo
  echo "[rvc_pod] terminate requested for $id $(date +%T)"
}

case "${1:-}" in
  launch) launch ;;
  upload) upload ;;
  status) status ;;
  download) download ;;
  stop) stop ;;
  all)
    launch || { [ -f "$PJ" ] && stop; exit 1; }
    upload || { echo "[rvc_pod] upload failed"; stop; exit 1; }
    if wait_done; then
      if download; then stop; echo "[rvc_pod] ALL_DONE $(date +%T)"; exit 0; fi
      if [ "${DL_PROBLEM:-0}" = 1 ]; then echo "[rvc_pod] a trained converter failed to download: pod left running for a retry of 'download' (then 'stop')"; exit 1; fi
      echo "[rvc_pod] training failed on the pod (see e6/logs/rvc_pod_*.log): stopping"; stop; exit 1
    fi
    echo "[rvc_pod] training did not finish: downloading what exists, then stopping"
    download || true; stop; exit 1 ;;
  *) echo "usage: rvc_pod.sh all|launch|upload|status|download|stop"; exit 1 ;;
esac
