#!/usr/bin/env bash
# E9 (local, Git Bash): launch the H100 pod, upload, and start bootstrap -> queue chained (no waiting on anyone).
# Derived from e8/pod_up.sh: E9's bundle (72 voices) and prebuilt row sources, E9's bootstrap/queue, pod state in e9/work,
# results in e9/results, and a resumable download (e9/sync_down.sh) of the candidate and listening renders (E9_PLAN.md §4).
#   bash e9/pod_up.sh launch     # creates the pod, writes e9/work/pod.json (id, ip, port)
#   bash e9/pod_up.sh upload     # code + bundles, starts bootstrap&&queue under nohup, then data, then UPLOAD_DONE
#   bash e9/pod_up.sh all > e9/logs/pod_up.log 2>&1   # launch -> upload -> wait (QUEUE_DONE) -> download -> stop
# Pod recipe (memory bangdream-pod-ops): RunPod REST, Secure H100 80GB HBM3, pytorch 2.8 / CUDA 12.8 image,
# 150 GB container disk, SSH key ~/.ssh/runpod_ed25519, no HF token on the pod.
set -euo pipefail
BD="E:/STEINS;GATEVoice/bangdream"
KP="E:/STEINS;GATEVoice/kurisu_v2"
W="$BD/e9/work"
KEY=~/.ssh/runpod_ed25519
# ip/port are read first, so ssh keeps the caller's stdin (tar streams); Windows python ends its lines with \r\n.
ssh_pod() { local ip port; read ip port < <(python -c "import json;d=json.load(open('$W/pod.json'));print(d['ip'], d['port'])" | tr -d '\r'); ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$ip" "$@"; }

case "$1" in
  launch)
    rm -f "$W/pod.json"
    # H100-class only (the queue is set for 80 GB: --vram_fraction 0.95, batch 16 fp32); up to 6 rounds, 5 min apart
    id=""
    for round in 1 2 3 4 5 6; do
      for gpu in "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe" "NVIDIA H100 NVL" "NVIDIA H200"; do
        body=$(GPU="$gpu" python - <<EOF
import json, os
print(json.dumps({"name": "bd-e9", "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "gpuTypeIds": [os.environ["GPU"]], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 150,
  "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": open(r"$(cygpath -w $KEY.pub)").read().strip()}}))
EOF
)
        resp=$(curl -s -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$body")
        id=$(echo "$resp" | python -c "import json,sys
try: print(json.load(sys.stdin).get('id') or '')
except Exception: print('')" | tr -d '\r')
        [ -n "$id" ] && break 2
        echo "[pod] round $round, $gpu: not created: ${resp:0:200}"
      done
      [ "$round" -lt 6 ] && sleep 300
    done
    [ -n "$id" ] || { echo "[pod] launch failed: no H100-class GPU in 6 rounds"; exit 1; }
    echo "{\"id\": \"$id\", \"gpu\": \"$gpu\"}" > "$W/pod.json"   # stop works from here on
    echo "[pod] created $id on $gpu $(date +%T)"
    for i in $(seq 1 60); do
      info=$(curl -s https://rest.runpod.io/v1/pods/$id -H "Authorization: Bearer $RUNPOD_API_KEY")
      ipport=$(echo "$info" | python -c "import json,sys; d=json.load(sys.stdin); ip=d.get('publicIp') or ''; pm=d.get('portMappings') or {}; print(ip, pm.get('22') or pm.get(22) or '')")
      set -- $ipport
      if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
        echo "{\"id\": \"$id\", \"gpu\": \"$gpu\", \"ip\": \"$1\", \"port\": $2}" > "$W/pod.json"; echo "[pod] ssh root@$1 -p $2"
        for j in $(seq 1 120); do   # the endpoint appears before sshd answers; a fresh host may first pull the image
          err=$(ssh_pod true 2>&1 < /dev/null) && { echo "[pod] ssh up after $((j * 10)) s"; exit 0; }
          [ $((j % 12)) -eq 0 ] && echo "[pod] $(date +%T) waiting for ssh: ${err:0:200}"
          sleep 10
        done
        echo "[pod] ssh not answering after 20 min (pod $id still exists: bash e9/pod_up.sh stop)"; exit 1
      fi
      sleep 10
    done
    echo "[pod] no ssh endpoint after 10 min: $info"; exit 1 ;;
  upload)
    python -I -X utf8 "$BD/e9/e9_upload_lists.py"
    ssh_pod "mkdir -p /workspace/bd/code /workspace/bd/logs /workspace/bd/eval" < /dev/null
    # Kurisu pod code (incl. train_sft.py with --spk_lr) -> code/
    (cd "$KP" && tar -cf - --transform 's,^pod,code,' pod/*.py pod/*.jsonl) | ssh_pod "tar --no-same-owner -xf - -C /workspace/bd"
    # code, the E9 bundle (-> bundle/), E9 row sources, listening items, E6's scored C0 (-> eval/C0)
    (cd "$BD" && tar -cf - --transform 's,^e9/bundle,bundle,;s,^e6/results/eval/C0,eval/C0,' -T "$W/upload_small.txt") | ssh_pod "tar --no-same-owner -xf - -C /workspace/bd"
    ssh_pod "ls /workspace/bd/bundle | head -20; wc -l /workspace/bd/bundle/ja_rows.jsonl; ls /workspace/bd/eval/C0 /workspace/bd/e9" < /dev/null
    # fully detached, so the data upload below is not blocked
    ssh_pod "cd /workspace/bd && (setsid nohup bash -c 'bash e9/bootstrap_e9.sh > logs/bootstrap.log 2>&1 && bash e9/queue_e9.sh > logs/queue.log 2>&1' > /dev/null 2>&1 < /dev/null &); sleep 2" < /dev/null
    echo "[pod] bootstrap + queue started (waits for UPLOAD_DONE)"
    t0=$(date +%s)
    (cd "$BD" && tar -cf - -T "$W/upload_data.txt") | ssh_pod "tar --no-same-owner -xf - -C /workspace/bd"
    ssh_pod "touch /workspace/bd/UPLOAD_DONE" < /dev/null
    echo "[pod] data uploaded in $(( ($(date +%s) - t0) / 60 )) min; UPLOAD_DONE set" ;;
  status)
    ssh_pod "tail -n 3 /workspace/bd/logs/bootstrap.log; tail -n 5 /workspace/bd/logs/queue.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" ;;
  download)
    # after QUEUE_DONE (E8_PLAN.md §4): numbers and logs, both arms' candidate checkpoints, every listening render, and
    # the pseudo rows and wavs (so a later run can skip ~1.3 h of bootstrap), each through e8/sync_down.sh
    # (resumable, size-checked; a local reset mid-transfer costs only a re-run of this step).
    R="$BD/e9/results"; mkdir -p "$R"
    ssh_pod "grep -q QUEUE_DONE /workspace/bd/logs/queue.log" < /dev/null || { echo "[pod] queue not finished"; exit 1; }
    ok=1
    bash e9/sync_down.sh /workspace/bd/eval "$R/eval" '*.wav' < /dev/null || ok=0
    bash e9/sync_down.sh /workspace/bd/logs "$R/logs" < /dev/null || ok=0
    bash e9/sync_down.sh /workspace/bd/exp "$R/exp" '*with_codes.jsonl' < /dev/null || ok=0
    bash e9/sync_down.sh /workspace/bd/listen_renders "$BD/e9/listen_renders" < /dev/null || ok=0
    if [ -s "$R/eval/candidates.txt" ]; then
      while read L path cond; do
        [ -n "$cond" ] || continue
        echo "[pod] candidate $L: $cond = $path"
        bash e9/sync_down.sh "$path" "$BD/models/$cond" 'optimizer*' '*.pt' < /dev/null || ok=0
        [ -f "$BD/models/$cond/model.safetensors" ] || ok=0
      done < <(tr -d '\r' < "$R/eval/candidates.txt")
    else
      echo "[pod] no candidate passed the screens: no checkpoint downloaded"
    fi
    [ "$ok" = 1 ] || { echo "[pod] download incomplete: run 'bash e9/pod_up.sh download' again (then 'stop')"; exit 1; }
    echo "[pod] downloaded; stop the pod with: bash e9/pod_up.sh stop" ;;
  wait)
    # until QUEUE_DONE (exit 0); the bootstrap/queue processes gone without it on two polls 5 min apart (exit 3);
    # ssh failing for 1 h (exit 4); 16 h (exit 2). Status every 30 min.
    t0=$(date +%s); n=0; dead=0; unreach=0
    while :; do
      st=$(ssh_pod "if grep -q QUEUE_DONE /workspace/bd/logs/queue.log 2>/dev/null; then echo done; elif pgrep -f '[b]ootstrap_e9.sh|[q]ueue_e9.sh' > /dev/null; then echo running; else echo dead; fi" < /dev/null 2>/dev/null | tr -d '\r') || st=unreachable
      [ -n "$st" ] || st=unreachable
      case "$st" in
        done) echo "[pod] QUEUE_DONE seen $(date +%T)"; exit 0 ;;
        dead) dead=$((dead + 1)); unreach=0; echo "[pod] $(date +%T) no bootstrap/queue process and no QUEUE_DONE ($dead)"
              [ "$dead" -ge 2 ] && { bash "$0" status < /dev/null || true; exit 3; } ;;
        running) dead=0; unreach=0 ;;
        *) unreach=$((unreach + 1)); echo "[pod] $(date +%T) ssh failed ($unreach)"; [ "$unreach" -ge 12 ] && exit 4 ;;
      esac
      [ $(( $(date +%s) - t0 )) -lt $(( 16 * 3600 )) ] || { echo "[pod] not done after 16 h"; exit 2; }
      n=$((n + 1)); [ $((n % 6)) -eq 1 ] && { echo "--- $(date +%T)"; bash "$0" status < /dev/null 2>/dev/null || echo "[pod] status failed"; }
      sleep 300
    done ;;
  partial)
    # after a failed queue: logs, eval numbers and mixes -> e9/results_partial_<time> (no audio, no checkpoints)
    P="$BD/e9/results_partial_$(date +%Y%m%d_%H%M)"; mkdir -p "$P"
    ssh_pod "cd /workspace/bd && tar -cf - --exclude='*.wav' logs eval exp/*/mix.json 2>/dev/null; true" < /dev/null | tar --no-same-owner -xf - -C "$P"
    echo "[pod] partial results in $P: $(ls "$P" | tr '\n' ' ')" ;;
  stop)
    id=$(python -c "import json; print(json.load(open(r'$(cygpath -w "$W")/pod.json'))['id'])" | tr -d '\r')
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/$id" -H "Authorization: Bearer $RUNPOD_API_KEY"; echo
    echo "[pod] terminate requested for $id $(date +%T)"
    sleep 20
    echo "[pod] pods on the account now: $(curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" | python -c "import json,sys; print([(p.get('id'), p.get('name'), p.get('desiredStatus')) for p in json.load(sys.stdin)])")" ;;
  all)
    # launch -> upload (bootstrap && queue start on the pod) -> wait -> download -> stop, unattended
    bash "$0" launch || { [ -f "$W/pod.json" ] && bash "$0" stop; exit 1; }
    bash "$0" upload || { echo "[pod] upload failed: stopping"; bash "$0" stop; exit 1; }
    rc=0; bash "$0" wait || rc=$?
    if [ "$rc" = 0 ]; then
      if bash "$0" download; then bash "$0" stop; echo "[pod] ALL_DONE $(date +%T)"; exit 0; fi
      echo "[pod] download failed: pod left running for a retry of 'download' (then 'stop')"; exit 1
    fi
    echo "[pod] queue did not finish (wait exit $rc): pulling logs and partial results, then stopping"
    bash "$0" partial || true; bash "$0" stop; exit 1 ;;
  *) echo "usage: pod_up.sh all|launch|upload|wait|status|download|partial|stop"; exit 1 ;;
esac
