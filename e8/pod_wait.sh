#!/usr/bin/env bash
# E8 (local, Git Bash; from e7/pod_wait.sh): watch a pod whose bootstrap/queue is already running, then download and stop it.
#   bash e8/pod_wait.sh >> e8/logs/pod_up.log 2>&1
# QUEUE_DONE -> download -> stop. If the queue dies, logs and partial results are pulled and the pod is kept
# for 60 min ("[pod] DEAD" in the log), so a fix can restart the queue without a new upload. If the queue
# runs again within that time, watching resumes; otherwise the pod is stopped. (2026-11-08: the first
# bootstrap died on p05's replay shortfall after a 75-min upload, which `pod_up.sh all` would have discarded.)
cd "E:/STEINS;GATEVoice/bangdream" || exit 1
P=e8/pod_up.sh
while :; do
  rc=0; bash $P wait || rc=$?
  echo "[pod] wait exit $rc $(date +%T)"
  if [ "$rc" = 0 ]; then
    if bash $P download; then bash $P stop; echo "[pod] ALL_DONE $(date +%T)"; exit 0; fi
    echo "[pod] download failed: pod left running for a retry of 'download' (then 'stop')"; exit 1
  fi
  echo "[pod] DEAD $(date +%T): queue not running (wait exit $rc); pulling logs; pod kept 60 min for a fix"
  bash $P partial || true
  back=0
  for i in $(seq 1 30); do
    sleep 120
    read ip port < <(python -c "import json;d=json.load(open('e8/work/pod.json'));print(d['ip'], d['port'])" | tr -d '\r')
    if ssh -i ~/.ssh/runpod_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p "$port" "root@$ip" "pgrep -f '[b]ootstrap_e8.sh|[q]ueue_e8.sh' > /dev/null" < /dev/null 2>/dev/null; then
      echo "[pod] queue running again $(date +%T): watching resumes"; back=1; break
    fi
  done
  [ "$back" = 1 ] && continue
  echo "[pod] no restart within 60 min: stopping"; bash $P stop; exit 1
done
