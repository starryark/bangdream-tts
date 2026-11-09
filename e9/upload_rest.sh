#!/usr/bin/env bash
# E9 (local, Git Bash; from e8/upload_rest.sh): finish an interrupted data upload, then watch the pod.
#   bash e9/upload_rest.sh >> e9/logs/pod_up.log 2>&1
# The local machine reset during the upload on 2026-11-08 (10:37), killing the tar stream; bootstrap waits for
# UPLOAD_DONE, so nothing on the pod ran on partial data. This compares every data file's size on the pod with
# the local copy, sends only missing or wrong-size files, repeats until none differ (at most 3 rounds), sets
# UPLOAD_DONE, and hands over to e9/pod_wait.sh.
cd "E:/STEINS;GATEVoice/bangdream" || exit 1
KEY=~/.ssh/runpod_ed25519
ssh_pod() { local ip port; read ip port < <(python -c "import json;d=json.load(open('e9/work/pod.json'));print(d['ip'], d['port'])" | tr -d '\r'); ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$ip" "$@"; }
for round in 1 2 3; do
  # find exits non-zero when a file vanishes mid-scan (replay's .part files); the listing is still usable
  ssh_pod "cd /workspace/bd && find . -type f -printf '%P\t%s\n' 2>/dev/null; true" < /dev/null > e9/work/pod_sizes.tsv
  [ -s e9/work/pod_sizes.tsv ] || { echo "[upload_rest] listing failed (ssh)"; sleep 30; continue; }
  n=$(python -I -X utf8 - <<'EOF'
import os
BD = r"E:/STEINS;GATEVoice/bangdream"
pod = {}
for l in open(BD + "/e9/work/pod_sizes.tsv", encoding="utf-8", errors="replace"):
    p, _, s = l.rstrip("\n").rpartition("\t")
    if p:
        pod[p] = int(s)
want = [l.strip() for l in open(BD + "/e9/work/upload_data.txt", encoding="utf-8") if l.strip()]
todo = [f for f in want if pod.get(f) != os.path.getsize(os.path.join(BD, f))]
open(BD + "/e9/work/upload_rest.txt", "w", encoding="utf-8", newline="\n").write("".join(f + "\n" for f in todo))
print(len(todo))
EOF
)
  n=$(echo "$n" | tr -d '\r')
  echo "[upload_rest] round $round $(date +%T): $n files to send"
  if [ "$n" = 0 ]; then
    ssh_pod "touch /workspace/bd/UPLOAD_DONE" < /dev/null && echo "[pod] data uploaded (resumed); UPLOAD_DONE set $(date +%T)" && exec bash e9/pod_wait.sh
    echo "[upload_rest] could not set UPLOAD_DONE"; exit 1
  fi
  (tar -cf - -T e9/work/upload_rest.txt) | ssh_pod "tar --no-same-owner -xf - -C /workspace/bd" || echo "[upload_rest] tar stream failed in round $round"
done
echo "[upload_rest] files still differ after 3 rounds: UPLOAD_DONE not set (pod left running)"; exit 1
