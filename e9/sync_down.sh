#!/usr/bin/env bash
# E9 (local, Git Bash; from e8/sync_down.sh): resumable download of one pod directory, size-checked (E8_PLAN.md §4).
#   bash e9/sync_down.sh REMOTE_DIR LOCAL_DIR [find -name pattern to exclude ...]
# Lists the remote files with sizes, fetches only missing or wrong-size ones, and repeats until none differ
# (at most 4 rounds). A local machine reset mid-transfer only costs a re-run. Pod address: e9/work/pod.json.
set -uo pipefail
cd "E:/STEINS;GATEVoice/bangdream" || exit 1
REM=$1; LOC=$2; shift 2
EXCL=""; for p in "$@"; do EXCL="$EXCL ! -name '$p'"; done
KEY=~/.ssh/runpod_ed25519
ssh_pod() { local ip port; read ip port < <(python -c "import json;d=json.load(open('e9/work/pod.json'));print(d['ip'], d['port'])" | tr -d '\r'); ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=30 -p "$port" "root@$ip" "$@"; }
mkdir -p "$LOC"
LIST="e9/work/sync_$(echo "$REM" | tr '/' '_').tsv"; TODO="$LIST.todo"
for round in 1 2 3 4; do
  ssh_pod "cd '$REM' 2>/dev/null && find . -type f $EXCL -printf '%P\t%s\n' 2>/dev/null; true" < /dev/null > "$LIST"
  [ -s "$LIST" ] || { echo "[sync] $REM: empty or unreachable listing (round $round)"; sleep 20; continue; }
  n=$(python -I -X utf8 - "$LIST" "$LOC" "$TODO" <<'EOF' | tr -d '\r'
import os, sys
lst, loc, todo = sys.argv[1], sys.argv[2], sys.argv[3]
want = {}
for l in open(lst, encoding="utf-8", errors="replace"):
    p, _, s = l.rstrip("\n").rpartition("\t")
    if p:
        want[p] = int(s)
miss = [p for p, s in want.items() if not os.path.isfile(os.path.join(loc, p)) or os.path.getsize(os.path.join(loc, p)) != s]
open(todo, "w", encoding="utf-8", newline="\n").write("".join(p + "\n" for p in miss))
print(len(miss), sum(want[p] for p in miss))
EOF
)
  set -- $n
  echo "[sync] $REM round $round $(date +%T): $1 files ($(( ${2:-0} / 1048576 )) MiB) to fetch"
  [ "$1" = 0 ] && { echo "[sync] $REM complete -> $LOC"; exit 0; }
  ssh_pod "cd '$REM' && tar -cf - -T -" < "$TODO" | tar --no-same-owner -xf - -C "$LOC" || echo "[sync] stream broke in round $round"
done
echo "[sync] $REM still incomplete after 4 rounds"; exit 1
