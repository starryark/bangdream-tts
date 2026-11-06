#!/usr/bin/env bash
# Stage the public GitHub repo (starryark/bangdream-tts) from this project folder.
#
#   bash release/stage_github.sh            -> release/bangdream-tts/ (a git work tree; .git is kept)
#
# Release form (user, 2026-11-06): code, plans, results and numeric reports only. Never staged:
# audio, model weights, vectors, game transcripts (data/*.jsonl, e1/bundle/ja_rows.jsonl,
# eval_rows.jsonl), reference-clip lists (e1/bundle/refs/), per-clip eval files (*/scored*.jsonl),
# listening sets, logs. The Kurisu pod code the pipeline imports is vendored as kurisu_pod/.
set -euo pipefail
BD="$(cd "$(dirname "$0")/.." && pwd)"
KP="$BD/../kurisu_v2/pod"
G="$BD/release/bangdream-tts"
mkdir -p "$G"
find "$G" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cd "$BD"
cpf() { for f in "$@"; do if [ -f "$f" ]; then mkdir -p "$G/$(dirname "$f")"; cp "$f" "$G/$f"; fi; done; }

cpf pipeline/*.py pipeline/rows.json pipeline/run_e0.sh
cpf e0/e0_results.md e0/floor_table.md e0/floor_table.json e0/codec_probe_report.json e0/vectors_report.json
cpf e1/*.py e1/*.sh e1/*.md e1/probes_ja_novel.jsonl \
    e1/bundle/speakers.txt e1/bundle/e2_baselines.json e1/bundle/rows.json e1/bundle/probes_ja_novel.jsonl \
    e1/results/eval/selection.json e1/results/eval/selection.md e1/results/eval/summary.json \
    e1/results/exp/E1b/mix.json e1/results/did_amendA.json
cpf e2/*.py e2/*.sh e2/*.md e2/results/selection_e2.json e2/results/selection_e2.md
cpf e3/*.py e3/*.sh e3/*.md
cpf e4/*.py e4/*.sh e4/*.md
cpf e5/*.py e5/*.sh e5/*.md e5/queue.txt e5/e1b_mix.json e5/e1_pseudo_report.json e5/e1_aux_filter.json \
    e5/vc/collect_report.json e5/vc/listening.json e5/vc/f0_targets.json \
    e5/vc/report_b1.json e5/vc/report_b2.json e5/vc/report_b3.json \
    e5/results/selection_e5.json e5/results/selection_e5.md e5/results/listening_e5.json \
    e5/results/mix.json e5/results/pod_rows.txt e5/results/listen_key.json
cpf research/*.md research/scripts/*.py research/scripts/spk_sep.json
cpf review/*.py review/*.md
cpf data/build_report.json data/asr_summary.json
cpf release/publish_hf.py release/stage_github.sh release/MODEL_CARD.md release/card_tables.py release/voice_table_E1b_ep2.md
cpf e1/results/pseudo/report.json
for f in README.md LICENSE CITATION.cff .gitignore .gitattributes; do
  [ -f "release/repo/$f" ] && cp "release/repo/$f" "$G/$f"
done
mkdir -p "$G/kurisu_pod"
cp "$KP"/*.py "$KP/probes_zh_v3.jsonl" "$KP/probes_en_v3.jsonl" "$G/kurisu_pod/"

# Audits: no audio/weights, no transcript-bearing jsonl, no secrets.
cd "$G"
bad=$(find . -path ./.git -prune -o -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.safetensors" \
      -o -name "*.pt" -o -name "*.npz" -o -name "*.pth" -o -name "*.index" \) -print)
[ -z "$bad" ] || { echo "refusing: binary/audio files staged: $bad"; exit 1; }
jl=$(find . -path ./.git -prune -o -name "*.jsonl" -print | grep -v -E "probes_ja_novel|probes_(zh|en)_v3" || true)
[ -z "$jl" ] || { echo "refusing: unexpected jsonl staged: $jl"; exit 1; }
if grep -rIlE "hf_[A-Za-z0-9]{30,}|rpa_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}" --exclude-dir=.git . ; then
  echo "refusing: token-like string found"; exit 1
fi
echo "[stage_github] $(find . -path ./.git -prune -o -type f -print | wc -l) files, $(du -sh --exclude=.git . | cut -f1) -> $G"
