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
# E6 and E7 (2026-11-08): code, plans, results, selection and per-condition numeric summaries. Never staged: bundles with
# line texts (ja_rows, eval_rows, eval_short, nv_rows, refs), scored/manifest files, listening sets, logs, audio.
cpf e6/*.py e6/*.sh e6/*.md e6/bundle/speakers.txt e6/bundle/rows_e6.json e6/bundle/probe_e6_calib.json \
    e6/vc/collect_report.json e6/vc/listening.json e6/results/eval/selection.json e6/results/eval/selection.md \
    e6/results/eval/*/summary.json e6/results/exp/*/mix.json e6/results/pseudo_on/report.json
cpf e7/*.py e7/*.sh e7/*.md e7/results/eval/selection.json e7/results/eval/selection.md \
    e7/results/eval/*/summary.json e7/results/exp/*/mix.json
cpf release/publish_hf_v2.py release/card_tables_v2.py release/voices_v2_ournotes.json release/MODEL_CARD_v2.md \
    release/space/app.py release/space/requirements.txt release/space/README.md
# E8 and E9 (2026-11-08/09): code, plans, results, screens, listening keys/answers/analyses. Never staged: listening
# items and pages (line texts), renders, bundles with line texts (ja_rows, eval_rows, eval_short, nv_rows, zh_texts,
# refs), e9/rows, pseudo rows, scored/manifest files, logs.
cpf e8/*.py e8/*.sh e8/*.md e8/listen/*_key.json e8/results/eval/*/summary.json e8/results/eval/*/screens.json \
    e8/results/exp/*/mix.json e8/results/listening/*.json e8/results/listening_round*.json \
    e8/results/pseudo_on/report.json e8/results/pseudo_v1_e1recipe/report.json
cpf e9/*.py e9/*.sh e9/*.md e9/listen_key.json e9/bundle/speakers.txt e9/bundle/rows_e6.json \
    e9/bundle/e9_bundle_report.json e9/bundle/probe_e6_calib.json e9/results/eval/*/summary.json \
    e9/results/eval/*/screens.json e9/results/exp/*/mix.json e9/results/listening.json e9/results/listening/*.json
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
# No game line in any staged text file: every line text of 8+ characters in the v1 and E6 bundles is searched for.
python -I -X utf8 - "$(cygpath -w "$BD")" <<'PYEOF' || { echo "refusing: a game line was found in a staged file"; exit 1; }
import json, pathlib, sys
bd = pathlib.Path(sys.argv[1]); lines = set()
for b in (bd / 'e1' / 'bundle', bd / 'e6' / 'bundle', bd / 'e9' / 'bundle'):
    for name, key in (('ja_rows.jsonl', 'text'), ('eval_rows.jsonl', 'text'), ('eval_short.jsonl', 'text')):
        f = b / name
        if f.exists():
            for l in f.read_text(encoding='utf-8').splitlines():
                t = json.loads(l).get(key) if l.strip() else None
                if t and len(t) >= 8 and not t.isascii():   # ASCII-only lines are band names such as 'millsage'
                    lines.add(t)
    zt = b / 'zh_texts.json'   # the game's own Mandarin translations of Our Notes lines: {row: [[clip id, text], ...]}
    if zt.exists():
        lines |= {t for pairs in json.loads(zt.read_text(encoding='utf-8')).values() for _, t in pairs
                  if t and len(t) >= 8 and not t.isascii()}
for f in (bd / 'e6' / 'results' / 'pseudo_on' / 'pseudo_rows.jsonl', bd / 'e8' / 'results' / 'pseudo_on' / 'pseudo_rows.jsonl'):
    if f.exists():   # pseudo-zh rows carry those translations as text
        lines |= {t for l in f.read_text(encoding='utf-8').splitlines() if l.strip()
                  for t in [json.loads(l).get('text')] if t and len(t) >= 8 and not t.isascii()}
hits = []
for p in pathlib.Path('.').rglob('*'):
    if '.git' in p.parts or not p.is_file() or p.stat().st_size > 5_000_000:
        continue
    try:
        txt = p.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    hits += [(str(p), t[:20]) for t in lines if t in txt]
print(f'[stage_github] game-line audit: {len(lines)} lines checked, {len(hits)} hits', hits[:5])
sys.exit(1 if hits or len(lines) < 1000 else 0)   # too few lines loaded = the audit itself failed
PYEOF
echo "[stage_github] $(find . -path ./.git -prune -o -type f -print | wc -l) files, $(du -sh --exclude=.git . | cut -f1) -> $G"
