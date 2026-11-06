#!/bin/bash
# Fresh pod -> environment, models, replay, pseudo-zh, codes, mixed arms.
# Expects (uploaded over SSH, never via HF):
#   /workspace/bd/data/wav24/<row>/*.wav     the 24 kHz Japanese clips (local bangdream/data/wav24)
#   /workspace/bd/bundle/                    e1_build.py output + probe_codec.npz
#   /workspace/bd/code/                      kurisu_v2/pod/*.py + probes_zh_v3.jsonl, probes_en_v3.jsonl
#   /workspace/bd/e1/, /workspace/bd/pipeline/   this folder, and bdcommon.py + rows.json
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
mkdir -p $W/logs
cd $W

# 0. Inputs present and consistent.
test -s bundle/ja_rows.jsonl && test -s bundle/speakers.txt && test -s bundle/probe_codec.npz
n_missing=$(python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/bd/bundle/ja_rows.jsonl", encoding="utf-8")]
refs = [p.strip() for f in __import__("glob").glob("/workspace/bd/bundle/refs/*.txt") for p in open(f) if p.strip()]
import os
print(sum(not os.path.exists(r["audio"]) for r in rows) + sum(not os.path.exists(p) for p in refs))
EOF
)
[ "$n_missing" -eq 0 ] || { echo "$n_missing audio files missing - re-upload data/wav24"; exit 1; }

# 1. Fail fast on a bad GPU (one Community H200 had an ECC fault).
python -c "import torch; x=torch.randn(8192,8192,device='cuda'); y=(x@x).float(); print('gpu ok', torch.cuda.get_device_name(0), bool(torch.isfinite(y).all()))"

# 2. Environment (Kurisu v3's; the leakage probe is applied by hand, so no scikit-learn).
python -m pip install -q --upgrade pip
pip install -q --ignore-installed blinker
pip install -q "qwen-asr==0.0.6" hf_transfer pyopenjtalk-plus speechbrain==1.0.3 "numpy<2.3" \
    soundfile pyloudnorm tensorboard pyarrow pypinyin scipy safetensors
pip install -q --no-deps "qwen-tts==0.1.1"
pip install -q onnxruntime einops sox
python -c "import torch, transformers, qwen_asr, qwen_tts, pyopenjtalk, pypinyin, speechbrain; print('env ok', torch.__version__, transformers.__version__)"

# 3. Models (pinned revisions for the accent metrics, as in Kurisu v3).
for m in Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-Tokenizer-12Hz; do
  python -c "from huggingface_hub import snapshot_download as s; import sys; s(sys.argv[1]); print('model ok', sys.argv[1])" "$m"
done
(cd code && python -c "import accent_metrics as A; from huggingface_hub import snapshot_download as s; [s(r, revision=v) for r, v in ((A.LID_REPO, A.LID_REV), (A.CHARSIU_REPO, A.CHARSIU_REV), (A.CHARSIU_TOK_REPO, A.CHARSIU_TOK_REV))]; print('model ok pinned')")
echo STAGE_ENV_DONE

# 4. Replay (native zh/en, other speakers). Need ~1,509 train rows (14% of 9,268 JA rows).
#    Defaults give 1,713 (Kurisu); the distance filter below may drop some speakers, so take
#    ~2,250: 30 zh speakers x up to 70 clips (2.0 h at 4.36 s/clip) + 25 en (1.0 h).
if [ ! -s replay/replay_rows.jsonl.unfiltered ] && [ ! -s replay/replay_rows.jsonl ]; then
  (cd code && python p05_replay.py --zh_hours 2.0 --en_hours 1.0 --max_clips_per_speaker 70) > logs/replay.log 2>&1
  tail -n 15 logs/replay.log
fi
python e1/e1_aux_filter.py 2>&1 | tee logs/aux_filter.log | grep aux_filter
echo STAGE_REPLAY_DONE

# 5. Pseudo-zh candidates + gates (E1b only: a failure here must not block E1a), codes, arms.
python e1/e1_pseudo.py > logs/pseudo.log 2>&1 || echo "WARN pseudo failed - E1b will not be mixed (see logs/pseudo.log)"
tail -n 5 logs/pseudo.log
python e1/e1_codes.py > logs/codes.log 2>&1; tail -n 10 logs/codes.log
python e1/e1_mix.py --arm E1a | tee logs/mix_E1a.log
if [ -s pseudo/pseudo_rows.jsonl ]; then
  python e1/e1_mix.py --arm E1b --pseudo | tee logs/mix_E1b.log
else
  echo "WARN no pseudo rows - E1b skipped"
fi
df -h /workspace | tail -1
echo STAGE_DATA_DONE
