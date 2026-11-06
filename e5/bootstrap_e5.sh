#!/bin/bash
# E5 pod bootstrap (E5_PLAN.md §5). Part 1 needs no data and runs while the upload is in flight;
# part 2 waits for /workspace/bd/UPLOAD_DONE. Expects (uploaded over SSH, never via HF):
#   data/wav24/<row>/*.wav, bundle/ (local e1/bundle), code/ (kurisu_v2/pod/*.py + probes),
#   e1/, pipeline/, e5/ (this folder's pod scripts + rvc_rows.jsonl + e1_pseudo_report.json +
#   e1_aux_filter.json) and e5/rows/zh/<row>/*.wav (the VC rows).
# Rebuilds E1b's data exactly as E1 did (replay, aux filter, Base pseudo-zh: same texts, seeds,
# gates and cap), checks it against E1's saved reports, then adds the VC rows and builds the E5 mix.
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
mkdir -p $W/logs
cd $W

# ---- part 1: GPU, environment, models, replay (as E1/E2) --------------------------------
python -c "import torch; x=torch.randn(8192,8192,device='cuda'); y=(x@x).float(); print('gpu ok', torch.cuda.get_device_name(0), bool(torch.isfinite(y).all()))"
python -m pip install -q --upgrade pip
pip install -q --ignore-installed blinker
pip install -q "qwen-asr==0.0.6" hf_transfer pyopenjtalk-plus speechbrain==1.0.3 "numpy<2.3" \
    soundfile pyloudnorm tensorboard pyarrow pypinyin scipy safetensors
pip install -q --no-deps "qwen-tts==0.1.1"
pip install -q onnxruntime einops sox
python -c "import torch, transformers, qwen_asr, qwen_tts, pyopenjtalk, pypinyin, speechbrain; print('env ok', torch.__version__, transformers.__version__)"
for m in Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-Tokenizer-12Hz; do
  python -c "from huggingface_hub import snapshot_download as s; import sys; s(sys.argv[1]); print('model ok', sys.argv[1])" "$m"
done
(cd code && python -c "import accent_metrics as A; from huggingface_hub import snapshot_download as s; [s(r, revision=v) for r, v in ((A.LID_REPO, A.LID_REV), (A.CHARSIU_REPO, A.CHARSIU_REV), (A.CHARSIU_TOK_REPO, A.CHARSIU_TOK_REV))]; print('model ok pinned')")
if [ ! -s replay/replay_rows.jsonl.unfiltered ] && [ ! -s replay/replay_rows.jsonl ]; then
  (cd code && python p05_replay.py --zh_hours 2.0 --en_hours 1.0 --max_clips_per_speaker 70) > logs/replay.log 2>&1
  tail -n 5 logs/replay.log
fi
echo STAGE_ENV_DONE

# ---- part 2: needs the uploaded audio --------------------------------------------------
while [ ! -f $W/UPLOAD_DONE ]; do sleep 20; done
n_missing=$(python - <<'EOF'
import glob, json, os
rows = [json.loads(l) for l in open("/workspace/bd/bundle/ja_rows.jsonl", encoding="utf-8")]
refs = [p.strip() for f in glob.glob("/workspace/bd/bundle/refs/*.txt") for p in open(f) if p.strip()]
need = ["/workspace/bd/e5/rvc_rows.jsonl", "/workspace/bd/e5/e1_pseudo_report.json", "/workspace/bd/e5/e1_aux_filter.json"]
print(sum(not os.path.exists(r["audio"]) for r in rows) + sum(not os.path.exists(p) for p in refs + need))
EOF
)
[ "$n_missing" -eq 0 ] || { echo "$n_missing inputs missing"; exit 1; }
python e1/e1_aux_filter.py 2>&1 | tee logs/aux_filter.log | grep aux_filter
python - <<'EOF'
import json
a = json.load(open("/workspace/bd/replay/aux_filter.json")); b = json.load(open("/workspace/bd/e5/e1_aux_filter.json"))
print("[e5] aux filter identical to E1:", a == b)
EOF
echo STAGE_REPLAY_DONE
python e1/e1_pseudo.py > logs/pseudo.log 2>&1; tail -n 2 logs/pseudo.log
python e5/e5_pod_rows.py | tee logs/e5_rows.log
python e1/e1_codes.py > logs/codes.log 2>&1; tail -n 8 logs/codes.log
python e1/e1_mix.py --arm E5 --pseudo | tee logs/mix_E5.log
python - <<'EOF'
import json
m = json.load(open("/workspace/bd/exp/E5/mix.json")); e = json.load(open("/workspace/bd/e5/e1b_mix.json"))
print("[e5] mix vs E1b:", {k: (m[k], e[k]) for k in ("train", "val", "ja_train", "replay", "pseudo", "aux_speakers_used")})
EOF
df -h /workspace | tail -1
echo STAGE_DATA_DONE
