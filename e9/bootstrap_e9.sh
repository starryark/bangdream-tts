#!/bin/bash
# E9 pod bootstrap (E9_PLAN.md §3-4). Part 1 (GPU, environment, models, replay) is E8's, unchanged, and runs while
# the upload is in flight. Part 2 waits for /workspace/bd/UPLOAD_DONE and does NOT regenerate pseudo-zh: the row
# sources were built locally by e9/e9_bundle.py (E1-recipe pseudo for all 49 v1 voices and the Our Notes pseudo,
# both from E8's run, relabelled; E5 + E6 VC rows), so it only checks inputs, filters replay, codes and mixes.
# Expects: bundle/ (= e9/bundle, 72 voices), e9/rows/*.jsonl, the audio they name (uploaded at the same relative
# paths), models/E1b_ep2/, code/, e1/, e6/, e8/, e9/, pipeline/, eval/C0/.
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
mkdir -p $W/logs
cd $W

# ---- part 1: GPU, environment, models, replay (E8's) ------------------------------------
python -c "import torch; x=torch.randn(8192,8192,device='cuda'); y=(x@x).float(); print('gpu ok', torch.cuda.get_device_name(0), bool(torch.isfinite(y).all()))"
python -m pip install -q --upgrade pip
pip install -q --ignore-installed blinker
pip install -q "qwen-asr==0.0.6" hf_transfer pyopenjtalk-plus speechbrain==1.0.3 "numpy<2.3" \
    soundfile pyloudnorm tensorboard pyarrow pypinyin scipy safetensors librosa
pip install -q --no-deps "qwen-tts==0.1.1"
pip install -q onnxruntime einops sox
python -c "import torch, transformers, qwen_asr, qwen_tts, pyopenjtalk, pypinyin, speechbrain, librosa; print('env ok', torch.__version__, transformers.__version__)"
for m in Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-Tokenizer-12Hz; do
  python -c "from huggingface_hub import snapshot_download as s; import sys; s(sys.argv[1]); print('model ok', sys.argv[1])" "$m"
done
(cd code && python -c "import accent_metrics as A; from huggingface_hub import snapshot_download as s; [s(r, revision=v) for r, v in ((A.LID_REPO, A.LID_REV), (A.CHARSIU_REPO, A.CHARSIU_REV), (A.CHARSIU_TOK_REPO, A.CHARSIU_TOK_REV))]; print('model ok pinned')")
if [ ! -s replay/replay_rows.jsonl.unfiltered ] && [ ! -s replay/replay_rows.jsonl ]; then
  (cd code && python p05_replay.py --zh_hours 5.5 --en_hours 2.5 --zh_speakers 70 --max_clips_per_speaker 70) > logs/replay.log 2>&1
  tail -n 5 logs/replay.log
fi
echo STAGE_ENV_DONE

# ---- part 2: needs the uploaded data ---------------------------------------------------
while [ ! -f $W/UPLOAD_DONE ]; do sleep 20; done
mkdir -p pseudo_v1 pseudo_on vc
cp e9/rows/pseudo_v1_rows.jsonl pseudo_v1/rows.jsonl
cp e9/rows/pseudo_on_rows.jsonl pseudo_on/pseudo_rows.jsonl
cp e9/rows/vc_rows.jsonl vc/vc_rows.jsonl
n_missing=$(python e9/e9_check_inputs.py)
[ "$n_missing" -eq 0 ] || { echo "$n_missing inputs missing (logs/missing.txt)"; exit 1; }
python e1/e1_aux_filter.py 2>&1 | tee logs/aux_filter.log | grep aux_filter
echo STAGE_REPLAY_DONE
python e6/e6_codes.py > logs/codes.log 2>&1; tail -n 12 logs/codes.log
BD_N_SPEAKERS=72 python e6/e6_mix.py --arm E9 --sources pseudo_v1,pseudo_on,vc | tee logs/mix_E9.log
df -h /workspace | tail -1
echo STAGE_DATA_DONE
