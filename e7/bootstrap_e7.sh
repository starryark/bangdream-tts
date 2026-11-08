#!/bin/bash
# E7 pod bootstrap (E7_PLAN.md §3, §7): E6's bootstrap (e6/bootstrap_e6.sh, with E6 amendments 9-10), building only the E7a mix.
# E6 pod bootstrap notes follow (E6_PLAN.md §3, §7). Part 1 needs no data and runs while the upload is in flight;
# part 2 waits for /workspace/bd/UPLOAD_DONE. Expects (uploaded over SSH, never via HF):
#   data/wav24/<v1 row>/*.wav      v1 audio (E1b-recipe rows for the 37 kept rows + the E1 pseudo context)
#   data_on/wav24/<row>/*.wav      Our Notes audio (training, refs, eval and nv clips)
#   bundle/                        e6/bundle (65 rows), e6up/bundle_e1/ (e1/bundle, for the E1b-recipe pseudo)
#   code/ (kurisu_v2/pod/*.py + probes), e1/, e6/, pipeline/ (incl. bdcommon.py)
#   e6up/{e1_pseudo_report.json, e5_rvc_rows.jsonl, e6_rvc_rows.jsonl}, e5/rows/zh/, e6/rows/zh/
#   models/E1b_ep2/                v1 as shipped (condition C0)
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
mkdir -p $W/logs
cd $W

# ---- part 1: GPU, environment, models, replay ------------------------------------------
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
# Replay pool sized for E6d's 25% share (~5.5k rows) so the conditional arm can't stop on a short pool.
if [ ! -s replay/replay_rows.jsonl.unfiltered ] && [ ! -s replay/replay_rows.jsonl ]; then
  # 2026-11-08: --zh_speakers 70 added and zh 5.0 -> 5.5 h. With p05's default 30 speakers x 70 clips the zh pool
  # stopped at 2,100 clips / 2.53 h (bootstrap exited on p05's shortfall check), short of E6d's ~5.2k-row draw.
  (cd code && python p05_replay.py --zh_hours 5.5 --en_hours 2.5 --zh_speakers 70 --max_clips_per_speaker 70) > logs/replay.log 2>&1
  tail -n 5 logs/replay.log
fi
echo STAGE_ENV_DONE

# ---- part 2: needs the uploaded data ---------------------------------------------------
while [ ! -f $W/UPLOAD_DONE ]; do sleep 20; done
n_missing=$(python - <<'EOF'
import glob, json, os
W = "/workspace/bd"
rows = [json.loads(l) for l in open(f"{W}/bundle/ja_rows.jsonl", encoding="utf-8")]
ev = [json.loads(l) for l in open(f"{W}/bundle/eval_rows.jsonl", encoding="utf-8")]
sh = [json.loads(l) for l in open(f"{W}/bundle/eval_short.jsonl", encoding="utf-8")]
nv = [json.loads(l) for l in open(f"{W}/bundle/nv_rows.jsonl", encoding="utf-8")]
refs = [p.strip() for f in glob.glob(f"{W}/bundle/refs/*.txt") + glob.glob(f"{W}/e6up/bundle_e1/refs/*.txt") for p in open(f) if p.strip()]
need = [f"{W}/e6up/{f}" for f in ("e1_pseudo_report.json", "e5_rvc_rows.jsonl", "e6_rvc_rows.jsonl")] + \
       [f"{W}/models/E1b_ep2/model.safetensors", f"{W}/bundle/probe_e6.npz", f"{W}/bundle/probe_e6_calib.json", f"{W}/bundle/zh_texts.json"]
paths = [r["audio"] for r in rows + nv] + [r["real_audio"] for r in ev] + [r["real_audio"] for r in sh if r.get("real_audio")] + refs + need
miss = [p for p in paths if not os.path.exists(p)]
print(len(miss))
if miss: open(f"{W}/logs/missing.txt", "w").write("\n".join(miss))
EOF
)
[ "$n_missing" -eq 0 ] || { echo "$n_missing inputs missing (logs/missing.txt)"; exit 1; }
python e1/e1_aux_filter.py 2>&1 | tee logs/aux_filter.log | grep aux_filter
echo STAGE_REPLAY_DONE

# E1b-recipe pseudo for the v1 rows: run e1_pseudo.py in the E1 bundle context (49 names, same texts and
# seeds), then keep the 37 kept rows (e6_pod_rows.py).
mkdir -p /workspace/bd1
for d in code e1 pipeline; do ln -sfn $W/$d /workspace/bd1/$d; done
rm -rf /workspace/bd1/bundle && cp -r $W/e6up/bundle_e1 /workspace/bd1/bundle
BD_POD=/workspace/bd1 python e1/e1_pseudo.py > logs/pseudo_v1.log 2>&1; tail -n 2 logs/pseudo_v1.log
python e6/e6_pseudo_on.py > logs/pseudo_on.log 2>&1; tail -n 2 logs/pseudo_on.log
python e6/e6_pod_rows.py | tee logs/pod_rows.log
python e6/e6_codes.py > logs/codes.log 2>&1; tail -n 12 logs/codes.log
python e6/e6_mix.py --arm E7a --sources pseudo_v1,pseudo_on,vc | tee logs/mix_E7a.log   # E6b's data (E7_PLAN.md §3)
df -h /workspace | tail -1
echo STAGE_DATA_DONE
