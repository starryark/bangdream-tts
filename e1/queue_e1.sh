#!/bin/bash
# E1 queue (E1_PLAN.md).  usage: queue_e1.sh <stage>...   run order: smoke E1a refs E1b select
#   smoke  tiny full-mode run (256 rows, a few steps) -> export -> eval plan on 2 voices: plumbing only
#   E1a/E1b  3-epoch cosine, checkpoints at 1.0/2.0/3.0 epochs, each scored on the eval plan
#   refs   the Base floor (from the E1a checkpoint's own speaker rows) and real held-out clips
#   select the pre-registered rule over all six checkpoints
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
cd $W

train () {  # train <run name> <arm> <extra train_sft flags...>
  local name=$1 arm=$2; shift 2
  if compgen -G "$W/runs/$name/checkpoint-step-*/config.json" > /dev/null && [ -f "$W/runs/$name/DONE" ]; then
    echo "[queue] $name already trained - skipped"; return 0
  fi
  rm -rf "$W/runs/$name"            # partial run from a crash: start clean (also drops the speaker cache)
  python code/train_sft.py \
    --train_jsonl $W/exp/$arm/train_with_codes.jsonl --val_jsonl $W/exp/$arm/val_with_codes.jsonl \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --speakers "$(cat $W/bundle/speakers.txt)" \
    --aux_speakers $W/replay/aux_speakers.json --p_auto_lang 0.25 --attn sdpa \
    --mode full --full_dtype fp32 --optimizer adamw --lr 5e-6 --weight_decay 0.01 \
    --batch_size 16 --effective_batch 32 --max_val_batches 1000 --eval_every 50 --log_every 10 \
    --keep_last 1 --vram_fraction 0.95 --tensorboard --seed 20261028 \
    --name $name --out $W/runs/$name "$@" 2>&1 | tee $W/logs/train_$name.log
  rm -rf $W/runs/$name/checkpoint-epoch-* $W/runs/$name/state.pt
  touch $W/runs/$name/DONE
}

score_run () {  # score_run <run name> <cond prefix>: every checkpoint, in step order = epochs 1, 2, 3
  local name=$1 prefix=$2 i=1
  local cks=($(ls -d $W/runs/$name/checkpoint-step-* | sort -t- -k3 -n))
  [ "${#cks[@]}" -eq 3 ] || { echo "[queue] $name: expected 3 checkpoints, found ${#cks[@]}"; exit 1; }
  for ck in "${cks[@]}"; do
    python e1/e1_eval.py --cond ${prefix}_ep$i --ckpt $ck 2>&1 | tee -a $W/logs/eval_$name.log
    i=$((i+1))
  done
}

for stage in "$@"; do
  case "$stage" in
    smoke)
      train smoke E1a --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0
      ck=$(ls -d $W/runs/smoke/checkpoint-step-* | head -1)
      python - "$ck" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
spk = cfg["talker_config"]["spk_id"]
names = [s.split("=")[0] for s in open("/workspace/bd/bundle/speakers.txt").read().strip().split(",")]
assert list(spk) == names, "spk_id order differs from speakers.txt"
assert sorted(spk.values()) == list(range(3000, 3000 + len(names))), spk
print("smoke export ok:", len(spk), "speakers, rows", min(spk.values()), "-", max(spk.values()))
EOF
      python - "$ck" <<'EOF'
import sys, torch, numpy as np
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained(sys.argv[1], device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
for spk in ("ppp_kasumi", "mujica_sakiko_tsuki"):
    for lang, text in (("Japanese", "今日はいい天気だね。"), ("Chinese", "今天天气很好。"), ("English", "The weather is nice today.")):
        w, sr = m.generate_custom_voice(text=[text], speaker=[spk], language=[lang], max_new_tokens=200, non_streaming_mode=True)
        a = np.asarray(w[0]); assert a.size > sr * 0.3 and np.isfinite(a).all(), (spk, lang, a.size)
print("smoke inference ok")
EOF
      python e1/e1_eval.py --cond smoke_eval --ckpt "$ck" --rows ppp_kasumi,mujica_umiri 2>&1 | tail -n 3
      # mixed-speaker generate_custom_voice must keep each item's speaker: umiri clones at the
      # real-clip level zero-shot, so a scrambled or broadcast speaker list shows up here
      python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/bd/eval/smoke_eval/scored.jsonl", encoding="utf-8")]
um = [r for r in rows if r["row"] == "mujica_umiri" and r["lang"] == "ja"]
ka = [r for r in rows if r["row"] == "ppp_kasumi"]
own = sum(r["pred"] == "mujica_umiri" for r in um) / len(um)
swap = sum(r["pred"] == "mujica_umiri" for r in ka) / len(ka)
print(f"smoke speaker check: umiri ja own-voice {own:.2f} (n={len(um)}), kasumi rendered as umiri {swap:.2f} (n={len(ka)})")
assert own >= 0.7 and swap <= 0.3, "mixed-speaker custom-voice batches look scrambled - stop"
EOF
      rm -rf $W/runs/smoke $W/eval/smoke_eval        # plumbing only: free the disk
      echo "STAGE_DONE smoke" ;;
    E1a)  train E1a E1a --epochs 3 --save_at 1.0,2.0,3.0 --stop_after 3.0
          score_run E1a E1a; echo "STAGE_DONE E1a" ;;
    E1b)  if [ ! -s $W/exp/E1b/train_with_codes.jsonl ]; then echo "[queue] no E1b mix (no pseudo rows) - skipped"; continue; fi
          # pre-registered: E1b only if >= 25 voices kept >= 15 gated pseudo rows; otherwise it
          # would be ~E1a and answer nothing (E1a's Mandarin guards then decide the question)
          if ! python - <<'EOF'
import json, sys
rep = json.load(open("/workspace/bd/pseudo/report.json", encoding="utf-8"))
ok = sorted(k for k, v in rep.items() if v["kept"] >= 15)
print("[pseudo yield] kept per voice:", {k: v["kept"] for k, v in sorted(rep.items(), key=lambda t: t[1]["kept"])})
print(f"[pseudo yield] voices with >= 15 rows: {len(ok)} / {len(rep)}; total {sum(v['kept'] for v in rep.values())}")
sys.exit(0 if len(ok) >= 25 else 1)
EOF
          then echo "[queue] E1b skipped: pseudo yield below the pre-registered bar"; continue; fi
          train E1b E1b --epochs 3 --save_at 1.0,2.0,3.0 --stop_after 3.0
          score_run E1b E1b; echo "STAGE_DONE E1b" ;;
    refs) ck=$(ls -d $W/runs/E1a/checkpoint-step-* | sort -t- -k3 -n | head -1)
          python e1/e1_eval.py --cond floor --vectors_from $ck 2>&1 | tee $W/logs/eval_floor.log
          python e1/e1_eval.py --cond real 2>&1 | tee $W/logs/eval_real.log
          echo "STAGE_DONE refs" ;;
    select) cands=$(ls -d $W/eval/E1?_ep? 2>/dev/null | xargs -n1 basename | paste -sd, -)
            [ -n "$cands" ] || { echo "[queue] nothing to select"; exit 1; }
            python e1/e1_select.py --cands "$cands" 2>&1 | tee $W/logs/select.log
            echo "STAGE_DONE select" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo QUEUE_DONE
