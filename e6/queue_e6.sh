#!/bin/bash
# E6 queue (E6_PLAN.md §2, §4-6), chained end to end:
#   smoke -> E6a (+eval) -> C0 -> E6b (+eval) -> E6c (+eval) -> real, floor_on -> select -> [E6d -> select]
# Arms: E1b's command, 3-epoch cosine laid out, checkpoints at 1.0 and 2.0, stopped at 2.0 (candidates ep1, ep2).
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
  rm -rf "$W/runs/$name"
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

score_run () {  # score_run <run name> <cond prefix> [--nv TEXT]
  local name=$1 prefix=$2 i=1; shift 2
  local cks=($(ls -d $W/runs/$name/checkpoint-step-* | sort -t- -k3 -n))
  [ "${#cks[@]}" -eq 2 ] || { echo "[queue] $name: expected 2 checkpoints, found ${#cks[@]}"; exit 1; }
  for ck in "${cks[@]}"; do
    python e6/e6_eval.py --cond ${prefix}_ep$i --ckpt $ck "$@" 2>&1 | tee -a $W/logs/eval_$name.log
    i=$((i+1))
  done
}

ARM="--epochs 3 --save_at 1.0,2.0 --stop_after 2.0"
printf '{"[nv]": ["息", "はぁ", "っ"]}' > $W/bundle/nv_tokens.json

for stage in "${@:-smoke E6a C0 E6b E6c refs select}"; do
 for st in $stage; do
  case "$st" in
    smoke)
      train smoke E6a --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0
      ck=$(ls -d $W/runs/smoke/checkpoint-step-* | head -1)
      python - "$ck" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json"))
spk = cfg["talker_config"]["spk_id"]
names = [s.split("=")[0] for s in open("/workspace/bd/bundle/speakers.txt").read().strip().split(",")]
assert list(spk) == names and len(names) == 65, "spk_id order differs from speakers.txt"
assert sorted(spk.values()) == list(range(3000, 3065)), spk
print("smoke export ok:", len(spk), "speakers, rows", min(spk.values()), "-", max(spk.values()))
EOF
      python - "$ck" <<'EOF'
import sys, torch, numpy as np
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained(sys.argv[1], device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
for spk in ("ppp_kasumi", "mujica_sakiko_tsuki", "ikka_raika"):
    for lang, text in (("Japanese", "今日はいい天気だね。"), ("Chinese", "今天天气很好。"), ("English", "The weather is nice today.")):
        w, sr = m.generate_custom_voice(text=[text], speaker=[spk], language=[lang], max_new_tokens=200, non_streaming_mode=True)
        a = np.asarray(w[0]); assert a.size > sr * 0.3 and np.isfinite(a).all(), (spk, lang, a.size)
print("smoke inference ok")
EOF
      python e6/e6_eval.py --cond smoke_eval --ckpt "$ck" --rows ppp_kasumi,mujica_umiri,millsage_natsume 2>&1 | tail -n 2
      python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/bd/eval/smoke_eval/scored.jsonl", encoding="utf-8")]
ja = lambda k: [r for r in rows if r["row"] == k and r["lang"] == "ja"]
own = sum(r["pred"] == "mujica_umiri" for r in ja("mujica_umiri")) / len(ja("mujica_umiri"))
swap = sum(r["pred"] == "mujica_umiri" for r in rows if r["row"] == "ppp_kasumi") / len([r for r in rows if r["row"] == "ppp_kasumi"])
nat = sum(r["pred"] == "millsage_natsume" for r in ja("millsage_natsume")) / len(ja("millsage_natsume"))
print(f"smoke speaker check: umiri ja own {own:.2f}, natsume ja own {nat:.2f}, kasumi rendered as umiri {swap:.2f}")
assert own >= 0.7 and swap <= 0.3, "mixed-speaker custom-voice batches look scrambled - stop"
EOF
      rm -rf $W/runs/smoke $W/eval/smoke_eval
      echo "STAGE_DONE smoke" ;;
    E6a) train E6a E6a $ARM; score_run E6a E6a --nv "……っ"; echo "STAGE_DONE E6a" ;;
    C0)  python e6/e6_eval.py --cond C0 --ckpt $W/models/E1b_ep2 2>&1 | tee $W/logs/eval_C0.log; echo "STAGE_DONE C0" ;;
    E6b) if [ ! -s $W/exp/E6b/train_with_codes.jsonl ]; then echo "[queue] no E6b mix - skipped"; continue; fi
         train E6b E6b $ARM; score_run E6b E6b; echo "STAGE_DONE E6b" ;;
    E6c) train E6c E6c $ARM --new_tokens $W/bundle/nv_tokens.json --new_token_lr 1e-4
         score_run E6c E6c --nv "[nv]"; echo "STAGE_DONE E6c" ;;
    refs) python e6/e6_eval.py --cond real 2>&1 | tee $W/logs/eval_real.log
          ck=$(ls -d $W/runs/E6a/checkpoint-step-* | sort -t- -k3 -n | head -1)
          python e6/e6_eval.py --cond floor_on --vectors_from $ck 2>&1 | tee $W/logs/eval_floor.log
          echo "STAGE_DONE refs" ;;
    select)
      python e6/e6_select.py 2>&1 | tee $W/logs/select.log
      if python -c "import json,sys; sys.exit(0 if json.load(open('$W/eval/selection.json'))['run_E6d'] else 1)"; then
        echo "[queue] E6d triggered by the rule"
        python e6/e6_mix.py --arm E6d --sources pseudo_v1,pseudo_on --replay_share 0.25 | tee $W/logs/mix_E6d.log
        train E6d E6d $ARM; score_run E6d E6d --nv "……っ"
        python e6/e6_select.py 2>&1 | tee $W/logs/select_final.log
      fi
      echo "STAGE_DONE select" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
 done
done
echo QUEUE_DONE
