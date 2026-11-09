#!/bin/bash
# E8 queue (E8_PLAN.md §2-6), chained end to end:
#   smoke (A and B) -> train A -> train B -> average -> screen evals (avg, fallback ep2.0) -> listening renders
# C0 is E6's scored C0 (uploaded to eval/C0), not re-rendered for the screens; it is rendered for listening.
# Training: E7's command and seed (20261108); A = E7a's data unchanged, B = the same + --spk_lr 1e-4.
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
    --keep_last 1 --vram_fraction 0.95 --tensorboard --seed 20261108 \
    --name $name --out $W/runs/$name "$@" 2>&1 | tee $W/logs/train_$name.log
  rm -rf $W/runs/$name/checkpoint-epoch-* $W/runs/$name/state.pt
  touch $W/runs/$name/DONE
}

ckpts () { ls -d $W/runs/$1/checkpoint-step-* | sort -t- -k3 -n; }

pick () {  # pick <arm letter>: average, screen; fall back to epoch 2.0; writes candidates.json entry
  local L=$1 run=E8$1
  local cks=($(ckpts $run))
  [ "${#cks[@]}" -eq 3 ] || { echo "[queue] $run: expected 3 checkpoints (1.0, 1.5, 2.0), found ${#cks[@]}"; exit 1; }
  [ -f $W/runs/$run/avg/model.safetensors ] || python e8/e8_average.py $W/runs/$run/avg "${cks[@]}"
  python e6/e6_eval.py --cond ${run}_avg --ckpt $W/runs/$run/avg --extra_on_seeds 2>&1 | tee -a $W/logs/eval_$run.log
  if python e8/e8_screens.py ${run}_avg | tee $W/logs/screens_${run}_avg.log; then
    echo "$L $W/runs/$run/avg ${run}_avg" >> $W/eval/candidates.txt; return 0
  fi
  echo "[queue] ${run}_avg failed a screen: evaluating epoch 2.0"
  python e6/e6_eval.py --cond ${run}_ep2 --ckpt ${cks[2]} --extra_on_seeds 2>&1 | tee -a $W/logs/eval_$run.log
  if python e8/e8_screens.py ${run}_ep2 | tee $W/logs/screens_${run}_ep2.log; then
    echo "$L ${cks[2]} ${run}_ep2" >> $W/eval/candidates.txt; return 0
  fi
  echo "[queue] arm $L has no candidate (avg and epoch 2.0 failed the screens)"
}

ARM="--epochs 3 --save_at 1.0,1.5,2.0 --stop_after 2.0"
[ -f $W/eval/C0/summary.json ] && [ -s $W/eval/C0/scored.jsonl ] || { echo "[queue] E6's C0 scores missing in eval/C0"; exit 1; }

for stage in "${@:-smoke A B pick listen}"; do
 for st in $stage; do
  case "$st" in
    smoke)
      train smoke E8 --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0
      train smokeB E8 --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0 --spk_lr 1e-4
      grep -q "learned offsets" $W/logs/train_smokeB.log || { echo "[queue] smokeB: no learned-offset report"; exit 1; }
      python - <<'EOF'
import json, torch, glob
from safetensors import safe_open
W = "/workspace/bd"
names = [s.split("=")[0] for s in open(f"{W}/bundle/speakers.txt").read().strip().split(",")]
for run, expect_same in (("smoke", True), ("smokeB", False)):
    locked = torch.load(f"{W}/runs/{run}/spk_embedding.pt", map_location="cpu")
    assert locked["names"][:65] == names, "speaker order"
    ck = sorted(glob.glob(f"{W}/runs/{run}/checkpoint-step-*"))[0]
    cfg = json.load(open(ck + "/config.json"))
    assert list(cfg["talker_config"]["spk_id"]) == names and sorted(cfg["talker_config"]["spk_id"].values()) == list(range(3000, 3065))
    with safe_open(ck + "/model.safetensors", "pt") as f:
        rows = f.get_tensor("talker.model.codec_embedding.weight")[3000:3065].float()
    ref = locked["table"][:65].to(torch.bfloat16).float()
    diff = (rows - ref).abs().max().item()
    print(f"smoke {run}: saved rows vs locked table, max |diff| {diff:.5f}")
    assert (diff == 0) if expect_same else (diff > 0), f"{run}: rows {'changed' if expect_same else 'did not change'}"
print("smoke rows ok: A keeps the locked rows, B saves locked + learned offset")
EOF
      ck=$(ls -d $W/runs/smokeB/checkpoint-step-* | head -1)
      python e6/e6_eval.py --cond smoke_eval --ckpt "$ck" --rows ppp_kasumi,mujica_umiri,millsage_natsume --extra_on_seeds 2>&1 | tail -n 2
      python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/bd/eval/smoke_eval/scored.jsonl", encoding="utf-8")]
mid = lambda k: [r for r in rows if r["row"] == k and r["kind"] in ("ja_heldout", "ja_novel")]
assert len(mid("mujica_umiri")) == 80 and len(mid("millsage_natsume")) == 80, "extra seeds"
own = sum(r["pred"] == "mujica_umiri" for r in mid("mujica_umiri")) / 80
print(f"smoke eval ok (smokeB): umiri own {own:.2f}")
assert own >= 0.7, "mixed-speaker batches look scrambled - stop"
EOF
      python e8/e8_listen_render.py --system A --ckpt "$ck" --limit 2 --out_name smoke 2>&1 | tail -n 1
      [ "$(ls $W/listen_renders/smoke/*.wav | wc -l)" -eq 2 ] || { echo "[queue] smoke listening render failed"; exit 1; }
      rm -rf $W/runs/smoke $W/runs/smokeB $W/eval/smoke_eval $W/listen_renders/smoke
      echo "STAGE_DONE smoke" ;;
    A) train E8A E8 $ARM; echo "STAGE_DONE A" ;;
    B) train E8B E8 $ARM --spk_lr 1e-4; echo "STAGE_DONE B" ;;
    pick) rm -f $W/eval/candidates.txt; pick A; pick B; cat $W/eval/candidates.txt 2>/dev/null || true; echo "STAGE_DONE pick" ;;
    listen)
      [ -s $W/eval/candidates.txt ] || { echo "[queue] no candidate passed the screens: nothing to listen to (v1 stays)"; echo "STAGE_DONE listen"; continue; }
      while read L path cond; do
        python e8/e8_listen_render.py --system $L --ckpt $path < /dev/null 2>&1 | tail -n 1
      done < $W/eval/candidates.txt
      python e8/e8_listen_render.py --system C0 --ckpt $W/models/E1b_ep2 2>&1 | tail -n 1
      echo "STAGE_DONE listen" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
 done
done
echo QUEUE_DONE
