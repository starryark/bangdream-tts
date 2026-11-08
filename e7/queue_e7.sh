#!/bin/bash
# E7 queue (E7_PLAN.md §2, §4-5), chained end to end:
#   smoke -> E7a (train + eval with --extra_on_seeds) -> select -> [E7d (mix, train, eval) -> select]
# C0 is E6's scored C0 (uploaded to eval/C0 by e7/pod_up.sh), not re-rendered.
# Training: E6's command (E1b's hyperparameters) with E7's seed 20261108; candidates ep1, ep2.
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

score_run () {  # score_run <run name> <cond prefix>
  local name=$1 prefix=$2 i=1
  local cks=($(ls -d $W/runs/$name/checkpoint-step-* | sort -t- -k3 -n))
  [ "${#cks[@]}" -eq 2 ] || { echo "[queue] $name: expected 2 checkpoints, found ${#cks[@]}"; exit 1; }
  for ck in "${cks[@]}"; do
    python e6/e6_eval.py --cond ${prefix}_ep$i --ckpt $ck --extra_on_seeds 2>&1 | tee -a $W/logs/eval_$name.log
    i=$((i+1))
  done
}

ARM="--epochs 3 --save_at 1.0,2.0 --stop_after 2.0"
[ -f $W/eval/C0/summary.json ] && [ -s $W/eval/C0/scored.jsonl ] || { echo "[queue] E6's C0 scores missing in eval/C0"; exit 1; }

for stage in "${@:-smoke E7a select}"; do
 for st in $stage; do
  case "$st" in
    smoke)
      train smoke E7a --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0
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
      python e6/e6_eval.py --cond smoke_eval --ckpt "$ck" --rows ppp_kasumi,mujica_umiri,millsage_natsume --extra_on_seeds 2>&1 | tail -n 2
      python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("/workspace/bd/eval/smoke_eval/scored.jsonl", encoding="utf-8")]
mid = lambda k, s: [r for r in rows if r["row"] == k and r["kind"] in ("ja_heldout", "ja_novel") and r["seed_base"] in s]
for k in ("mujica_umiri", "millsage_natsume"):
    b, x = len(mid(k, {6000, 11000})), len(mid(k, {16000, 21000}))
    assert b == 40 and x == 40, (k, b, x)
assert not mid("ppp_kasumi", {16000, 21000}), "extra seeds rendered for a v1 row"
assert len({r["pid"] for r in rows}) == len(rows), "pid collision"
own = sum(r["pred"] == "mujica_umiri" for r in mid("mujica_umiri", {6000, 11000, 16000, 21000})) / 80
print(f"smoke extra seeds ok: 40 + 40 ja-mid renders per Our Notes row, none for v1 rows; umiri own {own:.2f}")
assert own >= 0.7, "mixed-speaker batches look scrambled - stop"
EOF
      rm -rf $W/runs/smoke $W/eval/smoke_eval
      echo "STAGE_DONE smoke" ;;
    E7a) train E7a E7a $ARM; score_run E7a E7a; echo "STAGE_DONE E7a" ;;
    select)
      python e7/e7_select.py 2>&1 | tee $W/logs/select.log
      if python -c "import json,sys; sys.exit(0 if json.load(open('$W/eval/selection.json'))['run_E7d'] else 1)"; then
        echo "[queue] E7d triggered by the rule"
        python e6/e6_mix.py --arm E7d --sources pseudo_v1,pseudo_on,vc --replay_share 0.25 | tee $W/logs/mix_E7d.log
        train E7d E7d $ARM; score_run E7d E7d
        python e7/e7_select.py 2>&1 | tee $W/logs/select_final.log
      fi
      echo "STAGE_DONE select" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
 done
done
echo QUEUE_DONE
