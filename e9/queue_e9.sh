#!/bin/bash
# E9 queue (E9_PLAN.md §2-6), chained end to end:
#   smoke -> train E9 (recipe B: --spk_lr 1e-4, saves 1.0/1.5/2.0) -> average -> aliases -> screen eval (fallback ep2.0)
#   -> listening renders (E9 candidate, C0)
# C0 is E6's scored C0 (uploaded to eval/C0). Aliases (e9/e9_aliases.py) are added to every checkpoint before it is
# evaluated or rendered (eval and renderer read the checkpoint's spk_id).
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
ARM="--epochs 3 --save_at 1.0,1.5,2.0 --stop_after 2.0 --spk_lr 1e-4"
[ -f $W/eval/C0/summary.json ] && [ -s $W/eval/C0/scored.jsonl ] || { echo "[queue] E6's C0 scores missing in eval/C0"; exit 1; }

for stage in "${@:-smoke train pick listen}"; do
 for st in $stage; do
  case "$st" in
    smoke)
      train smokeB E9 --epochs 1 --max_train_samples 256 --allow_small_dataset --save_at 1.0 --stop_after 1.0 --spk_lr 1e-4
      grep -q "learned offsets" $W/logs/train_smokeB.log || { echo "[queue] smokeB: no learned-offset report"; exit 1; }
      ck=$(ls -d $W/runs/smokeB/checkpoint-step-* | head -1)
      python e9/e9_aliases.py "$ck"
      python - "$ck" <<'EOF'
import json, sys, torch, numpy as np
from safetensors import safe_open
W, ck = "/workspace/bd", sys.argv[1]
names = [s.split("=")[0] for s in open(f"{W}/bundle/speakers.txt").read().strip().split(",")]
cfg = json.load(open(ck + "/config.json")); spk = cfg["talker_config"]["spk_id"]
assert len(names) == 72 and all(spk[n] == 3000 + i for i, n in enumerate(names)), "spk_id order/rows"
assert spk["mygo_soyo_tsuki"] == spk["mygo_soyo"] and spk["mujica_sakiko_tsuki"] == spk["mujica_sakiko"], "aliases"
assert "mygo_soyo_tsuki" in cfg["talker_config"]["spk_is_dialect"], "alias dialect entry"
locked = torch.load(f"{W}/runs/smokeB/spk_embedding.pt", map_location="cpu")
with safe_open(ck + "/model.safetensors", "pt") as f:
    rows = f.get_tensor("talker.model.codec_embedding.weight")[3000:3072].float()
diff = (rows - locked["table"][:72].to(torch.bfloat16).float()).abs().max().item()
print(f"smoke export ok: 72 voices at rows 3000-3071, aliases in place, learned rows differ by {diff:.5f}")
assert diff > 0
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained(ck, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
for s in ("mygo_soyo_tsuki", "mujica_sakiko_tsuki", "mygo_anon_ournotes", "mygo_anon"):
    w, sr = m.generate_custom_voice(text=["今日はいい天気だね。"], speaker=[s], language=["Japanese"], max_new_tokens=200, non_streaming_mode=True)
    a = np.asarray(w[0]); assert a.size > sr * 0.3 and np.isfinite(a).all(), (s, a.size)
print("smoke inference ok (aliases and era ids)")
EOF
      python e6/e6_eval.py --cond smoke_eval --ckpt "$ck" --rows ppp_kasumi,mujica_umiri_ournotes,millsage_natsume --extra_on_seeds 2>&1 | tail -n 2
      python - <<'EOF'
import json
rows = [json.loads(l) for l in open("/workspace/bd/eval/smoke_eval/scored.jsonl", encoding="utf-8")]
mid = lambda k: [r for r in rows if r["row"] == k and r["kind"] in ("ja_heldout", "ja_novel")]
assert len(mid("mujica_umiri_ournotes")) >= 40 and len(mid("millsage_natsume")) == 80, ("extra seeds", len(mid("mujica_umiri_ournotes")))
own = sum(r["pred"] == "mujica_umiri" for r in mid("mujica_umiri_ournotes")) / len(mid("mujica_umiri_ournotes"))
print(f"smoke eval ok: umiri_ournotes assigned to Umiri {own:.2f}")
assert own >= 0.7, "mixed-speaker batches look scrambled - stop"
EOF
      python e8/e8_listen_render.py --system E9 --ckpt "$ck" --items e9/listen_items.json --limit 2 --out_name smoke 2>&1 | tail -n 1
      [ "$(ls $W/listen_renders/smoke/*.wav | wc -l)" -eq 2 ] || { echo "[queue] smoke listening render failed"; exit 1; }
      rm -rf $W/runs/smokeB $W/eval/smoke_eval $W/listen_renders/smoke
      echo "STAGE_DONE smoke" ;;
    train) train E9 E9 $ARM; echo "STAGE_DONE train" ;;
    pick)
      rm -f $W/eval/candidates.txt
      cks=($(ckpts E9))
      [ "${#cks[@]}" -eq 3 ] || { echo "[queue] E9: expected 3 checkpoints, found ${#cks[@]}"; exit 1; }
      [ -f $W/runs/E9/avg/model.safetensors ] || python e8/e8_average.py $W/runs/E9/avg "${cks[@]}"
      python e9/e9_aliases.py $W/runs/E9/avg "${cks[2]}"
      python e6/e6_eval.py --cond E9_avg --ckpt $W/runs/E9/avg --extra_on_seeds 2>&1 | tee -a $W/logs/eval_E9.log
      if python e9/e9_screens.py E9_avg | tee $W/logs/screens_E9_avg.log; then
        echo "E9 $W/runs/E9/avg E9_avg" > $W/eval/candidates.txt
      else
        echo "[queue] E9_avg failed a screen: evaluating epoch 2.0"
        python e6/e6_eval.py --cond E9_ep2 --ckpt ${cks[2]} --extra_on_seeds 2>&1 | tee -a $W/logs/eval_E9.log
        if python e9/e9_screens.py E9_ep2 | tee $W/logs/screens_E9_ep2.log; then
          echo "E9 ${cks[2]} E9_ep2" > $W/eval/candidates.txt
        else
          echo "[queue] no candidate (avg and epoch 2.0 failed the screens)"
        fi
      fi
      cat $W/eval/candidates.txt 2>/dev/null || true
      echo "STAGE_DONE pick" ;;
    listen)
      [ -s $W/eval/candidates.txt ] || { echo "[queue] no candidate passed the screens: nothing to listen to (v1 stays)"; echo "STAGE_DONE listen"; continue; }
      read L path cond < $W/eval/candidates.txt
      python e8/e8_listen_render.py --system E9 --ckpt $path --items e9/listen_items.json < /dev/null 2>&1 | tail -n 1
      python e8/e8_listen_render.py --system C0 --ckpt $W/models/E1b_ep2 --items e9/listen_items.json < /dev/null 2>&1 | tail -n 1
      echo "STAGE_DONE listen" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
 done
done
echo QUEUE_DONE
