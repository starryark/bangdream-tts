#!/bin/bash
# E2 queue (E2_PLAN.md): train E2 (E1b's command, E2 data; 3-epoch cosine stopped at 2.0,
# checkpoints at 1.0 and 2.0), then score both checkpoints on the unchanged E1 eval plan.
# Selection runs locally (e2_select.py) against the E1 floor / real / E1b numbers.
set -euo pipefail
export BD_POD=/workspace/bd KURISU_WORK=/workspace/bd HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
export BD_DIR=/workspace/bd BD_KPOD=/workspace/bd/code BD_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=/workspace/bd
cd $W
name=E2
if ! { compgen -G "$W/runs/$name/checkpoint-step-*/config.json" > /dev/null && [ -f "$W/runs/$name/DONE" ]; }; then
  rm -rf "$W/runs/$name"
  python code/train_sft.py \
    --train_jsonl $W/exp/E2/train_with_codes.jsonl --val_jsonl $W/exp/E2/val_with_codes.jsonl \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --speakers "$(cat $W/bundle/speakers.txt)" \
    --aux_speakers $W/replay/aux_speakers.json --p_auto_lang 0.25 --attn sdpa \
    --mode full --full_dtype fp32 --optimizer adamw --lr 5e-6 --weight_decay 0.01 \
    --batch_size 16 --effective_batch 32 --max_val_batches 1000 --eval_every 50 --log_every 10 \
    --keep_last 1 --vram_fraction 0.95 --tensorboard --seed 20261028 \
    --epochs 3 --save_at 1.0,2.0 --stop_after 2.0 \
    --name $name --out $W/runs/$name 2>&1 | tee $W/logs/train_$name.log
  rm -rf $W/runs/$name/checkpoint-epoch-* $W/runs/$name/state.pt
  touch $W/runs/$name/DONE
fi
cks=($(ls -d $W/runs/$name/checkpoint-step-* | sort -t- -k3 -n))
[ "${#cks[@]}" -eq 2 ] || { echo "[queue] expected 2 checkpoints, found ${#cks[@]}"; exit 1; }
i=1
for ck in "${cks[@]}"; do
  python e1/e1_eval.py --cond E2_ep$i --ckpt $ck 2>&1 | tee -a $W/logs/eval_E2.log
  i=$((i+1))
done
echo QUEUE_DONE
