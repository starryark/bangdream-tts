"""Publish the selected checkpoint to the model repo.

  python p08_publish.py --ckpt <checkpoint dir> --card <README.md> [--dry_run]

Before anything is overwritten, the repo's current head is tagged `v1-epoch10`
so the previous release stays one `revision=` away. Uploads, in one commit:
  - the checkpoint directory (model, tokenizer, processor, speech_tokenizer/)
  - README.md (model card)
  - training_data/: report.json, kept.jsonl (exact training rows and text),
    decisions.jsonl (every Japanese clip, kept or dropped, and why), ref_list.txt
  - eval/: table.json and the selected checkpoint's scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
REPO = "starrydark/Kurisu_Qwen3_TTS"
OLD_TAG = "v1-epoch10"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--eval_dir", default=None, help="eval/main/<tag> of the selected checkpoint")
    ap.add_argument("--message", default="v2: fine-tune on the re-verified Japanese corpus")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["tts_model_type"] == "custom_voice" and "kurisu" in cfg["talker_config"]["spk_id"]
    for f in ("model.safetensors", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json"):
        assert (ckpt / f).is_file(), f

    api = HfApi()
    head = api.model_info(REPO).sha
    tags = {t.name for t in api.list_repo_refs(REPO).tags}
    print(f"[publish] {REPO} head {head}; tags {sorted(tags)}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(ckpt, stage, ignore=shutil.ignore_patterns("*.pt", "optimizer*", "tb"))
        shutil.copy(args.card, stage / "README.md")
        td = stage / "training_data"
        td.mkdir()
        for f in ("report.json", "kept.jsonl", "decisions.jsonl", "ref_list.txt", "ref.json"):
            shutil.copy(WORK / "build" / f, td / f)
        ev = stage / "eval"
        ev.mkdir()
        shutil.copy(WORK / "eval" / "main" / "table.json", ev / "table_main.json")
        shutil.copy(WORK / "eval" / "confirm" / "table.json", ev / "table_confirm.json")
        shutil.copy(WORK / "eval" / "confirm" / "utmos_finalists.json", ev / "utmos_finalists.json")
        shutil.copy(WORK / "code" / "SELECTION_RULE.md", ev / "SELECTION_RULE.md")
        if args.eval_dir:
            shutil.copy(Path(args.eval_dir) / "scored.jsonl", ev / "selected_scored.jsonl")
        files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        print(f"[publish] {len(files)} files:", *files[:40], sep="\n  ")
        if args.dry_run:
            return
        if OLD_TAG not in tags:
            api.create_tag(REPO, tag=OLD_TAG, revision=head,
                           tag_message="Previous release: LoRA r32 epoch 10 (pre-v2)")
            print(f"[publish] tagged {head} as {OLD_TAG}")
        info = api.upload_folder(repo_id=REPO, folder_path=str(stage),
                                 commit_message=args.message)
        print(f"[publish] {info.commit_url}")


if __name__ == "__main__":
    main()
