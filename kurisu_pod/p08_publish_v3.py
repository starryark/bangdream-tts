"""Publish the selected v3 checkpoint to the model repo.

  python p08_publish_v3.py --ckpt <checkpoint dir> --card <README.md> [--dry_run]

Before anything is overwritten, v2 (the pinned commit below) is tagged `v2` so it
stays one `revision=` away (v1 is already `v1-epoch10`). The upload is one commit:
  - the checkpoint directory (model, extended tokenizer, processor, speech_tokenizer/,
    kurisu_v3.json - what it was trained with: language ids, tag tokens)
  - README.md (model card)
  - training_data/: build report, kept rows, every Japanese clip's decision,
    speaker reference lists, the mix summary, replay/augmentation reports and the
    SFX/replay attribution needed by their CC-BY licences
  - eval/: v3 table, adoption verdicts, selection output, E0 table, the selection
    rule with every amendment, and the selected checkpoint's per-clip scores
Old v2-only files at the repo root are left alone except training_data/ and eval/,
whose contents are replaced.
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
V2_SHA = "92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57"
V2_TAG = "v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--message", default="v3: language ids, native replay, Amadeus/phone voices, tags")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    spk = cfg["talker_config"]["spk_id"]
    assert cfg["tts_model_type"] == "custom_voice" and "kurisu" in spk, spk
    assert set(cfg["talker_config"]["spk_is_dialect"]) == set(spk)
    for f in ("model.safetensors", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json",
              "kurisu_v3.json", "added_tokens.json"):
        assert (ckpt / f).is_file(), f

    api = HfApi()
    head = api.model_info(REPO).sha
    tags = {t.name for t in api.list_repo_refs(REPO).tags}
    print(f"[publish] {REPO} head {head}; tags {sorted(tags)}")
    if head != V2_SHA:
        raise SystemExit(f"repo head {head} is not the v2 commit {V2_SHA}; refusing to overwrite")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(ckpt, stage, ignore=shutil.ignore_patterns("*.pt", "optimizer*", "tb"))
        shutil.copy(args.card, stage / "README.md")
        td = stage / "training_data"
        td.mkdir()
        for src, dst in ((WORK / "build_v3" / "report.json", "build_report.json"),
                         (WORK / "build_v3" / "kept.jsonl", "kept.jsonl"),
                         (WORK / "build_v3" / "decisions.jsonl", "decisions.jsonl"),
                         (WORK / "qwen_v3" / "exp" / "final" / "mix.json", "final_mix.json"),
                         (WORK / "augment" / "report.json", "augment_report.json"),
                         (WORK / "augment" / "phone_curve.json", "phone_curve.json"),
                         (WORK / "augment" / "sfx_attribution.json", "sfx_attribution.json"),
                         (WORK / "replay" / "report.json", "replay_report.json"),
                         (WORK / "pseudo" / "report.json", "pseudo_zh_report.json")):
            if src.is_file():
                shutil.copy(src, td / dst)
        for ref in (WORK / "build_v3").glob("ref_*.txt"):
            shutil.copy(ref, td / ref.name)
        ev = stage / "eval"
        ev.mkdir()
        for src, dst in ((WORK / "eval" / "v3" / "table.json", "table_v3.json"),
                         (WORK / "eval" / "v3" / "adoption.json", "adoption.json"),
                         (WORK / "eval" / "v3" / "selection.json", "selection.json"),
                         (WORK / "eval" / "e0" / "table.json", "e0_table.json"),
                         (WORK / "code" / "SELECTION_RULE.md", "SELECTION_RULE.md"),
                         (WORK / "logs" / "decisions.txt", "decisions.txt")):
            if src.is_file():
                shutil.copy(src, ev / dst)
        sel = WORK / "eval" / "v3" / ckpt.as_posix().rstrip("/").replace(":", "_").replace("/", "__")[-80:]
        if (sel / "scored.jsonl").is_file():
            shutil.copy(sel / "scored.jsonl", ev / "selected_scored.jsonl")
            shutil.copy(sel / "scored_accent.jsonl", ev / "selected_scored_accent.jsonl")
        files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        print(f"[publish] {len(files)} files:", *files, sep="\n  ")
        if args.dry_run:
            return
        if V2_TAG not in tags:
            api.create_tag(REPO, tag=V2_TAG, revision=V2_SHA,
                           tag_message="v2: full fine-tune, Japanese only, language Auto")
            print(f"[publish] tagged {V2_SHA} as {V2_TAG}")
        info = api.upload_folder(repo_id=REPO, folder_path=str(stage), commit_message=args.message,
                                 delete_patterns=["training_data/*", "eval/*"])
        print(f"[publish] {info.commit_url}")


if __name__ == "__main__":
    main()
