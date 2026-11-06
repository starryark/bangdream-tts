"""Publish the adopted 49-voice checkpoint to Hugging Face (run locally; the pod never holds an HF token).

  python release/publish_hf.py --ckpt models/<pick> --card release/MODEL_CARD.md [--dry_run]

Release form (user, 2026-11-06): public repo, the TTS checkpoint only (E1b_ep2, the pre-registered
rule's model; no RVC converters), no audio, no Space, and no training transcripts or reference-clip
lists. Uploads, in one commit:
  - the checkpoint directory (model, tokenizer, processor, speech_tokenizer/); its upstream Qwen
    README is replaced by the model card
  - voices.json: speaker id -> character, band, minutes of Japanese training audio
  - eval/: the pre-registered selection outputs and plans/results (numbers only, no per-clip text)
  - training_data/: counts only (build, ASR, mix, pseudo and VC-row reports), no transcripts
Refuses to stage any .wav/.mp3/.flac or any .jsonl, so per-clip text cannot leak in by accident.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

BD = Path(__file__).resolve().parent.parent
REPO = "starrydark/BanG_Dream_Qwen3_TTS"


def voices() -> dict:
    rows = json.loads((BD / "pipeline" / "rows.json").read_text(encoding="utf-8"))
    per = json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))["per_row"]
    out = {}
    for k, v in rows.items():
        if k.startswith("_") or v["row"] in out:
            continue
        out[v["row"]] = {"character": v["character"], "band": v["band"], "source": v.get("source"),
                         "ja_clips": per[v["row"]]["n"], "ja_minutes": per[v["row"]]["min"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--message", default="E1b epoch 2: 49 BanG Dream! voices, Japanese / Mandarin / English")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    spk = cfg["talker_config"]["spk_id"]
    assert cfg["tts_model_type"] == "custom_voice" and len(spk) == 49 and "ppp_kasumi" in spk, "not the 49-voice checkpoint"
    for f in ("model.safetensors", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json"):
        assert (ckpt / f).is_file(), f
    vs = voices()
    assert set(vs) == set(spk), "voices.json and the checkpoint's speakers differ"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(ckpt, stage, ignore=shutil.ignore_patterns("*.pt", "optimizer*", "tb", "README.md"))
        shutil.copy(args.card, stage / "README.md")
        (stage / "voices.json").write_text(json.dumps(vs, ensure_ascii=False, indent=1), encoding="utf-8")
        ev, td = stage / "eval", stage / "training_data"
        ev.mkdir()
        td.mkdir()
        # This checkpoint (E1b epoch 2): its plan, rule, results and training records. The follow-ups
        # that did not replace it (E2, E5) ship as evaluation evidence only.
        for src, dst in ((BD / "e1" / "results" / "eval" / "selection.json", ev / "selection_e1.json"),
                         (BD / "e1" / "results" / "eval" / "selection.md", ev / "selection_e1.md"),
                         (BD / "e1" / "results" / "did_amendA.json", ev / "did_amendA.json"),
                         (BD / "e1" / "E1_PLAN.md", ev / "E1_PLAN.md"),
                         (BD / "e1" / "E1_RESULTS.md", ev / "E1_RESULTS.md"),
                         (BD / "e2" / "results" / "selection_e2.md", ev / "selection_e2.md"),
                         (BD / "e2" / "E2_RESULTS.md", ev / "E2_RESULTS.md"),
                         (BD / "e5" / "results" / "selection_e5.json", ev / "selection_e5.json"),
                         (BD / "e5" / "results" / "selection_e5.md", ev / "selection_e5.md"),
                         (BD / "e5" / "E5_PLAN.md", ev / "E5_PLAN.md"),
                         (BD / "e5" / "E5_RESULTS.md", ev / "E5_RESULTS.md"),
                         (BD / "data" / "build_report.json", td / "build_report.json"),
                         (BD / "data" / "asr_summary.json", td / "asr_summary.json"),
                         (BD / "e1" / "results" / "exp" / "E1b" / "mix.json", td / "mix_E1b.json"),
                         (BD / "e1" / "results" / "pseudo" / "report.json", td / "pseudo_zh_report.json"),
                         (BD / "e5" / "e1_aux_filter.json", td / "replay_aux_filter.json")):
            if src.is_file():
                shutil.copy(src, dst)
            else:
                print(f"[publish] missing (skipped): {src}")
        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
        bad = [f for f in files if f.endswith((".wav", ".mp3", ".flac", ".jsonl"))]
        assert not bad, f"refusing to publish audio or per-clip files: {bad}"
        print(f"[publish] {REPO}: {len(files)} files:", *files, sep="\n  ")
        if args.dry_run:
            return
        api = HfApi()
        api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(repo_id=REPO, folder_path=str(stage), commit_message=args.message)
        print(f"[publish] {info.commit_url}")


if __name__ == "__main__":
    main()
