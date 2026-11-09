"""Publish E9_avg as the opt-in branch `ournotes` of the model repo (run locally; pods never hold an HF token).

  python -I -X utf8 release/publish_hf_ournotes.py --ckpt models/E9_avg [--dry_run]

Release form (user, 2026-11-09, after E9 failed its pre-registered rule): `main` stays v1 (E1b epoch 2, 49 voices)
and is not touched. E9_avg goes to the separate branch `ournotes`, recommended only for its 25 voices that passed
listening (16 new characters + 9 `*_ournotes`). The checkpoint only: no audio, transcripts or reference lists.
  - Refuses unless the checkpoint has 74 ids on rows 3000-3071 (v1's 49 ids, incl. the two aliases, plus the 25),
    and E9_RESULTS.md records the hybrid decision.
  - One commit on the branch: the checkpoint (upstream Qwen README replaced by the branch card), voices.json (74 ids,
    `use` = main / ournotes), eval/ (E9 plan, results, listening analysis, screens) and training_data/ (counts).
    v1's eval/ and training_data/ files are removed on the branch only.
  - Refuses any .wav/.mp3/.flac/.jsonl, and any added file containing a game line (Our Notes or v1 manifest text,
    original or official translation, >= 8 characters).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

BD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BD / "release"))
import card_tables_ournotes as T  # noqa: E402

REPO = "starrydark/BanG_Dream_Qwen3_TTS"
BRANCH = "ournotes"
E9 = BD / "e9"
R = E9 / "results"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def game_lines() -> list[str]:
    out = set()
    for p, fields in ((BD / "data_on" / "manifest.jsonl", ("text", "ja_raw", "zh", "en")),
                      (BD / "data" / "manifest.jsonl", ("text",))):
        if p.is_file():
            for r in jl(p):
                for f in fields:
                    t = (r.get(f) or "").strip()
                    if len(t) >= 8 and not (t.isascii() and " " not in t):   # single ASCII words are names ("millsage")
                        out.add(t)
    assert len(out) > 1000, "manifests missing: the transcript check would be empty"
    return sorted(out)


def leak_check(files: list[Path]) -> list[tuple[str, str]]:
    """Every game line >= 8 chars that appears verbatim in one of the files (8-gram prefilter)."""
    K = 8
    texts = {p: p.read_text(encoding="utf-8", errors="ignore") for p in files}
    grams = {s[i:i + K] for s in texts.values() for i in range(len(s) - K + 1)}
    hits = []
    for t in game_lines():
        if t[:K] in grams:
            hits += [(p.name, t) for p, s in texts.items() if t in s]
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--card", default=str(BD / "release" / "MODEL_CARD_ournotes.md"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    assert "## Decision after the verdict" in (E9 / "E9_RESULTS.md").read_text(encoding="utf-8"), "hybrid decision not recorded"
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    spk = cfg["talker_config"]["spk_id"]
    v1 = set(json.loads((BD / "release" / "voices_v1_main.json").read_text(encoding="utf-8")))
    ours = T.branch_voices()
    assert cfg["tts_model_type"] == "custom_voice" and set(spk) == v1 | set(ours) and len(spk) == 74, "not the 74-id E9 checkpoint"
    assert sorted(set(spk.values())) == list(range(3000, 3072)), "speaker rows are not 3000-3071"
    assert spk["mygo_soyo_tsuki"] == spk["mygo_soyo"] and spk["mujica_sakiko_tsuki"] == spk["mujica_sakiko"], "aliases"
    for f in ("model.safetensors", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json"):
        assert (ckpt / f).is_file(), f

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(ckpt, stage, ignore=shutil.ignore_patterns("*.pt", "optimizer*", "tb", "README.md", ".gitattributes"))
        shutil.copy(args.card, stage / "README.md")
        ev, td = stage / "eval", stage / "training_data"
        ev.mkdir()
        td.mkdir()
        (stage / "voices.json").write_text(json.dumps(T.voices(), ensure_ascii=False, indent=1), encoding="utf-8")
        for src, dst in ((E9 / "E9_PLAN.md", ev / "E9_PLAN.md"),
                         (E9 / "E9_RESULTS.md", ev / "E9_RESULTS.md"),
                         (R / "listening.json", ev / "listening_e9.json"),
                         (R / "eval" / "E9_avg" / "screens.json", ev / "screens_e9.json"),
                         (R / "eval" / "E9_avg" / "summary.json", ev / "summary_e9.json"),
                         (R / "exp" / "E9" / "mix.json", td / "mix_E9.json"),
                         (E9 / "bundle" / "e9_bundle_report.json", td / "bundle_report_e9.json"),
                         (BD / "e8" / "results" / "pseudo_on" / "report.json", td / "pseudo_zh_ournotes_report.json")):
            assert src.is_file(), f"missing {src}"
            shutil.copy(src, dst)

        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
        bad = [f for f in files if f.endswith((".wav", ".mp3", ".flac", ".jsonl"))]
        assert not bad, f"refusing to publish audio or per-clip files: {bad}"
        added = [stage / "README.md", stage / "voices.json"] + sorted(ev.iterdir()) + sorted(td.iterdir())
        hits = leak_check(added)
        assert not hits, f"refusing: game lines in {len(hits)} places, e.g. {hits[:5]}"
        print(f"[publish] {REPO}@{BRANCH}: {len(files)} files; transcript check passed on {len(added)} added files",
              *files, sep="\n  ")
        if args.dry_run:
            return
        from huggingface_hub import HfApi
        api = HfApi()
        main_sha = api.model_info(REPO, revision="main").sha
        if BRANCH not in [b.name for b in api.list_repo_refs(REPO).branches]:
            api.create_branch(REPO, branch=BRANCH, revision=main_sha)
            print(f"[publish] branch {BRANCH} created from main {main_sha}")
        info = api.upload_folder(repo_id=REPO, folder_path=str(stage), revision=BRANCH,
                                 delete_patterns=["eval/*", "training_data/*"],
                                 commit_message="ournotes (E9): 25 BanG Dream! Our Notes voices; opt-in, main stays v1")
        print(f"[publish] {info.commit_url}")
        assert api.model_info(REPO, revision="main").sha == main_sha, "main moved during the upload"


if __name__ == "__main__":
    main()
