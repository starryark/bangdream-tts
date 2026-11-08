"""Publish v2 (the checkpoint E6's pre-registered rule released) to Hugging Face (run locally; the pod
never holds an HF token).

  python -I -X utf8 release/publish_hf_v2.py --ckpt models/<release> --card release/MODEL_CARD_v2.md [--dry_run]

Release form (user, 2026-11-07, D4): public, same repo as v1, only if E6's rule adopts a candidate; the
TTS checkpoint only (no RVC converters), no audio, no Space, no transcripts or reference-clip lists.
  - Refuses unless e6/results/eval/selection.json "release" names this checkpoint and it has the 65
    E6 speakers at codec rows 3000-3064.
  - Tags the current main revision "v1" (E1b epoch 2, 49 voices) before the upload, if the tag is missing,
    so v1 stays downloadable with revision="v1".
  - One commit: the checkpoint (upstream Qwen README replaced by the card), voices.json, eval/ (E6 plan,
    results, selection, probe calibration) and training_data/ (counts only). The v1 eval/ and
    training_data/ files are removed from main in the same commit (they stay under the v1 tag).
  - Refuses any .wav/.mp3/.flac/.jsonl, and any added file containing a game line (Our Notes or v1
    manifest text, original or official translation, >= 8 characters).
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

BD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BD / "release"))
import card_tables_v2 as T  # noqa: E402

REPO = "starrydark/BanG_Dream_Qwen3_TTS"
E6 = BD / "e6"
R = E6 / "results"


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
    assert out, "no manifests found for the transcript check"
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
    ap.add_argument("--card", required=True)
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    sel = json.loads((R / "eval" / "selection.json").read_text(encoding="utf-8"))
    rel = sel.get("release")
    assert rel and ckpt.name == rel, f"E6's rule released {rel!r}, not {ckpt.name!r}: nothing to publish (v1 stays)"
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    spk = cfg["talker_config"]["spk_id"]
    names = [t["row"] for t in T.table_rows()]
    assert cfg["tts_model_type"] == "custom_voice" and set(spk) == set(names) and len(spk) == 65, "not the 65-voice E6 checkpoint"
    assert sorted(spk.values()) == list(range(3000, 3065)), "speaker rows are not 3000-3064"
    for f in ("model.safetensors", "speech_tokenizer/model.safetensors", "speech_tokenizer/config.json"):
        assert (ckpt / f).is_file(), f
    arm = rel.split("_ep")[0]

    voices = {t["row"]: {k: v for k, v in t.items() if k != "row" and v is not None} for t in T.table_rows()}
    man = jl(BD / "data_on" / "manifest.jsonl")
    on_counts = {row: dict(collections.Counter(f"{r['split']}_{r['kind']}" for r in man if r["row"] == row))
                 for row in sorted({r["row"] for r in man})}
    rep = json.loads((E6 / "bundle" / "selection_report.json").read_text(encoding="utf-8"))
    rep.pop("shared_short", None)            # carries eval line texts
    added = {"pseudo_v1": collections.Counter(r["speaker"] for r in jl(R / "pseudo_v1" / "rows.jsonl")),
             "pseudo_on": collections.Counter(r["speaker"] for r in jl(R / "pseudo_on" / "pseudo_rows.jsonl")),
             "vc": collections.Counter(r["speaker"] for r in jl(R / "vc" / "vc_rows.jsonl"))}

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(ckpt, stage, ignore=shutil.ignore_patterns("*.pt", "optimizer*", "tb", "README.md"))
        shutil.copy(args.card, stage / "README.md")
        ev, td = stage / "eval", stage / "training_data"
        ev.mkdir()
        td.mkdir()
        (stage / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=1), encoding="utf-8")
        for src, dst in ((R / "eval" / "selection.json", ev / "selection_e6.json"),
                         (R / "eval" / "selection.md", ev / "selection_e6.md"),
                         (E6 / "E6_PLAN.md", ev / "E6_PLAN.md"),
                         (E6 / "E6_RESULTS.md", ev / "E6_RESULTS.md"),
                         (E6 / "bundle" / "probe_e6_calib.json", ev / "probe_e6_calib.json"),
                         (R / "exp" / arm / "mix.json", td / f"mix_{arm}.json"),
                         (R / "pseudo_on" / "report.json", td / "pseudo_zh_ournotes_report.json")):
            assert src.is_file(), f"missing {src}"
            shutil.copy(src, dst)
        (td / "ournotes_build_counts.json").write_text(json.dumps(on_counts, indent=1), encoding="utf-8")
        (td / "row_selection_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        (td / "added_rows_per_voice.json").write_text(json.dumps({k: dict(sorted(v.items())) for k, v in added.items()}, indent=1), encoding="utf-8")

        files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
        bad = [f for f in files if f.endswith((".wav", ".mp3", ".flac", ".jsonl"))]
        assert not bad, f"refusing to publish audio or per-clip files: {bad}"
        ours = [stage / "README.md", stage / "voices.json"] + sorted(ev.iterdir()) + sorted(td.iterdir())
        hits = leak_check(ours)
        assert not hits, f"refusing: game lines in {len(hits)} places, e.g. {hits[:5]}"
        print(f"[publish] {REPO}: {len(files)} files; transcript check passed on {len(ours)} added files", *files, sep="\n  ")
        if args.dry_run:
            return
        from huggingface_hub import HfApi
        api = HfApi()
        tags = [t.name for t in api.list_repo_refs(REPO).tags]
        if "v1" not in tags:
            sha = api.model_info(REPO).sha
            api.create_tag(REPO, tag="v1", revision=sha, tag_message="v1: E1b epoch 2, 49 voices (2026-11-06)")
            print(f"[publish] tagged v1 at {sha}")
        info = api.upload_folder(repo_id=REPO, folder_path=str(stage), delete_patterns=["eval/*", "training_data/*"],
                                 commit_message=args.message or f"v2 ({rel}): 65 voices, MyGO!!!!! / Ave Mujica retrained on Our Notes, 16 new characters")
        print(f"[publish] {info.commit_url}")


if __name__ == "__main__":
    main()
