"""E1 (pod): compose one arm's train/val sets from the coded sources, with hard checks.

  python e1_mix.py --arm E1a                 # Japanese + replay
  python e1_mix.py --arm E1b --pseudo        # + every gated pseudo-zh row (capped per voice upstream)

Replay rows = round(share / (1 - share) * n_japanese_train): the share is of the final
train set and is computed from the Japanese rows only, so pseudo rows (E1b) do not change
the replay amount between arms. Refuses to write if any of the 49 shipped speakers has
fewer than MIN_ROWS train rows, if a row's speaker is unknown (not shipped, not aux), if a
language is not japanese/chinese/english, or if an E0 test clip appears anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
MIN_ROWS = 30


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--replay_share", type=float, default=0.14)
    ap.add_argument("--pseudo", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    Q = ROOT / "codes"
    shipped = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    aux = set(json.loads((ROOT / "replay" / "aux_speakers.json").read_text(encoding="utf-8")))
    test_ids = {r["real_id"] for r in read(ROOT / "bundle" / "eval_rows.jsonl")}
    assert len(shipped) == 49 and not (set(shipped) & aux), "speaker sets overlap or wrong size"

    rng = random.Random(args.seed)
    ja_train, ja_val = read(Q / "ja_train_with_codes.jsonl"), read(Q / "ja_val_with_codes.jsonl")
    train, val = list(ja_train), list(ja_val)
    pool = read(Q / "replay_train_with_codes.jsonl")
    want = int(round(args.replay_share / (1 - args.replay_share) * len(ja_train)))
    if want > len(pool):
        raise SystemExit(f"replay: need {want} rows, pool has {len(pool)} (raise p05 --zh_hours/--en_hours)")
    rng.shuffle(pool)
    train += pool[:want]
    val += read(Q / "replay_val_with_codes.jsonl")
    n_pseudo = 0
    if args.pseudo:
        ps = read(Q / "pseudo_train_with_codes.jsonl")
        if not ps:
            raise SystemExit("--pseudo: codes/pseudo_train_with_codes.jsonl missing or empty")
        train += ps
        val += read(Q / "pseudo_val_with_codes.jsonl")
        n_pseudo = len(ps)
    rng.shuffle(train)

    # checks -------------------------------------------------------------------------
    known = set(shipped) | aux
    for name, rows in (("train", train), ("val", val)):
        bad_spk = {r["speaker"] for r in rows} - known
        bad_lang = {r["language"] for r in rows} - {"japanese", "chinese", "english"}
        leaked = [r["id"] for r in rows if r.get("id") in test_ids or r.get("real_id") in test_ids]
        if bad_spk or bad_lang or leaked:
            raise SystemExit(f"{name}: unknown speakers {sorted(bad_spk)[:5]}, languages {bad_lang}, "
                             f"test clips {leaked[:5]}")
        if any("audio_codes" not in r for r in rows):
            raise SystemExit(f"{name}: rows without audio_codes")
    ids = Counter(r["id"] for r in train)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate train ids: {dup[:5]}")
    if {r["id"] for r in train} & {r["id"] for r in val}:
        raise SystemExit("train/val overlap")
    per = Counter(r["speaker"] for r in train if r["speaker"] in set(shipped))
    low = {k: per[k] for k in shipped if per[k] < MIN_ROWS}
    if low:
        raise SystemExit(f"shipped speakers below {MIN_ROWS} train rows: {low}")

    out = ROOT / "exp" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}_with_codes.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    langs = Counter(r["language"] for r in train)
    pseudo_per = Counter((r["speaker"], r["language"]) for r in train if str(r.get("source", "")).startswith("pseudo"))
    summary = {"arm": args.arm, "train": len(train), "val": len(val),
               "train_hours": round(sum(r["dur"] for r in train) / 3600, 3),
               "ja_train": len(ja_train), "replay": want, "pseudo": n_pseudo,
               "shares": {"replay": round(want / len(train), 4), "pseudo": round(n_pseudo / len(train), 4)},
               "languages": dict(langs), "shipped_rows_min": min(per.values()),
               "shipped_rows": dict(sorted(per.items())), "pseudo_per_speaker": {f"{k[0]}|{k[1]}": v for k, v in sorted(pseudo_per.items())},
               "aux_speakers_used": len({r["speaker"] for r in train} & aux)}
    (out / "mix.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("shipped_rows", "pseudo_per_speaker")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
