"""E6 (pod): compose one arm's train/val sets from the coded sources, with hard checks (as e1/e1_mix.py).

  python e6/e6_mix.py --arm E6a --sources pseudo_v1,pseudo_on
  python e6/e6_mix.py --arm E6b --sources pseudo_v1,pseudo_on,vc
  python e6/e6_mix.py --arm E6c --sources pseudo_v1,pseudo_on,nv
  python e6/e6_mix.py --arm E6d --sources pseudo_v1,pseudo_on --replay_share 0.25

Replay rows = round(share / (1 - share) * n_japanese_train), as E1 (E1b: share 0.14 -> 13.3% of the
final set once pseudo rows are added). The replay draw uses the same seed in every arm, so arms with
the same share get the same replay rows. Refuses to write if a shipped row has fewer than MIN_ROWS
train rows, a speaker is unknown, a language is unexpected, a test or eval clip appears, ids repeat,
or the replay pool is short.
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


def dedup(rows: list[dict], tag: str, dropped: Counter) -> list[dict]:
    """Collapse rows that repeat an id for the same clip (same speaker and audio): the Our Notes story reuses
    some voice files across scenes, and the bundle lists them once per scene (52 ja ids, 5 nv ids, 2026-11-08).
    An id naming two different clips still stops the mix."""
    seen, out = {}, []
    for r in rows:
        a = seen.get(r["id"])
        if a is None:
            seen[r["id"]] = r
            out.append(r)
        elif (a["speaker"], a.get("audio")) == (r["speaker"], r.get("audio")):
            dropped[tag] += 1
        else:
            raise SystemExit(f"{tag}: id {r['id']} names two different clips")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sources", default="")
    ap.add_argument("--replay_share", type=float, default=0.14)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    Q = ROOT / "codes"
    shipped = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    aux = set(json.loads((ROOT / "replay" / "aux_speakers.json").read_text(encoding="utf-8")))
    test_ids = {r["real_id"] for r in read(ROOT / "bundle" / "eval_rows.jsonl")} | \
               {r["real_id"] for r in read(ROOT / "bundle" / "eval_short.jsonl") if r.get("real_id")}
    assert len(shipped) == 65 and not (set(shipped) & aux), "speaker sets overlap or wrong size"

    rng = random.Random(args.seed)
    dropped = Counter()
    ja_train = dedup(read(Q / "ja_train_with_codes.jsonl"), "ja_train", dropped)
    ja_val = dedup(read(Q / "ja_val_with_codes.jsonl"), "ja_val", dropped)
    train, val = list(ja_train), list(ja_val)
    pool = read(Q / "replay_train_with_codes.jsonl")
    pool.sort(key=lambda r: r["id"])
    want = int(round(args.replay_share / (1 - args.replay_share) * len(ja_train)))
    if want > len(pool):
        raise SystemExit(f"replay: need {want} rows, pool has {len(pool)} (raise p05 --zh_hours/--en_hours)")
    rng.shuffle(pool)
    train += pool[:want]
    val += read(Q / "replay_val_with_codes.jsonl")
    added = Counter()
    for src in filter(None, args.sources.split(",")):
        tr = dedup(read(Q / f"{src}_train_with_codes.jsonl"), f"{src}_train", dropped)
        if not tr:
            raise SystemExit(f"--sources {src}: codes/{src}_train_with_codes.jsonl missing or empty")
        train += tr
        val += dedup(read(Q / f"{src}_val_with_codes.jsonl"), f"{src}_val", dropped)
        added[src] = len(tr)
    rng.shuffle(train)

    known = set(shipped) | aux
    for name, rows in (("train", train), ("val", val)):
        bad_spk = {r["speaker"] for r in rows} - known
        bad_lang = {r["language"] for r in rows} - {"japanese", "chinese", "english"}
        leaked = [r["id"] for r in rows if r.get("id") in test_ids or r.get("real_id") in test_ids]
        if bad_spk or bad_lang or leaked:
            raise SystemExit(f"{name}: unknown speakers {sorted(bad_spk)[:5]}, languages {bad_lang}, test clips {leaked[:5]}")
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
        (out / f"{name}_with_codes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    langs = Counter(r["language"] for r in train)
    extra = Counter((r["speaker"], r["language"], r.get("source", "")) for r in train
                    if str(r.get("source", "")).startswith(("pseudo", "nv")))
    summary = {"arm": args.arm, "train": len(train), "val": len(val), "steps_per_epoch": round(len(train) / 32, 1),
               "train_hours": round(sum(r["dur"] for r in train) / 3600, 3), "ja_train": len(ja_train), "replay": want,
               "replay_share": args.replay_share, "added": dict(added), "duplicates_dropped": dict(dropped),
               "shares": {"replay": round(want / len(train), 4), **{k: round(v / len(train), 4) for k, v in added.items()}},
               "languages": dict(langs), "shipped_rows_min": min(per.values()), "shipped_rows": dict(sorted(per.items())),
               "extra_per_speaker": {f"{a}|{b}|{c}": v for (a, b, c), v in sorted(extra.items())},
               "aux_speakers_used": len({r["speaker"] for r in train} & aux)}
    (out / "mix.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("shipped_rows", "extra_per_speaker")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
