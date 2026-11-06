"""E1 step 0 (local): pod-ready Japanese rows, reference lists and the upload bundle.

  python -I e1_build.py [--pod_root /workspace/bd]

Rows: every E0 "train" clip whose transcript verdict is ok, minus DROP_IDS. The E0
held-out "test" clips (414) never enter training: they are the paired Japanese eval
lines. From the train clips, max(2, round(1%)) per row become "val" (diagnostic val loss
only), chosen with a seed per row. Speaker = row id, language "japanese".

Reference lists for the trainer's locked vectors are the E0 lists (refs/ref_<row>.txt),
re-pointed at the pod paths. They are byte-for-byte the clips whose vectors E0 measured
(the floor), and no ref clip is a val or test clip.

Writes e1/bundle/{ja_rows.jsonl, refs/ref_<row>.txt, speakers.txt, eval_rows.jsonl,
probe_codec.npz, rows.json, probes_ja_novel.jsonl} and prints per-row counts. The audio itself is sent separately
(data/wav24 as-is, same relative layout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import bdcommon as C  # noqa: E402

DROP_IDS = {"mygo_tomori_d788f622e9"}   # E0 b02: script and audio disagree (絆創膏/楽屋 vs パソコン/学園)
MIN_TRAIN_ROWS = 30


def pod_audio(local: str, pod_root: str) -> str:
    rel = Path(local).resolve().relative_to(C.DATA.resolve())
    return f"{pod_root}/data/{rel.as_posix()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod_root", default="/workspace/bd")
    args = ap.parse_args()
    man = C.read_jsonl(C.DATA / "manifest.jsonl")
    split = {r["id"]: r["split"] for r in C.read_jsonl(C.DATA / "split.jsonl")}
    asr = {r["id"]: r for r in C.read_jsonl(C.DATA / "asr.jsonl")}
    rows_meta = C.load_rows()
    names = [m["row"] for m in rows_meta.values()]
    assert len(names) == len(set(names)) == 49, "rows.json must define 49 unique rows"

    out = HERE / "bundle"
    if out.exists():
        shutil.rmtree(out)
    (out / "refs").mkdir(parents=True)

    ref_ids = {}
    for k in names:
        lines = (C.OUT / "refs" / f"ref_{k}.txt").read_text(encoding="utf-8").split()
        ref_ids[k] = {Path(p).stem for p in lines}
        (out / "refs" / f"ref_{k}.txt").write_text(
            "".join(pod_audio(p, args.pod_root) + "\n" for p in lines), encoding="utf-8", newline="\n")

    ja, val_n, eval_rows = [], Counter(), []
    for k in names:
        cand = sorted((r for r in man if r["row"] == k and split[r["id"]] == "train"
                       and asr[r["id"]]["verdict"] == "ok" and r["id"] not in DROP_IDS),
                      key=lambda r: r["id"])
        # val never takes a reference clip, so the locked vector stays a train-only quantity
        pool = [r for r in cand if r["id"] not in ref_ids[k]]
        n_val = max(2, round(0.01 * len(cand)))
        rng = random.Random(int(hashlib.sha256(f"val:{k}".encode()).hexdigest()[:8], 16))
        val = {r["id"] for r in rng.sample(pool, n_val)}
        for r in cand:
            s = "val" if r["id"] in val else "train"
            val_n[k] += s == "val"
            ja.append({"id": r["id"], "audio": pod_audio(r["audio"], args.pod_root), "text": r["text"],
                       "language": "japanese", "speaker": k, "split": s, "dur": r["dur"], "source": "ja"})
        for r in man:
            if r["row"] == k and split[r["id"]] == "test":
                eval_rows.append({"real_id": r["id"], "row": k, "lang": "ja", "text": r["text"],
                                  "tid": f"ja_{r['id'].rsplit('_', 1)[1]}",
                                  "real_audio": pod_audio(r["audio"], args.pod_root)})

    counts = Counter(r["speaker"] for r in ja if r["split"] == "train")
    low = {k: counts[k] for k in names if counts[k] < MIN_TRAIN_ROWS}
    if low:
        raise SystemExit(f"rows below {MIN_TRAIN_ROWS} train clips: {low}")
    # no test clip, dropped clip or review clip may be in the training rows
    bad = [r["id"] for r in ja if split[r["id"]] != "train" or r["id"] in DROP_IDS
           or asr[r["id"]]["verdict"] != "ok"]
    assert not bad, bad[:5]
    assert not ({r["id"] for r in ja} & {e["real_id"] for e in eval_rows})

    C.write_jsonl(out / "ja_rows.jsonl", ja)
    C.write_jsonl(out / "eval_rows.jsonl", eval_rows)
    (out / "speakers.txt").write_text(
        ",".join(f"{k}={args.pod_root}/bundle/refs/ref_{k}.txt" for k in names), encoding="utf-8")
    # the codec-calibrated probe (b08): every rendered clip carries the codec channel
    shutil.copy(C.OUT / "vectors" / "probe_codec.npz", out / "probe_codec.npz")
    shutil.copy(C.BD / "pipeline" / "rows.json", out / "rows.json")
    shutil.copy(HERE / "probes_ja_novel.jsonl", out / "probes_ja_novel.jsonl")
    print(f"[e1_build] ja rows {len(ja)} (train {sum(counts.values())}, val {sum(val_n.values())}); "
          f"eval ja lines {len(eval_rows)}; rows {len(names)}; min train rows "
          f"{min(counts.values())} ({min(counts, key=counts.get)})")


if __name__ == "__main__":
    main()
