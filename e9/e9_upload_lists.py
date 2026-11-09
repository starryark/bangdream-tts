"""E9 (local): upload lists for the pod -> e9/work/upload_{data,small}.txt (paths relative to bangdream/).

  python -I -X utf8 e9/e9_upload_lists.py

data:  every audio file the pod reads, at the same relative path as here: the E9 bundle's training, eval, short-line
       and reference clips; the pseudo-zh wavs (e8/results/...); the VC wavs (e5/rows, e6/rows); models/E1b_ep2.
small: code and bundles. e9/bundle goes to bundle/ (pod_up.sh's tar transform); E6's scored C0 to eval/C0.
"""
from __future__ import annotations

import json
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
POD = "/workspace/bd/"


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def rel(p: str) -> str:
    p = p.replace("\\", "/")
    assert p.startswith(POD), p
    return p[len(POD):]


def main() -> None:
    B = BD / "e9" / "bundle"
    files = set()
    files |= {rel(r["audio"]) for r in rd(B / "ja_rows.jsonl")}
    files |= {rel(r["real_audio"]) for r in rd(B / "eval_rows.jsonl")}
    files |= {rel(r["real_audio"]) for r in rd(B / "eval_short.jsonl") if r.get("real_audio") not in (None, "None", "")}
    for f in (B / "refs").glob("*.txt"):
        files |= {rel(l) for l in f.read_text(encoding="utf-8").split()}
    for f in ("pseudo_v1_rows.jsonl", "pseudo_on_rows.jsonl", "vc_rows.jsonl"):
        files |= {rel(r["audio"]) for r in rd(BD / "e9" / "rows" / f)}
    files |= {p.relative_to(BD).as_posix() for p in (BD / "models" / "E1b_ep2").rglob("*") if p.is_file()}
    missing = sorted(f for f in files if not (BD / f).is_file())
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    size = sum((BD / f).stat().st_size for f in files) / 2 ** 30
    W = BD / "e9" / "work"
    W.mkdir(parents=True, exist_ok=True)
    (W / "upload_data.txt").write_text("".join(f + "\n" for f in sorted(files)), encoding="utf-8", newline="\n")
    small = ([p.relative_to(BD).as_posix() for d in ("e1", "e6", "pipeline") for p in (BD / d).glob("*.py")]
             + ["pipeline/rows.json", "e8/e8_average.py", "e8/e8_listen_render.py", "e9/listen_items.json"]
             + [p.relative_to(BD).as_posix() for p in (BD / "e9").glob("*.py")] + [p.relative_to(BD).as_posix() for p in (BD / "e9").glob("*.sh")]
             + [p.relative_to(BD).as_posix() for p in (BD / "e9" / "rows").glob("*.jsonl")]
             + [p.relative_to(BD).as_posix() for p in B.rglob("*") if p.is_file()]
             + ["e5/e1_pseudo_report.json"]
             + [f"e6/results/eval/C0/{f}" for f in ("manifest.jsonl", "scored.jsonl", "summary.json")])
    (W / "upload_small.txt").write_text("".join(f + "\n" for f in small), encoding="utf-8", newline="\n")
    print(f"[e9 upload] data: {len(files)} files, {size:.2f} GiB; small: {len(small)} entries")


if __name__ == "__main__":
    main()
