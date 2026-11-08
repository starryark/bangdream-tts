"""E6 (local): the upload lists for the pod, built from what the bundles reference.

  python -I -X utf8 e6/pod_files.py      # writes e6/work/upload_{data,small}.txt (paths relative to bangdream/)

data:  every audio file the pod reads - v1 rows' Japanese rows, eval clips and references (e6 bundle and the
       E1 bundle used for the E1b-recipe pseudo context), Our Notes rows, refs, eval and nv clips,
       the VC rows kept for E6 (E5 minus mygo_tomori / mujica_sakiko_tsuki, plus E6's), models/E1b_ep2.
small: code and bundles (pod_up.sh maps e6/bundle -> bundle, e1/bundle -> e6up/bundle_e1, and the e6up files).
"""
import json
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
POD = "/workspace/bd/"
DROP_VC = {"mygo_tomori", "mujica_sakiko_tsuki"}


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def rel(p: str) -> str:
    p = p.replace("\\", "/")
    if p.startswith(POD):
        return p[len(POD):]
    return Path(p).resolve().relative_to(BD.resolve()).as_posix()


def main() -> None:
    files = set()
    for b in (BD / "e6" / "bundle", BD / "e1" / "bundle"):
        files |= {rel(r["audio"]) for r in rd(b / "ja_rows.jsonl")}
        files |= {rel(r["real_audio"]) for r in rd(b / "eval_rows.jsonl")}
        for f in (b / "refs").glob("*.txt"):
            files |= {rel(l.strip()) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    files |= {rel(r["real_audio"]) for r in rd(BD / "e6" / "bundle" / "eval_short.jsonl") if r.get("real_audio")}
    files |= {rel(r["audio"]) for r in rd(BD / "e6" / "bundle" / "nv_rows.jsonl")}
    vc5 = [r for r in rd(BD / "e5" / "vc" / "rvc_rows.jsonl") if r["speaker"] not in DROP_VC]
    vc6 = rd(BD / "e6" / "vc" / "rvc_rows.jsonl")
    files |= {rel(r["audio"]) for r in vc5 + vc6}
    files |= {p.relative_to(BD).as_posix() for p in (BD / "models" / "E1b_ep2").rglob("*") if p.is_file()}
    missing = [f for f in files if not (BD / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing, e.g. {missing[:3]}")
    size = sum((BD / f).stat().st_size for f in files) / 2 ** 30
    (BD / "e6" / "work" / "upload_data.txt").write_text("\n".join(sorted(files)) + "\n", encoding="utf-8", newline="\n")
    small = ["e6/bundle", "e1/bundle", "e5/e1_pseudo_report.json", "e5/vc/rvc_rows.jsonl"] + \
            (["e6/vc/rvc_rows.jsonl"] if (BD / "e6" / "vc" / "rvc_rows.jsonl").exists() else []) + \
            [p.relative_to(BD).as_posix() for d in ("e1", "e6", "pipeline") for p in (BD / d).glob("*.py")] + \
            [p.relative_to(BD).as_posix() for p in (BD / "e6").glob("*.sh")] + ["pipeline/rows.json"]
    (BD / "e6" / "work" / "upload_small.txt").write_text("\n".join(small) + "\n", encoding="utf-8", newline="\n")
    print(f"[pod_files] data: {len(files)} files, {size:.2f} GiB; small: {len(small)} entries; VC rows E5 {len(vc5)} + E6 {len(vc6)}")


if __name__ == "__main__":
    main()
