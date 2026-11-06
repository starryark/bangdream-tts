"""E5 (pod): check the regenerated E1b pseudo rows against E1, then add the VC rows (E5_PLAN.md §5).

  python e5/e5_pod_rows.py

Runs after e1/e1_pseudo.py (which regenerates E1b's Base pseudo-zh rows: same texts, seeds, gates
and cap) and before e1/e1_codes.py. 1) Per voice, the regenerated kept count is compared with E1's
(e5/e1_pseudo_report.json, copied from e1/results/pseudo/report.json); differences are printed and
any voice off by more than 3 is listed for disclosure, not corrected. 2) The VC rows
(e5/rvc_rows.jsonl, written locally by e5_collect.py rows) are appended to pseudo/pseudo_rows.jsonl
with their audio paths moved under BD_POD; the E1-only file is kept as pseudo/pseudo_rows_e1.jsonl,
so a rerun rebuilds from it.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
LOCAL = "E:\\STEINS;GATEVoice\\bangdream"
TOL = 3


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ps, e1only = ROOT / "pseudo" / "pseudo_rows.jsonl", ROOT / "pseudo" / "pseudo_rows_e1.jsonl"
    if not e1only.exists():
        e1only.write_text(ps.read_text(encoding="utf-8"), encoding="utf-8")
    e1 = read(e1only)
    new = json.loads((ROOT / "pseudo" / "report.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "e5" / "e1_pseudo_report.json").read_text(encoding="utf-8"))
    diff = {k: new[k]["kept"] - old[k]["kept"] for k in old if new[k]["kept"] != old[k]["kept"]}
    print(f"[e5_rows] E1 pseudo regenerated: {len(e1)} rows (E1: {sum(v['kept'] for v in old.values())}); "
          f"per-voice differences {diff or 'none'}; off by > {TOL}: {[k for k, d in diff.items() if abs(d) > TOL] or 'none'}", flush=True)

    vc = read(ROOT / "e5" / "rvc_rows.jsonl")
    for r in vc:
        r["audio"] = str(ROOT / Path(r["audio"].replace(LOCAL + "\\", "").replace("\\", "/")))
    missing = [r["audio"] for r in vc if not os.path.exists(r["audio"])]
    if missing:
        raise SystemExit(f"{len(missing)} VC row audio files missing, e.g. {missing[:3]}")
    ids = Counter(r["id"] for r in e1 + vc)
    if any(v > 1 for v in ids.values()):
        raise SystemExit(f"duplicate ids: {[k for k, v in ids.items() if v > 1][:5]}")
    ps.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e1 + vc), encoding="utf-8")
    per = Counter(r["speaker"] for r in vc)
    print(f"[e5_rows] + {len(vc)} VC rows ({sum(r['split'] == 'val' for r in vc)} val): {dict(sorted(per.items()))}", flush=True)


if __name__ == "__main__":
    main()
