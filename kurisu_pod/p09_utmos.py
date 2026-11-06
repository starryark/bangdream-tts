"""UTMOSv2 for already-generated eval clips of a few finalists (select-set rows only).

  python p09_utmos.py --suite confirm --tag <dir> --tag <dir> ...

Too slow to run on every checkpoint (<3 clips/s on the pod's CPU feature path),
so it is a naturalness proxy for the finalists, reported next to CER/SIM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="confirm")
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()
    import utmosv2
    model = utmosv2.create_model(pretrained=True)
    out = {}
    for tag in args.tag:
        d = WORK / "eval" / args.suite / tag
        rows = [json.loads(l) for l in (d / "scored.jsonl").read_text(encoding="utf-8").splitlines() if l]
        rows = [r for r in rows if r.get("kind") in ("ja_heldout", "ja_novel")]
        scores = [float(model.predict(input_path=str(d / r["file"]), verbose=False)) for r in rows]
        out[tag] = {"n": len(scores), "utmos_mean": round(float(np.mean(scores)), 3),
                    "utmos_p10": round(float(np.percentile(scores, 10)), 3)}
        print(f"[utmos] {tag}: {out[tag]}", flush=True)
    (WORK / "eval" / args.suite / "utmos_finalists.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
