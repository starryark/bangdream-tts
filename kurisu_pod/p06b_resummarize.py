"""Recompute every summary in eval/<suite> from its scored.jsonl (no regeneration).

Used when an aggregate definition changes (e.g. the runaway calibration), so all
entries in table.json are computed by the same code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import metrics
from p06_eval import tag_of

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    args = ap.parse_args()
    suite = WORK / "eval" / args.suite
    table = json.loads((suite / "table.json").read_text())
    for ckpt in table:
        d = suite / tag_of(ckpt)
        if (d / "scored.jsonl").exists():
            table[ckpt] = metrics.summarize(d)
            print(f"[resummarize] {ckpt}: {json.dumps(table[ckpt].get('select'))}")
    (suite / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
