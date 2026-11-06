"""Apply SELECTION_RULE.md to eval/<suite>/table.json; optionally prune a run.

  python p07_select.py --suite main                     # ranked table
  python p07_select.py --suite main --prune <run_dir> --keep 2

Pruning deletes checkpoint directories of <run_dir> that are not among its best
`--keep` by the rule (scored ones only; unscored checkpoints are left alone).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
TIE = 0.01
GUARD = 1.5


def rank(table: dict) -> list[tuple[str, dict]]:
    clone = next((v for k, v in table.items() if k.startswith("clone:")), None)
    rows = []
    for ckpt, s in table.items():
        if ckpt == "real" or ckpt.startswith("clone:") or not s.get("select"):
            continue
        sel = s["select"]
        guard_ok = True
        if clone:
            for lang in ("en", "zh"):
                if s.get(lang) and clone.get(lang) and clone[lang]["cer_mean"] > 0:
                    guard_ok &= s[lang]["cer_mean"] <= GUARD * clone[lang]["cer_mean"] + 0.02
        rows.append((ckpt, {"eligible": sel["runaway"] == 0, "guard_ok": guard_ok, **sel,
                            "en_cer": (s.get("en") or {}).get("cer_mean"),
                            "zh_cer": (s.get("zh") or {}).get("cer_mean")}))
    eligible = [r for r in rows if r[1]["eligible"]]
    best_cer = min((r[1]["cer_mean"] for r in eligible), default=None)

    def key(r):
        m = r[1]
        tied = best_cer is not None and m["eligible"] and m["cer_mean"] <= best_cer + TIE
        return (not m["eligible"], not m["guard_ok"], not tied,
                -m["sim_mean"] if tied else m["cer_mean"], m["cer_mean"])
    return sorted(rows, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--prune", default=None)
    ap.add_argument("--keep", type=int, default=2)
    args = ap.parse_args()
    table = json.loads((WORK / "eval" / args.suite / "table.json").read_text())
    ranked = rank(table)
    for k in ("real", *[k for k in table if k.startswith("clone:")]):
        if k in table:
            print(f"{'baseline':>8}  {k[-60:]:60}  {json.dumps(table[k].get('select') or table[k].get('all'))}")
    for i, (ckpt, m) in enumerate(ranked):
        print(f"{i + 1:>8}  {ckpt[-60:]:60}  cer {m['cer_mean']:.4f}  sim {m['sim_mean']:.4f}  "
              f"utmos {m.get('utmos_mean')}  runaway {m['runaway']}  en {m['en_cer']}  zh {m['zh_cer']}"
              f"{'' if m['guard_ok'] else '  [guard fail]'}")
    if args.prune:
        run = Path(args.prune).resolve()
        mine = [c for c, _ in ranked if Path(c).resolve().parent == run]
        for c in mine[args.keep:]:
            shutil.rmtree(c, ignore_errors=True)
            print(f"[prune] removed {c}")


if __name__ == "__main__":
    main()
