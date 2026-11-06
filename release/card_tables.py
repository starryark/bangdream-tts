"""Print the model card's voice table for a given eval condition (the published checkpoint's scored.jsonl).

  python release/card_tables.py <scored.jsonl>

Per voice: speaker id, character, band, Japanese minutes, Japanese identity (1 - ja leak) and
Mandarin identity (1 - zh leak) over that checkpoint's renders, and a Mandarin note:
"own voice" if zh leak <= 0.15 (the pre-registered threshold), "partly generic" up to 0.4, else "generic"
(the voice speaks Mandarin in a generic timbre).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

BD = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
    leak = collections.defaultdict(list)
    for r in rows:
        leak[(r["row"], r["lang"])].append(r["pred"] != r["row"])
    L = {k: sum(v) / len(v) for k, v in leak.items()}
    spec = json.loads((BD / "pipeline" / "rows.json").read_text(encoding="utf-8"))
    per = json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))["per_row"]
    print("| `speaker` | Character | Band | Japanese data | Japanese identity | Mandarin identity | Mandarin voice |")
    print("|---|---|---|---:|---:|---:|---|")
    seen = set()
    for k, v in spec.items():
        if k.startswith("_") or v["row"] in seen:
            continue
        r = v["row"]
        seen.add(r)
        zh = L[(r, "zh")]
        print(f"| `{r}` | {v['character']} | {v['band']} | {per[r]['min']:.0f} min | {1 - L[(r, 'ja')]:.2f} | {1 - zh:.2f} | "
              f"{'own voice' if zh <= 0.15 else ('partly generic' if zh <= 0.4 else '**generic**')} |")


if __name__ == "__main__":
    main()
