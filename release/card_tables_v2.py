"""Print the v2 model card's voice table from the released condition's scored renders (65 voices).

  python -I -X utf8 release/card_tables_v2.py e6/results/eval/<release>/scored.jsonl

Per voice: speaker id, character, band, data source, minutes of Japanese training audio (the E6 bundle's
train split), Japanese identity (1 - leak over held-out game lines + new sentences), short-line identity
(1 - leak over the short lines: the row's own held-out short game lines for the 28 Our Notes rows, one
shared 16-line set for the 37 v1 rows), Mandarin identity (1 - zh leak) and the v1 card's Mandarin note
("own voice" if zh leak <= 0.15, "partly generic" up to 0.4, else "generic"). Leak = the probe's top
voice is not this row (persona pairs counted as errors, as in the v1 card).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

BD = Path(__file__).resolve().parent.parent
B = BD / "e6" / "bundle"


def table_rows() -> list[dict]:
    names = [s.split("=")[0] for s in (B / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    v1 = {v["row"]: v for k, v in json.loads((BD / "pipeline" / "rows.json").read_text(encoding="utf-8")).items()
          if not k.startswith("_")}
    on = {k: v for k, v in json.loads((BD / "release" / "voices_v2_ournotes.json").read_text(encoding="utf-8")).items()
          if not k.startswith("_")}
    mins = collections.Counter()
    for l in (B / "ja_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == "train":
            mins[r["speaker"]] += r["dur"] / 60
    out = []
    for k in names:
        if k in on:
            m = {**on[k], "source": "Our Notes story"}
        else:
            src = v1[k].get("source")
            m = {"character": v1[k]["character"], "band": v1[k]["band"], "persona": v1[k].get("persona"),
                 "source": {"game": "Girls Band Party", "anime": "anime", "mixed": "anime + game"}.get(src, src)}
        out.append({"row": k, **m, "ja_train_min": round(mins[k], 1)})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sc = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
    group = {"ja_heldout": "ja", "ja_novel": "ja", "ja_short": "short", "zh": "zh", "en": "en"}
    leak = collections.defaultdict(list)
    for r in sc:
        if r["kind"] in group and r.get("cond", "") not in ("real_raw", "real_resynth"):
            leak[(r["row"], group[r["kind"]])].append(r["pred"] != r["row"])
    L = {k: sum(v) / len(v) for k, v in leak.items()}
    print("| `speaker` | Character | Band | Source | Japanese data | Japanese identity | Short-line identity | Mandarin identity | Mandarin voice |")
    print("|---|---|---|---|---:|---:|---:|---:|---|")
    ident = lambda r, g: f"{1 - L[(r, g)]:.2f}" if (r, g) in L else "—"   # noqa: E731
    missing = [t["row"] for t in table_rows() if (t["row"], "zh") not in L]
    for t in table_rows():
        r = t["row"]
        zh = L.get((r, "zh"))
        note = "—" if zh is None else ("own voice" if zh <= 0.15 else ("partly generic" if zh <= 0.4 else "**generic**"))
        ch = t["character"] + (f" ({t['persona']})" if t.get("persona") else "")
        print(f"| `{r}` | {ch} | {t['band']} | {t['source']} | {t['ja_train_min']:.0f} min | {ident(r, 'ja')} | "
              f"{ident(r, 'short')} | {ident(r, 'zh')} | {note} |")
    if missing:
        print(f"\nWARNING: no renders for {len(missing)} rows: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
