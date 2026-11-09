"""Voice table and voices.json for the `ournotes` branch (E9_avg; the user's hybrid decision of 2026-11-09).

  python -I -X utf8 release/card_tables_ournotes.py table          # markdown table of the 25 branch voices
  python -I -X utf8 release/card_tables_ournotes.py voices OUT.json  # voices.json for all 74 ids on the branch

Sources: E9's eval scores (identity = the share of renders the probe gives to the voice's character; the two Sakiko
classes both count for Sakiko), the listening answers (e9/results/listening.json), E9's training rows (minutes) and
the labels of v1 (main's voices.json, saved in release/voices_v1_main.json) and of the Our Notes voices.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

BD = Path(__file__).resolve().parent.parent
E9 = BD / "e9"
ERA9 = ["mygo_tomori", "mygo_anon", "mygo_soyo", "mygo_taki", "mujica_uika", "mujica_mutsumi", "mujica_mortis",
        "mujica_umiri", "mujica_sakiko"]
SAME = {"mujica_sakiko_tsuki": "mujica_sakiko", "mygo_soyo_tsuki": "mygo_soyo"}


def branch_voices() -> list[str]:
    """The 25 ids this branch is recommended for, in E9's speaker order."""
    names = [s.split("=")[0] for s in (E9 / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    v1 = set(json.loads((BD / "release" / "voices_v1_main.json").read_text(encoding="utf-8")))
    out = [n for n in names if n not in v1]
    assert len(out) == 25, len(out)
    return out


def character(row: str) -> str:
    return row[:-len("_ournotes")] if row.endswith("_ournotes") else row


def identity() -> dict[tuple[str, str], float]:
    rows = [json.loads(l) for l in (E9 / "results" / "eval" / "E9_avg" / "scored.jsonl").open(encoding="utf-8")]
    hit, n = collections.Counter(), collections.Counter()
    for r in rows:
        lang = {"ja_heldout": "ja", "ja_novel": "ja", "zh": "zh", "en": "en"}.get(r["kind"])
        if lang:
            n[(r["row"], lang)] += 1
            hit[(r["row"], lang)] += SAME.get(r["pred"], r["pred"]) == character(r["row"])
    return {k: hit[k] / n[k] for k in n}


def minutes() -> dict[str, float]:
    m = collections.Counter()
    for l in (E9 / "bundle" / "ja_rows.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == "train":
            m[r["speaker"]] += r["dur"] / 60
    return m


def labels_on() -> dict[str, dict]:
    on = {k: v for k, v in json.loads((BD / "release" / "voices_v2_ournotes.json").read_text(encoding="utf-8")).items()
          if not k.startswith("_")}
    return {r: on[character(r)] for r in branch_voices()}


def zh_voice(x: float) -> str:   # the v1 card's bands: own voice >= 0.85, partly generic >= 0.60
    return "own voice" if x >= 0.85 else ("partly generic" if x >= 0.60 else "**generic**")


def table() -> str:
    idt, mins, lab = identity(), minutes(), labels_on()
    absolute = json.loads((E9 / "results" / "listening.json").read_text(encoding="utf-8"))["absolute"]
    out = ["| `speaker` | Character | Band | Japanese data | Listening (2 answers) | Japanese identity | Mandarin identity | Mandarin voice |",
           "|---|---|---|---:|---|---:|---:|---|"]
    for r in branch_voices():
        grp = "ournotes_abs" if r.endswith("_ournotes") else "new_abs"
        ans = ", ".join(absolute[f"{grp}|{r}"])
        out.append(f"| `{r}` | {lab[r]['character']} | {lab[r]['band']} | {mins[r]:.0f} min | {ans} | "
                   f"{idt[(r, 'ja')]:.2f} | {idt[(r, 'zh')]:.2f} | {zh_voice(idt[(r, 'zh')])} |")
    return "\n".join(out)


def voices() -> dict:
    v1 = json.loads((BD / "release" / "voices_v1_main.json").read_text(encoding="utf-8"))
    mins, lab = minutes(), labels_on()
    out = {"_note": "Branch `ournotes` (E9). Recommended only for the 25 ids marked use=ournotes: 16 new characters and the "
                    "BanG Dream! Our Notes era of nine MyGO!!!!! / Ave Mujica characters (`*_ournotes`). The other 49 ids "
                    "exist here too, but blind listening judged this branch's MyGO!!!!! / Ave Mujica versions worse than "
                    "v1, so use revision main for them."}
    for r, v in v1.items():
        out[r] = {**v, "use": "main"}
    for r in branch_voices():
        out[r] = {**lab[r], "source": "Our Notes story", "ja_minutes": round(mins[r], 2), "use": "ournotes"}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "table":
        print(table())
    elif sys.argv[1] == "voices":
        Path(sys.argv[2]).write_text(json.dumps(voices(), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {sys.argv[2]}")
    else:
        raise SystemExit(__doc__)
