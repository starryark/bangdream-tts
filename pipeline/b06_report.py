"""E0 step 6: per-voice floor table and bucket summary from e0/floor_table.json.

  python -I b06_report.py

Prints a markdown table (one line per row, sorted by minutes of data) and writes it to
e0/floor_table.md. Columns:
  min       minutes of unique audio in the row (data/build_report.json)
  sim real  centred sim of real held-out clips to the locked vector (the target)
  sim ja/zh/en   same for the Base zero-shot floor renders
  p_true    leakage probe P(own row), floor ja | real ja
  leak      share of floor clips (all languages) the probe assigns to another row
  ja CER, zh CER, en WER   Qwen3-ASR, language forced
  zh SER    shared-char toneless syllable error (Kurisu native anchor 0.048)
  zh lid    VoxLingua log P(ja) - log P(zh) (Kurisu native anchor -22.6; higher = more Japanese)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402

BUCKETS = [("<10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30+", 30, float("inf"))]


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def main() -> None:
    import numpy as np

    t = json.loads((C.OUT / "e0" / "floor_table.json").read_text(encoding="utf-8"))
    mins = {}
    for r in C.read_jsonl(C.DATA / "manifest.jsonl"):     # unpadded source minutes, all clips
        mins[r["row"]] = mins.get(r["row"], 0.0) + r["dur_src"] / 60
    rows = sorted((k for k in t if not k.startswith("_")), key=lambda k: mins[k])
    lines = ["| row | min | sim real | sim ja | sim zh | sim en | p_true ja (real) | leak | ja CER (real) | zh CER | zh SER | zh lid | en WER | runaway |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per = {}
    for k in rows:
        r, j, z, e = t[k]["real_ja"], t[k]["floor_ja"], t[k]["floor_zh"], t[k]["floor_en"]
        n_all = j["n"] + z["n"] + e["n"]
        leak = (j.get("leak", 0) * j["n"] + z.get("leak", 0) * z["n"] + e.get("leak", 0) * e["n"]) / max(n_all, 1)
        per[k] = {"min": mins[k], "sim_real": r.get("sim"), "sim_ja": j.get("sim"), "sim_zh": z.get("sim"),
                  "sim_en": e.get("sim"), "leak": leak, "ser": z.get("ser_shared"), "lid": z.get("lid_ratio"),
                  "cer_ja": j.get("cer"), "cer_zh": z.get("cer"), "wer_en": e.get("cer"),
                  "p_ja": j.get("p_true"), "p_real": r.get("p_true"),
                  "runaway": j.get("runaway", 0) + z.get("runaway", 0) + e.get("runaway", 0)}
        p = per[k]
        lines.append(f"| {k} | {p['min']:.1f} | {f(p['sim_real'])} | {f(p['sim_ja'])} | {f(p['sim_zh'])} | {f(p['sim_en'])} | "
                     f"{f(p['p_ja'], 2)} ({f(p['p_real'], 2)}) | {p['leak']:.2f} | {f(p['cer_ja'])} ({f(r.get('cer'))}) | "
                     f"{f(p['cer_zh'])} | {f(p['ser'])} | {f(p['lid'], 1)} | {f(p['wer_en'])} | {p['runaway']} |")
    lines += ["", "| bucket | rows | sim real | sim ja | sim zh | sim en | leak | ja CER | zh SER | zh lid | en WER |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, lo, hi in BUCKETS:
        sel = [per[k] for k in rows if lo <= per[k]["min"] < hi]
        if not sel:
            continue
        m = lambda key: float(np.mean([s[key] for s in sel if s[key] is not None]))  # noqa: E731
        lines.append(f"| {name} min | {len(sel)} | {m('sim_real'):.3f} | {m('sim_ja'):.3f} | {m('sim_zh'):.3f} | "
                     f"{m('sim_en'):.3f} | {m('leak'):.2f} | {m('cer_ja'):.3f} | {m('ser'):.3f} | {m('lid'):.1f} | {m('wer_en'):.3f} |")
    a = t["_all"]
    lines += ["", f"All rows: real ja sim {f(a['real_ja'].get('sim'))}, CER {f(a['real_ja'].get('cer'))}; "
              f"floor ja sim {f(a['floor_ja'].get('sim'))}, CER {f(a['floor_ja'].get('cer'))}, leak {a['floor_ja'].get('leak')}; "
              f"floor zh sim {f(a['floor_zh'].get('sim'))}, CER {f(a['floor_zh'].get('cer'))}, SER {f(a['floor_zh'].get('ser_shared'))}, "
              f"lid {f(a['floor_zh'].get('lid_ratio'), 1)}, leak {a['floor_zh'].get('leak')}; "
              f"floor en sim {f(a['floor_en'].get('sim'))}, WER {f(a['floor_en'].get('cer'))}, leak {a['floor_en'].get('leak')}"]
    out = "\n".join(lines)
    (C.OUT / "e0" / "floor_table.md").write_text(out + "\n", encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
