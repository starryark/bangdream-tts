"""Amendment A analysis (pre-registered before any E1b score): per-voice E1b - E1a at the same
epoch, by pseudo-row group, with bootstrap CIs over voices. Also the Mandarin identity (zh leak,
p_true) by group, since E1b's Mandarin identity fell back to the floor in the pooled numbers.

  python -I e1_did.py            (reads e1/results)
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def load(c):
    return [json.loads(l) for l in open(R / "eval" / c / "scored.jsonl", encoding="utf-8")]


def per_voice(rows, lang, key):
    d = collections.defaultdict(list)
    for r in rows:
        if r["lang"] != lang:
            continue
        if key == "leak":
            d[r["row"]].append(float(r["pred"] != r["row"]))
        elif r.get(key) is not None and (key != "ser_shared" or r.get("m23_valid", True)):
            d[r["row"]].append(r[key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def boot_mean(vals, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(vals)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return round(float(v.mean()), 4), [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rep = json.load(open(R / "pseudo" / "report.json", encoding="utf-8"))
    g15 = sorted(k for k, v in rep.items() if v["kept"] >= 15)
    g0 = sorted(k for k, v in rep.items() if v["kept"] == 0)
    floor = load("floor")
    out = {"groups": {"ge15": g15, "zero": g0}, "epochs": {}}
    for ep in (1, 2, 3):
        a, b = load(f"E1a_ep{ep}"), load(f"E1b_ep{ep}")
        res = {}
        for metric, lang, key in (("zh_ser", "zh", "ser_shared"), ("zh_lid", "zh", "lid_ratio"),
                                  ("zh_leak", "zh", "leak"), ("zh_p_true", "zh", "p_true"),
                                  ("ja_leak", "ja", "leak"), ("en_lid", "en", "lid_ratio")):
            pa, pb, pf = per_voice(a, lang, key), per_voice(b, lang, key), per_voice(floor, lang, key)
            row = {}
            for gname, g in (("ge15", g15), ("zero", g0)):
                d = [pb[k] - pa[k] for k in g]
                row[gname] = {"delta_E1b_minus_E1a": boot_mean(d), "E1a": round(st.mean(pa[k] for k in g), 4),
                              "E1b": round(st.mean(pb[k] for k in g), 4), "floor": round(st.mean(pf[k] for k in g), 4)}
            dd = [pb[k] - pa[k] for k in g15]
            dz = [pb[k] - pa[k] for k in g0]
            rng = np.random.default_rng(1)
            gaps = [rng.choice(dd, len(dd)).mean() - rng.choice(dz, len(dz)).mean() for _ in range(4000)]
            row["gap_ge15_minus_zero"] = [round(float(np.mean(dd) - np.mean(dz)), 4),
                                          [round(float(np.percentile(gaps, 2.5)), 4), round(float(np.percentile(gaps, 97.5)), 4)]]
            res[metric] = row
        out["epochs"][ep] = res
    (R / "did_amendA.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for ep, res in out["epochs"].items():
        print(f"== epoch {ep}")
        for metric, row in res.items():
            print(f"  {metric:9s} >=15: E1a {row['ge15']['E1a']:.3f} -> E1b {row['ge15']['E1b']:.3f} "
                  f"(floor {row['ge15']['floor']:.3f}) d={row['ge15']['delta_E1b_minus_E1a']} | "
                  f"zero: E1a {row['zero']['E1a']:.3f} -> E1b {row['zero']['E1b']:.3f} (floor {row['zero']['floor']:.3f}) "
                  f"d={row['zero']['delta_E1b_minus_E1a']} | gap {row['gap_ge15_minus_zero']}")


if __name__ == "__main__":
    main()
