"""E3 step 3 (pod, main env): score VC outputs, gate them as in E2, merge with E2's pseudo rows.

  python e3_collect.py --feasibility    # pass rates only (E3_PLAN.md feasibility gate)
  python e3_collect.py                  # + write the merged pseudo rows for the E3 mix

Gates are E2's: identity pred == row and p_true >= 0.5 * real_p_true[row]; accent zh P(zh) >= 0.9
and lid <= -19 / en P(en) >= 0.9 and lid <= -12; CER/WER <= 0.05; not a runaway; 2-12 s after
the training audio policy. Merge: every E2 row is kept; passing VC clips (best per text by
p_true / real_p_true, texts the voice does not already have) fill each voice up to zh 25 / en 12.
Writes e3/vc/scored.jsonl, e3/vc/report.json and (full mode) e3/pseudo_rows_e3.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
for p in (ROOT / "code", ROOT / "e1", ROOT / "pipeline"):
    sys.path.insert(0, str(p))
os.environ.setdefault("KURISU_WORK", str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_vc_run import out_path  # noqa: E402

CAP = {"zh": 25, "en": 12}
LID_MAX = {"zh": -19.0, "en": -12.0}


def main() -> None:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import bdscore

    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    V = ROOT / "e3" / "vc"
    jobs = json.loads((V / "jobs.json").read_text(encoding="utf-8"))
    base_p = json.loads((ROOT / "bundle" / "e2_baselines.json").read_text(encoding="utf-8"))["real_resynth_p_true"]
    rows = [{**{k: j[k] for k in ("row", "lang", "utt", "text", "src_row", "src_source", "src_lid")},
             "source": "vc", "path": out_path(j)} for j in jobs if os.path.exists(out_path(j))]
    bdscore.score(rows, ROOT / "bundle" / "probe_codec.npz")
    for r in rows:
        r["ratio"] = r["p_true"] / base_p[r["row"]]
        r["gates"] = {"identity": r["pred"] == r["row"] and r["ratio"] >= 0.5,
                      "accent": r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= LID_MAX[r["lang"]],
                      "text": r.get("cer", 1) <= 0.05,
                      "length": 1.5 <= r["dur"] <= 13.0 and not r.get("runaway")}
        r["pass"] = all(r["gates"].values())
    (V / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {}
    for (v, lang) in sorted({(r["row"], r["lang"]) for r in rows}):
        mine = [r for r in rows if r["row"] == v and r["lang"] == lang]
        report[f"{v}|{lang}"] = {
            "n": len(mine), "pass": round(float(np.mean([r["pass"] for r in mine])), 3),
            "gates": {g: round(float(np.mean([r["gates"][g] for r in mine])), 3) for g in mine[0]["gates"]},
            "lid_median_vc": round(st.median(r["lid_ratio"] for r in mine if r.get("lid_ratio") is not None), 2),
            "lid_median_src": round(st.median(r["src_lid"] for r in mine if r.get("src_lid") is not None), 2),
            "p_true_mean": round(float(np.mean([r["p_true"] for r in mine])), 3),
            "pred_top": Counter(r["pred"] for r in mine).most_common(3)}
    (V / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in report.items():
        print(f"[e3_collect] {k}: pass {v['pass']} gates {v['gates']} lid vc/src {v['lid_median_vc']}/{v['lid_median_src']} "
              f"p_true {v['p_true_mean']} top {v['pred_top']}", flush=True)
    if args.feasibility:
        return

    import pyloudnorm
    import p03_build as V2
    meter = pyloudnorm.Meter(V2.SR)
    e2 = [json.loads(l) for l in (ROOT / "pseudo" / "pseudo_rows.jsonl").read_text(encoding="utf-8").splitlines() if l]
    lang_of = {"chinese": "zh", "english": "en"}
    have = defaultdict(set)            # texts each voice already has as an E2 pseudo row
    for r in e2:
        have[(r["speaker"], lang_of[r["language"]])].add(r["text"])
    count = Counter((r["speaker"], lang_of[r["language"]]) for r in e2)
    rng = random.Random(9)
    added = []
    for (v, lang) in sorted({(r["row"], r["lang"]) for r in rows}):
        best = {}
        for r in rows:
            if r["row"] == v and r["lang"] == lang and r["pass"] and r["text"] not in have[(v, lang)]:
                if r["utt"] not in best or r["ratio"] > best[r["utt"]]["ratio"]:
                    best[r["utt"]] = r
        room = CAP[lang] - count[(v, lang)]
        for r in sorted(best.values(), key=lambda r: -r["ratio"]):
            if room <= 0:
                break
            wav, sr = sf.read(r["path"], dtype="float32", always_2d=False)
            if sr != V2.SR:
                from math import gcd
                g = gcd(sr, V2.SR)
                wav = resample_poly(wav, V2.SR // g, sr // g).astype("float32")
            new, _ = V2.normalise_audio(wav, meter)
            if new is None or not (2.0 <= len(new) / V2.SR <= 12.0):
                continue
            rid = f"pseudo_{lang}_{v}_vc_{r['utt']}"
            p = ROOT / "e3" / "wavs" / lang / v / f"{rid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), new, V2.SR, subtype="PCM_16")
            added.append({"id": rid, "audio": str(p), "text": r["text"], "speaker": v,
                          "language": "chinese" if lang == "zh" else "english",
                          "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                          "source": "pseudo_vc", "p_true": r["p_true"], "ratio": round(r["ratio"], 3),
                          "lid_ratio": r.get("lid_ratio"), "cer": r.get("cer"), "src_row": r["src_row"]})
            room -= 1
    merged = e2 + added
    (ROOT / "e3" / "pseudo_rows_e3.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    c = Counter((r["speaker"], r["language"]) for r in added)
    print(f"[e3_collect] added {len(added)} VC rows: {dict(sorted(c.items()))}; merged {len(merged)}", flush=True)


if __name__ == "__main__":
    main()
