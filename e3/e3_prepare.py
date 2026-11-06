"""E3 step 1 (pod, main env): VC jobs = native-accented E2 candidates -> each target voice.

  python e3_prepare.py --feasibility      # 20 Kasumi + 20 Eve Mandarin sources only
  python e3_prepare.py                    # all target voices, zh and en

E3_PLAN.md "Method". Sources are E2 candidates (pseudo/candidates.jsonl) of the target language
that pass the accent, text and length gates, regardless of identity: the voice's own texts
first, then other target voices' texts; one per text, the most native (lowest lid) first, at
most N_SRC per voice and language. References: the trainer's reference list for the voice,
concatenated until ~25 s. Writes e3/vc/jobs.json and e3/vc/ref/<row>.wav.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
N_SRC = 60
REF_SECONDS = 25.0


def main() -> None:
    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility", action="store_true")
    args = ap.parse_args()
    rep = json.loads((ROOT / "pseudo" / "report.json").read_text(encoding="utf-8"))
    cands = [json.loads(l) for l in (ROOT / "pseudo" / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if args.feasibility:
        targets = {"zh": ["ppp_kasumi", "pp_eve"], "en": []}
        n_src = 20
    else:
        targets = {"zh": sorted(k for k, v in rep.items() if v["zh"]["kept"] < 10),
                   "en": sorted(k for k, v in rep.items() if v["en"]["kept"] < 6)}
        n_src = N_SRC
    out = ROOT / "e3" / "vc"
    (out / "ref").mkdir(parents=True, exist_ok=True)
    jobs = []
    lid_max = {"zh": -19.0, "en": -12.0}

    def native(r):   # E2's accent + text + length gates, recomputed from the stored scores
        return (r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= lid_max[r["lang"]]
                and r.get("cer", 1) <= 0.05 and 2.0 <= r["dur"] <= 12.0 and not r.get("runaway"))

    for lang, voices in targets.items():
        ok = [r for r in cands if r["lang"] == lang and native(r)]
        for v in voices:
            if not (out / "ref" / f"{v}.wav").exists():
                paths = [p.strip() for p in (ROOT / "bundle" / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
                chunks, tot, sr = [], 0.0, None
                for p in paths:
                    w, sr = sf.read(p, dtype="float32", always_2d=False)
                    chunks.append(w)
                    tot += len(w) / sr
                    if tot >= REF_SECONDS:
                        break
                sf.write(str(out / "ref" / f"{v}.wav"), np.concatenate(chunks), sr)
            seen, picked = set(), []
            for group in ([r for r in ok if r["row"] == v], [r for r in ok if r["row"] != v and r["row"] in voices]):
                for r in sorted(group, key=lambda r: r.get("lid_ratio", 0)):
                    if r["utt"] in seen:
                        continue
                    seen.add(r["utt"])
                    picked.append(r)
                    if len(picked) >= n_src:
                        break
                if len(picked) >= n_src:
                    break
            for r in picked:
                jobs.append({"row": v, "lang": lang, "utt": r["utt"], "text": r["text"], "src": r["path"],
                             "src_row": r["row"], "src_source": r["source"], "src_lid": r.get("lid_ratio"),
                             "ref": str(out / "ref" / f"{v}.wav"), "out_dir": str(out / "out" / lang / v)})
            print(f"[e3_prepare] {lang} {v}: {len(picked)} sources ({sum(r['row'] == v for r in picked)} own texts)", flush=True)
    (out / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    print(f"[e3_prepare] {len(jobs)} jobs; targets zh {targets['zh']} en {targets['en']}")


if __name__ == "__main__":
    main()
