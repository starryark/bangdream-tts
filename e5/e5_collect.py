"""E5 step 3d (local, TTS venv): gate the converted rows, build the listening sets, write the VC rows.

  python -I e5_collect.py listen     # per-voice report + blind listening sets (listen/<row>/), no rows written
  python -I e5_collect.py rows       # after the user's verdicts (vc/listening.json): the training rows

E5_PLAN.md §4. Input: every vc/scored_<batch>.jsonl from e4_score (E4_VC=e5/vc, E4_ROWS + E4_TAG per batch of voices;
cells 200, shift, 200_noshift, source). A converted clip (cell 200) is a passing row if: identity raw (pred == row
and p_true >= 0.5 * real_resynth_p_true), identity after the codec (codec pred == row and codec ratio
>= 0.5), CER <= 0.05, 1.5-13 s and not a runaway (e4_score's length gate), and 2-12 s after the
training audio policy. The source already passed the native gate (e5_sources). No post-conversion
accent metric gates anything. Per voice: at most 20 rows, highest raw ratio first (the trained rows).
A (cell, row, utt) scored in several batch files: the newest file wins (Amendment E5-A4).
Listening: 3 of the trained rows per voice (random, seed 5; if they mix own-row and donor sources,
at least one of each: E5-A3), played as their codec resynthesis; a voice with > 25% text-gate
failures is flagged and gets 6 (3 of them text failures). No scores shown. A listening set whose
pick is unchanged is left as it is.
Rows: Kurisu v3 audio policy (p03_build.normalise_audio) at 24 kHz, id pseudo_zh_<row>_rvc_<utt>,
source pseudo_vc, 3% val (seed 9). Voices the user judged accented or not in character get none.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import statistics as st
import sys
from collections import Counter
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.path.insert(0, r"E:\STEINS;GATEVoice\kurisu_v2\pod")   # p03_build: the training audio policy
VC = BD / "e5" / "vc"
CAP, MIN_ROWS, FLAG_TEXT_FAIL = 20, 10, 0.25


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def policy(path, V2, meter):
    """The training audio path: 24 kHz, Kurisu v3 normalise_audio, 2-12 s; None if it fails."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != V2.SR:
        g = gcd(sr, V2.SR)
        wav = resample_poly(wav, V2.SR // g, sr // g).astype("float32")
    new, _ = V2.normalise_audio(wav, meter)
    return new if new is not None and 2.0 <= len(new) / V2.SR <= 12.0 else None


def main() -> None:
    import pyloudnorm
    import p03_build as V2
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1]
    by = {}
    for f in sorted(VC.glob("scored*.jsonl"), key=lambda f: f.stat().st_mtime):   # one file per batch (E4_TAG)
        for l in f.read_text(encoding="utf-8").splitlines():
            if l:
                r = json.loads(l)
                by[(r["epoch"], r["row"], r["utt"])] = r                         # E5-A4: newest file wins
    vc = [r for k, r in sorted(by.items()) if k[0] == "200"]
    meter = pyloudnorm.Meter(V2.SR)
    offsets = json.loads((VC / "offsets.json").read_text())
    for r in vc:
        r["codec_identity"] = r["codec_pred"] == r["row"] and r["codec_ratio"] >= 0.5
        r["row_pass"] = r["gates"]["identity"] and r["codec_identity"] and r["gates"]["text"] and r["gates"]["length"]

    report, trained = {}, {}
    for v in sorted({r["row"] for r in vc}):
        m = [r for r in vc if r["row"] == v]
        sh = [by.get(("shift", v, r["utt"])) for r in m]
        pairs = [(r.get("ser_valid"), s.get("ser_valid")) for r, s in zip(m, sh) if s]
        cell = lambda c, k: mean([by[(c, v, r["utt"])].get(k) for r in m if (c, v, r["utt"]) in by])   # noqa: E731
        text_fail = 1 - mean([r["gates"]["text"] for r in m])
        report[v] = {"n": len(m), "row_pass": mean([r["row_pass"] for r in m]),
                     "identity_raw": mean([r["gates"]["identity"] for r in m]), "identity_codec": mean([r["codec_identity"] for r in m]),
                     "text": mean([r["gates"]["text"] for r in m]), "length": mean([r["gates"]["length"] for r in m]),
                     "text_fail": round(text_fail, 3), "flagged": text_fail > FLAG_TEXT_FAIL,
                     "cer_vc": mean([r["cer"] for r in m]), "cer_src": mean([r["src_cer"] for r in m]),
                     "ser_vc_minus_shift": mean([a - b for a, b in pairs if a is not None and b is not None]),
                     "lid_median": {c: round(st.median(by[(c, v, r["utt"])]["lid_ratio"] for r in m), 2)
                                    for c in ("source", "shift", "200_noshift", "200") if all((c, v, r["utt"]) in by for r in m)},
                     "ser_mean": {c: cell(c, "ser_valid") for c in ("source", "shift", "200_noshift", "200")},
                     "pred_top": Counter(r["pred"] for r in m).most_common(2)}
        report[v]["passing"] = sum(r["row_pass"] for r in m)
        report[v]["weak"] = report[v]["passing"] < MIN_ROWS
        T = []                                     # the trained rows: passing, highest raw ratio first, policy length
        for r in sorted([r for r in m if r["row_pass"]], key=lambda r: -r["ratio"]):
            if len(T) >= CAP:
                break
            new = policy(r["path"], V2, meter)
            if new is not None:
                T.append((r, new))
        trained[v] = T
        report[v]["trained"] = len(T)
        report[v]["src_donor"] = {"sources": sum(r["src_row"] != v for r in m), "passing": sum(r["src_row"] != v and r["row_pass"] for r in m),
                                  "trained": sum(r["src_row"] != v for r, _ in T)}
        off = [offsets.get(f"{v}|{r['utt']}") for r in m]
        report[v]["pitch_clipped"] = mean([abs(o) >= 12 for o in off if o is not None])   # E5-A2
    (VC / "collect_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for v, x in report.items():
        print(f"[e5_collect] {v}: {json.dumps(x, ensure_ascii=False)}", flush=True)

    if mode == "listen":
        for v, x in report.items():
            m = sorted([r for r in vc if r["row"] == v], key=lambda r: r["utt"])
            ok = sorted([r for r, _ in trained[v]], key=lambda r: r["utt"])
            rng = random.Random(5)
            pick = rng.sample(ok, min(3, len(ok)))
            kinds = {r["src_row"] == v for r in ok}
            if len(kinds) == 2 and len({r["src_row"] == v for r in pick}) == 1:   # E5-A3: hear both source kinds
                pick[-1] = rng.choice([r for r in ok if (r["src_row"] == v) != (pick[0]["src_row"] == v)])
            if x["flagged"]:
                bad = [r for r in m if not r["gates"]["text"]]
                pick += rng.sample(bad, min(3, len(bad)))
                rng.shuffle(pick)
            key = {str(i): r["utt"] for i, r in enumerate(pick, 1)}
            kpath = BD / "e5" / "listen" / f"{v}_key.json"
            if kpath.exists() and json.loads(kpath.read_text()) == key:
                print(f"[e5_collect] listen/{v}: unchanged", flush=True)
                continue
            out = BD / "e5" / "listen" / v
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            lines = [f"{v}: Mandarin training clips (RVC epoch 200, after the TTS codec; LOCAL ONLY, never share)", "",
                     "Questions: 1. native Mandarin, or Japanese-accented?  2. does it sound like the character?", ""]
            for i, r in enumerate(pick, 1):
                shutil.copy2(VC / "codec" / "200" / v / f"{r['utt']}.wav", out / f"{i}.wav")
                lines.append(f"{i}: {r['text']}")
            refs = [l.strip() for l in (BD / "e1" / "bundle" / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").splitlines()
                    if l.strip()][:2]
            for j, p in enumerate(refs, 1):
                shutil.copy2(p.replace("/workspace/bd", str(BD)), out / f"ref_ja_{j}.wav")
            (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            json.dump(key, open(kpath, "w"), indent=0)
            print(f"[e5_collect] listen/{v}: {len(pick)} clips{' (flagged)' if x['flagged'] else ''}", flush=True)
        return

    import soundfile as sf
    verdict = json.loads((VC / "listening.json").read_text(encoding="utf-8"))   # {row: {"native": bool, "character": bool}}
    rng = random.Random(9)
    added, counts = [], {}
    for v in sorted(report):
        ok = verdict.get(v, {})
        if not (ok.get("native") and ok.get("character")):
            counts[v] = f"0 (listening: {ok or 'missing'})"
            continue
        n = 0
        for r, new in trained[v]:
            rid = f"pseudo_zh_{v}_rvc_{r['utt']}"
            p = BD / "e5" / "rows" / "zh" / v / f"{rid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), new, V2.SR, subtype="PCM_16")
            added.append({"id": rid, "audio": str(p), "text": r["text"], "speaker": v, "language": "chinese",
                          "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                          "source": "pseudo_vc", "p_true": r["p_true"], "ratio": r["ratio"], "codec_ratio": r["codec_ratio"],
                          "src_lid": r["src_lid"], "cer": r["cer"], "src_row": r["src_row"]})
            n += 1
        counts[v] = n
    (VC / "rvc_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in added), encoding="utf-8")
    print(f"[e5_collect] {len(added)} VC rows: {counts}", flush=True)


if __name__ == "__main__":
    main()
