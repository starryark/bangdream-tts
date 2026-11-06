"""E4 step 4 (local, TTS venv): score the RVC conversions, apply E3's gates, decide the feasibility gate.

  python -I e4_score.py 200 100 200_noshift shift source   (first cell listed = the gated checkpoint)

Cells (Amendment B): "<epoch>" = RVC as planned, "<epoch>_noshift" = RVC without the pitch shift,
"shift" = the sources pitch-shifted only (e4_shift.py), "source" = the native sources themselves.

E4_PLAN.md "Scoring and gates". Primary: bdscore on the raw converted audio, with the gates of
e3_collect.py (identity pred == row and p_true >= 0.5 * real_resynth_p_true; accent P(zh) >= 0.9
and lid <= -19; CER <= 0.05; 1.5-13 s and not a runaway). Feasibility (gated epoch only): for at
least one voice, pass rate >= 20%, median lid <= -19 and mean shared-char SER (m23-valid) <= 0.10.
Diagnostic only: identity after a Qwen3-TTS-Tokenizer-12Hz encode -> decode of each clip, identity
split into pred == row and ratio, paired source -> VC deltas, the other epochs.
Writes e4/vc/scored.jsonl and e4/vc/report.json.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import Counter
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
os.environ.setdefault("BD_POD", str(BD))
os.environ.setdefault("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                 r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
os.environ.setdefault("KURISU_WORK", str(BD / "e4" / "work"))
sys.path.insert(0, str(BD / ".pydeps_sb"))
sys.path.insert(0, str(BD / "e1"))

VC = Path(os.environ.get("E4_VC", BD / "e4" / "vc"))   # E5 reuses this with E4_VC=e5/vc
PROBE = BD / "e1" / "bundle" / "probe_codec.npz"


def gates(r: dict, base_p: dict) -> dict:
    return {"identity": r["pred"] == r["row"] and r["p_true"] >= 0.5 * base_p[r["row"]],
            "accent": r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= -19.0,
            "text": r.get("cer", 1) <= 0.05,
            "length": 1.5 <= r["dur"] <= 13.0 and not r.get("runaway")}


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import bdscore
    sys.stdout.reconfigure(encoding="utf-8")
    # cells: "200" / "200_noshift" = out/ep<cell>; "source" = the sources; anything else = controls/<cell>
    cells = sys.argv[1:] or ["200"]
    base_p = json.loads((BD / "e1" / "bundle" / "e2_baselines.json").read_text(encoding="utf-8"))["real_resynth_p_true"]
    src = [json.loads(l) for l in (VC / "sources.jsonl").read_text(encoding="utf-8").splitlines() if l]
    only = set(filter(None, os.environ.get("E4_ROWS", "").split(",")))   # smoke test: subset of voices
    src = [s for s in src if not only or s["row"] in only]
    tag = os.environ.get("E4_TAG", "")                                   # smoke test: separate output names

    def cell_path(c: str, s: dict) -> str:
        if c == "source":
            return s["src"]
        d = VC / "out" / f"ep{c}" if c[0].isdigit() else VC / "controls" / c
        return str(d / s["row"] / f"{s['utt']}.wav")
    rows = [{**s, "epoch": c, "path": cell_path(c, s)} for c in cells for s in src]
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    if missing:
        raise SystemExit(f"{len(missing)} conversions missing, e.g. {missing[0]}")
    bdscore.score(rows, PROBE)

    # diagnostic: identity on the codec-resynthesised clip (what the TTS would learn from)
    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    codec = []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        wavs, sr = tok.decode(tok.encode([r["path"] for r in chunk]))
        for r, w in zip(chunk, wavs):
            p = VC / "codec" / r["epoch"] / r["row"] / f"{r['utt']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
            codec.append({"row": r["row"], "lang": "zh", "text": r["text"], "path": str(p)})
    del tok
    torch.cuda.empty_cache()
    bdscore.score(codec, PROBE, accent=False, asr=False)
    for r, c in zip(rows, codec):
        r["codec_pred"], r["codec_p_true"] = c["pred"], c["p_true"]
        r["gates"] = gates(r, base_p)
        r["pass"] = all(r["gates"].values())
        r["ratio"] = round(r["p_true"] / base_p[r["row"]], 3)
        r["codec_ratio"] = round(r["codec_p_true"] / base_p[r["row"]], 3)
        r["ser_valid"] = r.get("ser_shared") if r.get("m23_valid", True) else None
    (VC / f"scored{tag}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    report = {}
    for ep in cells:
        for v in sorted({r["row"] for r in rows}):
            m = [r for r in rows if r["epoch"] == ep and r["row"] == v]
            report[f"{ep}|{v}"] = rep = {
                "n": len(m), "pass": mean([r["pass"] for r in m]),
                "gates": {g: mean([r["gates"][g] for r in m]) for g in m[0]["gates"]},
                "pred_is_row": mean([r["pred"] == v for r in m]), "ratio_ge_0.5": mean([r["ratio"] >= 0.5 for r in m]),
                "codec_pred_is_row": mean([r["codec_pred"] == v for r in m]),
                "codec_ratio_ge_0.5": mean([r["codec_ratio"] >= 0.5 for r in m]),
                "lid_median_vc": round(st.median(r["lid_ratio"] for r in m), 2),
                "lid_median_src": round(st.median(r["src_lid"] for r in m), 2),
                "ser_mean_vc": mean([r["ser_valid"] for r in m]), "ser_mean_src": mean([r["src_ser"] for r in m]),
                "cer_mean_vc": mean([r["cer"] for r in m]), "p_true_mean": mean([r["p_true"] for r in m]),
                "pred_top": Counter(r["pred"] for r in m).most_common(3)}
            rep["feasible"] = bool(rep["pass"] >= 0.20 and rep["lid_median_vc"] <= -19.0
                                   and rep["ser_mean_vc"] is not None and rep["ser_mean_vc"] <= 0.10)
    gated = cells[0]                                   # the first cell listed is the gated checkpoint
    report["feasibility_gate"] = {"cell": gated, "passed": any(v["feasible"] for k, v in report.items()
                                                                if k.startswith(f"{gated}|"))}
    (VC / f"report{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in report.items():
        print(f"[e4_score] {k}: {json.dumps(v, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
