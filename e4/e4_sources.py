"""E4 step 1 (local, main env): re-render the E3 feasibility sources with E1b and keep the most native.

  python -I e4_sources.py

E4_PLAN.md "Sources". E3's source wavs were deleted with the pod, so each (src_row, text) in
e2/results/e3/vc/jobs.json is rendered 6 times with E1b_ep2 (generate_custom_voice, Chinese,
fixed seeds), scored with bdscore, and per job the lowest-lid sample passing E2's accent, text
and length gates is kept (identity ignored, as in e3_prepare.native). Writes e4/vc/src_cands.jsonl
(every scored render) and e4/vc/sources.jsonl (one per job, jobs without a passing sample dropped).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
os.environ.setdefault("BD_POD", str(BD))
os.environ.setdefault("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                 r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
os.environ.setdefault("KURISU_WORK", str(BD / "e4" / "work"))
sys.path.insert(0, str(BD / ".pydeps_sb"))   # speechbrain 1.1.2 (Windows k2_fsa fix), as b05_score
sys.path.insert(0, str(BD / "e1"))

E1B = BD / "models" / "E1b_ep2"
JOBS = BD / "e2" / "results" / "e3" / "vc" / "jobs.json"
OUT = BD / "e4" / "vc"
N_SAMPLES, BATCH = 6, 20


def native(r: dict) -> bool:   # e3_prepare.native: E2's accent + text + length gates
    return (r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= -19.0 and r.get("cer", 1) <= 0.05
            and 2.0 <= r["dur"] <= 12.0 and not r.get("runaway"))


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    cpath = OUT / "src_cands.jsonl"
    if not cpath.exists():
        from qwen_tts import Qwen3TTSModel
        import bdscore
        pairs = sorted({(j["src_row"], j["utt"], j["text"]) for j in jobs})
        model = Qwen3TTSModel.from_pretrained(str(E1B), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        cands = []
        for s in range(N_SAMPLES):
            for b in range(0, len(pairs), BATCH):
                chunk = pairs[b:b + BATCH]
                torch.manual_seed(410000 + 1000 * s + b // BATCH)
                wavs, sr = model.generate_custom_voice(text=[t for *_, t in chunk], speaker=[r for r, *_ in chunk],
                                                       language=["Chinese"] * len(chunk), max_new_tokens=400,
                                                       non_streaming_mode=True)
                assert len(wavs) == len(chunk)
                for (row, u, t), w in zip(chunk, wavs):
                    p = OUT / "src_raw" / row / f"{u}_s{s}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                    cands.append({"row": row, "lang": "zh", "utt": u, "text": t, "sample": s, "path": str(p)})
            print(f"[e4_sources] sample {s + 1}/{N_SAMPLES}: {len(cands)} renders", flush=True)
        del model
        torch.cuda.empty_cache()
        bdscore.score(cands, BD / "e1" / "bundle" / "probe_codec.npz")
        cpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cands), encoding="utf-8")
    cands = [json.loads(l) for l in cpath.read_text(encoding="utf-8").splitlines() if l]

    out, dropped = [], []
    for j in jobs:
        ok = [r for r in cands if r["row"] == j["src_row"] and r["utt"] == j["utt"] and native(r)]
        if not ok:
            dropped.append((j["row"], j["utt"]))
            continue
        r = min(ok, key=lambda r: r["lid_ratio"])
        out.append({"row": j["row"], "lang": "zh", "utt": j["utt"], "text": j["text"], "src_row": j["src_row"],
                    "src": r["path"], "src_sample": r["sample"], "src_lid": r["lid_ratio"],
                    "src_ser": r.get("ser_shared") if r.get("m23_valid", True) else None,
                    "src_cer": r["cer"], "src_pred": r["pred"], "e3_src_lid": j["src_lid"]})
    (OUT / "sources.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    for v in sorted({j["row"] for j in jobs}):
        mine = [r for r in out if r["row"] == v]
        ser = [r["src_ser"] for r in mine if r["src_ser"] is not None]
        print(f"[e4_sources] {v}: {len(mine)}/{sum(j['row'] == v for j in jobs)} jobs kept; "
              f"median lid {np.median([r['src_lid'] for r in mine]):.1f} (E3 sources "
              f"{np.median([j['src_lid'] for j in jobs if j['row'] == v]):.1f}); mean SER {np.mean(ser):.3f}", flush=True)
    native_rate = np.mean([native(r) for r in cands])
    print(f"[e4_sources] {len(cands)} renders, {native_rate:.0%} native; dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()
