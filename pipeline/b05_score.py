"""E0 step 5: score the floor renders and the real held-out clips with the same metrics.

  python -I b05_score.py

Per clip (kurisu_v2/pod/metrics.py conventions, text_norm "v3"):
  cer        Qwen3-ASR-1.7B with the language forced, no context. ja: reading-space CER
             (BanG Dream readings table); zh: hanzi CER; en: WER (numbers spelled out)
  dur_ratio, runaway   metrics.is_runaway (calibrated on real Kurisu takes)
  sim        centred cosine of the Qwen speaker-encoder vector to the row's locked vector
  p_true, pred         the leakage probe (vectors/probe.npz): P(own row), argmax row
zh/en clips also get accent_metrics (VoxLingua107 lid_ratio; zh shared-char SER).

The real held-out Japanese clips ("real" condition) go through the same scoring, so every
floor number has a real-audio counterpart.

Writes e0/floor/<row>/scored.jsonl, e0/real/scored.jsonl, e0/floor_table.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# speechbrain 1.1.0 (in the shared venv) has a Windows-only lazy-import bug (k2_fsa); 1.1.2 lives here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".pydeps_sb"))
os.environ.setdefault("KURISU_WORK", str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e0", "work")))
import bdcommon as C  # noqa: E402

LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "en": "English"}


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import metrics
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    floor = C.OUT / "e0" / "floor"
    rows_all = []
    for d in sorted(p for p in floor.iterdir() if (p / "manifest.jsonl").exists()):
        for r in C.read_jsonl(d / "manifest.jsonl"):
            rows_all.append({**r, "cond": "floor", "path": str(d / r["file"])})
    real = []
    seen = set()
    for r in rows_all:
        if r["lang"] == "ja" and r["real_id"] not in seen:
            seen.add(r["real_id"])
            real.append({"row": r["row"], "tid": r["tid"], "lang": "ja", "text": r["text"],
                         "pid": f"{r['row']}~{r['tid']}", "cond": "real", "path": r["real_audio"]})
    allr = rows_all + real

    # ASR --------------------------------------------------------------------------
    asr = metrics._asr()
    for lang in ("ja", "zh", "en"):
        idx = [i for i, r in enumerate(allr) if r["lang"] == lang]
        for s in range(0, len(idx), 32):
            chunk = idx[s:s + 32]
            res = asr.transcribe([allr[i]["path"] for i in chunk], language=LANG_NAME[lang])
            for i, h in zip(chunk, res):
                allr[i]["asr"] = h.text.strip()
        print(f"[score] asr {lang}: {len(idx)}", flush=True)
    del metrics._models["asr"]
    torch.cuda.empty_cache()
    for r in allr:
        ref = metrics._norm_text(r["text"], r["lang"], "v3")
        hyp = metrics._norm_text(r.get("asr", ""), r["lang"], "v3")
        r["cer"] = round(metrics._err(ref, hyp, r["lang"]), 4)
        r["dur"] = round(sf.info(r["path"]).duration, 3)
        units = len(ref) if r["lang"] != "en" else len(ref.split())
        r["dur_ratio"] = round(r["dur"] / max(units / metrics.UNITS_PER_S[r["lang"]], 0.5), 3)
        r["runaway"] = bool(metrics.is_runaway(r))

    # speaker: centred Qwen encoder vector, sim to locked row vector, probe ------------
    w = Qwen3TTSModel.from_pretrained(str(C.QWEN_BASE), dtype=torch.bfloat16, attn_implementation="sdpa")
    enc = w.model.speaker_encoder.to("cuda").eval()
    pr = np.load(C.OUT / "vectors" / "probe.npz")
    mu, coef, icpt, classes = pr["mu"], pr["coef"], pr["intercept"], list(pr["classes"])
    locked = torch.load(C.OUT / "vectors" / "locked.pt", map_location="cpu")
    L = locked["table"].float().numpy() - mu
    L = L / np.linalg.norm(L, axis=-1, keepdims=True)
    names = locked["names"]
    from math import gcd
    from scipy.signal import resample_poly
    for i, r in enumerate(allr):
        wav, sr = sf.read(r["path"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(-1)
        if sr != 24000:
            g = gcd(sr, 24000)
            wav = resample_poly(wav, 24000 // g, sr // g).astype("float32")
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                                hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            v = enc(m.to("cuda").to(torch.bfloat16))[:1].float().cpu().numpy()[0]
        x = v - mu
        x = x / np.linalg.norm(x)
        r["sim"] = round(float(x @ L[names.index(r["row"])]), 4)
        z = coef @ x + icpt
        p = np.exp(z - z.max())
        p /= p.sum()
        r["p_true"] = round(float(p[classes.index(r["row"])]), 4)
        r["pred"] = classes[int(p.argmax())]
    del w, enc
    torch.cuda.empty_cache()

    # accent metrics on zh/en floor clips --------------------------------------------
    import accent_metrics
    acc_rows = [r for r in allr if r["lang"] in ("zh", "en")]
    scored = accent_metrics.score_rows([{**r, "file": r["path"]} for r in acc_rows], floor, verbose=False)
    by_pid = {s["pid"]: s for s in scored}
    for r in acc_rows:
        s = by_pid.get(r["pid"], {})
        for k in ("lid_ratio", "p_target", "ser", "ser_shared", "ter"):
            if k in s:
                r[k] = s[k]

    # write ---------------------------------------------------------------------------
    for row in sorted({r["row"] for r in allr}):
        C.write_jsonl(floor / row / "scored.jsonl", [r for r in rows_all if r["row"] == row])
    (C.OUT / "e0" / "real").mkdir(exist_ok=True)
    C.write_jsonl(C.OUT / "e0" / "real" / "scored.jsonl", real)

    def agg(sel, keys=("cer", "sim", "p_true", "lid_ratio", "ser_shared")):
        out = {"n": len(sel)}
        for k in keys:
            vals = [r[k] for r in sel if r.get(k) is not None]
            if vals:
                out[k] = round(float(np.mean(vals)), 4)
        if sel:
            out["leak"] = round(float(np.mean([r["pred"] != r["row"] for r in sel])), 3)
            out["runaway"] = int(sum(r.get("runaway", False) for r in sel))
        return out

    table = {}
    for row in sorted({r["row"] for r in allr}):
        table[row] = {"real_ja": agg([r for r in real if r["row"] == row], ("cer", "sim", "p_true")),
                      **{f"floor_{l}": agg([r for r in rows_all if r["row"] == row and r["lang"] == l])
                         for l in ("ja", "zh", "en")}}
    table["_all"] = {"real_ja": agg(real, ("cer", "sim", "p_true")),
                     **{f"floor_{l}": agg([r for r in rows_all if r["lang"] == l]) for l in ("ja", "zh", "en")}}
    (C.OUT / "e0" / "floor_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    a = table["_all"]
    print("[score] all: " + json.dumps(a, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
