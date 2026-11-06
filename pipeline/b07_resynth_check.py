"""E0 step 7: is the leakage probe fooled by the codec channel?

  python -I b07_resynth_check.py

The probe (vectors/probe.npz) was fit on real recordings only. Rendered audio also carries
the Qwen3-TTS 12 Hz codec's channel, so a probe that keys on channel would route every
rendered clip to the most "codec-like" real class, whatever the voice. Control: pass the real
held-out clips through the same codec (Qwen3-TTS-Tokenizer-12Hz encode -> decode) and score
them with the same probe. The voice is unchanged, only the channel moves, so any drop in
p_true / rise in leak is the channel effect, and its destination shows where synthetic audio
is drawn.

Writes e0/resynth/<row>/<tid>.wav and e0/resynth/scored.jsonl, and prints real vs resynth.
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    real = C.read_jsonl(C.OUT / "e0" / "real" / "scored.jsonl")
    out = C.OUT / "e0" / "resynth"
    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    rows = []
    for s in range(0, len(real), 8):
        chunk = real[s:s + 8]
        wavs, sr = tok.decode(tok.encode([r["path"] for r in chunk]))
        for r, w in zip(chunk, wavs):
            p = out / r["row"] / f"{r['tid']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
            keep = {k: r[k] for k in ("row", "tid", "lang", "text", "pid")}
            rows.append({**keep, "cond": "resynth", "path": str(p), "real_p_true": r["p_true"], "real_pred": r["pred"]})
    del tok
    torch.cuda.empty_cache()

    w = Qwen3TTSModel.from_pretrained(str(C.QWEN_BASE), dtype=torch.bfloat16, attn_implementation="sdpa")
    enc = w.model.speaker_encoder.to("cuda").eval()
    pr = np.load(C.OUT / "vectors" / "probe.npz")
    mu, coef, icpt, classes = pr["mu"], pr["coef"], pr["intercept"], list(pr["classes"])
    from math import gcd
    from scipy.signal import resample_poly
    for r in rows:
        x, sr = sf.read(r["path"], dtype="float32", always_2d=False)
        if sr != 24000:
            g = gcd(sr, 24000)
            x = resample_poly(x, 24000 // g, sr // g).astype("float32")
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                                hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            v = enc(m.to("cuda").to(torch.bfloat16))[:1].float().cpu().numpy()[0]
        v = (v - mu) / np.linalg.norm(v - mu)
        z = coef @ v + icpt
        p = np.exp(z - z.max())
        p /= p.sum()
        r["p_true"] = round(float(p[classes.index(r["row"])]), 4)
        r["pred"] = classes[int(p.argmax())]
    C.write_jsonl(out / "scored.jsonl", rows)

    sys.stdout.reconfigure(encoding="utf-8")
    pt_r = np.mean([r["real_p_true"] for r in rows])
    pt_s = np.mean([r["p_true"] for r in rows])
    lk_r = np.mean([r["real_pred"] != r["row"] for r in rows])
    lk_s = np.mean([r["pred"] != r["row"] for r in rows])
    print(f"[resynth] n={len(rows)}  p_true real {pt_r:.3f} -> resynth {pt_s:.3f}   leak real {lk_r:.3f} -> resynth {lk_s:.3f}")
    dest = collections.Counter(r["pred"] for r in rows if r["pred"] != r["row"])
    print("[resynth] leaked to:", dest.most_common(8))
    per = collections.defaultdict(list)
    for r in rows:
        per[r["row"]].append(r)
    worst = sorted(per.items(), key=lambda t: np.mean([x["p_true"] for x in t[1]]))[:12]
    print("[resynth] lowest p_true rows:", [(k, round(float(np.mean([x['p_true'] for x in v])), 2),
                                            round(float(np.mean([x['real_p_true'] for x in v])), 2)) for k, v in worst])


if __name__ == "__main__":
    main()
