"""Shared scorer for rendered or real clips (E1 eval and pseudo-zh gating), identical to E0.

score(rows, probe_path) adds, per row {path, lang, text, row}:
  cer        Qwen3-ASR-1.7B, language forced, no context; metrics._norm_text "v3"
             (ja: reading-space CER with the BanG Dream readings; zh: hanzi CER; en: WER)
  dur, dur_ratio, runaway   metrics.is_runaway
  p_true, pred              codec-calibrated leakage probe (vectors/probe_codec.npz, b08):
                            Qwen3-TTS-Base speaker encoder, trainer mel recipe, centred by the
                            probe's own mean, multinomial softmax
  lid_ratio, p_target, ser_shared, m23_valid   accent_metrics, zh/en rows only
SER is only meaningful when m23_valid; aggregate() ignores invalid rows for SER.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

POD = Path(os.environ.get("BD_POD", "/workspace/bd"))
for p in (POD / "code", POD / "e1", POD / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("KURISU_WORK", str(POD))
import bdcommon  # noqa: E402,F401  (BanG Dream readings into p03_build before metrics imports it)

LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "en": "English"}
BASE = os.environ.get("BD_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


def _embedder():
    import numpy as np
    import torch
    from math import gcd
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from scipy.signal import resample_poly

    w = Qwen3TTSModel.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    enc = w.model.speaker_encoder.to("cuda").eval()
    del w

    def embed(wav, sr):
        if wav.ndim > 1:
            wav = wav.mean(-1)
        if sr != 24000:
            g = gcd(sr, 24000)
            wav = resample_poly(wav, 24000 // g, sr // g)
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0), n_fft=1024,
                                num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0,
                                fmax=12000).transpose(1, 2)
            return enc(m.to("cuda").to(torch.bfloat16))[:1].float().cpu().numpy()[0]
    return embed, enc


def score(rows: list[dict], probe_path: Path, accent: bool = True, asr: bool = True) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    import metrics

    if asr:
        model = metrics._asr()
        for lang in ("ja", "zh", "en"):
            idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
            for s in range(0, len(idx), 32):
                chunk = idx[s:s + 32]
                res = model.transcribe([rows[i]["path"] for i in chunk], language=LANG_NAME[lang])
                for i, h in zip(chunk, res):
                    rows[i]["asr"] = h.text.strip()
        metrics._models.pop("asr", None)
        del model
        torch.cuda.empty_cache()
    for r in rows:
        r["dur"] = round(sf.info(r["path"]).duration, 3)
        if asr:
            ref = metrics._norm_text(r["text"], r["lang"], "v3")
            hyp = metrics._norm_text(r.get("asr", ""), r["lang"], "v3")
            r["cer"] = round(metrics._err(ref, hyp, r["lang"]), 4)
            units = len(ref) if r["lang"] != "en" else len(ref.split())
            r["dur_ratio"] = round(r["dur"] / max(units / metrics.UNITS_PER_S[r["lang"]], 0.5), 3)
            r["runaway"] = bool(metrics.is_runaway(r))

    pr = np.load(probe_path)
    mu, coef, icpt, classes = pr["mu"], pr["coef"], pr["intercept"], [str(c) for c in pr["classes"]]
    embed, enc = _embedder()
    for r in rows:
        x, sr = sf.read(r["path"], dtype="float32", always_2d=False)
        v = embed(x, sr) - mu
        v = v / np.linalg.norm(v)
        z = coef @ v + icpt
        p = np.exp(z - z.max())
        p /= p.sum()
        r["p_true"] = round(float(p[classes.index(r["row"])]), 4)
        r["pred"] = classes[int(p.argmax())]
    del enc
    torch.cuda.empty_cache()

    if accent:
        import accent_metrics
        acc = [r for r in rows if r["lang"] in ("zh", "en")]
        scored = accent_metrics.score_rows([{**r, "file": r["path"], "pid": f"{i}"} for i, r in enumerate(acc)],
                                           Path("/"), verbose=False)
        by = {int(s["pid"]): s for s in scored}
        for i, r in enumerate(acc):
            s = by.get(i, {})
            for k in ("lid_ratio", "p_target", "ser_shared", "ser", "m23_valid"):
                if k in s:
                    r[k] = s[k]
    return rows


def aggregate(sel: list[dict]) -> dict:
    import numpy as np
    out = {"n": len(sel)}
    if not sel:
        return out
    for k in ("cer", "p_true", "lid_ratio"):
        v = [r[k] for r in sel if r.get(k) is not None]
        if v:
            out[k] = round(float(np.mean(v)), 4)
    v = [r["ser_shared"] for r in sel if r.get("ser_shared") is not None and r.get("m23_valid", True)]
    if v:
        out["ser_shared"] = round(float(np.mean(v)), 4)
    out["leak"] = round(float(np.mean([r["pred"] != r["row"] for r in sel])), 4)
    out["runaway"] = int(sum(bool(r.get("runaway")) for r in sel))
    return out
