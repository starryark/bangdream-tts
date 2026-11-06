"""E0 step 8: a leakage probe calibrated to the codec channel, and the floor re-scored with it.

  python -I b08_codec_probe.py

Why: b07 showed that the Qwen3-TTS 12 Hz codec moves every clip's speaker vector by a
largely shared shift. Real clips survive it, but low-margin renders tip over to the class
that the shift favours (e0 review C1). Every rendered clip passes through that codec, so
the probe must be trained and baselined on audio with the same channel:
  * train clips are resynthesised (tokenizer encode -> decode, no disk) and embedded with
    the trainer's mel recipe;
  * the probe (multinomial logistic regression, balanced, on vectors centred by the
    resynth-train mean) is fit on those;
  * the real baseline is the resynthesised held-out clips (e0/resynth from b07);
  * floor renders are scored with the same probe.
Clusters: renders of one voice are not independent, so pooled CIs bootstrap over voices.

Writes vectors/probe_codec.npz, vectors/clipvec_resynth.npz, e0/codec_probe_report.json,
and prints the corrected Japanese leak table.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from math import gcd
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    sys.stdout.reconfigure(encoding="utf-8")
    man = C.read_jsonl(C.DATA / "manifest.jsonl")
    split = {r["id"]: r["split"] for r in C.read_jsonl(C.DATA / "split.jsonl")}
    rows = sorted({r["row"] for r in man})
    train = [r for r in man if split[r["id"]] == "train"]
    cache = C.OUT / "vectors" / "clipvec_resynth.npz"

    w = Qwen3TTSModel.from_pretrained(str(C.QWEN_BASE), dtype=torch.bfloat16, attn_implementation="sdpa")
    enc = w.model.speaker_encoder.to("cuda").eval()

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
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

    if cache.exists():
        z = np.load(cache)
        ids, V = list(z["ids"]), z["V"]
        print(f"[codec] reusing {cache.name} ({len(ids)} clips)", flush=True)
    else:
        tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
        ids, vecs = [], []
        order = sorted(train, key=lambda r: r["dur"])
        for s in range(0, len(order), 8):
            chunk = order[s:s + 8]
            with torch.inference_mode():
                wavs, sr = tok.decode(tok.encode([r["audio"] for r in chunk]))
            torch.cuda.empty_cache()
            for r, x in zip(chunk, wavs):
                ids.append(r["id"])
                vecs.append(embed(np.asarray(x, dtype=np.float32), sr))
            if (s // 8) % 200 == 0:
                print(f"[codec] resynth {s + len(chunk)}/{len(order)}", flush=True)
        del tok
        torch.cuda.empty_cache()
        V = np.stack(vecs)
        np.savez(cache, ids=np.array(ids), V=V)

    def nrm(a):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)

    y = np.array([rows.index(i.rsplit("_", 1)[0]) for i in ids])
    mu = V.mean(0)
    X = nrm(V - mu)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X, y)
    cvp = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), X, y, cv=5)
    np.savez(C.OUT / "vectors" / "probe_codec.npz", coef=clf.coef_, intercept=clf.intercept_,
             classes=np.array(rows), mu=mu)

    def score(paths):
        out = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32", always_2d=False)
            v = nrm(embed(x, sr) - mu)
            pr = clf.predict_proba(v[None])[0]
            out.append((pr, rows[int(pr.argmax())]))
        return out

    # baseline: resynthesised held-out clips (b07); floor renders (b04)
    res = C.read_jsonl(C.OUT / "e0" / "resynth" / "scored.jsonl")
    floor = []
    for d in sorted((C.OUT / "e0" / "floor").iterdir()):
        if (d / "scored.jsonl").exists():
            for r in C.read_jsonl(d / "scored.jsonl"):
                floor.append({**r, "path": str(d / r["file"])})
    for group in (res, floor):
        for r, (pr, pred) in zip(group, score([r["path"] for r in group])):
            r["p_true_c"] = round(float(pr[rows.index(r["row"])]), 4)
            r["pred_c"] = pred

    rng = np.random.default_rng(0)

    def pooled(sel):
        by = {}
        for r in sel:
            by.setdefault(r["row"], []).append(r["pred_c"] != r["row"])
        keys = sorted(by)
        k = sum(sum(v) for v in by.values())
        n = sum(len(v) for v in by.values())
        boot = []
        for _ in range(2000):
            pick = rng.choice(len(keys), len(keys))
            kk = sum(sum(by[keys[i]]) for i in pick)
            nn = sum(len(by[keys[i]]) for i in pick)
            boot.append(kk / nn)
        return {"k": k, "n": n, "leak": round(k / n, 3),
                "ci_voice_bootstrap": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                "p_true_mean": round(float(np.mean([r["p_true_c"] for r in sel])), 3)}

    rep = {"probe_cv5_acc_resynth_train": round(float((cvp == y).mean()), 4),
           "baseline_resynth_heldout_ja": pooled(res),
           "floor": {l: pooled([r for r in floor if r["lang"] == l]) for l in ("ja", "zh", "en")},
           "per_row": {}}
    for k in rows:
        fj = [r for r in floor if r["row"] == k and r["lang"] == "ja"]
        fa = [r for r in floor if r["row"] == k]
        rb = [r for r in res if r["row"] == k]
        kj = sum(r["pred_c"] != k for r in fj)
        dest = {}
        for r in fa:
            if r["pred_c"] != k:
                dest[r["pred_c"]] = dest.get(r["pred_c"], 0) + 1
        rep["per_row"][k] = {"ja_leak": f"{kj}/{len(fj)}", "ja_ci": wilson(kj, len(fj)),
                             "ja_p_true": round(float(np.mean([r["p_true_c"] for r in fj])), 3) if fj else None,
                             "baseline_p_true": round(float(np.mean([r["p_true_c"] for r in rb])), 3) if rb else None,
                             "baseline_leak": f"{sum(r['pred_c'] != k for r in rb)}/{len(rb)}",
                             "all_lang_leak": f"{sum(r['pred_c'] != k for r in fa)}/{len(fa)}",
                             "top_dest": sorted(dest.items(), key=lambda t: -t[1])[:3]}
    (C.OUT / "e0" / "codec_probe_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    C.write_jsonl(C.OUT / "e0" / "floor_codecprobe.jsonl",
                  [{k: r[k] for k in ("row", "tid", "lang", "pred_c", "p_true_c")} for r in floor])
    print(f"[codec] probe CV5 on resynth train: {rep['probe_cv5_acc_resynth_train']}")
    print(f"[codec] baseline (resynth held-out ja): {rep['baseline_resynth_heldout_ja']}")
    for l in ("ja", "zh", "en"):
        print(f"[codec] floor {l}: {rep['floor'][l]}")
    worst = sorted(rep["per_row"].items(), key=lambda t: -(int(t[1]['ja_leak'].split('/')[0]) / max(1, int(t[1]['ja_leak'].split('/')[1]))))
    print("[codec] ja leak by voice (worst 14): " + "; ".join(
        f"{k} {v['ja_leak']} p{v['ja_p_true']} (base p{v['baseline_p_true']}) -> {v['top_dest'][:1]}" for k, v in worst[:14]))


if __name__ == "__main__":
    main()
