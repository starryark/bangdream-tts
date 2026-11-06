"""Robustness of the codec-shift correction: estimate delta on the held-out clips of one half of the rows, apply it
to the floor renders of the OTHER half (no row contributes to its own correction). CPU-only ECAPA, read-only.
Caches centred unit vectors in review/_vecs.npz."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open

torch.set_num_threads(8)
BD = Path(r"E:\STEINS;GATEVoice\bangdream")
QB = Path(r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
sys.stdout.reconfigure(encoding="utf-8")
CACHE = Path(__file__).with_name("_vecs.npz")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


pr = np.load(BD / "vectors" / "probe.npz")
mu, coef, icpt, classes = pr["mu"].astype(np.float64), pr["coef"], pr["intercept"], list(pr["classes"])
real = rj(BD / "e0" / "real" / "scored.jsonl")
res = {r["pid"]: r for r in rj(BD / "e0" / "resynth" / "scored.jsonl")}
floor = []
for d in sorted((BD / "e0" / "floor").iterdir()):
    if (d / "scored.jsonl").exists():
        floor += rj(d / "scored.jsonl")

if CACHE.exists():
    z = np.load(CACHE)
    Xr, Xs, Xf = z["Xr"], z["Xs"], z["Xf"]
else:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram
    cfg = json.loads((QB / "config.json").read_text(encoding="utf-8"))["speaker_encoder_config"]
    enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**cfg)).eval()
    with safe_open(str(QB / "model.safetensors"), "pt") as f:
        enc.load_state_dict({k[16:]: f.get_tensor(k).float() for k in f.keys() if k.startswith("speaker_encoder.")})

    def cvec(path):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                                hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            v = enc(m)[0].numpy().astype(np.float64) - mu
        return v / np.linalg.norm(v)
    Xr = np.array([cvec(r["path"]) for r in real])
    Xs = np.array([cvec(res[r["pid"]]["path"]) for r in real])
    Xf = np.array([cvec(r["path"]) for r in floor])
    np.savez(CACHE, Xr=Xr, Xs=Xs, Xf=Xf)

yr = np.array([classes.index(r["row"]) for r in real])
yf = np.array([classes.index(r["row"]) for r in floor])
lang = np.array([r["lang"] for r in floor])


def pred(X):
    return (X @ coef.T + icpt).argmax(1)


def corr(X, d):
    Y = X - d
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


rng = np.random.default_rng(0)
res_ja = []
for rep in range(200):
    perm = rng.permutation(len(classes))
    A = set(perm[: len(classes) // 2])
    mA = np.array([y in A for y in yr])
    dA = (Xs[mA] - Xr[mA]).mean(0)
    dB = (Xs[~mA] - Xr[~mA]).mean(0)
    fB = np.array([y not in A for y in yf])          # renders of rows NOT used for dA
    fA = ~fB
    out = np.empty(len(yf), dtype=int)
    out[fB] = pred(corr(Xf[fB], dA))
    out[fA] = pred(corr(Xf[fA], dB))
    res_ja.append([(out[lang == l] != yf[lang == l]).mean() for l in ("ja", "zh", "en")])
res_ja = np.array(res_ja)
base = [(pred(Xf)[lang == l] != yf[lang == l]).mean() for l in ("ja", "zh", "en")]
full = [(pred(corr(Xf, (Xs - Xr).mean(0)))[lang == l] != yf[lang == l]).mean() for l in ("ja", "zh", "en")]
for i, l in enumerate(("ja", "zh", "en")):
    print(f"{l}: leak uncorrected {base[i]:.3f} | delta from all rows {full[i]:.3f} | cross-half delta "
          f"median {np.median(res_ja[:, i]):.3f} (2.5-97.5%: {np.percentile(res_ja[:, i], 2.5):.3f}-{np.percentile(res_ja[:, i], 97.5):.3f})")
# also: delta estimated only from the 'well-cloned' rows (floor ja leak 0) applied to all
fl_leak = collections.defaultdict(list)
for r, p in zip(floor, pred(Xf)):
    if r["lang"] == "ja":
        fl_leak[r["row"]].append(classes[p] != r["row"])
good = {classes.index(k) for k, v in fl_leak.items() if not any(v)}
mG = np.array([y in good for y in yr])
dG = (Xs[mG] - Xr[mG]).mean(0)
fG = pred(corr(Xf, dG))
print(f"delta from the {len(good)} rows with zero floor-ja leak only, applied to all: ja leak {(fG[lang=='ja'] != yf[lang=='ja']).mean():.3f}")
# resynth itself with real-margin: fraction of real clips whose margin < codec cost
