"""Sensitivity: re-score floor renders with the mean codec shift (from check6) removed. CPU-only ECAPA, read-only."""
import collections
import json
import math
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
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram  # noqa: E402


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


cfg = json.loads((QB / "config.json").read_text(encoding="utf-8"))["speaker_encoder_config"]
enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**cfg)).eval()
with safe_open(str(QB / "model.safetensors"), "pt") as f:
    enc.load_state_dict({k[16:]: f.get_tensor(k).float() for k in f.keys() if k.startswith("speaker_encoder.")})


def vec(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 24000
    with torch.inference_mode():
        m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                            hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        return enc(m)[0].numpy().astype(np.float64)


pr = np.load(BD / "vectors" / "probe.npz")
mu, coef, icpt, classes = pr["mu"].astype(np.float64), pr["coef"], pr["intercept"], list(pr["classes"])
delta = np.load(Path(__file__).with_name("_codec_delta.npz"))["delta"]
floor = []
for d in sorted((BD / "e0" / "floor").iterdir()):
    if (d / "scored.jsonl").exists():
        floor += rj(d / "scored.jsonl")
X = []
for n, r in enumerate(floor):
    v = vec(r["path"]) - mu
    X.append(v / np.linalg.norm(v))
    if n % 300 == 0:
        print("..", n, flush=True)
X = np.array(X)
y = np.array([classes.index(r["row"]) for r in floor])


def score(X):
    Z = X @ coef.T + icpt
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z); P = E / E.sum(1, keepdims=True)
    return P[np.arange(len(y)), y], P.argmax(1)


pt0, pd0 = score(X)
stored = np.array([r["p_true"] for r in floor])
print("CPU recompute vs b05 stored p_true: max |diff| %.4f; pred agree %d/%d" % (
    np.abs(pt0 - stored).max(), sum(classes[p] == r["pred"] for p, r in zip(pd0, floor)), len(floor)))
Xc = X - delta
Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
pt1, pd1 = score(Xc)
lang = np.array([r["lang"] for r in floor])
rows = np.array([r["row"] for r in floor])
for l in ("ja", "zh", "en"):
    m = lang == l
    k0, k1, n = int((pd0[m] != y[m]).sum()), int((pd1[m] != y[m]).sum()), int(m.sum())
    print(f"{l}: leak {k0}/{n}={k0/n:.3f} -> codec-shift removed {k1}/{n}={k1/n:.3f};  p_true {pt0[m].mean():.3f} -> {pt1[m].mean():.3f}")
print("leak destinations after correction:", collections.Counter(classes[p] for p, t in zip(pd1, y) if p != t).most_common(6))
print("\nper-row ja leak (b05 -> shift removed), rows with any ja leak:")
for k in sorted(set(rows)):
    m = (rows == k) & (lang == "ja")
    a, b, n = int((pd0[m] != y[m]).sum()), int((pd1[m] != y[m]).sum()), int(m.sum())
    if a or b:
        lo, hi = wilson(b, n)
        print(f"   {k:20s} {a}/{n} -> {b}/{n} (CI {lo:.2f}-{hi:.2f})   p_true {pt0[m].mean():.2f} -> {pt1[m].mean():.2f}")
