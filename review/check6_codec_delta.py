"""Codec channel shift delta (resynth - real) in the probe's centred space; does it point at the sink class?
Loads ONLY the speaker encoder (ECAPA) weights on CPU via safetensors; no TTS model, no GPU. Read-only."""
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
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram  # noqa: E402


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


cfg = json.loads((QB / "config.json").read_text(encoding="utf-8"))["speaker_encoder_config"]
enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**cfg)).eval()
sd = {}
with safe_open(str(QB / "model.safetensors"), "pt") as f:
    for k in f.keys():
        if k.startswith("speaker_encoder."):
            sd[k[len("speaker_encoder."):]] = f.get_tensor(k).float()
missing, unexpected = enc.load_state_dict(sd, strict=True), None
print("loaded", len(sd), "tensors")


def vec(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(-1)
    assert sr == 24000, (path, sr)
    with torch.inference_mode():
        m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                            hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        return enc(m)[0].numpy().astype(np.float64)


pr = np.load(BD / "vectors" / "probe.npz")
mu, coef, icpt, classes = pr["mu"].astype(np.float64), pr["coef"], pr["intercept"], list(pr["classes"])
cv = np.load(BD / "vectors" / "clipvec.npz")
cid = {i: n for n, i in enumerate(cv["ids"])}
real = rj(BD / "e0" / "real" / "scored.jsonl")
res = {r["pid"]: r for r in rj(BD / "e0" / "resynth" / "scored.jsonl")}
Xr, Xs, rows, agree = [], [], [], []
for n, r in enumerate(real):
    vr, vs = vec(r["path"]), vec(res[r["pid"]]["path"])
    ref = cv["V"][cid[f"{r['row']}_{r['tid'].split('_', 1)[1]}"]].astype(np.float64)
    agree.append(vr @ ref / np.linalg.norm(vr) / np.linalg.norm(ref))
    Xr.append((vr - mu) / np.linalg.norm(vr - mu))
    Xs.append((vs - mu) / np.linalg.norm(vs - mu))
    rows.append(r["row"])
    if n % 100 == 0:
        print("..", n, flush=True)
Xr, Xs = np.array(Xr), np.array(Xs)
print("CPU fp32 encoder vs b03 bf16 clipvec: cos min %.4f median %.4f" % (min(agree), np.median(agree)))
delta = (Xs - Xr).mean(0)
print("|delta| = %.3f ; mean per-clip |xs-xr| = %.3f ; consistency |mean|/mean|.| = %.3f" % (
    np.linalg.norm(delta), np.linalg.norm(Xs - Xr, axis=1).mean(),
    np.linalg.norm(delta) / np.linalg.norm(Xs - Xr, axis=1).mean()))
g = coef @ delta
o = np.argsort(-g)
print("logit change per class from the mean codec shift, top:", [(classes[i], round(float(g[i]), 3)) for i in o[:6]])
print("   bottom:", [(classes[i], round(float(g[i]), 3)) for i in o[-4:]])


def P(X):
    Z = X @ coef.T + icpt
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


y = np.array([classes.index(k) for k in rows])
for name, X in (("real", Xr), ("resynth", Xs)):
    p = P(X)
    print(f"{name}: p_true {p[np.arange(len(y)), y].mean():.3f}  leak {(p.argmax(1) != y).mean():.3f}")
# how much margin does the codec shift cost vs how much margin do floor renders lack?  logit margin own - best other
def margin(X):
    Z = X @ coef.T + icpt
    own = Z[np.arange(len(y)), y]
    Z2 = Z.copy(); Z2[np.arange(len(y)), y] = -np.inf
    return own - Z2.max(1)
mr, ms = margin(Xr), margin(Xs)
print("logit margin own-vs-best-other: real median %.2f, resynth median %.2f (codec costs %.2f)" % (
    np.median(mr), np.median(ms), np.median(mr - ms)))
# sink: cosine of the shift with the sink centroid direction
lab = np.array([i.rsplit("_", 1)[0] for i in cv["ids"]])
Vc = cv["V"].astype(np.float64) - mu
Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
for k in ("mujica_sakiko_tsuki", "mujica_mortis", "pp_aya"):
    c = Vc[lab == k].mean(0)
    print(f"cos(delta, centroid {k}) = {float(delta @ c / np.linalg.norm(delta) / np.linalg.norm(c)):.3f}")
np.savez(Path(__file__).with_name("_codec_delta.npz"), delta=delta)
