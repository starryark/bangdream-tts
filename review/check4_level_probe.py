"""Loudness of renders vs real clips (pyloudnorm, CPU) and the probe's 'default class' geometry. No models."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.path.append(str(BD / ".pydeps"))
import pyloudnorm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


meter = pyloudnorm.Meter(24000)
floor = []
for d in sorted((BD / "e0" / "floor").iterdir()):
    if (d / "scored.jsonl").exists():
        floor += rj(d / "scored.jsonl")
real = rj(BD / "e0" / "real" / "scored.jsonl")
res = rj(BD / "e0" / "resynth" / "scored.jsonl")


def lufs(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    try:
        return meter.integrated_loudness(x), float(np.abs(x).max())
    except Exception:
        return float("nan"), float("nan")


for name, sel in (("real", real[::6]), ("resynth", res[::6]), ("floor ja", [r for r in floor if r["lang"] == "ja"][::6]),
                  ("floor zh", [r for r in floor if r["lang"] == "zh"][::12])):
    L = np.array([lufs(r["path"]) for r in sel])
    print(f"{name:9s} n={len(sel)} LUFS median {np.nanmedian(L[:, 0]):.1f} p10 {np.nanpercentile(L[:, 0], 10):.1f} "
          f"p90 {np.nanpercentile(L[:, 0], 90):.1f}  peak median {np.nanmedian(L[:, 1]):.2f}")
# leaked vs non-leaked floor ja loudness
fj = [r for r in floor if r["lang"] == "ja"]
Lk = [lufs(r["path"])[0] for r in fj if r["pred"] != r["row"]]
Lo = [lufs(r["path"])[0] for r in fj if r["pred"] == r["row"]]
print(f"floor ja LUFS: leaked median {np.nanmedian(Lk):.1f} (n={len(Lk)})  own-row median {np.nanmedian(Lo):.1f} (n={len(Lo)})")

# probe geometry: which class wins for inputs with little evidence?
pr = np.load(BD / "vectors" / "probe.npz")
coef, icpt, classes = pr["coef"], pr["intercept"], list(pr["classes"])
order = np.argsort(-icpt)
print("top intercepts:", [(classes[i], round(float(icpt[i]), 3)) for i in order[:5]])
print("coef row norms top:", sorted(((classes[i], round(float(np.linalg.norm(coef[i])), 2)) for i in range(len(classes))),
                                     key=lambda t: -t[1])[:5])
rng = np.random.default_rng(0)
for scale in (0.0, 0.5, 1.0):
    X = rng.standard_normal((5000, coef.shape[1]))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = scale * X @ coef.T + icpt
    c = collections.Counter(classes[i] for i in Z.argmax(1))
    print(f"random unit inputs x{scale}: argmax classes", c.most_common(4))
# within-class spread of real clips (centred), as a proxy for how 'broad' each class is
cv = np.load(BD / "vectors" / "clipvec.npz")
ids, V = list(cv["ids"]), cv["V"].astype(np.float64)
mu = pr["mu"].astype(np.float64)
X = V - mu
X /= np.linalg.norm(X, axis=1, keepdims=True)
lab = np.array([i.rsplit("_", 1)[0] for i in ids])
spread = {}
for k in classes:
    Xk = X[lab == k]
    c = Xk.mean(0)
    spread[k] = float(np.linalg.norm(c))   # mean resultant length: low = broad class
print("broadest classes (lowest mean resultant length of centred unit vectors):",
      sorted(((k, round(v, 3)) for k, v in spread.items()), key=lambda t: t[1])[:5])
# distance of each class centroid to the origin direction mu: cos(centroid_raw, mu) -- 'generic-ness'
raw_c = {k: V[lab == k].mean(0) for k in classes}
print("rows whose raw centroid is closest to the global mean (cos to mu):",
      sorted(((k, round(float(raw_c[k] @ mu / np.linalg.norm(raw_c[k]) / np.linalg.norm(mu)), 4)) for k in classes),
             key=lambda t: -t[1])[:5])
