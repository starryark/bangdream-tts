"""Premise check for C1: does the per-row render shift (floor ja minus real, centred unit space) align with the
codec shift delta? Uses the cached vectors from check10 (_vecs.npz). No models."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


z = np.load(Path(__file__).with_name("_vecs.npz"))
Xr, Xs, Xf = z["Xr"], z["Xs"], z["Xf"]
real = rj(BD / "e0" / "real" / "scored.jsonl")
floor = []
for d in sorted((BD / "e0" / "floor").iterdir()):
    if (d / "scored.jsonl").exists():
        floor += rj(d / "scored.jsonl")
delta = (Xs - Xr).mean(0)
rr = np.array([r["row"] for r in real])
fr = np.array([r["row"] for r in floor])
fl = np.array([r["lang"] for r in floor])
leak0 = {k for k in set(fr) if not any(r["pred"] != r["row"] for r in floor if r["row"] == k and r["lang"] == "ja")}
cos_all, cos_res, cos_good, cos_bad = [], [], [], []
for k in sorted(set(rr)):
    s = Xf[(fr == k) & (fl == "ja")].mean(0) - Xr[rr == k].mean(0)
    c = float(s @ delta / np.linalg.norm(s) / np.linalg.norm(delta))
    # the row's own codec shift, for reference
    dk = (Xs[rr == k] - Xr[rr == k]).mean(0)
    cos_res.append(float(dk @ delta / np.linalg.norm(dk) / np.linalg.norm(delta)))
    cos_all.append(c)
    (cos_good if k in leak0 else cos_bad).append(c)
q = lambda v: f"median {np.median(v):.3f} (min {np.min(v):.3f}, max {np.max(v):.3f}, n={len(v)})"
print("cos(render shift_k, delta), all rows:", q(cos_all))
print("   zero-ja-leak rows:", q(cos_good))
print("   rows with ja leak:", q(cos_bad))
print("reference: cos(row's own codec shift, delta):", q(cos_res))
# share of the render shift explained by delta: projection length / shift length
proj = []
for k in sorted(set(rr)):
    s = Xf[(fr == k) & (fl == "ja")].mean(0) - Xr[rr == k].mean(0)
    proj.append(float(s @ delta / np.linalg.norm(delta)) / np.linalg.norm(delta))
print("projection of render shift on delta, in units of |delta|: median %.2f (min %.2f, max %.2f)" % (
    np.median(proj), np.min(proj), np.max(proj)))
# null: random direction cos in 2048-D ~ N(0, 1/sqrt(2048)) = 0.022
print("null sd of cos between random 2048-D directions: %.3f" % (1 / np.sqrt(2048)))
