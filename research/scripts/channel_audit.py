"""Channel audit: do v1 (GBP game / anime) and Our Notes recordings differ in bandwidth, noise floor or level?
Also re-checks speaker purity with per-source centring (removes the shared channel offset).

  python -I -X utf8 research/scripts/channel_audit.py     (TTS venv)
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
ALIAS = {"mygo_soyo_tsuki": "mygo_soyo", "mujica_sakiko_tsuki": "mujica_sakiko"}


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def local(p):
    p = p.replace("/workspace/bd/", "")
    return BD / p if not p.startswith("E:") else Path(p)


def feats(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(-1)
    h = int(0.02 * sr)
    n = len(y) // h
    if n < 20:
        return None
    fr = y[: n * h].reshape(n, h)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-20)
    nz = rms > 1e-5                                   # exclude digital-silence padding
    if nz.sum() < 15:
        return None
    db = 20 * np.log10(rms[nz])
    act = np.percentile(db, 90)
    floor = np.percentile(db, 10)
    # spectrum of active frames
    act_fr = fr[nz][db > act - 20]
    spec = np.abs(np.fft.rfft(act_fr * np.hanning(h), axis=1)) ** 2
    ps = spec.mean(0)
    f = np.fft.rfftfreq(h, 1 / sr)
    c = np.cumsum(ps) / ps.sum()
    roll99 = float(f[np.searchsorted(c, 0.99)])
    hf = float(ps[f >= 8000].sum() / ps.sum())
    # highest frequency with energy within 60 dB of the spectral peak (bandwidth / low-pass cutoff)
    pdb = 10 * np.log10(ps + 1e-20)
    above = np.where(pdb > pdb.max() - 60)[0]
    bw = float(f[above.max()]) if len(above) else 0.0
    return {"active_db": float(act), "floor_rel_db": float(act - floor), "rolloff99": roll99, "hf_ratio": hf, "bandwidth": bw,
            "peak": float(np.abs(y).max())}


def main():
    e6 = rd(BD / "e6/bundle/ja_rows.jsonl")
    e1 = rd(BD / "e1/bundle/ja_rows.jsonl")
    on_ids = {r["id"] for r in rd(BD / "data_on/manifest.jsonl")}
    man = {r["id"]: r for r in rd(BD / "data/manifest.jsonl")}
    rows = {r["id"]: r for r in e6 + e1}
    src = {i: ("on" if i in on_ids else "v1_" + man.get(i, {}).get("source", "?")) for i in rows}
    rng = random.Random(0)
    out = {}
    for s in ("v1_game", "v1_anime", "on"):
        ids = [i for i in rows if src[i] == s and rows[i]["split"] == "train"]
        pick = rng.sample(ids, min(400, len(ids)))
        fs = [x for x in (feats(local(rows[i]["audio"])) for i in pick) if x]
        out[s] = {k: {"p10": round(float(np.percentile([x[k] for x in fs], 10)), 3), "p50": round(float(np.percentile([x[k] for x in fs], 50)), 3),
                      "p90": round(float(np.percentile([x[k] for x in fs], 90)), 3)} for k in fs[0]}
        out[s]["n"] = len(fs)
    # per merged character, the same features for v1 vs ON
    per = {}
    for ch in ("mygo_anon", "mujica_umiri", "mygo_tomori", "mujica_uika"):
        for s in ("v1", "on"):
            ids = [i for i, r in rows.items() if ALIAS.get(r["speaker"], r["speaker"]) == ch and (src[i] == "on") == (s == "on") and r["split"] == "train"]
            fs = [x for x in (feats(local(rows[i]["audio"])) for i in rng.sample(ids, min(60, len(ids)))) if x]
            per[f"{ch}:{s}"] = {k: round(float(np.median([x[k] for x in fs])), 2) for k in ("active_db", "floor_rel_db", "bandwidth", "hf_ratio")}
    out["per_character_median"] = per
    # speaker purity with per-source centring: nearest centroid among the same source's voices
    vec = {}
    d = np.load(BD / "vectors/clipvec.npz", allow_pickle=True)
    for i, v in zip(d["ids"].tolist(), d["V"]):
        vec[i] = v
    e = np.load(BD / "e6/work/on_vec.npz", allow_pickle=True)
    EV = e["V"]                                   # manifest 'vec' is the row index into V
    for m in rd(BD / "data_on/manifest.jsonl"):
        if m.get("vec") not in (None, "None", ""):
            vec[m["id"]] = EV[int(m["vec"])]
    pur = {}
    for s in ("v1", "on"):
        tr = [i for i, r in rows.items() if r["split"] == "train" and i in vec and ((src[i] == "on") == (s == "on"))]
        X = np.stack([vec[i] for i in tr]).astype(np.float64); X -= X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
        lab = [ALIAS.get(rows[i]["speaker"], rows[i]["speaker"]) for i in tr]
        names = sorted(set(lab)); li = np.array([names.index(x) for x in lab])
        C = np.stack([X[li == j].mean(0) for j in range(len(names))]); C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = X @ C.T; own = S[np.arange(len(tr)), li]; S[np.arange(len(tr)), li] = -9
        closer = own < S.max(1)
        worst = collections.Counter(np.array(lab)[closer].tolist()); tot = collections.Counter(lab)
        pur[s] = {"n": len(tr), "share_closer_to_other": round(float(closer.mean()), 3),
                  "worst": sorted(((k, round(worst[k] / tot[k], 3)) for k in worst), key=lambda x: -x[1])[:8]}
    out["purity_per_source_centring"] = pur
    (BD / "research/channel_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
