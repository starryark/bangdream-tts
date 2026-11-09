"""Curation audit of the BanG Dream training data (2026-11-09; CPU only, reads existing artifacts).

  python -I -X utf8 research/scripts/curation_audit.py     (TTS venv: numpy, sklearn, soundfile)

Training set = what E9 would train on: E6 bundle ja_rows (37 v1 rows + 28 Our Notes rows) plus E1 bundle ja_rows
for the 12 MyGO!!!!! / Ave Mujica v1 rows (merged), Soyo/Sakiko persona rows folded. Reports, per source:
A composition, B transcript mismatch (best-of-two ASR CER), C speaker purity (clip vs own centroid, nearest other
centroid), D speaking-rate outliers (kana per second), E noise floor (Our Notes from on_meta; v1 measured from wav),
F duplicate texts, G v1-vs-Our-Notes distance per merged character. Writes research/curation_audit.json.
"""
from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

import numpy as np

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
ALIAS = {"mygo_soyo_tsuki": "mygo_soyo", "mujica_sakiko_tsuki": "mujica_sakiko"}
MERGED = ["mygo_tomori", "mygo_anon", "mygo_raana", "mygo_soyo", "mygo_taki", "mujica_uika", "mujica_mutsumi",
          "mujica_mortis", "mujica_umiri", "mujica_nyamu", "mujica_sakiko"]


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def q(xs, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": round(float(np.percentile(xs, p)), 3) for p in ps} if len(xs) else {}


def local(p):
    p = p.replace("/workspace/bd/", "")
    return BD / p if not p.startswith("E:") else Path(p)


def main():
    out = {}
    e6 = rd(BD / "e6/bundle/ja_rows.jsonl")
    e1 = rd(BD / "e1/bundle/ja_rows.jsonl")
    rep12 = set(MERGED) | {"mujica_sakiko_tsuki"}
    rows, seen = [], set()
    for r in e6 + [x for x in e1 if x["speaker"] in rep12]:
        if r["id"] in seen:
            continue
        seen.add(r["id"]); rows.append({**r, "spk": ALIAS.get(r["speaker"], r["speaker"])})
    on_ids = {r["id"] for r in rd(BD / "data_on/manifest.jsonl")}
    man_v1 = {r["id"]: r for r in rd(BD / "data/manifest.jsonl")}
    for r in rows:
        r["src"] = "on" if r["id"] in on_ids else ("v1_" + man_v1.get(r["id"], {}).get("source", "?"))
    train = [r for r in rows if r["split"] == "train"]

    # A composition
    comp = collections.defaultdict(lambda: [0, 0.0])
    for r in train:
        comp[r["src"]][0] += 1; comp[r["src"]][1] += r["dur"] / 60
    out["A_composition"] = {k: {"clips": v[0], "minutes": round(v[1], 1)} for k, v in comp.items()}
    out["A_duration_quantiles_s"] = {s: q([r["dur"] for r in train if r["src"] == s]) for s in comp}
    out["A_short_share"] = {s: round(sum(r["dur"] < 1.5 for r in train if r["src"] == s) / max(1, comp[s][0]), 3) for s in comp}
    per_spk = collections.Counter(r["spk"] for r in train)
    out["A_clips_per_voice"] = {"min": min(per_spk.values()), "median": st.median(per_spk.values()), "max": max(per_spk.values()),
                                "merged": {k: per_spk[k] for k in MERGED}}

    # B transcript mismatch (best of two ASR systems, as the gate used)
    asr = {}
    for p in ("data/asr.jsonl", "data_on/asr.jsonl"):
        for a in rd(BD / p):
            cs = [float(a[k]) for k in ("cer_qwen", "cer_anime") if a.get(k) not in (None, "None", "")]
            if cs:
                asr[a["id"]] = {"cer": min(cs), "bar": float(a.get("bar") or 0), "kana": int(float(a.get("ref_kana") or 0)),
                                "qwen": a.get("qwen", ""), "anime": a.get("anime", "")}
    bc = collections.defaultdict(list)
    for r in train:
        if r["id"] in asr:
            bc[r["src"]].append(asr[r["id"]]["cer"])
    out["B_train_best_cer"] = {s: {"n": len(v), "share_0": round(sum(x == 0 for x in v) / len(v), 3),
                                    "share_gt_0.10": round(sum(x > 0.10 for x in v) / len(v), 3),
                                    "share_gt_0.20": round(sum(x > 0.20 for x in v) / len(v), 3), **q(v, (50, 90, 99))}
                               for s, v in bc.items()}
    worst = sorted((r for r in train if r["id"] in asr and asr[r["id"]]["cer"] > 0.2), key=lambda r: -asr[r["id"]]["cer"])
    out["B_examples_gt_0.20"] = [{"spk": r["spk"], "cer": asr[r["id"]]["cer"], "text": r["text"][:40],
                                  "asr": (asr[r["id"]]["qwen"] or asr[r["id"]]["anime"])[:40]} for r in worst[:12]]

    # C speaker purity
    vec = {}
    d = np.load(BD / "vectors/clipvec.npz", allow_pickle=True)
    for i, v in zip(d["ids"].tolist(), d["V"]):
        vec[i] = v
    on_man = rd(BD / "data_on/manifest.jsonl")
    e = np.load(BD / "e6/work/on_vec.npz", allow_pickle=True)
    EV = e["V"]                                   # load once; manifest 'vec' is the row index into V (checked: cos 0.99 vs probe_e6_vectors)
    for m in on_man:
        if m.get("vec") not in (None, "None", ""):
            vec[m["id"]] = EV[int(m["vec"])]
    have = [r for r in train if r["id"] in vec]
    X = np.stack([vec[r["id"]] for r in have]).astype(np.float64)
    mu = X.mean(0); Xc = X - mu; Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    spks = sorted({r["spk"] for r in have}); si = {s: i for i, s in enumerate(spks)}
    lab = np.array([si[r["spk"]] for r in have])
    C = np.stack([Xc[lab == i].mean(0) for i in range(len(spks))]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = Xc @ C.T
    own = S[np.arange(len(have)), lab]
    S2 = S.copy(); S2[np.arange(len(have)), lab] = -9
    other = S2.max(1)
    closer = own < other
    out["C_vectors_found"] = f"{len(have)} of {len(train)} training clips"
    by = collections.defaultdict(lambda: [0, 0])
    for r, c in zip(have, closer):
        by[r["src"]][0] += 1; by[r["src"]][1] += int(c)
    out["C_closer_to_another_voice"] = {s: {"n": v[0], "share": round(v[1] / v[0], 3)} for s, v in by.items()}
    per = collections.Counter(have[i]["spk"] for i in np.where(closer)[0])
    tot = collections.Counter(r["spk"] for r in have)
    out["C_worst_voices"] = sorted(((k, round(per[k] / tot[k], 3), per[k]) for k in per), key=lambda x: -x[1])[:12]
    zs = []
    for i in range(len(spks)):
        o = own[lab == i]; med = np.median(o); mad = np.median(np.abs(o - med)) + 1e-9
        zs.append(((o - med) / (1.4826 * mad)))
    allz = np.concatenate(zs)
    out["C_within_voice_outliers_robust_z_lt_-3"] = round(float((allz < -3).mean()), 4)

    # D speaking rate (kana per second, on the clip's own text length)
    rate = collections.defaultdict(list); flagged = 0; ex = []
    for s in spks:
        rs = [(r, asr[r["id"]]["kana"] / max(r["dur"], 0.3)) for r in train if r["spk"] == s and r["id"] in asr and asr[r["id"]]["kana"] >= 3]
        if len(rs) < 10:
            continue
        v = np.array([x for _, x in rs]); med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
        for r, x in rs:
            z = (x - med) / (1.4826 * mad)
            if abs(z) > 3.5:
                flagged += 1
                if len(ex) < 8:
                    ex.append({"spk": s, "kana_per_s": round(x, 2), "median": round(float(med), 2), "dur": r["dur"], "text": r["text"][:30]})
        rate[s] = round(float(med), 2)
    out["D_rate_outliers_|z|>3.5"] = {"n": flagged, "share": round(flagged / len(train), 4), "examples": ex}

    # E noise floor: Our Notes from on_meta (active - floor25); v1 measured from wav (10th-pct frame RMS vs 90th)
    import soundfile as sf
    def floor_db(path):
        y, srr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(-1)
        h = int(0.02 * srr)
        if len(y) < h * 10:
            return None
        fr = np.sqrt(np.mean(y[: len(y) // h * h].reshape(-1, h) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(fr)
        return float(np.percentile(db, 90) - np.percentile(db, 10))
    snr = collections.defaultdict(list)
    for r in train:
        if r["src"] != "on":
            if r["src"] == "v1_game" and len(snr["v1_game"]) >= 600:
                continue
            x = floor_db(local(r["audio"]))
            if x is not None:
                snr[r["src"]].append(x)
    meta = {m["path"].split("/")[-1].rsplit(".", 1)[0]: m for m in rd(BD / "e6/work/on_meta.jsonl")}
    for r in train:
        if r["src"] == "on":
            m = meta.get(r["id"])
            if m and m.get("floor25") not in (None, "None"):
                snr["on (active-floor25)"].append(float(m["active"]) - max(float(m["floor25"]), -90))
    out["E_dynamic_range_p90_minus_p10_dB"] = {k: {"n": len(v), **q(v, (5, 50, 95)), "share_lt_25dB": round(sum(x < 25 for x in v) / len(v), 3)} for k, v in snr.items()}
    anime_low = collections.Counter()
    for r in train:
        if r["src"] in ("v1_anime", "v1_mixed"):
            x = floor_db(local(r["audio"]))
            if x is not None and x < 25:
                anime_low[r["spk"]] += 1
    out["E_anime_low_range_by_voice"] = dict(anime_low)

    # F duplicate texts within a voice
    dup = collections.Counter((r["spk"], r["text"]) for r in train)
    out["F_duplicate_text_rows"] = sum(v - 1 for v in dup.values() if v > 1)

    # G v1 vs Our Notes per merged character
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    g = {}
    for s in MERGED:
        a = [Xc[i] for i, r in enumerate(have) if r["spk"] == s and r["src"].startswith("v1")]
        b = [Xc[i] for i, r in enumerate(have) if r["spk"] == s and r["src"] == "on"]
        if len(a) < 15 or len(b) < 15:
            g[s] = {"v1": len(a), "on": len(b)}; continue
        A, B = np.stack(a), np.stack(b); ca, cb = A.mean(0), B.mean(0)
        n = min(len(A), len(B), 250); rng = np.random.default_rng(0)
        Xp = np.vstack([A[rng.choice(len(A), n, replace=False)], B[rng.choice(len(B), n, replace=False)]]); y = np.r_[np.zeros(n), np.ones(n)]
        acc = cross_val_score(LogisticRegression(max_iter=2000), Xp, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        g[s] = {"v1": len(A), "on": len(B), "centroid_cos": round(float(ca @ cb / np.linalg.norm(ca) / np.linalg.norm(cb)), 3), "v1_vs_on_separability": round(float(acc), 3)}
    out["G_v1_vs_ournotes_per_merged_character"] = g
    (BD / "research/curation_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
