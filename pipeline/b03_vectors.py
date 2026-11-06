"""E0 step 3: held-out split, locked speaker vectors, separability, and the leakage probe.

  python -I b03_vectors.py

1. Held-out (split "test"), chosen BEFORE any vector: per row ~5% (floor 4, cap 10) of the
   clips that passed the transcript check, 2-12 s, no Latin letters, >= 8 kana. Seeded by
   row name, so the split is reproducible.
2. Clip vectors: the Qwen3-TTS-Base speaker encoder with the trainer's mel recipe, on the
   normalised 24 kHz train clips.
3. Reference list per row = the trainer's --speakers input (refs/ref_<row>.txt): train
   clips with an ok transcript and >= 2 s, minus the 20% least typical by centred cosine to
   the row's mean (the "dominant cluster", CosyVoice2 practice), capped at 64 clips. The
   locked vector is then computed by kurisu_v2/pod/train_sft.speaker_embedding itself, so
   it is exactly what training will use.
4. Separability on centred vectors, the Mujica "cluster vs all" A/B, and a logistic-
   regression probe on centred clip vectors (fit on train, scored on held-out real clips).
   The probe's confusion on real clips is the baseline that rendered clips are compared to.

Writes data/split.jsonl, refs/ref_<row>.txt, vectors/{clipvec.npz, locked.pt, probe.npz},
e0/vectors_report.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402

LATIN = re.compile(r"[A-Za-z]")
REF_CAP, REF_DROP = 64, 0.20


def nrm(a):
    import numpy as np
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from sklearn.linear_model import LogisticRegression

    man = C.read_jsonl(C.DATA / "manifest.jsonl")
    asr = {r["id"]: r for r in C.read_jsonl(C.DATA / "asr.jsonl")}
    rows = sorted({r["row"] for r in man})
    by_row = {k: [r for r in man if r["row"] == k] for k in rows}

    # 1. held-out split ------------------------------------------------------------
    split = {}
    for k in rows:
        elig = [r for r in by_row[k] if asr[r["id"]]["verdict"] == "ok" and 2.0 <= r["dur_src"] <= 12.0
                and not LATIN.search(r["text"]) and asr[r["id"]]["ref_kana"] >= 8]
        n = min(10, max(4, round(0.05 * len(by_row[k]))))
        rng = random.Random(int(hashlib.sha256(k.encode()).hexdigest()[:8], 16))
        test = set(r["id"] for r in rng.sample(sorted(elig, key=lambda r: r["id"]), min(n, len(elig))))
        for r in by_row[k]:
            split[r["id"]] = "test" if r["id"] in test else "train"
    C.write_jsonl(C.DATA / "split.jsonl", [{"id": r["id"], "row": r["row"], "split": split[r["id"]]} for r in man])

    # 2. clip vectors ---------------------------------------------------------------
    vec_dir = C.OUT / "vectors"
    vec_dir.mkdir(exist_ok=True)
    w = Qwen3TTSModel.from_pretrained(str(C.QWEN_BASE), dtype=torch.bfloat16, attn_implementation="sdpa")
    model = w.model
    enc = model.speaker_encoder.to("cuda").eval()
    ids, vecs = [], []
    for i, r in enumerate(man):
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                                hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            vecs.append(enc(m.to("cuda").to(torch.bfloat16))[:1].float().cpu().numpy()[0])
        ids.append(r["id"])
        if i % 2000 == 0:
            print(f"[vec] {i}/{len(man)}", flush=True)
    V = np.stack(vecs)
    np.savez(vec_dir / "clipvec.npz", ids=np.array(ids), V=V)
    idx = {i: n for n, i in enumerate(ids)}
    train_ids = [r["id"] for r in man if split[r["id"]] == "train"]
    mu = V[[idx[i] for i in train_ids]].mean(0)

    # 3. reference lists + locked vectors (the trainer's own function) ---------------
    from train_sft import speaker_embedding
    ref_dir = C.OUT / "refs"
    ref_dir.mkdir(exist_ok=True)
    report = {"rows": {}, "mujica_ab": {}}
    table, ref_lists = [], {}
    for k in rows:
        cand = [r for r in by_row[k] if split[r["id"]] == "train" and asr[r["id"]]["verdict"] == "ok"
                and r["dur_src"] >= 2.0]
        cv = nrm(V[[idx[r["id"]] for r in cand]] - mu)
        typ = cv @ nrm(cv.mean(0))
        keep = [cand[j] for j in np.argsort(-typ)[: max(1, int(round(len(cand) * (1 - REF_DROP))))]]
        rng = random.Random(1)
        keep = sorted(rng.sample(keep, min(REF_CAP, len(keep))), key=lambda r: r["id"])
        ref_lists[k] = keep
        (ref_dir / f"ref_{k}.txt").write_text("".join(r["audio"] + "\n" for r in keep), encoding="utf-8", newline="\n")
        from pathlib import Path
        e = speaker_embedding(model, k, [Path(r["audio"]) for r in keep], "cuda", torch.bfloat16)
        table.append(e.float().cpu())
        report["rows"][k] = {"n_clips": len(by_row[k]), "n_test": sum(split[r["id"]] == "test" for r in by_row[k]),
                             "n_ref": len(keep), "ref_seconds": round(sum(r["dur"] for r in keep), 1),
                             "n_candidates": len(cand)}
        if k.startswith("mujica_"):
            report["mujica_ab"][k] = {"all": nrm(V[[idx[r["id"]] for r in cand]].mean(0) - mu).tolist(),
                                      "cluster": nrm(V[[idx[r["id"]] for r in keep]].mean(0) - mu).tolist()}
    T = torch.cat(table, 0)
    torch.save({"names": rows, "table": T}, vec_dir / "locked.pt")

    # 4a. separability of the locked vectors (centred) --------------------------------
    L = nrm(T.numpy() - mu)
    S = L @ L.T
    n = len(rows)
    off = S[~np.eye(n, dtype=bool)]
    pairs = sorted(((float(S[i, j]), rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)), reverse=True)
    report["locked_centred"] = {"median": round(float(np.median(off)), 3), "p95": round(float(np.percentile(off, 95)), 3),
                                "max": round(float(off.max()), 3),
                                "top_pairs": [(a, b, round(s, 3)) for s, a, b in pairs[:15]]}
    # 4b. Mujica: does the dominant-cluster vector move away from its Mujica neighbours?
    muj = [k for k in rows if k.startswith("mujica_")]
    ab = report.pop("mujica_ab")
    report["mujica_ab"] = {}
    for k in muj if len(muj) > 1 else []:
        res = {}
        for mode in ("all", "cluster"):
            a = np.array(ab[k][mode])
            others = [(float(a @ np.array(ab[o][mode])), o) for o in muj if o != k]
            res[mode] = max(others)
        report["mujica_ab"][k] = {"nearest_mujica_all": [res["all"][1], round(res["all"][0], 3)],
                                  "nearest_mujica_cluster": [res["cluster"][1], round(res["cluster"][0], 3)]}
    # 4c. probe: centred clip vectors -> row; fit on train, confusion on held-out real clips
    Xtr = nrm(V[[idx[i] for i in train_ids]] - mu)
    ytr = np.array([rows.index(i.rsplit("_", 1)[0]) for i in train_ids])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    test_ids = [r["id"] for r in man if split[r["id"]] == "test"]
    Xte = nrm(V[[idx[i] for i in test_ids]] - mu)
    yte = np.array([rows.index(i.rsplit("_", 1)[0]) for i in test_ids])
    pred = clf.predict(Xte)
    P = clf.predict_proba(Xte)
    np.savez(vec_dir / "probe.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(rows), mu=mu)
    real = {}
    for j, k in enumerate(rows):
        m = yte == j
        wrong = [rows[p] for p in pred[m] if p != j]
        real[k] = {"n": int(m.sum()), "acc": round(float((pred[m] == j).mean()), 3),
                   "p_true_mean": round(float(P[m, j].mean()), 3),
                   "confused_with": max(set(wrong), key=wrong.count) if wrong else None}
    report["probe_real_heldout"] = {"acc_mean": round(float((pred == yte).mean()), 3), "per_row": real}
    # held-out sets are 4-10 clips per row, so also a 5-fold estimate on the train clips
    from sklearn.model_selection import cross_val_predict
    cvp = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), Xtr, ytr, cv=5)
    report["probe_real_cv5"] = {"acc_mean": round(float((cvp == ytr).mean()), 3), "per_row": {
        k: {"acc": round(float((cvp[ytr == j] == j).mean()), 3),
            "confused_with": (lambda w: max(set(w), key=w.count) if w else None)(
                [rows[p] for p in cvp[ytr == j] if p != j])} for j, k in enumerate(rows)}}
    out = C.OUT / "e0"
    out.mkdir(exist_ok=True)
    (out / "vectors_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    worst = sorted(real.items(), key=lambda t: t[1]["acc"])[:8]
    print(f"[vec] locked centred median {report['locked_centred']['median']} max {report['locked_centred']['max']}; "
          f"probe real held-out acc {report['probe_real_heldout']['acc_mean']}; worst "
          + ", ".join(f"{k} {v['acc']}->{v['confused_with']}" for k, v in worst), flush=True)


if __name__ == "__main__":
    main()
