"""E6 step 4 (local, TTS venv): probeE6, its matched identity bar, and per-row real-resynth p_true.

  python -I -X utf8 e6_probe.py

E6_PLAN.md §3.4 / §4. Classes: the 65 rows of e6/bundle/speakers.txt.
  Our Notes rows: up to 150 of the row's selected training clips >= 1.5 s (seed 71), through the
    Qwen3-TTS-Tokenizer-12Hz (encode -> decode), embedded with bdscore's speaker-encoder embedder.
  v1 rows: their stored resynthesised vectors (vectors/clipvec_resynth.npz, b08), all clips.
Logistic regression (C 1, balanced) on vectors centred by the global mean and length-normalised,
as b08 and probe63 (facts §9.2).
Matched bar (facts §9.2 method): the p_true value passed by the same fraction of real clips (5-fold
out-of-fold) as E1's 0.5 bar passed under the 49-class probe (0.910, e6/work/gate_calib.json).
base_p[row] = mean out-of-fold p_true of the row's real resynthesised clips (the VC identity gate:
p_true >= 0.5 * base_p, as E4/E5).
Writes e6/bundle/probe_e6.npz {coef, intercept, classes, mu} and e6/bundle/probe_e6_calib.json.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
os.environ.setdefault("BD_POD", str(BD))
os.environ.setdefault("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                 r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
os.environ.setdefault("KURISU_WORK", str(HERE / "work"))
sys.path.insert(0, str(BD / ".pydeps_sb"))
sys.path.insert(0, str(BD / "e1"))
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402

B = HERE / "bundle"
PER_ROW = 150


def local(p: str) -> str:
    return str(BD / p.replace("/workspace/bd/", "").replace("/", os.sep))


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    import bdscore
    from qwen_tts import Qwen3TTSTokenizer
    sys.stdout.reconfigure(encoding="utf-8")
    names = [s.split("=")[0] for s in (B / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    sel = json.loads((B / "rows_e6.json").read_text())
    on_rows = sel["replaced"] + sel["new"]
    ja = [r for r in C.read_jsonl(B / "ja_rows.jsonl") if r["source"] == "ja_on" and r["split"] == "train" and r["dur"] >= 1.5]
    cache = HERE / "work" / "probe_e6_vectors.npz"
    if cache.exists():
        z = np.load(cache)
        ids, VN, yn = list(z["ids"]), z["V"], list(z["row"])
    else:
        pick = []
        for row in on_rows:
            mine = sorted([r for r in ja if r["speaker"] == row], key=lambda r: r["id"])
            random.Random(71).shuffle(mine)
            pick += mine[:PER_ROW]
        pick.sort(key=lambda r: (r["dur"], r["id"]))
        # Batches of <= 8 clips and <= 40 s of audio, cache freed after each (b08's local recipe): batches
        # of 16 long clips filled the 16 GB card and spilled ~25 GB into system RAM (2026-11-07).
        batches, cur = [], []
        for r in pick:
            if cur and (len(cur) >= 8 or sum(x["dur"] for x in cur) + r["dur"] > 40.0):
                batches.append(cur)
                cur = []
            cur.append(r)
        batches += [cur] if cur else []
        part = cache.with_suffix(".partial.npz")
        ids, vs, yn = [], [], []
        if part.exists():
            z = np.load(part)
            ids, vs, yn = list(z["ids"]), list(z["V"]), list(z["row"])
        done = set(ids)
        tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
        embed, enc = bdscore._embedder()
        n0 = len(ids)
        for bi, ch in enumerate(batches):
            if all(r["id"] in done for r in ch):
                continue
            with torch.inference_mode():
                wavs, sr = tok.decode(tok.encode([local(r["audio"]) for r in ch]))
            torch.cuda.empty_cache()
            for r, w in zip(ch, wavs):
                ids.append(r["id"])
                vs.append(embed(np.asarray(w, dtype=np.float32), sr))
                yn.append(r["speaker"])
            del wavs
            if (len(ids) - n0) // 400 != (len(ids) - n0 - len(ch)) // 400:
                np.savez(part, ids=np.array(ids), V=np.stack(vs), row=np.array(yn))
                print(f"[probe] resynth {len(ids)}/{len(pick)} (batch {bi + 1}/{len(batches)}, "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak)", flush=True)
        assert sorted(ids) == sorted(r["id"] for r in pick)
        VN = np.stack(vs)
        np.savez(cache, ids=np.array(ids), V=VN, row=np.array(yn))
        part.unlink(missing_ok=True)
        del tok, enc
        torch.cuda.empty_cache()
    z = np.load(BD / "vectors" / "clipvec_resynth.npz")
    oid = [str(i) for i in z["ids"]]
    orow = np.array([i.rsplit("_", 1)[0] for i in oid])
    keep = np.isin(orow, sel["v1_kept"])
    Vall = np.concatenate([VN, z["V"][keep]])
    yall = np.array(list(yn) + list(orow[keep]))
    assert sorted(set(yall)) == sorted(names), set(names) ^ set(yall)
    y = np.array([names.index(k) for k in yall])
    mu = Vall.mean(0)
    X = Vall - mu
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X, y)
    P = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), X, y, cv=5, method="predict_proba")
    pt = P[np.arange(len(y)), y]
    f49 = json.loads((HERE / "work" / "gate_calib.json").read_text())["f49"]
    bar = float(np.quantile(pt, 1 - f49))
    acc = {k: round(float((P[y == j].argmax(1) == j).mean()), 3) for j, k in enumerate(names)}
    base_p = {k: round(float(pt[y == j].mean()), 4) for j, k in enumerate(names)}
    np.savez(B / "probe_e6.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(names), mu=mu)
    cal = {"classes": len(names), "n_clips": int(len(y)), "cv5_acc": round(float((P.argmax(1) == y).mean()), 4),
           "f49": f49, "bar": round(bar, 4), "frac_pass_0.5": round(float((pt >= 0.5).mean()), 4),
           "per_row_acc": acc, "base_p": base_p,
           "low_rows": {k: v for k, v in acc.items() if v < 0.9}}
    (B / "probe_e6_calib.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(f"[probe] {len(names)} classes, {len(y)} clips, cv5 acc {cal['cv5_acc']}, matched bar {cal['bar']} "
          f"(0.5 passes {cal['frac_pass_0.5']}); rows below 0.9: {cal['low_rows']}", flush=True)


if __name__ == "__main__":
    main()
