"""E6 VC rows for the four roles Base can't reach (local, TTS venv; E6_PLAN.md §3.6, E5's procedure).

  python -I -X utf8 e6_vc.py texts      # 40 AISHELL-3 test texts per role, unused by E1-E5 and the probes
  python -I -X utf8 e6_vc.py sources    # E1b_ep2 renders in the donor row, native gate, one source per text
  (Applio venv, cwd E:\\bd_rvc\\Applio:  E4_VC=<e6/vc> rvcenv python e4/e4_convert.py 200)
  python -I -X utf8 e6_vc.py score      # bdscore with probeE6 on the conversions (cell 200) and sources
  python -I -X utf8 e6_vc.py listen     # report + blind listening sets e6/listen/<row>/ (3 clips each)
  python -I -X utf8 e6_vc.py rows       # after e6/vc/listening.json: the VC training rows

Texts: E5's rule (e5_texts.py) - p11's pool in E1/E2's order (seed 31), excluding E1's texts, E2's
candidates, E4's and E5's sources, and the eval probes; 40 per role, disjoint.
Sources: the four roles have no E1b row, so every text takes E5-A's donor path: 6 renders in
ppp_kasumi (E1b_ep2, Chinese, seed 2510000 + 100000 * role index + 1000 * sample + batch), scored
with bdscore (row = donor), lowest-lid sample passing E2's native gate (e4_sources.native).
Gates on conversions (E5): identity raw (probeE6 pred == row and p_true >= 0.5 * base_p[row]),
identity after the codec (codec pred == row and codec p_true >= 0.5 * base_p[row]), CER <= 0.05,
1.5-13 s and not a runaway, 2-12 s after the training audio policy. No post-conversion accent gate.
At most 20 rows per role, highest raw ratio first. Listening (E5): 3 trained rows per role, codec
resynthesis, questions native? / in character?; a role judged no on either contributes 0 rows.
Rows: Kurisu v3 audio policy at 24 kHz, id pseudo_zh_<row>_rvc_<utt>, source pseudo_vc, 3% val (seed 9).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
KPOD = Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
os.environ.setdefault("BD_POD", str(BD))
os.environ.setdefault("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                 r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
os.environ.setdefault("KURISU_WORK", str(BD / "e5" / "work"))
for p in (BD / ".pydeps_sb", BD / "e1", BD / "e4", BD / "e5", KPOD, BD / "pipeline"):
    sys.path.insert(0, str(p))

VC = HERE / "vc"
ROLES = ["mewtype_arale", "mewtype_nonoka", "millsage_hotaru", "ikka_yomogi"]
DONOR = "ppp_kasumi"
PER, N_SAMPLES, BATCH, SEED = 40, 6, 20, 2510000
CAP, MIN_ROWS = 20, 10
E1B = BD / "models" / "E1b_ep2"
PROBE = HERE / "bundle" / "probe_e6.npz"


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def wj(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def texts() -> None:
    import p11_pseudo_zh as P11
    pool = P11.texts(10 ** 6, 31)
    norm = lambda t: t.rstrip("。？")   # noqa: E731
    used = {norm(t) for _, t in pool[:60 * 49]}
    used |= {norm(json.loads(l)["text"]) for l in (BD / "e2" / "results" / "pseudo" / "candidates.jsonl").open(encoding="utf-8")
             if json.loads(l)["lang"] == "zh"}
    used |= {norm(r["text"]) for r in rd(BD / "e4" / "vc" / "sources.jsonl")}
    used |= {norm(t) for v in json.loads((BD / "e5" / "vc" / "texts.json").read_text(encoding="utf-8")).values() for _, t in v}
    probes = {norm(r["text"]) for r in rd(KPOD / "probes_zh_v3.jsonl")}
    free = [(u, t) for u, t in pool[60 * 49:] if norm(t) not in used]
    assert not ({norm(t) for _, t in free} & probes)
    out = {row: free[i * PER:(i + 1) * PER] for i, row in enumerate(ROLES)}
    VC.mkdir(parents=True, exist_ok=True)
    (VC / "texts.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[e6_vc] texts: {PER} x {len(ROLES)} from {len(free)} free", flush=True)


def sources() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import bdscore
    from e4_sources import native
    from qwen_tts import Qwen3TTSModel
    tx = json.loads((VC / "texts.json").read_text(encoding="utf-8"))
    cpath = VC / "src_cands.jsonl"
    cands = rd(cpath)
    if not cands:
        model = Qwen3TTSModel.from_pretrained(str(E1B), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        for vi, row in enumerate(ROLES):
            pairs = tx[row]
            for s in range(N_SAMPLES):
                for b in range(0, len(pairs), BATCH):
                    ch = pairs[b:b + BATCH]
                    torch.manual_seed(SEED + 100000 * vi + 1000 * s + b // BATCH)
                    wavs, sr = model.generate_custom_voice(text=[t for _, t in ch], speaker=[DONOR] * len(ch),
                                                           language=["Chinese"] * len(ch), max_new_tokens=400, non_streaming_mode=True)
                    for (u, t), w in zip(ch, wavs):
                        p = VC / "src_raw" / row / f"{u}_s{s}.wav"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                        cands.append({"row": DONOR, "target": row, "lang": "zh", "utt": u, "text": t, "sample": s, "path": str(p)})
            print(f"[e6_vc] sources {row}: {N_SAMPLES * len(pairs)} renders", flush=True)
        del model
        torch.cuda.empty_cache()
        bdscore.score(cands, BD / "e1" / "bundle" / "probe_codec.npz")
        wj(cpath, cands)
    out = []
    for row in ROLES:
        mine = [r for r in cands if r["target"] == row]
        for u, t in tx[row]:
            ok = [r for r in mine if r["utt"] == u and native(r)]
            if ok:
                r = min(ok, key=lambda r: r["lid_ratio"])
                out.append({"row": row, "lang": "zh", "utt": u, "text": t, "src_row": DONOR, "src": r["path"],
                            "src_sample": r["sample"], "src_lid": r["lid_ratio"],
                            "src_ser": r.get("ser_shared") if r.get("m23_valid", True) else None, "src_cer": r["cer"],
                            "src_pred": r["pred"]})
        n = sum(r["row"] == row for r in out)
        print(f"[e6_vc] {row}: {n}/{len(tx[row])} texts with a native source; renders native "
              f"{np.mean([native(r) for r in mine]):.0%}", flush=True)
    wj(VC / "sources.jsonl", out)


def base_p() -> dict:
    return json.loads((HERE / "bundle" / "probe_e6_calib.json").read_text())["base_p"]


def score() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import bdscore
    from qwen_tts import Qwen3TTSTokenizer
    bp = base_p()
    src = rd(VC / "sources.jsonl")
    rows = [{**s, "epoch": c, "path": s["src"] if c == "source" else str(VC / "out" / "ep200" / s["row"] / f"{s['utt']}.wav")}
            for c in ("200", "source") for s in src]
    missing = [r["path"] for r in rows if not os.path.exists(r["path"])]
    if missing:
        raise SystemExit(f"{len(missing)} conversions missing, e.g. {missing[0]}")
    bdscore.score(rows, PROBE)
    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    codec = []
    for i in range(0, len(rows), 8):          # 8 + cache freed per batch: the local 16 GB card (see e6_probe)
        ch = rows[i:i + 8]
        with torch.inference_mode():
            wavs, sr = tok.decode(tok.encode([r["path"] for r in ch]))
        torch.cuda.empty_cache()
        for r, w in zip(ch, wavs):
            p = VC / "codec" / r["epoch"] / r["row"] / f"{r['utt']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
            codec.append({"row": r["row"], "lang": "zh", "text": r["text"], "path": str(p)})
    del tok
    torch.cuda.empty_cache()
    bdscore.score(codec, PROBE, accent=False, asr=False)
    for r, c in zip(rows, codec):
        r["codec_pred"], r["codec_p_true"] = c["pred"], c["p_true"]
        r["ratio"] = round(r["p_true"] / bp[r["row"]], 3)
        r["codec_ratio"] = round(r["codec_p_true"] / bp[r["row"]], 3)
        r["gates"] = {"identity": r["pred"] == r["row"] and r["ratio"] >= 0.5,
                      "codec_identity": r["codec_pred"] == r["row"] and r["codec_ratio"] >= 0.5,
                      "text": r.get("cer", 1) <= 0.05, "length": 1.5 <= r["dur"] <= 13.0 and not r.get("runaway")}
        r["row_pass"] = all(r["gates"].values())
    wj(VC / "scored.jsonl", rows)
    for row in ROLES:
        m = [r for r in rows if r["row"] == row and r["epoch"] == "200"]
        s = [r for r in rows if r["row"] == row and r["epoch"] == "source"]
        print(f"[e6_vc] score {row}: pass {np.mean([r['row_pass'] for r in m]):.2f} "
              f"(identity {np.mean([r['gates']['identity'] for r in m]):.2f}, codec {np.mean([r['gates']['codec_identity'] for r in m]):.2f}, "
              f"text {np.mean([r['gates']['text'] for r in m]):.2f}); pred {Counter(r['pred'] for r in m).most_common(2)}; "
              f"source pred {Counter(r['pred'] for r in s).most_common(2)}", flush=True)


def collect(mode: str) -> None:
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    import p03_build as V2
    from e5_collect import policy
    rows = [r for r in rd(VC / "scored.jsonl") if r["epoch"] == "200"]
    meter = pyloudnorm.Meter(V2.SR)
    trained, report = {}, {}
    for v in ROLES:
        m = [r for r in rows if r["row"] == v]
        T = []
        for r in sorted([r for r in m if r["row_pass"]], key=lambda r: -r["ratio"]):
            if len(T) >= CAP:
                break
            new = policy(r["path"], V2, meter)
            if new is not None:
                T.append((r, new))
        trained[v] = T
        report[v] = {"n": len(m), "passing": sum(r["row_pass"] for r in m), "trained": len(T), "weak": len(T) < MIN_ROWS,
                     "gates": {g: round(float(np.mean([r["gates"][g] for r in m])), 3) for g in m[0]["gates"]} if m else {}}
    (VC / "collect_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[e6_vc] collect: {json.dumps(report)}", flush=True)
    if mode == "listen":
        for v in ROLES:
            ok = sorted([r for r, _ in trained[v]], key=lambda r: r["utt"])
            pick = random.Random(5).sample(ok, min(3, len(ok)))
            out = HERE / "listen" / v
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            lines = [f"{v}: Mandarin training clips (RVC epoch 200, after the TTS codec; LOCAL ONLY, never share)", "",
                     "Questions: 1. native Mandarin, or Japanese-accented?  2. does it sound like the character?", ""]
            for i, r in enumerate(pick, 1):
                shutil.copy2(VC / "codec" / "200" / v / f"{r['utt']}.wav", out / f"{i}.wav")
                lines.append(f"{i}: {r['text']}")
            refs = [l.strip() for l in (HERE / "bundle" / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").splitlines() if l.strip()][:2]
            for j, p in enumerate(refs, 1):
                shutil.copy2(BD / p.replace("/workspace/bd/", ""), out / f"ref_ja_{j}.wav")
            (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            json.dump({str(i): r["utt"] for i, r in enumerate(pick, 1)}, open(HERE / "listen" / f"{v}_key.json", "w"), indent=0)
            print(f"[e6_vc] listen/{v}: {len(pick)} clips", flush=True)
        return
    verdict = json.loads((VC / "listening.json").read_text(encoding="utf-8"))
    rng = random.Random(9)
    added, counts = [], {}
    for v in ROLES:
        ok = verdict.get(v, {})
        if not (ok.get("native") and ok.get("character")):
            counts[v] = f"0 (listening: {ok or 'missing'})"
            continue
        for r, new in trained[v]:
            rid = f"pseudo_zh_{v}_rvc_{r['utt']}"
            p = HERE / "rows" / "zh" / v / f"{rid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), new, V2.SR, subtype="PCM_16")
            added.append({"id": rid, "audio": str(p), "text": r["text"], "speaker": v, "language": "chinese",
                          "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                          "source": "pseudo_vc", "p_true": r["p_true"], "ratio": r["ratio"], "codec_ratio": r["codec_ratio"],
                          "src_lid": r["src_lid"], "cer": r["cer"], "src_row": r["src_row"]})
        counts[v] = len(trained[v])
    wj(VC / "rvc_rows.jsonl", added)
    print(f"[e6_vc] {len(added)} VC rows: {counts}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"texts": texts, "sources": sources, "score": score}.get(sys.argv[1], lambda: collect(sys.argv[1]))()
