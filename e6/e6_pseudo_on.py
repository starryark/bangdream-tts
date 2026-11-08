"""E6 (pod): Base-generated Mandarin for the 28 Our Notes rows (E6_PLAN.md §3.4).

  python e6/e6_pseudo_on.py [--samples 6] [--cap 40]

As e1/e1_pseudo.py, with three pre-registered differences:
  texts     bundle/zh_texts.json - official zh-Hans translations of the row's own training lines
            (60 per row, built locally by e6_rows.py; asserted disjoint from the eval probes)
  samples   6 per text (E1: 2)
  identity  probeE6 (bundle/probe_e6.npz): pred == row exactly and p_true >= the matched bar in
            bundle/probe_e6_calib.json (E1: probe_codec, 0.5)
Accent (P(zh) >= 0.9, lid <= -19), text (CER <= 0.05) and length (1.5-13 s raw, not a runaway, 2-12 s
after the training audio policy) are E1's. Per text the passing sample with the highest p_true is kept,
then at most --cap per row. Locked vectors are the trainer's own (train_sft.speaker_embedding of
bundle/refs). Writes pseudo_on/{pseudo_rows.jsonl, report.json, candidates.jsonl}. Skipped if done.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
for p in (ROOT / "code", ROOT / "e1", ROOT / "pipeline"):
    sys.path.insert(0, str(p))
os.environ.setdefault("KURISU_WORK", str(ROOT))
BASE = os.environ.get("BD_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
BATCH = 48


def main() -> None:
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    import torch
    import bdscore
    import p03_build as V2
    import train_sft
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--rows", default=None, help="comma list (smoke tests only)")
    ap.add_argument("--texts", type=int, default=None, help="texts per row (smoke tests only)")
    ap.add_argument("--out", default="pseudo_on")
    args = ap.parse_args()
    out = ROOT / args.out
    if (out / "pseudo_rows.jsonl").exists() and (out / "report.json").exists():
        print(f"[pseudo_on] {out / 'pseudo_rows.jsonl'} exists, skipped")
        return
    B = ROOT / "bundle"
    texts = json.loads((B / "zh_texts.json").read_text(encoding="utf-8"))
    if args.rows:
        texts = {k: v for k, v in texts.items() if k in args.rows.split(",")}
    if args.texts:
        texts = {k: v[:args.texts] for k, v in texts.items()}
    bar = json.loads((B / "probe_e6_calib.json").read_text())["bar"]
    probe_zh = {json.loads(l)["text"].rstrip("。？！") for l in (ROOT / "code" / "probes_zh_v3.jsonl").read_text(encoding="utf-8").splitlines() if l}
    assert not ({t.rstrip("。？！") for v in texts.values() for _, t in v} & probe_zh), "pseudo texts overlap the eval probes"
    names = list(texts)

    raw = out / "raw"
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts, jobs = {}, []
    for row in names:
        vec = train_sft.speaker_embedding(base.model, row, train_sft.ref_paths(B / "refs" / f"ref_{row}.txt"), "cuda:0", torch.bfloat16)[0]
        prompts[row] = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec, x_vector_only_mode=True, icl_mode=False)
        (raw / row).mkdir(parents=True, exist_ok=True)
        jobs += [(row, u, t) for u, t in texts[row]]
    cands = []
    for s in range(args.samples):
        for b in range(0, len(jobs), BATCH):
            chunk = jobs[b:b + BATCH]
            torch.manual_seed(90000 + 1000 * s + b // BATCH)
            wavs, sr = base.generate_voice_clone(text=[t for _, _, t in chunk], language=["Chinese"] * len(chunk),
                                                 voice_clone_prompt=[prompts[row] for row, _, _ in chunk], max_new_tokens=400,
                                                 non_streaming_mode=True)
            assert len(wavs) == len(chunk)
            for (row, u, t), w in zip(chunk, wavs):
                p = raw / row / f"{u}_s{s}.wav"
                sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                cands.append({"row": row, "utt": u, "sample": s, "text": t, "lang": "zh", "path": str(p)})
            if (b // BATCH) % 10 == 0:
                print(f"[pseudo_on] sample {s}: {b + len(chunk)}/{len(jobs)}", flush=True)
    del base
    torch.cuda.empty_cache()

    bdscore.score(cands, B / "probe_e6.npz")
    gates = {"identity": lambda r: r["pred"] == r["row"] and r["p_true"] >= bar,
             "accent": lambda r: r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= -19,
             "text": lambda r: r["cer"] <= 0.05,
             "length": lambda r: 1.5 <= r["dur"] <= 13.0 and not r["runaway"]}
    for r in cands:
        r["gates"] = {g: bool(f(r)) for g, f in gates.items()}
        r["pass"] = all(r["gates"].values())
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cands), encoding="utf-8")

    meter = pyloudnorm.Meter(V2.SR)
    rng = random.Random(5)
    rows, report = [], {}
    for row in names:
        mine = [r for r in cands if r["row"] == row]
        best = {}
        for r in mine:
            if r["pass"] and (r["utt"] not in best or r["p_true"] > best[r["utt"]]["p_true"]):
                best[r["utt"]] = r
        (out / "wavs" / row).mkdir(parents=True, exist_ok=True)
        n_kept, n_trim = 0, 0
        for r in sorted(best.values(), key=lambda r: -r["p_true"]):
            if n_kept >= args.cap:
                break
            wav, sr = sf.read(r["path"], dtype="float32", always_2d=False)
            new, _ = V2.normalise_audio(wav, meter)
            if new is None or not (2.0 <= len(new) / V2.SR <= 12.0):
                n_trim += 1
                continue
            n_kept += 1
            rid = f"pseudo_on_{row}_{r['utt']}"
            p = out / "wavs" / row / f"{rid}.wav"
            sf.write(str(p), new, V2.SR, subtype="PCM_16")
            rows.append({"id": rid, "audio": str(p), "text": r["text"], "language": "chinese", "speaker": row,
                         "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                         "source": "pseudo_base_on", "p_true": r["p_true"], "lid_ratio": r.get("lid_ratio"), "cer": r["cer"]})
        report[row] = {"rendered": len(mine), "pass_any_sample": len(best), "kept": n_kept, "dropped_after_trim": n_trim,
                       "gate_pass_rate": {g: round(float(np.mean([r["gates"][g] for r in mine])), 3) for g in gates}}
    (out / "pseudo_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"bar": bar, "rows": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = Counter(r["speaker"] for r in rows)
    print(f"[pseudo_on] bar {bar}; kept {len(rows)} rows: {dict(sorted(kept.items()))}; zero: {[k for k in names if kept[k] == 0]}", flush=True)


if __name__ == "__main__":
    main()
