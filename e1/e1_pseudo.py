"""E1 (pod): Base-generated Mandarin in each voice, gated on identity AND accent (E1b rows).

  python e1_pseudo.py [--texts_per_voice 60] [--samples 2] [--cap 40]

Per voice: the trainer's locked vector (train_sft.speaker_embedding of bundle/refs/ref_<row>.txt,
the same call the trainer makes), Base x-vector-only, language Chinese, non-streaming,
upstream sampling defaults. Each voice gets its own AISHELL-3 *test* texts (p11's pool:
8-30 hanzi, <= 1 numeral, not in AISHELL-3 train; disjoint across voices; fewer per voice if
the pool is short, minimum 20). Eval probes are probes_zh_v3 texts, which are not AISHELL
sentences (asserted). Batches mix voices (one prompt per item). Skipped if its output exists.

Gates (all required), scored by bdscore.score (identical to the E1 eval):
  identity  probe_codec assigns the clip to its own row, p_true >= 0.5
  accent    VoxLingua P(zh) >= 0.9 and lid_ratio <= -19 (the E0 floor mean)
  text      Qwen3-ASR CER <= 0.05
  length    not a runaway, and 2-12 s after the training audio policy (trim + pad)
Per text the passing sample with the highest p_true is kept; then at most --cap per voice
(highest p_true first). Kept clips get the Kurisu v3 audio policy (p03_build.normalise_audio).
Writes pseudo/pseudo_rows.jsonl and pseudo/report.json (pass rates per voice and gate).
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
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BATCH = 48


def main() -> None:
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    import torch
    import bdscore
    import p03_build as V2
    import p11_pseudo_zh as P11
    import train_sft
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_per_voice", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    out = ROOT / "pseudo"
    if (out / "pseudo_rows.jsonl").exists() and (out / "report.json").exists():
        print(f"[pseudo] {out / 'pseudo_rows.jsonl'} exists, skipped")
        return
    names = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    pool = P11.texts(args.texts_per_voice * len(names), args.seed)
    per_voice = min(args.texts_per_voice, len(pool) // len(names))
    if per_voice < 20:
        raise SystemExit(f"only {len(pool)} AISHELL-3 test texts available ({per_voice} per voice)")
    if per_voice < args.texts_per_voice:
        print(f"[pseudo] WARNING: {len(pool)} texts -> {per_voice} per voice (asked {args.texts_per_voice})")
    probe_zh = {json.loads(l)["text"].rstrip("。？") for l in (ROOT / "code" / "probes_zh_v3.jsonl").read_text(encoding="utf-8").splitlines() if l}
    assert not ({t.rstrip("。？") for _, t in pool} & probe_zh), "pseudo texts overlap the eval probes"

    raw = out / "raw"
    base = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts, jobs = {}, []
    for vi, row in enumerate(names):
        vec = train_sft.speaker_embedding(base.model, row, train_sft.ref_paths(ROOT / "bundle" / "refs" / f"ref_{row}.txt"),
                                          "cuda:0", torch.bfloat16)[0]
        prompts[row] = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec, x_vector_only_mode=True, icl_mode=False)
        (raw / row).mkdir(parents=True, exist_ok=True)
        for u, t in pool[vi * per_voice:(vi + 1) * per_voice]:      # disjoint texts per voice
            jobs.append((row, u, t))
    cands = []
    for s in range(args.samples):            # batches mix voices; one prompt per item
        for b in range(0, len(jobs), BATCH):
            chunk = jobs[b:b + BATCH]
            torch.manual_seed(70000 + 1000 * s + b // BATCH)
            wavs, sr = base.generate_voice_clone(
                text=[t for _, _, t in chunk], language=["Chinese"] * len(chunk),
                voice_clone_prompt=[prompts[row] for row, _, _ in chunk], max_new_tokens=400,
                non_streaming_mode=True)
            assert len(wavs) == len(chunk)
            for (row, u, t), w in zip(chunk, wavs):
                p = raw / row / f"{u}_s{s}.wav"
                sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                cands.append({"row": row, "utt": u, "sample": s, "text": t, "lang": "zh", "path": str(p)})
            if (b // BATCH) % 10 == 0:
                print(f"[pseudo] sample {s}: {b + len(chunk)}/{len(jobs)}", flush=True)
    del base
    torch.cuda.empty_cache()

    bdscore.score(cands, ROOT / "bundle" / "probe_codec.npz")
    gates = {"identity": lambda r: r["pred"] == r["row"] and r["p_true"] >= 0.5,
             "accent": lambda r: r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= -19,
             "text": lambda r: r["cer"] <= 0.05,
             "length": lambda r: 1.5 <= r["dur"] <= 13.0 and not r["runaway"]}   # final 2-12 s check after trim
    for r in cands:
        r["gates"] = {g: bool(f(r)) for g, f in gates.items()}
        r["pass"] = all(r["gates"].values())
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cands), encoding="utf-8")

    meter = pyloudnorm.Meter(V2.SR)
    rng = random.Random(5)
    wav_dir = out / "wavs"
    rows, report = [], {}
    for row in names:
        mine = [r for r in cands if r["row"] == row]
        best = {}
        for r in mine:
            if r["pass"] and (r["utt"] not in best or r["p_true"] > best[r["utt"]]["p_true"]):
                best[r["utt"]] = r
        ranked = sorted(best.values(), key=lambda r: -r["p_true"])
        (wav_dir / row).mkdir(parents=True, exist_ok=True)
        n_kept, n_trim_fail = 0, 0
        for r in ranked:
            if n_kept >= args.cap:
                break
            wav, sr = sf.read(r["path"], dtype="float32", always_2d=False)
            if sr != V2.SR:
                raise SystemExit(f"{r['path']}: {sr} Hz")
            new, _ = V2.normalise_audio(wav, meter)
            # length gate on the clip as it will be trained (trimmed + padded), not the raw render
            if new is None or not (2.0 <= len(new) / V2.SR <= 12.0):
                n_trim_fail += 1
                continue
            n_kept += 1
            rid = f"pseudo_{row}_{r['utt']}"
            p = wav_dir / row / f"{rid}.wav"
            sf.write(str(p), new, V2.SR, subtype="PCM_16")
            rows.append({"id": rid, "audio": str(p), "text": r["text"], "language": "chinese", "speaker": row,
                         "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                         "source": "pseudo_base", "p_true": r["p_true"], "lid_ratio": r.get("lid_ratio"),
                         "cer": r["cer"]})
        report[row] = {"rendered": len(mine), "pass_any_sample": len(best), "kept": n_kept,
                       "dropped_after_trim": n_trim_fail,
                       "gate_pass_rate": {g: round(float(np.mean([r["gates"][g] for r in mine])), 3) for g in gates}}
    (out / "pseudo_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = Counter(r["speaker"] for r in rows)
    print(f"[pseudo] kept {len(rows)} rows; voices with 0: {[k for k in names if kept[k] == 0]}; "
          f"with < 10: {[k for k in names if 0 < kept[k] < 10]}")


if __name__ == "__main__":
    main()
