"""E2 (pod): in-voice Mandarin + English pseudo rows from two sources, with per-voice gates.

  python e2_pseudo.py

E2_PLAN.md "Data". Stages, each skipped if its output exists (so a crash resumes):
  1 render  -> pseudo/cands_raw.jsonl     Base (x-vector, the trainer's locked vectors) and E1b
                                         (generate_custom_voice); batches of 48 mix voices
  2 score   -> pseudo/candidates.jsonl    bdscore.score (identical to the E1/E2 eval)
  3 select  -> pseudo/pseudo_rows.jsonl + pseudo/report.json
Gates: identity pred == row and p_true >= 0.5 * real_p_true[row] (bundle/e2_baselines.json);
accent zh P(zh) >= 0.9 and lid <= -19, en P(en) >= 0.9 and lid <= -12; CER/WER <= 0.05;
not a runaway and 2-12 s after the Kurisu v3 audio policy. Per (voice, language, text) the
passing candidate with the highest p_true / real_p_true is kept; caps zh 25, en 12 per voice.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
for p in (ROOT / "code", ROOT / "e1", ROOT / "e2", ROOT / "pipeline"):
    sys.path.insert(0, str(p))
os.environ.setdefault("KURISU_WORK", str(ROOT))
BASE = os.environ.get("BD_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
E1B = Path(os.environ.get("BD_E1B", ROOT / "models" / "E1b_ep2"))
# smoke tests only: render a subset of voices / texts (text assignment still uses all 49)
ONLY = set(filter(None, os.environ.get("BD_E2_ONLY", "").split(",")))
MAXT = int(os.environ.get("BD_E2_MAXTEXTS", "0"))
BATCH = 48
ZH_PER_VOICE_E1, ZH_EXTRA_PROBLEM, EN_PER_VOICE = 60, 100, 30
CAP = {"zh": 25, "en": 12}
OUT = ROOT / "pseudo"


def jl_read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def jl_write(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def en_texts(n: int, seed: int, exclude: set[str]) -> list[tuple[str, str]]:
    """LibriTTS-R test.clean text_normalized: 6-20 words, no digits, p05's text filter."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    import p05_replay as RP
    fs = HfFileSystem(token=False)
    shards = sorted(fs.glob(f"datasets/{RP.EN_REPO}@{RP.EN_REV}/data/test.clean/*.parquet"))
    seen, pool = set(), []
    for sh in shards:
        with fs.open(sh, "rb") as f:
            tab = pq.ParquetFile(f).read(columns=["text_normalized", "id"]).to_pylist()
        for r in tab:
            t = RP.en_norm(r["text_normalized"])
            k = t.lower()
            if (6 <= len(t.split()) <= 20 and not re.search(r"\d", t) and RP.en_text_problem(t) is None
                    and k not in seen and k not in exclude):
                seen.add(k)
                pool.append((str(r["id"]), t))
    pool.sort()
    random.Random(seed).shuffle(pool)
    return pool[:n]


def render(names: list[str], problem: set[str]) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    import p11_pseudo_zh as P11
    import train_sft
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    n_extra = ZH_EXTRA_PROBLEM * len(problem)
    zh_pool = P11.texts(ZH_PER_VOICE_E1 * len(names) + n_extra, 31)     # prefix = E1's texts (same seed)
    extra_per = min(ZH_EXTRA_PROBLEM, max(0, len(zh_pool) - ZH_PER_VOICE_E1 * len(names)) // max(1, len(problem)))
    print(f"[e2_pseudo] zh pool {len(zh_pool)}; extra texts per problem voice {extra_per}", flush=True)
    probes_en = {json.loads(l)["text"].lower() for l in (ROOT / "code" / "probes_en_v3.jsonl").read_text(encoding="utf-8").splitlines() if l}
    en_pool = en_texts(EN_PER_VOICE * len(names), 17, probes_en)
    if len(en_pool) < EN_PER_VOICE * len(names):
        raise SystemExit(f"only {len(en_pool)} English texts")
    probe_zh = {json.loads(l)["text"].rstrip("。？") for l in (ROOT / "code" / "probes_zh_v3.jsonl").read_text(encoding="utf-8").splitlines() if l}
    assert not ({t.rstrip("。？") for _, t in zh_pool} & probe_zh), "pseudo zh texts overlap the eval probes"

    zh_texts, en_txt = {}, {}
    pi = 0
    for vi, row in enumerate(names):
        zh_texts[row] = list(zh_pool[vi * ZH_PER_VOICE_E1:(vi + 1) * ZH_PER_VOICE_E1])
        if row in problem:
            s = ZH_PER_VOICE_E1 * len(names) + pi * extra_per
            zh_texts[row] += zh_pool[s:s + extra_per]
            pi += 1
        en_txt[row] = en_pool[vi * EN_PER_VOICE:(vi + 1) * EN_PER_VOICE]
    jobs = {"base": [], "e1b": []}
    for row in names:
        if row not in problem:
            for s in (0, 1):
                jobs["base"] += [(row, "zh", u, t, s) for u, t in zh_texts[row]]
            jobs["e1b"] += [(row, "zh", u, t, 0) for u, t in zh_texts[row]]
        else:
            for s in (0, 1):
                jobs["e1b"] += [(row, "zh", u, t, s) for u, t in zh_texts[row]]
        for src in ("base", "e1b"):
            jobs[src] += [(row, "en", u, t, 0) for u, t in en_txt[row]]
    if ONLY or MAXT:
        keep_t = {}
        for src in jobs:
            out = []
            for j in jobs[src]:
                if ONLY and j[0] not in ONLY:
                    continue
                k = (src, j[0], j[1], j[4])
                keep_t[k] = keep_t.get(k, 0) + 1
                if not MAXT or keep_t[k] <= MAXT:
                    out.append(j)
            jobs[src] = out
    print(f"[e2_pseudo] jobs: base {len(jobs['base'])}, e1b {len(jobs['e1b'])}", flush=True)

    cands = []
    raw = OUT / "raw"
    for src_i, src in enumerate(("base", "e1b")):
        if src == "base":
            model = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
            prompts = {}
            for row in names:
                vec = train_sft.speaker_embedding(model.model, row, train_sft.ref_paths(ROOT / "bundle" / "refs" / f"ref_{row}.txt"),
                                                  "cuda:0", torch.bfloat16)[0]
                prompts[row] = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec, x_vector_only_mode=True, icl_mode=False)
        else:
            model = Qwen3TTSModel.from_pretrained(str(E1B), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        for lang in ("zh", "en"):
            js = [j for j in jobs[src] if j[1] == lang]
            for b in range(0, len(js), BATCH):
                chunk = js[b:b + BATCH]
                torch.manual_seed(900000 + 100000 * src_i + (0 if lang == "zh" else 50000) + b // BATCH)
                text = [t for _, _, _, t, _ in chunk]
                language = ["Chinese" if lang == "zh" else "English"] * len(chunk)
                if src == "base":
                    wavs, sr = model.generate_voice_clone(text=text, language=language,
                                                          voice_clone_prompt=[prompts[r] for r, *_ in chunk],
                                                          max_new_tokens=400, non_streaming_mode=True)
                else:
                    wavs, sr = model.generate_custom_voice(text=text, speaker=[r for r, *_ in chunk], language=language,
                                                           max_new_tokens=400, non_streaming_mode=True)
                assert len(wavs) == len(chunk)
                for (row, lg, u, t, s), w in zip(chunk, wavs):
                    p = raw / src / lg / row / f"{u}_s{s}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                    cands.append({"row": row, "lang": lg, "utt": u, "text": t, "source": src, "sample": s, "path": str(p)})
                if (b // BATCH) % 20 == 0:
                    print(f"[e2_pseudo] {src} {lang}: {b + len(chunk)}/{len(js)}", flush=True)
        del model
        torch.cuda.empty_cache()
    return cands


def main() -> None:
    import numpy as np
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "pseudo_rows.jsonl").exists() and (OUT / "report.json").exists():
        print("[e2_pseudo] done, skipped")
        return
    names = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    bl = json.loads((ROOT / "bundle" / "e2_baselines.json").read_text(encoding="utf-8"))
    base_p, problem = bl["real_resynth_p_true"], set(bl["problem_voices"])

    if not (OUT / "cands_raw.jsonl").exists():
        jl_write(OUT / "cands_raw.jsonl", render(names, problem))
    if not (OUT / "candidates.jsonl").exists():
        import bdscore
        cands = jl_read(OUT / "cands_raw.jsonl")
        bdscore.score(cands, ROOT / "bundle" / "probe_codec.npz")
        jl_write(OUT / "candidates.jsonl", cands)
    cands = jl_read(OUT / "candidates.jsonl")

    lid_max = {"zh": -19.0, "en": -12.0}
    gates = {"identity": lambda r: r["pred"] == r["row"] and r["p_true"] >= 0.5 * base_p[r["row"]],
             "accent": lambda r: r.get("p_target", 0) >= 0.9 and r.get("lid_ratio", 0) <= lid_max[r["lang"]],
             "text": lambda r: r.get("cer", 1) <= 0.05,
             "length": lambda r: 1.5 <= r["dur"] <= 13.0 and not r.get("runaway")}
    for r in cands:
        r["gates"] = {g: bool(f(r)) for g, f in gates.items()}
        r["pass"] = all(r["gates"].values())
        r["ratio"] = r["p_true"] / base_p[r["row"]]

    import pyloudnorm
    import soundfile as sf
    import p03_build as V2
    meter = pyloudnorm.Meter(V2.SR)
    rng = random.Random(5)
    rows, report = [], {}
    for row in [n for n in names if not ONLY or n in ONLY]:
        report[row] = {"problem": row in problem}
        for lang in ("zh", "en"):
            mine = [r for r in cands if r["row"] == row and r["lang"] == lang]
            best = {}
            for r in mine:
                if r["pass"] and (r["utt"] not in best or r["ratio"] > best[r["utt"]]["ratio"]):
                    best[r["utt"]] = r
            kept, trim_fail = 0, 0
            for r in sorted(best.values(), key=lambda r: -r["ratio"]):
                if kept >= CAP[lang]:
                    break
                wav, sr = sf.read(r["path"], dtype="float32", always_2d=False)
                if sr != V2.SR:
                    raise SystemExit(f"{r['path']}: {sr} Hz")
                new, _ = V2.normalise_audio(wav, meter)
                if new is None or not (2.0 <= len(new) / V2.SR <= 12.0):
                    trim_fail += 1
                    continue
                kept += 1
                rid = f"pseudo_{lang}_{row}_{r['utt']}"
                p = OUT / "wavs" / lang / row / f"{rid}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), new, V2.SR, subtype="PCM_16")
                rows.append({"id": rid, "audio": str(p), "text": r["text"], "speaker": row,
                             "language": "chinese" if lang == "zh" else "english",
                             "split": "val" if rng.random() < 0.03 else "train", "dur": round(len(new) / V2.SR, 3),
                             "source": f"pseudo_{r['source']}", "p_true": r["p_true"], "ratio": round(r["ratio"], 3),
                             "lid_ratio": r.get("lid_ratio"), "cer": r.get("cer")})
            by_src = {s: [r for r in mine if r["source"] == s] for s in ("base", "e1b")}
            report[row][lang] = {
                "rendered": len(mine), "texts_passing": len(best), "kept": kept, "dropped_after_trim": trim_fail,
                "kept_by_source": dict(Counter(r["source"] for r in rows if r["speaker"] == row and
                                               r["language"] == ("chinese" if lang == "zh" else "english"))),
                "pass_rate_by_source": {s: round(float(np.mean([r["pass"] for r in v])), 3) for s, v in by_src.items() if v},
                "gate_pass_rate": {g: round(float(np.mean([r["gates"][g] for r in mine])), 3) for g in gates} if mine else {}}
    jl_write(OUT / "pseudo_rows.jsonl", rows)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for lang in ("zh", "en"):
        k = {row: report[row][lang]["kept"] for row in report}
        print(f"[e2_pseudo] {lang}: kept {sum(k.values())}; problem voices "
              f"{ {r: k[r] for r in sorted(problem) if r in k} }; voices with < 10: {[r for r in k if k[r] < 10]}", flush=True)


if __name__ == "__main__":
    main()
