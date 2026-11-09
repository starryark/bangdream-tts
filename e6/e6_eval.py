"""E6 (pod): render + score one condition on the fixed E6 eval plan (E6_PLAN.md §4; e1/e1_eval.py extended).

  python e6/e6_eval.py --cond C0 --ckpt <E1b_ep2 dir>        # v1 as shipped (rows it has)
  python e6/e6_eval.py --cond E6a_ep1 --ckpt <ckpt> [--nv "……っ"|"[nv]"]
  python e6/e6_eval.py --cond floor_on --vectors_from <ckpt>  # Base x-vector-only, Our Notes rows, zh only
  python e6/e6_eval.py --cond real                            # real held-out ja + short: raw and codec resynth

Plan per row (identical in every condition, so texts and seeds are paired):
  ja_heldout  v1 rows: E1's held-out lines; Our Notes rows: 8 test-episode lines of 3-8 s (e6_rows.py)
  ja_novel    the 12 novel sentences (E1)
  ja_short    Our Notes rows: 16 own test-episode lines of 0.7-1.6 s; v1 rows: the shared 16-line set
  zh          the first 16 "shared" probes_zh_v3 (E1)
  en          the first 8 "conversational" probes_en_v3 (E1)
  nv          only with --nv: the given text x 4 per row (each seed base: 8 renders per row)
Two seed bases (6000, 11000). Per (seed base, group) all rows' texts are laid out in a fixed order and
cut into batches of BATCH mixing rows, seeded base * 100000 + 1000 * group + batch.
Scoring: bdscore.score with bundle/probe_e6.npz (identity, ASR CER, accent on zh/en). Added here:
dur_trim (-40 dB, 10 ms frames) of render and real, F0 range (10th-90th percentile, semitones, yin
90-950 Hz on frames within 15% of peak energy) of render and real, for ja_short and Our Notes
ja_heldout; nv_kana (kana in the ASR transcript) for nv renders.
Writes eval/<cond>/{wavs, manifest.jsonl, scored.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
for p in (ROOT / "code", ROOT / "e1", ROOT / "pipeline"):
    sys.path.insert(0, str(p))
os.environ.setdefault("KURISU_WORK", str(ROOT))
LANG = {"ja": "Japanese", "zh": "Chinese", "en": "English"}
SEED_BASES = (6000, 11000)
# E7 (E7_PLAN.md): two more seed bases for the Our Notes rows' ja group (held-out + novel lines) only, so G3's
# per-row ja-mid leak rests on 80 renders instead of 40. The base layout above is unchanged (paired with C0).
EXTRA_SEED_BASES = (16000, 21000)
BATCH = 48
GROUPS = ("ja", "ja_short", "zh", "en", "nv")


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def names() -> list[str]:
    return [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]


def plan(nv_text: str | None = None, extra_rows: set[str] | None = None) -> list[dict]:
    B = ROOT / "bundle"
    held, short = read(B / "eval_rows.jsonl"), read(B / "eval_short.jsonl")
    novel = read(B / "probes_ja_novel.jsonl")
    zh = [r for r in read(ROOT / "code" / "probes_zh_v3.jsonl") if r["category"] == "shared"][:16]
    en = [r for r in read(ROOT / "code" / "probes_en_v3.jsonl") if r["category"] == "conversational"][:8]
    batches = []
    for base in SEED_BASES:
        for gi, group in enumerate(GROUPS):
            sel = []
            for row in names():
                if group == "ja":
                    texts = [{"tid": h["tid"], "lang": "ja", "text": h["text"], "kind": "ja_heldout", "real_audio": h["real_audio"]}
                             for h in held if h["row"] == row] + \
                            [{"tid": n["tid"], "lang": "ja", "text": n["text"], "kind": "ja_novel"} for n in novel]
                elif group == "ja_short":
                    texts = [{"tid": s["tid"], "lang": "ja", "text": s["text"], "kind": "ja_short", "real_audio": s.get("real_audio"),
                              "real_dur": s.get("real_dur")} for s in short if s["row"] == row]
                elif group == "zh":
                    texts = [{"tid": z["pid"], "lang": "zh", "text": z["text"], "kind": "zh", "shared_positions": z.get("shared_positions")}
                             for z in zh]
                elif group == "en":
                    texts = [{"tid": e["pid"], "lang": "en", "text": e["text"], "kind": "en"} for e in en]
                else:
                    texts = [{"tid": f"nv{k}", "lang": "ja", "text": nv_text, "kind": "nv"} for k in range(4)] if nv_text else []
                sel += [{**t, "row": row, "seed_base": base, "pid": f"{row}~{t['tid']}~s{base}"} for t in texts]
            for b in range(0, len(sel), BATCH):
                batches.append({"seed": base * 100000 + gi * 1000 + b // BATCH, "lang": sel[0]["lang"], "group": group,
                                "rows": sel[b:b + BATCH]})
    if extra_rows:
        for base in EXTRA_SEED_BASES:
            sel = []
            for row in names():
                if row not in extra_rows:
                    continue
                texts = [{"tid": h["tid"], "lang": "ja", "text": h["text"], "kind": "ja_heldout", "real_audio": h["real_audio"]}
                         for h in held if h["row"] == row] + \
                        [{"tid": n["tid"], "lang": "ja", "text": n["text"], "kind": "ja_novel"} for n in novel]
                sel += [{**t, "row": row, "seed_base": base, "pid": f"{row}~{t['tid']}~s{base}"} for t in texts]
            for b in range(0, len(sel), BATCH):
                batches.append({"seed": base * 100000 + b // BATCH, "lang": "ja", "group": "ja", "rows": sel[b:b + BATCH]})
    return batches


def locked_vectors(ckpt: Path) -> dict:
    from safetensors import safe_open
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    out = {}
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        emb = f.get_slice("talker.model.codec_embedding.weight")
        for name, idx in cfg["talker_config"]["spk_id"].items():
            out[name] = emb[idx:idx + 1][0].float()
    return out


def acoustics(path: str):
    """(trimmed duration s, F0 range semitones or None)."""
    import numpy as np
    import soundfile as sf
    import librosa
    y, s = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(-1)
    h = int(0.01 * s)
    e = np.array([np.sqrt(np.mean(y[i:i + h] ** 2) + 1e-12) for i in range(0, max(len(y) - h, 1), h)])
    on = np.where(20 * np.log10(e / e.max()) > -40)[0]
    dtrim = (on[-1] - on[0] + 1) * 0.01 if len(on) else len(y) / s
    y16 = librosa.resample(y, orig_sr=s, target_sr=16000)
    f = librosa.yin(y16, fmin=90, fmax=950, sr=16000, frame_length=1024, hop_length=160)
    en = librosa.feature.rms(y=y16, frame_length=1024, hop_length=160)[0][: len(f)]
    f = f[en > en.max() * 0.15]
    if len(f) < 10:
        return round(float(dtrim), 3), None
    l = 12 * np.log2(f)
    return round(float(dtrim), 3), round(float(np.percentile(l, 90) - np.percentile(l, 10)), 3)


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import bdscore
    import bdcommon as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vectors_from", default=None)
    ap.add_argument("--nv", default=None)
    ap.add_argument("--rows", default=None, help="comma list (smoke tests only)")
    ap.add_argument("--extra_on_seeds", action="store_true", help="E7: EXTRA_SEED_BASES for the Our Notes rows' ja group")
    args = ap.parse_args()
    out = ROOT / "eval" / args.cond
    if (out / "summary.json").exists():
        print(f"[eval] {args.cond}: done, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "bundle" / "rows_e6.json").read_text())
    on_rows = set(sel["replaced"] + sel["new"])
    batches = plan(args.nv, on_rows if args.extra_on_seeds else None)
    keep = set(args.rows.split(",")) if args.rows else None
    rows = []
    if args.cond == "real":
        from qwen_tts import Qwen3TTSTokenizer
        seen = {}
        for b in batches:
            for r in b["rows"]:
                if r.get("real_audio") and r["seed_base"] == SEED_BASES[0] and (keep is None or r["row"] in keep):
                    seen[r["pid"]] = r
        real = list(seen.values())
        tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
        for s in range(0, len(real), 16):
            chunk = real[s:s + 16]
            with torch.inference_mode():
                wavs, sr = tok.decode(tok.encode([r["real_audio"] for r in chunk]))
            for r, w in zip(chunk, wavs):
                p = out / f"{r['pid']}~resynth.wav"
                sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                rows.append({**r, "cond": "real_resynth", "path": str(p)})
                rows.append({**r, "cond": "real_raw", "path": r["real_audio"]})
        del tok
        torch.cuda.empty_cache()
    else:
        from qwen_tts import Qwen3TTSModel
        if args.cond.startswith("floor"):
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
            vecs = {k: v for k, v in locked_vectors(Path(args.vectors_from)).items() if k in on_rows}
            model = Qwen3TTSModel.from_pretrained(os.environ.get("BD_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"), device_map="cuda:0",
                                                  dtype=torch.bfloat16, attn_implementation="sdpa")
            items = {k: VoiceClonePromptItem(ref_code=None, ref_spk_embedding=v.to("cuda:0").to(torch.bfloat16),
                                             x_vector_only_mode=True, icl_mode=False) for k, v in vecs.items()}
            batches = [{**b, "rows": [r for r in b["rows"] if r["row"] in vecs]} for b in batches if b["group"] == "zh"]
        else:
            model = Qwen3TTSModel.from_pretrained(str(args.ckpt), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
            have = set(json.loads((Path(args.ckpt) / "config.json").read_text())["talker_config"]["spk_id"])
            batches = [{**b, "rows": [r for r in b["rows"] if r["row"] in have]} for b in batches]
        if keep:
            batches = [{**b, "rows": [r for r in b["rows"] if r["row"] in keep]} for b in batches]
        batches = [b for b in batches if b["rows"]]
        for k, b in enumerate(batches):
            torch.manual_seed(b["seed"])
            text = [r["text"] for r in b["rows"]]
            lang = [LANG[b["lang"]]] * len(text)
            if args.cond.startswith("floor"):
                wavs, sr = model.generate_voice_clone(text=text, language=lang, voice_clone_prompt=[items[r["row"]] for r in b["rows"]],
                                                      max_new_tokens=400, non_streaming_mode=True)
            else:
                wavs, sr = model.generate_custom_voice(text=text, speaker=[r["row"] for r in b["rows"]], language=lang,
                                                       max_new_tokens=400, non_streaming_mode=True)
            assert len(wavs) == len(b["rows"]), (len(wavs), len(b["rows"]))
            for r, w in zip(b["rows"], wavs):
                p = out / f"{r['pid']}.wav"
                sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                rows.append({**r, "cond": args.cond, "path": str(p)})
            if k % 10 == 0:
                print(f"[eval] {args.cond}: batch {k + 1}/{len(batches)}", flush=True)
        del model
        torch.cuda.empty_cache()
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # E9: an Our-Notes-era id X_ournotes is scored against probe class X (probeE6 has no _ournotes classes);
    # the id is restored afterwards, so scored.jsonl keeps the voice id and the probe class appears as pred.
    orig_rows = [r["row"] for r in rows]
    for r in rows:
        if r["row"].endswith("_ournotes"):
            r["row"] = r["row"][: -len("_ournotes")]
    bdscore.score(rows, ROOT / "bundle" / "probe_e6.npz", accent=args.cond != "real")
    for r, o in zip(rows, orig_rows):
        r["row"] = o
    for r in rows:
        if r["kind"] == "nv":
            r["nv_kana"] = len(C.reading(r.get("asr", "")))
        if r["kind"] in ("ja_short", "nv") or (r["kind"] == "ja_heldout" and r["row"] in on_rows):
            r["dur_trim"], r["f0r"] = acoustics(r["path"])
            if r.get("real_audio") and r["cond"] != "real_raw":
                r["real_dur_trim"], r["f0r_real"] = acoustics(r["real_audio"])
    (out / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['cond']}|{r['kind']}", []).append(r)
    summary = {g: bdscore.aggregate(v) for g, v in sorted(groups.items())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[eval] {args.cond}: {len(rows)} rows " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
