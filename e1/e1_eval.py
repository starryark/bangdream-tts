"""E1 (pod): render + score one condition on the fixed eval plan.

  python e1_eval.py --cond floor --vectors_from <ckpt dir>   # Base x-vector-only, the ckpt's own rows
  python e1_eval.py --cond <name> --ckpt <ckpt dir>           # fine-tuned, generate_custom_voice
  python e1_eval.py --cond real                               # real held-out ja: raw (CER) + codec
                                                              # resynth (probe baseline)

Plan (identical for every condition, so texts are paired across conditions):
  per voice: its held-out Japanese lines (bundle/eval_rows.jsonl), the 12 novel Japanese
  sentences, the first 16 "shared" zh probes and the first 8 "conversational" en probes
  (as in E0); seed bases 6000 and 11000. Per (seed base, language) all voices' texts are
  laid out in a fixed order (voice, then text) and cut into batches of BATCH that mix voices
  (one speaker per item), seeded torch.manual_seed(base * 100000 + 1000 * language + batch):
  unique per batch, and the same batch composition and seed in every condition.
Writes eval/<cond>/{wav files, manifest.jsonl, scored.jsonl, summary.json}.
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
BATCH = 48


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def plan() -> list[dict]:
    names = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    held = read(ROOT / "bundle" / "eval_rows.jsonl")
    novel = read(ROOT / "bundle" / "probes_ja_novel.jsonl")
    zh = [r for r in read(ROOT / "code" / "probes_zh_v3.jsonl") if r["category"] == "shared"][:16]
    en = [r for r in read(ROOT / "code" / "probes_en_v3.jsonl") if r["category"] == "conversational"][:8]
    batches = []
    for base in SEED_BASES:
        for li, lang in enumerate(("ja", "zh", "en")):
            sel = []
            for row in names:
                texts = ([{"tid": h["tid"], "lang": "ja", "text": h["text"], "kind": "ja_heldout",
                           "real_audio": h["real_audio"]} for h in held if h["row"] == row] +
                         [{"tid": n["tid"], "lang": "ja", "text": n["text"], "kind": "ja_novel"} for n in novel] +
                         [{"tid": z["pid"], "lang": "zh", "text": z["text"], "kind": "zh",
                           "shared_positions": z.get("shared_positions")} for z in zh] +
                         [{"tid": e["pid"], "lang": "en", "text": e["text"], "kind": "en"} for e in en])
                sel += [{**t, "row": row, "seed_base": base, "pid": f"{row}~{t['tid']}~s{base}"}
                        for t in texts if t["lang"] == lang]
            for b in range(0, len(sel), BATCH):
                batches.append({"seed": base * 100000 + li * 1000 + b // BATCH, "lang": lang,
                                "rows": sel[b:b + BATCH]})
    return batches


def locked_vectors(ckpt: Path) -> dict:
    """The speaker rows the fine-tune was trained with (codec_embedding rows from spk_id)."""
    from safetensors import safe_open
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    spk = cfg["talker_config"]["spk_id"]
    out = {}
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        emb = f.get_slice("talker.model.codec_embedding.weight")
        for name, idx in spk.items():
            out[name] = emb[idx:idx + 1][0].float()
    return out


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import bdscore

    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vectors_from", default=None)
    ap.add_argument("--rows", default=None, help="comma list of voices (smoke tests only)")
    args = ap.parse_args()
    out = ROOT / "eval" / args.cond
    if (out / "summary.json").exists():
        print(f"[eval] {args.cond}: done, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.cond == "real":
        from qwen_tts import Qwen3TTSTokenizer
        seen = {}
        for b in plan():
            for r in b["rows"]:
                if r["kind"] == "ja_heldout" and r["seed_base"] == SEED_BASES[0]:
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
        if args.cond == "floor":
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
            vecs = locked_vectors(Path(args.vectors_from))
            model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                                  dtype=torch.bfloat16, attn_implementation="sdpa")
        else:
            model = Qwen3TTSModel.from_pretrained(str(args.ckpt), device_map="cuda:0", dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
        batches = plan()
        if args.rows:   # smoke tests: same seeds, fewer items per batch
            keep = set(args.rows.split(","))
            batches = [{**b, "rows": [r for r in b["rows"] if r["row"] in keep]} for b in batches]
            batches = [b for b in batches if b["rows"]]
        if args.cond == "floor":
            items = {k: VoiceClonePromptItem(ref_code=None, ref_spk_embedding=v.to("cuda:0").to(torch.bfloat16),
                                             x_vector_only_mode=True, icl_mode=False) for k, v in vecs.items()}
        for k, b in enumerate(batches):
            torch.manual_seed(b["seed"])
            text = [r["text"] for r in b["rows"]]
            lang = [LANG[b["lang"]]] * len(text)
            if args.cond == "floor":
                wavs, sr = model.generate_voice_clone(text=text, language=lang,
                                                      voice_clone_prompt=[items[r["row"]] for r in b["rows"]],
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
    bdscore.score(rows, ROOT / "bundle" / "probe_codec.npz", accent=args.cond != "real")
    (out / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['cond']}|{r['lang']}", []).append(r)
    summary = {g: bdscore.aggregate(v) for g, v in sorted(groups.items())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[eval] {args.cond}: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
