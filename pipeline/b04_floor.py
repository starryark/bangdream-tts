"""E0 step 4: the per-voice floor = Base zero-shot from the trainer's own locked vector.

  python -I b04_floor.py [--rows a,b,...] [--batch 8]

Condition X of kurisu_v2/pod/p10_e0.py, once per row: Qwen3-TTS-12Hz-1.7B-Base,
x-vector-only prompt (vectors/locked.pt, i.e. train_sft.speaker_embedding of refs/ref_<row>.txt),
explicit language id, non-streaming, upstream sampling defaults, max_new_tokens 400.
Texts per row: that row's held-out Japanese lines (real audio exists for them), the first 16
"shared" rows of probes_zh_v3.jsonl (shared-char SER needs them) and the first 8
"conversational" rows of probes_en_v3.jsonl. One sampling seed; batch b of a language is
seeded torch.manual_seed(SEED * 100003 + b), identical for every row, so a text is rendered
with the same seed in every voice.

Writes e0/floor/<row>/<tid>.wav and e0/floor/<row>/manifest.jsonl (written last, so a
crashed row is regenerated, not half-scored).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402

SEED = 1
LANG = {"ja": "Japanese", "zh": "Chinese", "en": "English"}
POD = C.ROOT / "kurisu_v2" / "pod"


def texts_for(row: str, man: dict, split: list[dict]) -> list[dict]:
    ja = [{"tid": f"ja_{s['id'].rsplit('_', 1)[1]}", "lang": "ja", "text": man[s["id"]]["text"],
           "real_audio": man[s["id"]]["audio"], "real_id": s["id"]}
          for s in split if s["row"] == row and s["split"] == "test"]
    zh = [r for r in C.read_jsonl(POD / "probes_zh_v3.jsonl") if r["category"] == "shared"][:16]
    en = [r for r in C.read_jsonl(POD / "probes_en_v3.jsonl") if r["category"] == "conversational"][:8]
    return ja + [{**r, "tid": r["pid"]} for r in zh] + [{**r, "tid": r["pid"]} for r in en]


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    man = {r["id"]: r for r in C.read_jsonl(C.DATA / "manifest.jsonl")}
    split = C.read_jsonl(C.DATA / "split.jsonl")
    locked = torch.load(C.OUT / "vectors" / "locked.pt", map_location="cpu")
    names = locked["names"]
    todo = args.rows.split(",") if args.rows else names
    base = Qwen3TTSModel.from_pretrained(str(C.QWEN_BASE), device_map="cuda:0", dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    root = C.OUT / "e0" / "floor"
    for row in todo:
        out = root / row
        if (out / "manifest.jsonl").exists():
            print(f"[floor] {row}: exists, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        vec = locked["table"][names.index(row)].to("cuda:0").to(torch.bfloat16)
        item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec, x_vector_only_mode=True, icl_mode=False)
        rows = texts_for(row, man, split)
        t0, man_out, secs = time.time(), [], 0.0
        for lang in ("ja", "zh", "en"):
            sel = [r for r in rows if r["lang"] == lang]
            for b in range(0, len(sel), args.batch):
                chunk = sel[b:b + args.batch]
                torch.manual_seed(SEED * 100003 + b)
                wavs, sr = base.generate_voice_clone(
                    text=[r["text"] for r in chunk], language=[LANG[lang]] * len(chunk),
                    voice_clone_prompt=[item] * len(chunk), max_new_tokens=400, non_streaming_mode=True)
                for r, w in zip(chunk, wavs):
                    f = f"{r['tid']}.wav"
                    w = np.asarray(w, dtype=np.float32)
                    sf.write(str(out / f), w, sr)
                    secs += len(w) / sr
                    man_out.append({**r, "row": row, "file": f, "pid": f"{row}~{r['tid']}", "seed": SEED})
        C.write_jsonl(out / "manifest.jsonl", man_out)
        el = time.time() - t0
        print(f"[floor] {row}: {len(man_out)} clips, {secs:.0f} s audio in {el:.0f} s (RTF {el / max(secs, 1):.2f})",
              flush=True)


if __name__ == "__main__":
    main()
