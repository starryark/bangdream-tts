"""E6 step 2 (local, TTS venv): two independent transcripts per Our Notes clip, as v1's b02_asr.py.

  python -I -X utf8 e6_asr.py

Same systems and settings as pipeline/b02_asr.py: Qwen3-ASR-1.7B (forced Japanese, a fixed
proper-noun context, now with the Our Notes cast and terms added) and litagin/anime-whisper (no
prompt). A Qwen3-ASR hypothesis containing three or more context names is discarded as an echo.
Runs on every kind == "line" clip of data_on/manifest.jsonl (train and test). Resumable. anime-whisper
runs only where Qwen3-ASR misses the bar (or echoed): the verdict is unchanged from b02's rule.

Writes data_on/asr.jsonl {id, qwen, anime, ref_kana, cer_qwen, cer_anime, bar, verdict}:
  ok      either system within cer_bar (v1's bar, on the source duration)
  review  both above the bar
The 1-2 kana exception (E6_PLAN §3.3) is applied later, in e6_rows.py, not here.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402
import b02_asr as B   # noqa: E402  (NAMES, TERMS, batches, save)

ON_NAMES = ["仲町あられ", "宮永ののか", "峰月律", "藤都子", "千石ユノ", "汐見蛍", "伊沢なつめ", "琴平凪", "浜崎まほろ",
            "和泉朋花", "須賀蕾叶", "馬橋心玖", "矢倉蓬咲", "梅里ちえり", "四宮寧月", "沢海奏多", "モーティス"]
ON_TERMS = ["夢限大みゅーたいぷ", "millsage", "一家Dumb Rock!", "CRYCHIC"]
CONTEXT = "、".join(B.NAMES + ON_NAMES + B.TERMS + ON_TERMS)
ECHO_NAMES = B.NAMES + ON_NAMES + B.TERMS + ON_TERMS
DATA = BD / "data_on"


def main() -> None:
    import soundfile as sf
    import torch
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [r for r in C.read_jsonl(DATA / "manifest.jsonl") if r["kind"] == "line"]
    rows.sort(key=lambda r: -r["dur"])
    order = [r["id"] for r in rows]
    out = DATA / "asr.jsonl"
    done = {r["id"]: r for r in C.read_jsonl(out)} if out.exists() else {}

    todo = [r for r in rows if "qwen" not in done.get(r["id"], {})]
    if todo:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                              attn_implementation="sdpa", max_inference_batch_size=8, max_new_tokens=256)
        for i, chunk in enumerate(B.batches(todo, 8)):
            res = model.transcribe([r["audio"] for r in chunk], language="Japanese", context=[CONTEXT] * len(chunk))
            for r, h in zip(chunk, res):
                text = h.text.strip()
                echo = sum(n in text for n in ECHO_NAMES) >= 3
                done.setdefault(r["id"], {"id": r["id"]})["qwen"] = None if echo else text
            if i % 200 == 0:
                print(f"[qwen-asr] {min((i + 1) * 8, len(todo))}/{len(todo)}", flush=True)
            if i % 200 == 199:
                B.save(done, order, out)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        B.save(done, order, out)

    # The verdict is "either system within the bar", so the second system only decides clips the first
    # fails (or echoed): identical verdicts to b02, ~6% of the anime-whisper work.
    def qwen_ok(r):
        q = done[r["id"]].get("qwen")
        return q is not None and C.cer(C.reading(r["text"]), C.reading(q)) <= C.cer_bar(len(C.reading(r["text"])), r["dur_src"])
    todo = [r for r in rows if "anime" not in done[r["id"]] and not qwen_ok(r)]
    if todo:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="litagin/anime-whisper", torch_dtype=torch.float16,
                       device="cuda:0", chunk_length_s=30.0)
        for i, chunk in enumerate(B.batches(todo, 8)):
            audio = []
            for r in chunk:
                wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
                audio.append({"raw": wav, "sampling_rate": sr})
            res = asr(audio, batch_size=8, generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0,
                                                            "repetition_penalty": 1.0})
            for r, h in zip(chunk, res):
                done[r["id"]]["anime"] = h["text"].strip()
            if i % 200 == 0:
                print(f"[anime-whisper] {min((i + 1) * 8, len(todo))}/{len(todo)}", flush=True)
            if i % 200 == 199:
                B.save(done, order, out)
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        B.save(done, order, out)

    n_ok = 0
    for r in rows:
        d = done[r["id"]]
        ref = C.reading(r["text"])
        cq = C.cer(ref, C.reading(d["qwen"])) if d.get("qwen") is not None else None
        ca = C.cer(ref, C.reading(d["anime"])) if "anime" in d else None
        bar = C.cer_bar(len(ref), r["dur_src"])
        best = min(x for x in (cq, ca) if x is not None) if (cq is not None or ca is not None) else 1.0
        d.update(ref_kana=len(ref), cer_qwen=None if cq is None else round(cq, 3), cer_anime=None if ca is None else round(ca, 3),
                 bar=bar, verdict="ok" if best <= bar else "review")
        n_ok += d["verdict"] == "ok"
    B.save(done, order, out)
    print(f"[e6_asr] {len(rows)} line clips, ok {n_ok} ({100 * n_ok / len(rows):.1f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
