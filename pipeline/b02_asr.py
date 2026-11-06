"""E0 step 2: two independent transcripts per clip, compared with the filename text in kana.

  python -I b02_asr.py

Same two systems and settings as kurisu_v2/pod/p02_asr.py: Qwen3-ASR-1.7B (forced
Japanese, a fixed proper-noun context identical for every clip) and litagin/anime-whisper
(no prompt). Qwen3-ASR echoed its context instead of transcribing on 163 Kurisu clips, so
a hypothesis that contains three or more context names is discarded as an echo.

Writes data/asr.jsonl {id, qwen, anime, ref_kana, cer_qwen, cer_anime, bar, verdict}:
  ok       either system within the Kurisu cer_bar (dur/kana-dependent, 0.25-0.5)
  review   both above the bar (likely a transcript problem, or both ASRs failed)
Plus data/asr_summary.json per row.
"""
from __future__ import annotations

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402

NAMES = ["戸山香澄", "花園たえ", "牛込りみ", "山吹沙綾", "市ヶ谷有咲", "美竹蘭", "青葉モカ", "上原ひまり",
         "宇田川巴", "羽沢つぐみ", "丸山彩", "氷川日菜", "白鷺千聖", "大和麻弥", "若宮イヴ", "湊友希那",
         "氷川紗夜", "今井リサ", "宇田川あこ", "白金燐子", "弦巻こころ", "瀬田薫", "北沢はぐみ", "松原花音",
         "奥沢美咲", "倉田ましろ", "桐ヶ谷透子", "広町七深", "二葉つくし", "八潮瑠唯", "和奏レイ", "朝日六花",
         "佐藤ますき", "鳰原令王那", "珠手ちゆ", "高松燈", "千早愛音", "要楽奈", "長崎そよ", "椎名立希",
         "三角初華", "若葉睦", "八幡海鈴", "祐天寺にゃむ", "豊川祥子"]
TERMS = ["Poppin'Party", "Afterglow", "Pastel*Palettes", "Roselia", "ハロー、ハッピーワールド！",
         "Morfonica", "RAISE A SUILEN", "MyGO!!!!!", "Ave Mujica", "CiRCLE", "RiNG", "花咲川", "羽丘",
         "月ノ森", "ポピパ", "パスパレ", "ハロハピ", "モニカ", "まりな", "ミッシェル"]
CONTEXT = "、".join(NAMES + TERMS)
ECHO_NAMES = NAMES + TERMS


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def save(done: dict, order: list[str], path) -> None:
    C.write_jsonl(path, [done[i] for i in order if i in done])


def main() -> None:
    import soundfile as sf
    import torch

    rows = C.read_jsonl(C.DATA / "manifest.jsonl")
    rows.sort(key=lambda r: -r["dur"])           # bounded padding per batch
    order = [r["id"] for r in rows]
    out = C.DATA / "asr.jsonl"
    done = {r["id"]: r for r in C.read_jsonl(out)} if out.exists() else {}

    todo = [r for r in rows if "qwen" not in done.get(r["id"], {})]
    if todo:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", max_inference_batch_size=8, max_new_tokens=256)
        for i, chunk in enumerate(batches(todo, 8)):
            res = model.transcribe([r["audio"] for r in chunk], language="Japanese",
                                   context=[CONTEXT] * len(chunk))
            for r, h in zip(chunk, res):
                text = h.text.strip()
                echo = sum(n in text for n in ECHO_NAMES) >= 3
                done.setdefault(r["id"], {"id": r["id"]})["qwen"] = None if echo else text
            if i % 100 == 0:
                print(f"[qwen-asr] {min((i + 1) * 8, len(todo))}/{len(todo)}", flush=True)
            if i % 200 == 199:
                save(done, order, out)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        save(done, order, out)

    todo = [r for r in rows if "anime" not in done[r["id"]]]
    if todo:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="litagin/anime-whisper",
                       torch_dtype=torch.float16, device="cuda:0", chunk_length_s=30.0)
        for i, chunk in enumerate(batches(todo, 8)):
            audio = []
            for r in chunk:
                wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
                audio.append({"raw": wav, "sampling_rate": sr})
            res = asr(audio, batch_size=8,
                      generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0,
                                       "repetition_penalty": 1.0})
            for r, h in zip(chunk, res):
                done[r["id"]]["anime"] = h["text"].strip()
            if i % 100 == 0:
                print(f"[anime-whisper] {min((i + 1) * 8, len(todo))}/{len(todo)}", flush=True)
            if i % 200 == 199:
                save(done, order, out)
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        save(done, order, out)

    by_row: dict[str, dict] = {}
    for r in rows:
        d = done[r["id"]]
        ref = C.reading(r["text"])
        cq = C.cer(ref, C.reading(d["qwen"])) if d.get("qwen") is not None else None
        ca = C.cer(ref, C.reading(d.get("anime") or ""))
        bar = C.cer_bar(len(ref), r["dur_src"])
        best = min(x for x in (cq, ca) if x is not None)
        d.update(ref_kana=len(ref), cer_qwen=None if cq is None else round(cq, 3),
                 cer_anime=round(ca, 3), bar=bar, verdict="ok" if best <= bar else "review")
        s = by_row.setdefault(r["row"], {"n": 0, "review": 0, "echo": 0, "cer_best_sum": 0.0})
        s["n"] += 1
        s["review"] += d["verdict"] == "review"
        s["echo"] += d.get("qwen") is None
        s["cer_best_sum"] += best
    save(done, order, out)
    summary = {k: {"n": v["n"], "review": v["review"], "review_pct": round(100 * v["review"] / v["n"], 1),
                   "qwen_echo": v["echo"], "cer_best_mean": round(v["cer_best_sum"] / v["n"], 3)}
               for k, v in sorted(by_row.items())}
    (C.DATA / "asr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    tot = sum(v["review"] for v in summary.values())
    print(f"[asr] {len(rows)} clips, review {tot} ({100 * tot / len(rows):.1f}%) -> {out}")


if __name__ == "__main__":
    main()
