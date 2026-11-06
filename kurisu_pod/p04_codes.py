"""Codec-tokenise the kept clips and pick the speaker-reference set.

Writes:
  qwen/{train,val,test}_with_codes.jsonl   {id, audio, text, audio_codes}
  build/ref_list.txt                        clips averaged into the speaker vector
"""
from __future__ import annotations

import json
import os
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
SPLIT = {"train": "train", "validation": "val", "test": "test"}
BATCH = 48
N_REF = 64


def main() -> None:
    from qwen_tts import Qwen3TTSTokenizer

    kept = [json.loads(l) for l in (WORK / "build" / "kept.jsonl").read_text(encoding="utf-8").splitlines() if l]

    # Speaker reference: the cleanest, most typical studio takes of ordinary Kurisu,
    # from training rows only, spread over the three main games.
    pool = [k for k in kept if k["split"] == "train" and k["medium"] == "game"
            and k["quality_tier"] == "A" and 3.0 <= k["dur"] <= 10.0
            and "[" not in k["text"] and (k.get("speaker_margin") or -1) > 0]
    pool.sort(key=lambda k: -(k.get("speaker_margin") or 0))
    refs = []
    for series in ("SG", "SGRE", "PHENO"):
        refs += [k for k in pool if k["series"] == series][: N_REF // 3]
    (WORK / "build" / "ref_list.txt").write_text("".join(k["audio"] + "\n" for k in refs), encoding="utf-8")
    print(f"[ref] {len(refs)} reference clips, {sum(k['dur'] for k in refs):.0f}s")
    # The single clip the dataset card pins (highest speaker margin of 1,860 candidates),
    # used by the zero-shot clone baseline.
    pinned = next((k for k in kept if k["id"] == "game_ja_SGRE_kurisu_01686"), refs[0])
    (WORK / "build" / "ref.json").write_text(json.dumps(
        {"id": pinned["id"], "audio": pinned["audio"], "text": pinned["text"]}, ensure_ascii=False),
        encoding="utf-8")

    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    out_dir = WORK / "qwen"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_split, name in SPLIT.items():
        rows = sorted([k for k in kept if k["split"] == src_split], key=lambda k: k["dur"])
        lines = []
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            enc = tok.encode([r["audio"] for r in chunk])
            for r, codes in zip(chunk, enc.audio_codes):
                codes = codes.cpu().tolist()
                expect = r["dur"] * 12.5
                if abs(len(codes) - expect) > 3:
                    raise SystemExit(f"{r['id']}: {len(codes)} frames for {r['dur']}s")
                lines.append(json.dumps({"id": r["id"], "audio": r["audio"], "text": r["text"],
                                         "audio_codes": codes}, ensure_ascii=False))
        (out_dir / f"{name}_with_codes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[codes] {name}: {len(lines)} rows")


if __name__ == "__main__":
    main()
