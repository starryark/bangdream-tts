"""E1 (pod): codec-tokenise each row source once; e1_mix.py composes the arms from them.

  python e1_codes.py

kurisu_v2/pod/p04_codes_v3.py with BanG Dream sources (same tokenizer, batch, frame check):
  bundle/ja_rows.jsonl         Japanese rows (e1_build.py)
  replay/replay_rows.jsonl     native zh/en rows from other speakers (p05_replay.py)
  pseudo/pseudo_rows.jsonl     gated Base Mandarin in each voice (e1_pseudo.py; optional)
Writes codes/<source>_<split>_with_codes.jsonl (source fields + audio_codes).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
sys.path.insert(0, str(ROOT / "code"))
import p04_codes_v3 as P  # noqa: E402  (signature + the encode loop's conventions)

SOURCES = {"ja": ROOT / "bundle" / "ja_rows.jsonl",
           "replay": ROOT / "replay" / "replay_rows.jsonl",
           "pseudo": ROOT / "pseudo" / "pseudo_rows.jsonl"}


def main() -> None:
    import json
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    out_dir = ROOT / "codes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"[codes] {name}: {path} missing, skipped")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        for r in rows:
            if "dur" not in r:
                info = sf.info(r["audio"])
                r["dur"] = info.frames / info.samplerate
        by_split: dict[str, list] = {}
        for r in rows:
            by_split.setdefault(P.SPLIT[r.get("split", "train")], []).append(r)
        for split, items in by_split.items():
            out = out_dir / f"{name}_{split}_with_codes.jsonl"
            sig_path = out.with_suffix(".sig")
            sig = P.signature(items)
            if out.exists() and sig_path.exists() and sig_path.read_text() == sig:
                print(f"[codes] {out.name}: up to date ({len(items)} rows)")
                continue
            items.sort(key=lambda r: r["dur"])
            lines = []
            for i in range(0, len(items), P.BATCH):
                chunk = items[i:i + P.BATCH]
                enc = tok.encode([r["audio"] for r in chunk])
                for r, codes in zip(chunk, enc.audio_codes):
                    codes = codes.cpu().tolist()
                    if abs(len(codes) - r["dur"] * 12.5) > 3:
                        raise SystemExit(f"{r.get('id')}: {len(codes)} frames for {r['dur']}s")
                    lines.append(json.dumps({**r, "audio_codes": codes}, ensure_ascii=False))
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            sig_path.write_text(sig)
            print(f"[codes] {out.name}: {len(lines)} rows")


if __name__ == "__main__":
    main()
