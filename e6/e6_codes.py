"""E6 (pod): codec-tokenise each row source once; e6_mix.py composes the arms (as e1/e1_codes.py).

  python e6/e6_codes.py

Sources: ja (bundle/ja_rows.jsonl: 37 v1 rows + 28 Our Notes rows), replay (replay/replay_rows.jsonl),
pseudo_v1 (pseudo_v1/rows.jsonl), pseudo_on (pseudo_on/pseudo_rows.jsonl), vc (vc/vc_rows.jsonl),
nv (bundle/nv_rows.jsonl). Same tokenizer, batch, frame check and signature cache as e1_codes.py.
Writes codes/<source>_<split>_with_codes.jsonl.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
sys.path.insert(0, str(ROOT / "code"))
import p04_codes_v3 as P  # noqa: E402

SOURCES = {"ja": ROOT / "bundle" / "ja_rows.jsonl", "replay": ROOT / "replay" / "replay_rows.jsonl",
           "pseudo_v1": ROOT / "pseudo_v1" / "rows.jsonl", "pseudo_on": ROOT / "pseudo_on" / "pseudo_rows.jsonl",
           "vc": ROOT / "vc" / "vc_rows.jsonl", "nv": ROOT / "bundle" / "nv_rows.jsonl"}


def main() -> None:
    import json
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer
    only = set(sys.argv[1:])
    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    out_dir = ROOT / "codes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        if only and name not in only:
            continue
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
            print(f"[codes] {out.name}: {len(lines)} rows", flush=True)


if __name__ == "__main__":
    main()
