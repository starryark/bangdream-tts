"""E8 (pod): render the listening items (e8/listen_items.json) for one system (E8_PLAN.md §6).

  python e8/e8_listen_render.py --system A --ckpt <dir>      # writes listen_renders/A/<item id>.wav

One take per item, batch size 1, torch.manual_seed(item seed): every system renders an item with the same
seed, and the first take is the one listened to (no cherry-picking). Skips items already rendered.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N items (smoke test)")
    ap.add_argument("--out_name", default=None, help="output folder name (default: the system)")
    ap.add_argument("--items", default="e8/listen_items.json", help="items file under BD_POD (E9: e9/listen_items.json)")
    args = ap.parse_args()
    items = [it for it in json.loads((ROOT / args.items).read_text(encoding="utf-8"))["items"]
             if args.system in it["systems"]]
    if args.limit:
        items = items[:args.limit]
    out = ROOT / "listen_renders" / (args.out_name or args.system)
    out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(args.ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    have = set(json.loads((Path(args.ckpt) / "config.json").read_text())["talker_config"]["spk_id"])
    missing = sorted({it["voice"] for it in items} - have)
    if missing:
        raise SystemExit(f"{args.system}: checkpoint lacks voices {missing}")
    done = 0
    for it in items:
        p = out / f"{it['id']}.wav"
        if p.exists():
            continue
        torch.manual_seed(it["seed"])
        wavs, sr = model.generate_custom_voice(text=[it["text"]], speaker=[it["voice"]], language=[it["language"]],
                                               max_new_tokens=400, non_streaming_mode=True)
        w = np.asarray(wavs[0], dtype=np.float32)
        if w.size == 0 or not np.isfinite(w).all():
            raise SystemExit(f"{args.system} {it['id']}: empty or non-finite render")
        sf.write(str(p), w, sr)
        done += 1
    print(f"[listen_render] {args.system}: {done} new renders, {len(items)} items in {out}", flush=True)


if __name__ == "__main__":
    main()
