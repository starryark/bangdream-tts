"""E1 (pod): drop replay speakers that sit too close to any shipped BanG Dream voice.

  python e1_aux_filter.py [--max_cos 0.7]

E1_PLAN.md: "Replay speakers far from all 49 rows by centred cosine". A replay speaker whose
vector is close to a character's locked vector could pull that character's timbre toward a
Mandarin/English reader during the shared fine-tune.

Vectors are the trainer's own (train_sft.speaker_embedding): the 49 shipped rows from
bundle/refs, each aux speaker from its aux_speakers.json clips. They are centred by the mean
of the 49 shipped vectors (the BanG Dream average voice), the same idea as E0's centred
cosine. An aux speaker is dropped if its centred cosine to any shipped row exceeds --max_cos
(pre-registered 0.7: above E0's p95 between characters, 0.57, below its closest pair, 0.82).
Rewrites replay/{aux_speakers.json, replay_rows.jsonl} (originals kept as *.unfiltered) and
writes replay/aux_filter.json. Idempotent: always filters from the *.unfiltered originals.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("KURISU_WORK", str(ROOT))


def main() -> None:
    import torch
    import train_sft
    from qwen_tts import Qwen3TTSModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cos", type=float, default=0.7)
    args = ap.parse_args()
    R = ROOT / "replay"
    for f in ("aux_speakers.json", "replay_rows.jsonl"):
        if not (R / f"{f}.unfiltered").exists():
            shutil.copy(R / f, R / f"{f}.unfiltered")
    aux = json.loads((R / "aux_speakers.json.unfiltered").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (R / "replay_rows.jsonl.unfiltered").read_text(encoding="utf-8").splitlines() if l]
    names = [s.split("=")[0] for s in (ROOT / "bundle" / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]

    w = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = w.model.to("cuda")

    def vec(name, paths):
        return train_sft.speaker_embedding(model, name, [Path(p) for p in paths], "cuda", torch.bfloat16)[0].float().cpu()

    S = torch.stack([vec(n, train_sft.ref_paths(ROOT / "bundle" / "refs" / f"ref_{n}.txt")) for n in names])
    A = torch.stack([vec(n, ps) for n, ps in aux.items()])
    mu = S.mean(0)
    Sc = torch.nn.functional.normalize(S - mu, dim=-1)
    Ac = torch.nn.functional.normalize(A - mu, dim=-1)
    cos = Ac @ Sc.T
    rep, drop = {}, set()
    for i, a in enumerate(aux):
        j = int(cos[i].argmax())
        rep[a] = {"max_cos": round(float(cos[i, j]), 3), "nearest": names[j]}
        if cos[i, j] > args.max_cos:
            drop.add(a)
    kept_aux = {a: ps for a, ps in aux.items() if a not in drop}
    kept_rows = [r for r in rows if r["speaker"] not in drop]
    (R / "aux_speakers.json").write_text(json.dumps(kept_aux, indent=1, ensure_ascii=False), encoding="utf-8")
    (R / "replay_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept_rows), encoding="utf-8")
    (R / "aux_filter.json").write_text(json.dumps({"max_cos": args.max_cos, "dropped": sorted(drop),
                                                   "rows_before": len(rows), "rows_after": len(kept_rows),
                                                   "speakers": rep}, indent=1, ensure_ascii=False), encoding="utf-8")
    top = sorted(rep.items(), key=lambda t: -t[1]["max_cos"])[:5]
    print(f"[aux_filter] dropped {len(drop)}/{len(aux)} speakers ({len(rows) - len(kept_rows)} rows); "
          f"closest: {[(a, v['max_cos'], v['nearest']) for a, v in top]}")


if __name__ == "__main__":
    main()
