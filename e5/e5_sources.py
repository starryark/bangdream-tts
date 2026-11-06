"""E5 step 3b (local, TTS venv): render each target voice's texts with E1b and keep the most native.

  E5_ROWS=ppp_kasumi,pp_eve python -I e5_sources.py      (default: every voice in texts.json)

E5_PLAN.md §3. Per voice, its 40 texts (vc/texts.json) are rendered 6 times with E1b_ep2
(generate_custom_voice in the voice's own row, Chinese; batches of 20 within one voice, seed
510000 + 100000 * voice index + 1000 * sample + batch, so a voice's renders do not depend on which
other voices run with it), scored with bdscore, and per text the lowest-lid sample passing E2's
accent, text and length gates (e4_sources.native) is kept. Amendment E5-A: a text with no native
own-row sample is rendered 6 times in the donor row (ppp_kasumi; seed 2510000 + ...; scored with
row = donor) and the lowest-lid native donor sample is kept, src_row = donor. Kasumi gets no fill.
Amendment E5-B: an own-row text keeps its native sample with the highest source p_true instead of the
lowest lid, except for Kasumi and Eve (their listening sets were issued under lowest-lid). When a
(row, utt)'s chosen source changes, its converted outputs and logged offset are deleted (E5-B2).
Voices already scored are skipped. Writes vc/src_cands/<row>.jsonl and <row>__donor.jsonl (every
scored render) and rebuilds vc/sources.jsonl from all of them, in the field layout e4_convert /
e4_shift / e4_score read (run those with E4_VC=e5/vc).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
os.environ.setdefault("BD_POD", str(BD))
os.environ.setdefault("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                 r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
os.environ.setdefault("KURISU_WORK", str(BD / "e5" / "work"))
sys.path.insert(0, str(BD / ".pydeps_sb"))   # speechbrain 1.1.2 (Windows k2_fsa fix), as b05_score
sys.path.insert(0, str(BD / "e1"))
sys.path.insert(0, str(BD / "e4"))

E1B = BD / "models" / "E1b_ep2"
OUT = BD / "e5" / "vc"
N_SAMPLES, BATCH = 6, 20
from e4_sources import native              # noqa: E402  (E2's accent + text + length gates)
DONOR = "ppp_kasumi"                       # E5-A1: E3/E4's donor for Eve; generic, low-pitched E1b Mandarin
SEED, DONOR_SEED = 510000, 2510000
LID_PICK = {"ppp_kasumi", "pp_eve"}        # E5-B scope: sets issued before the amendment keep lowest-lid


def render(model, speaker, pairs, seed0, vi, folder, extra):
    import numpy as np
    import soundfile as sf
    import torch
    cands = []
    for s in range(N_SAMPLES):
        for b in range(0, len(pairs), BATCH):
            chunk = pairs[b:b + BATCH]
            torch.manual_seed(seed0 + 100000 * vi + 1000 * s + b // BATCH)
            wavs, sr = model.generate_custom_voice(text=[t for _, t in chunk], speaker=[speaker] * len(chunk),
                                                   language=["Chinese"] * len(chunk), max_new_tokens=400,
                                                   non_streaming_mode=True)
            assert len(wavs) == len(chunk)
            for (u, t), w in zip(chunk, wavs):
                p = folder / f"{u}_s{s}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
                cands.append({"row": speaker, "lang": "zh", "utt": u, "text": t, "sample": s, "path": str(p), **extra})
    return cands


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l] if path.exists() else None


def best(cands, u, by="lid"):
    ok = [r for r in cands if r["utt"] == u and native(r)]
    if not ok:
        return None
    return min(ok, key=lambda r: r["lid_ratio"]) if by == "lid" else max(ok, key=lambda r: r["p_true"])   # E5-B1


def invalidate(old, new):
    """E5-B2: delete the converted outputs and logged offset of every (row, utt) whose chosen source changed."""
    was = {(r["row"], r["utt"]): r["src"] for r in old}
    stale = [(r["row"], r["utt"]) for r in new if was.get((r["row"], r["utt"]), r["src"]) != r["src"]]
    off_path = OUT / "offsets.json"
    offsets = json.loads(off_path.read_text()) if off_path.exists() else {}
    for row, u in stale:
        for p in list((OUT / "out").glob(f"*/{row}/{u}.wav")) + [OUT / "controls" / "shift" / row / f"{u}.wav"]:
            p.unlink(missing_ok=True)
        offsets.pop(f"{row}|{u}", None)
    if stale:
        off_path.write_text(json.dumps(offsets, indent=0))
    return stale


def main() -> None:
    import numpy as np
    import torch
    sys.stdout.reconfigure(encoding="utf-8")
    texts = json.loads((OUT / "texts.json").read_text(encoding="utf-8"))
    order = list(texts)
    only = set(filter(None, os.environ.get("E5_ROWS", "").split(",")))
    sel = [r for r in order if not only or r in only]
    cdir = OUT / "src_cands"
    cdir.mkdir(exist_ok=True)
    for phase in ("own", "donor"):
        if phase == "own":
            todo = {r: texts[r] for r in sel if not (cdir / f"{r}.jsonl").exists()}
        else:                                      # E5-A1: texts with no native own-row sample
            todo = {r: [(u, t) for u, t in texts[r] if best(load(cdir / f"{r}.jsonl"), u) is None]
                    for r in sel if r != DONOR and not (cdir / f"{r}__donor.jsonl").exists()}
        need = {r: p for r, p in todo.items() if p}
        cands = {r: [] for r in todo}
        if need:
            from qwen_tts import Qwen3TTSModel
            import bdscore
            model = Qwen3TTSModel.from_pretrained(str(E1B), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
            for row, pairs in need.items():
                vi = order.index(row)
                if phase == "own":
                    cands[row] = render(model, row, pairs, SEED, vi, OUT / "src_raw" / row, {})
                else:
                    cands[row] = render(model, DONOR, pairs, DONOR_SEED, vi, OUT / "src_raw" / row / "donor", {"target": row})
                print(f"[e5_sources] {row} {phase}: {N_SAMPLES * len(pairs)} renders", flush=True)
            del model
            torch.cuda.empty_cache()
            bdscore.score([c for v in cands.values() for c in v], BD / "e1" / "bundle" / "probe_codec.npz")
        for row, c in cands.items():
            (cdir / (f"{row}.jsonl" if phase == "own" else f"{row}__donor.jsonl")).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in c), encoding="utf-8")

    out = []
    for row in order:
        own, donor = load(cdir / f"{row}.jsonl"), load(cdir / f"{row}__donor.jsonl") or []
        if own is None:
            continue
        mine = []
        for u, t in texts[row]:
            r = best(own, u, "lid" if row in LID_PICK else "p_true") or best(donor, u, "lid")
            if r:
                mine.append({"row": row, "lang": "zh", "utt": u, "text": t, "src_row": r["row"], "src": r["path"],
                             "src_sample": r["sample"], "src_lid": r["lid_ratio"],
                             "src_ser": r.get("ser_shared") if r.get("m23_valid", True) else None,
                             "src_cer": r["cer"], "src_pred": r["pred"]})
        out += mine
        ser = [r["src_ser"] for r in mine if r["src_ser"] is not None]
        n_d = sum(r["src_row"] != row for r in mine)
        print(f"[e5_sources] {row}: {len(mine)}/{len(texts[row])} texts kept ({len(mine) - n_d} own, {n_d} donor); "
              f"own renders native {np.mean([native(r) for r in own]):.0%}"
              + (f", donor {np.mean([native(r) for r in donor]):.0%}" if donor else "")
              + f"; median lid {np.median([r['src_lid'] for r in mine]):.1f}, mean SER {np.mean(ser):.3f}", flush=True)
    stale = invalidate(load(OUT / "sources.jsonl") or [], out)
    if stale:
        print(f"[e5_sources] source changed for {len(stale)} texts; their old conversions were deleted", flush=True)
    (OUT / "sources.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
