"""E4 step 3 (local, Applio venv, cwd = E:\\bd_rvc\\Applio): convert the E4 sources with each voice's RVC.

  E:\\bd_rvc\\rvcenv\\Scripts\\python.exe "E:\\STEINS;GATEVoice\\bangdream\\e4\\e4_convert.py" 200 100

E4_PLAN.md "Inference": index_rate 0 (the FAISS index is never loaded), rmvpe, pitch 0 with
proposed_pitch, threshold = the voice's median voiced F0 over its training slices (pooled frames
of the extract step's Hz-valued f0 files), protect 0.33 and volume envelope 1.0 (defaults), contentvec,
no autotune / formant / cleanup. One output per (source, epoch): e4/vc/out/ep<E>/<row>/<utt>.wav
(32 kHz). Writes e4/vc/f0_targets.json. Existing outputs are skipped.
"""
import contextlib
import glob
import io
import json
import os
import re
import sys

import numpy as np

BD = r"E:\STEINS;GATEVoice\bangdream"
APPLIO = r"E:\bd_rvc\Applio"
VC = os.environ.get("E4_VC", os.path.join(BD, "e4", "vc"))   # E5 reuses this with E4_VC=e5/vc


def hz_dir(model_dir: str) -> str:
    """The extract step writes coarse (1-255 ints) and Hz F0 to f0/ and f0_voiced/; pick the Hz one."""
    for d in ("f0_voiced", "f0"):
        a = np.load(sorted(glob.glob(os.path.join(model_dir, d, "*.npy")))[0])
        if np.any(np.abs(a - np.round(a)) > 1e-3):
            return d
    raise SystemExit(f"no Hz-valued f0 folder in {model_dir}")


def median_f0(model_dir: str) -> float:
    d = hz_dir(model_dir)
    f = np.concatenate([np.load(p) for p in glob.glob(os.path.join(model_dir, d, "*.npy"))])
    return float(np.median(f[f > 0]))


def main() -> None:
    epochs = [int(e) for e in sys.argv[1:]] or [200]
    os.chdir(APPLIO)
    sys.path.insert(0, APPLIO)
    from rvc.infer.infer import VoiceConverter
    src = [json.loads(l) for l in open(os.path.join(VC, "sources.jsonl"), encoding="utf-8") if l.strip()]
    only = set(filter(None, os.environ.get("E4_ROWS", "").split(",")))   # smoke test: subset of voices
    src = [r for r in src if not only or r["row"] in only]
    rows = sorted({r["row"] for r in src})
    targets = {row: round(median_f0(os.path.join(APPLIO, "logs", f"bd_{row}")), 1) for row in rows}
    json.dump(targets, open(os.path.join(VC, "f0_targets.json"), "w"), indent=1)
    print(f"[e4_convert] target median F0 (Hz): {targets}", flush=True)
    # Amendment B: E4_NOSHIFT=1 -> the "VC, no shift" control cell (proposed_pitch off, pitch 0)
    noshift = os.environ.get("E4_NOSHIFT") == "1"
    cell = "_noshift" if noshift else ""
    off_path = os.path.join(VC, "offsets.json")      # per-clip semitone offsets Applio printed (shift control)
    offsets = json.load(open(off_path)) if os.path.exists(off_path) else {}
    vc = VoiceConverter()
    for ep in epochs:
        for row in rows:
            mdir = os.path.join(APPLIO, "logs", f"bd_{row}")
            pth = glob.glob(os.path.join(mdir, f"bd_{row}_{ep}e_*s.pth"))
            if len(pth) != 1:
                raise SystemExit(f"expected one epoch-{ep} weight file in {mdir}, found {pth}")
            out_dir = os.path.join(VC, "out", f"ep{ep}{cell}", row)
            os.makedirs(out_dir, exist_ok=True)
            for r in [r for r in src if r["row"] == row]:
                out = os.path.join(out_dir, f"{r['utt']}.wav")
                if os.path.exists(out):
                    continue
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    vc.convert_audio(audio_input_path=r["src"], audio_output_path=out, model_path=pth[0], index_path="",
                                     pitch=0, f0_method="rmvpe", index_rate=0.0, volume_envelope=1.0, protect=0.33,
                                     f0_autotune=False, embedder_model="contentvec", export_format="WAV", sid=0,
                                     proposed_pitch=not noshift, proposed_pitch_threshold=targets[row])
                m = re.search(r"calculated pitch offset: (-?\d+)", buf.getvalue())
                if m:
                    offsets[f"{row}|{r['utt']}"] = int(m.group(1))
                if not os.path.exists(out):
                    raise SystemExit(f"expected output missing: {out}\n{buf.getvalue()[-2000:]}")
            json.dump(offsets, open(off_path, "w"), indent=0)
            print(f"[e4_convert] ep{ep}{cell} {row}: {len(os.listdir(out_dir))} clips", flush=True)


if __name__ == "__main__":
    main()
