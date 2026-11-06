"""E4 Amendment B "shift" control (local, Applio venv): pitch-shift the native sources, no conversion.

  E:\\bd_rvc\\rvcenv\\Scripts\\python.exe "E:\\STEINS;GATEVoice\\bangdream\\e4\\e4_shift.py"

Each source with an offset in e4/vc/offsets.json (the semitones Applio's proposed-pitch applied in
the gated conversion) goes through Praat PSOLA "Change gender" with formant shift ratio 1.0 (formants
and articulation kept), new pitch median = source median x 2^(offset/12), range factor 1, duration 1.
Writes e4/vc/controls/shift/<row>/<utt>.wav and e4/vc/controls/shift/f0_check.json (median F0 in/out).
"""
import json
import os

import parselmouth
from parselmouth.praat import call

VC = os.environ.get("E4_VC", r"E:\STEINS;GATEVoice\bangdream\e4\vc")   # E5 reuses this with E4_VC=e5/vc


def median_f0(snd) -> float:
    return call(snd.to_pitch(pitch_floor=75, pitch_ceiling=900), "Get quantile", 0, 0, 0.5, "Hertz")


def main() -> None:
    src = [json.loads(l) for l in open(os.path.join(VC, "sources.jsonl"), encoding="utf-8") if l.strip()]
    offsets = json.load(open(os.path.join(VC, "offsets.json")))
    check = {}
    for r in src:
        k = offsets.get(f"{r['row']}|{r['utt']}")
        if k is None:
            continue
        snd = parselmouth.Sound(r["src"])
        f_in = median_f0(snd)
        out = call(snd, "Change gender", 75, 600, 1.0, f_in * 2 ** (k / 12), 1.0, 1.0)
        p = os.path.join(VC, "controls", "shift", r["row"], f"{r['utt']}.wav")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out.save(p, "WAV")
        check[f"{r['row']}|{r['utt']}"] = {"offset": k, "f0_in": round(f_in, 1), "f0_target": round(f_in * 2 ** (k / 12), 1),
                                            "f0_out": round(median_f0(out), 1)}
    json.dump(check, open(os.path.join(VC, "controls", "shift", "f0_check.json"), "w"), indent=1)
    for row in sorted({k.split("|")[0] for k in check}):
        v = [c for k, c in check.items() if k.startswith(row + "|")]
        med = sorted(c["f0_out"] for c in v)[len(v) // 2]
        print(f"[e4_shift] {row}: {len(v)} clips, median F0 out {med} Hz", flush=True)


if __name__ == "__main__":
    main()
