"""E4 Amendment C listening set (any python): the first 5 sources per voice in four cells, no scores.

  python -I e4_listen.py 200        -> e4/listen/ep200/<row>/

Cells: a = the native source, b = shift only (controls/shift), c = RVC without shift (out/ep<E>_noshift),
d = RVC as planned (out/ep<E>). Plus two real Japanese reference clips per voice (e1/bundle/refs).
The README carries no scores, so the verdict is blind to the metrics (C1(b)).
"""
import json
import os
import shutil
import sys

BD = r"E:\STEINS;GATEVoice\bangdream"
VC = os.path.join(BD, "e4", "vc")
NAMES = {"ppp_kasumi": "Kasumi", "pp_eve": "Eve"}


def main() -> None:
    ep = sys.argv[1] if len(sys.argv) > 1 else "200"
    src = [json.loads(l) for l in open(os.path.join(VC, "sources.jsonl"), encoding="utf-8") if l.strip()]
    cells = [("a_source", lambda r: r["src"]),
             ("b_shift_only", lambda r: os.path.join(VC, "controls", "shift", r["row"], f"{r['utt']}.wav")),
             ("c_rvc_noshift", lambda r: os.path.join(VC, "out", f"ep{ep}_noshift", r["row"], f"{r['utt']}.wav")),
             ("d_rvc_shift", lambda r: os.path.join(VC, "out", f"ep{ep}", r["row"], f"{r['utt']}.wav"))]
    for row in dict.fromkeys(r["row"] for r in src):
        name = NAMES.get(row, row)
        out = os.path.join(BD, "e4", "listen", f"ep{ep}", row)
        os.makedirs(out, exist_ok=True)
        lines = [f"{name} Mandarin listening set (epoch-{ep} RVC; LOCAL ONLY, never share)", "",
                 "a_source      native E1b render (generic timbre)",
                 f"b_shift_only  the same clip pitch-shifted to {name}'s pitch, no conversion",
                 f"c_rvc_noshift RVC into {name}'s voice, source pitch kept",
                 f"d_rvc_shift   RVC into {name}'s voice at her pitch (the planned setup)",
                 f"ref_ja_*      real {name} Japanese clips, for timbre", "",
                 "Questions:",
                 "  1. Do c and d sound native Mandarin, or Japanese-accented?",
                 f"  2. Does d sound like {name}?", ""]
        for i, r in enumerate([r for r in src if r["row"] == row][:5], 1):
            lines.append(f"{i}: {r['text']}")
            for tag, f in cells:
                shutil.copy2(f(r), os.path.join(out, f"{i}_{tag}.wav"))
        refs = [l.strip() for l in open(os.path.join(BD, "e1", "bundle", "refs", f"ref_{row}.txt"), encoding="utf-8")
                if l.strip()][:2]
        for j, p in enumerate(refs, 1):
            shutil.copy2(p.replace("/workspace/bd", BD), os.path.join(out, f"ref_ja_{j}.wav"))
        open(os.path.join(out, "README.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[e4_listen] {out}: {len(os.listdir(out))} files", flush=True)


if __name__ == "__main__":
    main()
