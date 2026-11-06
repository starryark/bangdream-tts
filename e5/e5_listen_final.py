"""E5 §6 condition 4 (local): the target-accent listening set for the picked checkpoint.

  python -I e5_listen_final.py <cond>        e.g. E5_ep2

2 blind zh renders per target voice (15 voices, 30 clips), drawn at random (seed 11) from the pick's
eval renders (e5/results/eval/<cond>/wav/<pid>.wav, copied from the pod), two different texts per
voice. No scores are shown. Each voice folder also gets 2 of its Japanese reference clips (as in the
row-listening sets) so "sounds like the character" has an anchor. Writes e5/listen_final/<voice>/
and e5/results/listen_key.json. LOCAL ONLY: the folder holds game-derived reference audio.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
sys.path.insert(0, str(HERE))
from e5_select import TARGETS  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cond = sys.argv[1]
    ev = HERE / "results" / "eval" / cond
    rows = [json.loads(l) for l in (ev / "scored.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(11)
    out_root = HERE / "listen_final"
    if out_root.exists():
        shutil.rmtree(out_root)
    key = {"cond": cond, "voices": {}}
    for v in TARGETS:
        zh = sorted([r for r in rows if r["row"] == v and r["lang"] == "zh"], key=lambda r: r["pid"])
        first = rng.choice(zh)
        second = rng.choice([r for r in zh if r["tid"] != first["tid"]])
        out = out_root / v
        out.mkdir(parents=True)
        lines = [f"{v}: Mandarin renders from the retrained TTS model (LOCAL ONLY, never share)", "",
                 "Questions: 1. native Mandarin, or Japanese-accented?  2. does it sound like the character?", ""]
        for i, r in enumerate((first, second), 1):
            shutil.copy2(ev / "wav" / f"{r['pid']}.wav", out / f"{i}.wav")
            lines.append(f"{i}: {r['text']}")
        refs = [l.strip() for l in (BD / "e1" / "bundle" / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").splitlines()
                if l.strip()][:2]
        for j, p in enumerate(refs, 1):
            shutil.copy2(p.replace("/workspace/bd", str(BD)), out / f"ref_ja_{j}.wav")
        (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        key["voices"][v] = {"1": first["pid"], "2": second["pid"]}
    (HERE / "results" / "listen_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[listen_final] {cond}: {2 * len(TARGETS)} clips in {out_root}")


if __name__ == "__main__":
    main()
