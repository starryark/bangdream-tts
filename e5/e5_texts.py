"""E5 step 3a (local, TTS venv): the Mandarin texts each target voice's VC rows are rendered from.

  python -I e5_texts.py

E5_PLAN.md §3: AISHELL-3 test texts from p11's pool (8-30 hanzi, <= 1 numeral, not in AISHELL-3
train), the same pool order as E1/E2 (seed 31). Excluded: E1's texts (the first 60 x 49 of that
order, which the E1b mix's pseudo rows came from), every E2 candidate text, the E4 sources, and
the eval probes (asserted). 40 texts per target voice, disjoint across voices, in pool order.
Writes e5/vc/texts.json {row: [[utt, text], ...]}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
KPOD = Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
os.environ.setdefault("KURISU_WORK", str(BD / "e5" / "work"))   # p05's download cache (AISHELL-3 content.txt)
sys.path.insert(0, str(KPOD))

TARGETS = ["ppp_rimi", "ppp_kasumi", "ro_ako_maou", "mygo_tomori", "mygo_soyo_tsuki", "ras_lock", "ro_rinko", "pp_eve",
           "hhw_kanon", "ras_chuchu", "mujica_sakiko_tsuki", "ras_masking", "hhw_hagumi", "pp_aya", "hhw_kaoru"]
PER_VOICE, E1_TEXTS = 40, 60 * 49


def main() -> None:
    import p11_pseudo_zh as P11
    sys.stdout.reconfigure(encoding="utf-8")
    pool = P11.texts(10 ** 6, 31)                 # the whole pool, in E1/E2's order
    norm = lambda t: t.rstrip("。？")             # noqa: E731
    used = {norm(t) for _, t in pool[:E1_TEXTS]}
    used |= {norm(json.loads(l)["text"]) for l in (BD / "e2" / "results" / "pseudo" / "candidates.jsonl").open(encoding="utf-8")
             if json.loads(l)["lang"] == "zh"}
    used |= {norm(json.loads(l)["text"]) for l in (BD / "e4" / "vc" / "sources.jsonl").open(encoding="utf-8") if l.strip()}
    probes = {norm(json.loads(l)["text"]) for l in (KPOD / "probes_zh_v3.jsonl").open(encoding="utf-8") if l.strip()}
    free = [(u, t) for u, t in pool[E1_TEXTS:] if norm(t) not in used]
    assert not ({norm(t) for _, t in free} & probes), "E5 texts overlap the eval probes"
    if len(free) < PER_VOICE * len(TARGETS):
        raise SystemExit(f"only {len(free)} unused texts for {PER_VOICE * len(TARGETS)} needed")
    texts = {row: free[i * PER_VOICE:(i + 1) * PER_VOICE] for i, row in enumerate(TARGETS)}
    out = BD / "e5" / "vc" / "texts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[e5_texts] pool {len(pool)}, excluded {len(pool) - len(free)}, free {len(free)}; "
          f"{PER_VOICE} x {len(TARGETS)} written to {out}", flush=True)


if __name__ == "__main__":
    main()
