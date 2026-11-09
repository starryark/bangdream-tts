"""E9 (pod or local): give merged persona ids their merged voice's speaker row (E9_PLAN.md §2).

  python e9/e9_aliases.py CKPT_DIR [CKPT_DIR ...]

Soyo/Soyo-tsuki and Sakiko/Sakiko-tsuki are one voice each in E9 (user, 2026-11-09). v1's ids keep working:
talker_config.spk_id[alias] = spk_id[target], and spk_is_dialect[alias] = spk_is_dialect[target] (generate() looks
the dialect up for every speaker name). Idempotent. Refuses if an alias already names a different row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ALIASES = {"mygo_soyo_tsuki": "mygo_soyo", "mujica_sakiko_tsuki": "mujica_sakiko"}


def add_aliases(ckpt: Path) -> dict:
    p = ckpt / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    tc = cfg["talker_config"]
    spk, dia = tc["spk_id"], tc.setdefault("spk_is_dialect", {})
    for alias, target in ALIASES.items():
        if target not in spk:
            raise SystemExit(f"{ckpt}: target voice {target} not in spk_id")
        if alias in spk and spk[alias] != spk[target]:
            raise SystemExit(f"{ckpt}: {alias} already names row {spk[alias]}, not {target}'s {spk[target]}")
        spk[alias] = spk[target]
        dia[alias] = dia.get(target, False)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return {a: spk[a] for a in ALIASES}


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(f"[aliases] {c}: {add_aliases(Path(c))}")
