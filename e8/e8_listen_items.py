"""E8 (local): the fixed listening items (E8_PLAN.md §6) -> e8/listen_items.json.

  python -I -X utf8 e8/e8_listen_items.py

Texts come from E6's eval bundle (held-out and short lines per voice), E1's novel sentences and the
zh/en probes. Choices use random.Random(8). Each item has its own render seed (900000 + index), the same
for every system that renders it. Systems: round 1 -> A, B; round 2 -> A, B, C0 (the winner of round 1
is not known on the pod); new-voice check -> A, B. The reference clip for the new-voice check is the
first clip of the voice's E6 reference list (local path).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
KP = Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
R1_VOICES = ["ikka_yomogi", "mewtype_arale", "mewtype_nonoka", "ikka_chieri", "ikka_raika", "mewtype_yuno",
             "millsage_nagi", "ournotes_kanata"]
R2_REPLACED = ["mygo_tomori", "mygo_anon", "mygo_taki", "mujica_uika", "mujica_mutsumi", "mujica_umiri",
               "mujica_nyamu", "mujica_sakiko"]
R2_UNCHANGED = ["ppp_kasumi", "ag_ran", "pp_aya", "ro_yukina", "hhw_kokoro", "mo_mashiro", "ras_chuchu",
                "mygo_soyo_tsuki"]
R2_ZH = ["mygo_tomori", "mygo_anon", "mujica_uika", "mujica_sakiko", "pp_eve", "hhw_hagumi", "ppp_kasumi",
         "ro_rinko"]
LANG = {"ja": "Japanese", "zh": "Chinese", "en": "English"}


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def local(p: str) -> str:
    return str(BD / p.replace("/workspace/bd/", ""))


def main() -> None:
    rng = random.Random(8)
    B = BD / "e6" / "bundle"
    sel = json.loads((B / "rows_e6.json").read_text(encoding="utf-8"))
    held, short, novel = rd(B / "eval_rows.jsonl"), rd(B / "eval_short.jsonl"), rd(B / "probes_ja_novel.jsonl")
    zh = [r for r in rd(KP / "probes_zh_v3.jsonl") if r["category"] == "shared"][:16]
    en = [r for r in rd(KP / "probes_en_v3.jsonl") if r["category"] == "conversational"][:8]
    items = []

    def add(rnd, voice, kind, systems):
        pool, key, lang = {"ja_heldout": ([h for h in held if h["row"] == voice], "tid", "ja"),
                           "ja_short": ([s for s in short if s["row"] == voice], "tid", "ja"),
                           "ja_novel": (novel, "tid", "ja"), "zh": (zh, "pid", "zh"), "en": (en, "pid", "en")}[kind]
        if not pool:
            raise SystemExit(f"no {kind} line for {voice}")
        t = rng.choice(pool)
        text, tid = t["text"], t[key]
        i = len(items)
        items.append({"id": f"it{i:03d}", "round": rnd, "voice": voice, "kind": kind, "lang": lang,
                      "language": LANG[lang], "text": text, "tid": tid, "seed": 900000 + i, "systems": systems})

    for v in R1_VOICES:
        add(1, v, "ja_heldout", ["A", "B"]); add(1, v, "ja_short", ["A", "B"])
    for v in R1_VOICES[:4]:
        add(1, v, "zh", ["A", "B"])
    for v in R2_REPLACED + R2_UNCHANGED:
        for k in ("ja_heldout", "ja_novel", "ja_short"):
            add(2, v, k, ["A", "B", "C0"])
    for v in R2_ZH:
        add(2, v, "zh", ["A", "B", "C0"]); add(2, v, "zh", ["A", "B", "C0"])
    for v in R2_UNCHANGED:
        add(2, v, "en", ["A", "B", "C0"])
    for v in sel["new"]:
        for k in ("ja_heldout", "ja_short", "zh"):
            add("new", v, k, ["A", "B"])
    refs = {}
    for v in sel["new"]:
        paths = [l.strip() for l in (B / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        refs[v] = local(paths[0])
        if not Path(refs[v]).is_file():
            raise SystemExit(f"reference clip missing: {refs[v]}")
    n = {r: sum(1 for it in items if it["round"] == r) for r in (1, 2, "new")}
    renders = sum(len(it["systems"]) for it in items)
    out = {"items": items, "new_voice_refs": refs, "counts": {str(k): v for k, v in n.items()}, "renders": renders}
    (BD / "e8" / "listen_items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[listen_items] {len(items)} items {n}; {renders} renders; refs for {len(refs)} new voices")


if __name__ == "__main__":
    main()
