"""E9 (local): the fixed listening items (E9_PLAN.md §6) -> e9/listen_items.json.

  python -I -X utf8 e9/e9_listen_items.py

Pairs (E9 vs v1 = C0, same voice id in both; texts and voices fixed here, random.Random(9)):
  v1era_ja  11 characters (the 9 v1-era ids + mygo_raana, mujica_nyamu) x 3: one v1 held-out line (E1's test lines),
            one novel sentence, one short line (the shared 16-line set v1 rows use)
  persona   mygo_soyo_tsuki, mujica_sakiko_tsuki x 2 v1 held-out lines (E9 renders them through the alias)
  zh        8 voices x 1 probe: v1-era tomori, anon, uika, sakiko, soyo, taki + unchanged pp_eve, hhw_hagumi
  unch_ja   6 unchanged voices x 1 held-out line
Absolute (E9 only, next to a real clip of the character):
  ournotes_abs  the 9 *_ournotes ids x 2 (one Our Notes held-out line, one own short line)
  new_abs       the 16 new voices x 2 (held-out line, short line)
Render seed per item: 910000 + index, identical for both systems.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
B9, B1, KP = BD / "e9" / "bundle", BD / "e1" / "bundle", Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
V1ERA = ["mygo_tomori", "mygo_anon", "mygo_soyo", "mygo_taki", "mujica_uika", "mujica_mutsumi", "mujica_mortis",
         "mujica_umiri", "mujica_sakiko", "mygo_raana", "mujica_nyamu"]
ZH = ["mygo_tomori", "mygo_anon", "mujica_uika", "mujica_sakiko", "mygo_soyo", "mygo_taki", "pp_eve", "hhw_hagumi"]
UNCH = ["ppp_kasumi", "ag_ran", "ro_yukina", "hhw_kokoro", "pp_aya", "ras_chuchu"]
LANG = {"ja": "Japanese", "zh": "Chinese", "en": "English"}


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def local(p: str) -> str:
    return str(BD / p.replace("/workspace/bd/", ""))


def main() -> None:
    rng = random.Random(9)
    sel = json.loads((B9 / "rows_e6.json").read_text(encoding="utf-8"))
    ev9, sh9, ev1 = rd(B9 / "eval_rows.jsonl"), rd(B9 / "eval_short.jsonl"), rd(B1 / "eval_rows.jsonl")
    novel = rd(B9 / "probes_ja_novel.jsonl")
    zh = [r for r in rd(KP / "probes_zh_v3.jsonl") if r["category"] == "shared"][:16]
    shared = sorted({(r["tid"], r["text"]) for r in sh9 if r["row"] == "mygo_soyo"})       # the shared v1 short set
    items = []

    def add(group, voice, kind, text, tid, lang, systems, ref=None):
        i = len(items)
        items.append({"id": f"e9i{i:03d}", "group": group, "voice": voice, "kind": kind, "lang": lang, "language": LANG[lang],
                      "text": text, "tid": tid, "seed": 910000 + i, "systems": systems, **({"ref": ref} if ref else {})})

    for v in V1ERA:
        pool = [r for r in ev1 if r["row"] in (v, {"mygo_soyo": "mygo_soyo", "mujica_sakiko": "mujica_sakiko"}.get(v, v))]
        h = rng.choice(pool); add("v1era_ja", v, "ja_heldout", h["text"], h["tid"], "ja", ["E9", "C0"])
        n = rng.choice(novel); add("v1era_ja", v, "ja_novel", n["text"], n["tid"], "ja", ["E9", "C0"])
        s = rng.choice(shared); add("v1era_ja", v, "ja_short", s[1], s[0], "ja", ["E9", "C0"])
    for p in ("mygo_soyo_tsuki", "mujica_sakiko_tsuki"):
        for h in rng.sample([r for r in ev1 if r["row"] == p], 2):
            add("persona", p, "ja_heldout", h["text"], h["tid"], "ja", ["E9", "C0"])
    for v in ZH:
        z = rng.choice(zh); add("zh", v, "zh", z["text"], z["pid"], "zh", ["E9", "C0"])
    for v in UNCH:
        h = rng.choice([r for r in ev9 if r["row"] == v]); add("unch_ja", v, "ja_heldout", h["text"], h["tid"], "ja", ["E9", "C0"])

    def first_ref(v):
        return local((B9 / "refs" / f"ref_{v}.txt").read_text(encoding="utf-8").split()[0])
    for v in [n for n in sel["replaced"] if n.endswith("_ournotes")]:
        h = rng.choice([r for r in ev9 if r["row"] == v]); add("ournotes_abs", v, "ja_heldout", h["text"], h["tid"], "ja", ["E9"], first_ref(v))
        s = rng.choice([r for r in sh9 if r["row"] == v]); add("ournotes_abs", v, "ja_short", s["text"], s["tid"], "ja", ["E9"], first_ref(v))
    for v in sel["new"]:
        h = rng.choice([r for r in ev9 if r["row"] == v]); add("new_abs", v, "ja_heldout", h["text"], h["tid"], "ja", ["E9"], first_ref(v))
        s = rng.choice([r for r in sh9 if r["row"] == v]); add("new_abs", v, "ja_short", s["text"], s["tid"], "ja", ["E9"], first_ref(v))
    for it in items:
        if it.get("ref") and not Path(it["ref"]).is_file():
            raise SystemExit(f"reference clip missing: {it['ref']}")
    counts = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    out = {"items": items, "counts": counts, "renders": sum(len(it["systems"]) for it in items)}
    (BD / "e9" / "listen_items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[e9 listen_items] {len(items)} items {counts}; {out['renders']} renders")


if __name__ == "__main__":
    main()
