"""E9 (local): blind listening pages and the decision analysis (E9_PLAN.md §6). Pages and clips stay on this machine.

  python e9/e9_listen_page.py pages                         # -> e9/listen/{p1,p2,abs}.html ; key e9/listen/e9_key.json
  python e9/e9_listen_page.py analyze P1.json P2.json ABS.json

Reuses e8_listen_page (page template, loudness normalisation, blind randomised pairs, repeats, nulls). Pairs: E9 vs
v1 (C0) on the same voice id and text; CCR -3..+3. Absolute: E9 next to a real clip; yes / partly / no.
"""
from __future__ import annotations

import json
import secrets
import shutil
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.path.insert(0, str(BD / "e8"))
import e8_listen_page as P  # noqa: E402

E9 = BD / "e9"
LISTEN, RENDERS = E9 / "listen", E9 / "listen_renders"
PAIR_GROUPS = ("v1era_ja", "persona", "zh", "unch_ja")


def items() -> list[dict]:
    return json.loads((E9 / "listen_items.json").read_text(encoding="utf-8"))["items"]


def render_path(system: str, it: dict) -> Path:
    p = RENDERS / system / f"{it['id']}.wav"
    if not p.is_file():
        raise SystemExit(f"missing render {p}")
    return p


INTRO = ("Each trial plays the same text twice, as A and B. Which sounds better <b>for this character</b>: voice identity "
         "first, then naturalness, pronunciation and accent? Please take your time (about 30 s per trial), use "
         "headphones, and replay as often as you like. Some trials repeat.")


def pages() -> None:
    its = items()
    if LISTEN.exists():
        shutil.rmtree(LISTEN)
    LISTEN.mkdir(parents=True)
    pairs = [{**it, "round": it["group"]} for it in its if it["group"] in PAIR_GROUPS]
    trials, key = P.pair_trials(pairs, "E9", "C0", LISTEN, 6, 3, render_path)
    half = (len(trials) + 1) // 2
    P.write_page(LISTEN / "p1.html", "E9 listening, part 1 of 3", INTRO, trials[:half], "e9_p1")
    P.write_page(LISTEN / "p2.html", "E9 listening, part 2 of 3", INTRO, trials[half:], "e9_p2")
    audio = LISTEN / "audio"
    refs, abst = {}, []
    for it in [x for x in its if x["group"] in ("ournotes_abs", "new_abs")]:
        if it["ref"] not in refs:
            refs[it["ref"]] = P.norm_copy(Path(it["ref"]), audio)
        tid = secrets.token_hex(4)
        abst.append({"tid": tid, "type": "abs", "a": P.norm_copy(render_path("E9", it), audio), "ref": refs[it["ref"]],
                     "text": it["text"], "lang": it["language"]})
        key[tid] = {"kind": "abs", "group": it["group"], "item": it["id"], "voice": it["voice"]}
    P.SR.shuffle(abst)
    P.write_page(LISTEN / "abs.html", "E9 listening, part 3 of 3: does it sound like the character?",
                 "Each trial plays a real clip of the character, then a generated one. Does the generated clip sound like this character?",
                 abst, "e9_abs")
    (E9 / "listen_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"[e9 pages] {len(trials)} pair trials (p1 {half}, p2 {len(trials) - half}), {len(abst)} absolute -> {LISTEN}")


def analyze(paths: list[str]) -> None:
    from scipy.stats import wilcoxon
    key = json.loads((E9 / "listen_key.json").read_text(encoding="utf-8"))
    ans = P.load_answers(paths)
    missing = [t for t in key if t not in ans or "score" not in ans[t]]
    if missing:
        raise SystemExit(f"{len(missing)} trials unanswered")
    main = [(t, k) for t, k in key.items() if k["kind"] == "main"]
    pref = {t: P.pref(k, ans[t], "E9") for t, k in main}

    def test(x):
        nz = [v for v in x if v != 0]
        return float(wilcoxon(x).pvalue) if nz else 1.0
    g1 = [pref[t] for t, k in main if k["round"] == "v1era_ja"]
    g2 = [pref[t] for t, k in main if k["round"] in ("persona", "zh", "unch_ja")]
    per_voice = {}
    for t, k in main:
        if k["round"] == "v1era_ja":
            per_voice[k["voice"]] = per_voice.get(k["voice"], 0) + pref[t]
    wrong = {}
    for t, k in main:
        side = "A" if k["left"] == "E9" else "B"
        if f"wrong_voice_{side}" in ans[t].get("flags", []):
            wrong[k["voice"]] = wrong.get(k["voice"], 0) + 1
    absv = {}
    for t, k in key.items():
        if k["kind"] == "abs":
            absv.setdefault((k["group"], k["voice"]), []).append(ans[t]["score"])
    weak = lambda grp: sorted(v for (g, v), s in absv.items() if g == grp and "yes" not in s and "no" in s)   # noqa: E731
    cons = P.consistency({t: k for t, k in key.items() if k["kind"] in ("main", "repeat", "null")}, ans, "E9")
    by_group = {}
    for t, k in main:
        by_group.setdefault(k["round"], []).append(pref[t])
    rules = {
        "v1era_not_worse": (sum(g1) / len(g1) >= 0) or test(g1) >= 0.05,
        "no_v1era_voice_sum_le_-5": all(s > -5 for s in per_voice.values()),
        "others_not_worse": (sum(g2) / len(g2) >= 0) or test(g2) >= 0.05,
        "no_voice_wrong_twice": all(n < 2 for n in wrong.values()),
        "new_weak_le_3": len(weak("new_abs")) <= 3,
        "ournotes_weak_le_2": len(weak("ournotes_abs")) <= 2,
        "consistent": cons["repeats_agree"] >= 4 and cons["nulls_within_1"],
    }
    out = {"groups": {g: {"n": len(x), "mean": round(sum(x) / len(x), 3), "p_two_sided": round(test(x), 4)} for g, x in by_group.items()},
           "v1era_ja": {"n": len(g1), "mean": round(sum(g1) / len(g1), 3), "p_two_sided": round(test(g1), 4)},
           "others": {"n": len(g2), "mean": round(sum(g2) / len(g2), 3), "p_two_sided": round(test(g2), 4)},
           "per_v1era_voice_sum": dict(sorted(per_voice.items(), key=lambda x: x[1])), "wrong_voice_flags": wrong,
           "absolute": {f"{g}|{v}": s for (g, v), s in sorted(absv.items())}, "weak_new": weak("new_abs"), "weak_ournotes": weak("ournotes_abs"),
           "consistency": cons, "rules": rules, "adopt_by_listening": all(rules.values()),
           "note": "Screens (E9_PLAN §5) must also pass. If only 'consistent' fails, the pair pages are repeated once (§6)."}
    (E9 / "results").mkdir(exist_ok=True)
    (E9 / "results" / "listening.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "pages":
        pages()
    elif sys.argv[1] == "analyze":
        analyze(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
