"""E5 selection (E5_PLAN.md §6), run locally on copied results.

  python -I e5_select.py

Reads e1/results/eval/{floor,E1b_ep2} and e5/results/eval/{E5_ep1,E5_ep2}; writes
e5/results/selection_e5.{json,md}. The E1 guards as in E2, except that zh_ser, zh_lid and
zh_ser_voice_max are computed over the 34 non-target voices only (the 15 targets now speak at
their own pitch, where those metrics fail on native speech: E4 shift-only control). Pick: the
eligible checkpoint with the lowest pooled zh leak. Adopt over E1b_ep2 if pooled zh leak <= 0.08,
>= 43/49 voices at zh leak <= 0.15, and the targets' pooled zh leak <= 0.15; the fourth condition
(>= 12/15 targets native by ear, 2 blind renders each) is the user's and is recorded separately.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E1R = HERE.parent / "e1" / "results" / "eval"
E5R = HERE / "results"
sys.path.insert(0, str(HERE.parent / "e1"))
import e1_select as S  # noqa: E402  (stats(): the E1 guard metrics, unchanged)

TARGETS = ["ppp_rimi", "ppp_kasumi", "ro_ako_maou", "mygo_tomori", "mygo_soyo_tsuki", "ras_lock", "ro_rinko", "pp_eve",
           "hhw_kanon", "ras_chuchu", "mujica_sakiko_tsuki", "ras_masking", "hhw_hagumi", "pp_aya", "hhw_kaoru"]
ZH_NON_TARGET = ("zh_ser", "zh_lid", "zh_ser_voice_max")


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summary(rows: list[dict]) -> dict:
    s = S.stats(rows)
    non = S.stats([r for r in rows if not (r["lang"] == "zh" and r["row"] in TARGETS)])
    s.update({k: non[k] for k in ZH_NON_TARGET})
    zh = [r for r in rows if r["lang"] == "zh"]
    leak = collections.defaultdict(list)
    for r in zh:
        leak[r["row"]].append(r["pred"] != r["row"])
    lv = {k: sum(v) / len(v) for k, v in leak.items()}
    tz = [r for r in zh if r["row"] in TARGETS]
    for r in tz:                                   # the SER rows e1_select.stats counts
        r["ser_ok"] = r["ser_shared"] if r.get("ser_shared") is not None and r.get("m23_valid", True) else None
    by = lambda key, rs: {k: round(sum(v) / len(v), 3) for k, v in   # noqa: E731
                          ((k, [r[key] for r in rs if r["row"] == k and r.get(key) is not None]) for k in TARGETS) if v}
    en = [r["lid_ratio"] for r in rows if r["lang"] == "en" and r.get("lid_ratio") is not None]
    en_t = [r["lid_ratio"] for r in rows if r["lang"] == "en" and r["row"] in TARGETS and r.get("lid_ratio") is not None]
    return {**s, "zh_leak": sum(r["pred"] != r["row"] for r in zh) / len(zh),
            "voices_zh_leak_le_0.15": sum(v <= 0.15 for v in lv.values()),
            "target_zh_leak": sum(r["pred"] != r["row"] for r in tz) / len(tz),
            "target_voice_zh_leak": {k: round(lv[k], 3) for k in TARGETS},
            "target_voice_zh_lid": by("lid_ratio", tz), "target_voice_zh_ser": by("ser_ok", tz),
            "en_lid": sum(en) / len(en), "en_lid_targets": sum(en_t) / len(en_t)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    floor_rows = read(E1R / "floor" / "scored.jsonl")
    floor = S.stats(floor_rows)
    ref = summary(read(E1R / "E1b_ep2" / "scored.jsonl"))
    out = {"E1b_ep2": {k: v for k, v in ref.items() if k != "voice_ja_leak"}, "candidates": {}}
    for c in ("E5_ep1", "E5_ep2"):
        p = E5R / "eval" / c / "scored.jsonl"
        if not p.exists():
            continue
        s = summary(read(p))
        worse = {k: round(v - floor["voice_ja_leak"].get(k, 0), 3) for k, v in s["voice_ja_leak"].items()
                 if v - floor["voice_ja_leak"].get(k, 0) > S.GUARDS["voice_ja_leak_over_floor"]}
        checks = {"zh_ser_non_target": s["zh_ser"] <= S.GUARDS["zh_ser"], "zh_lid_non_target": s["zh_lid"] <= S.GUARDS["zh_lid"],
                  "zh_ser_voice_max_non_target": s["zh_ser_voice_max"] <= S.GUARDS["zh_ser_voice_max"],
                  "ja_cer": s["ja_cer"] <= S.GUARDS["ja_cer"], "en_wer": s["en_wer"] <= S.GUARDS["en_wer"],
                  "runaway_share": s["runaway_share"] <= S.GUARDS["runaway_share"], "no_voice_worse": not worse,
                  "ja_leak_le_0.03": s["ja_leak"] <= 0.03, "en_lid_no_regression": s["en_lid"] <= -9.3}
        out["candidates"][c] = {**{k: v for k, v in s.items() if k != "voice_ja_leak"}, "checks": checks,
                                "eligible": all(checks.values()), "voices_worse_than_floor": worse}
    elig = [c for c, v in out["candidates"].items() if v["eligible"]]
    pick = min(elig, key=lambda c: out["candidates"][c]["zh_leak"]) if elig else None
    v = out["candidates"].get(pick, {})
    metric_ok = {"zh_leak_le_0.08": v.get("zh_leak", 1) <= 0.08, "voices_le_0.15_ge_43": v.get("voices_zh_leak_le_0.15", 0) >= 43,
                 "target_zh_leak_le_0.15": v.get("target_zh_leak", 1) <= 0.15} if pick else {}
    out["pick"], out["adopt_metric_conditions"] = pick, metric_ok
    out["adopt_over_E1b_ep2"] = ("pending the user's target listening (>= 12/15 native)" if metric_ok and all(metric_ok.values())
                                 else False)
    E5R.mkdir(parents=True, exist_ok=True)
    (E5R / "selection_e5.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    lines = ["| cond | ja leak | ja CER | zh SER (34) | zh lid (34) | zh leak | voices zh leak ≤ 0.15 | target zh leak | en lid (targets) | en WER | eligible |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in [("E1b_ep2", out["E1b_ep2"])] + list(out["candidates"].items()):
        lines.append(f"| {name} | {f(s['ja_leak'])} | {f(s['ja_cer'])} | {f(s['zh_ser'])} | {f(s['zh_lid'])} | {f(s['zh_leak'])} | "
                     f"{s['voices_zh_leak_le_0.15']}/49 | {f(s['target_zh_leak'])} | {f(s['en_lid'])} ({f(s['en_lid_targets'])}) | "
                     f"{f(s['en_wer'])} | {s.get('eligible', '')} |")
    lines.append(f"\npick: {pick}; metric conditions: {metric_ok}; adopt: {out['adopt_over_E1b_ep2']}")
    (E5R / "selection_e5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for c, s in out["candidates"].items():
        print(c, "failed:", [k for k, ok in s["checks"].items() if not ok], "worse voices:", s["voices_worse_than_floor"])
        print("  target zh leak:", s["target_voice_zh_leak"])
    print("E1b_ep2 target zh leak:", out["E1b_ep2"]["target_voice_zh_leak"])


if __name__ == "__main__":
    main()
