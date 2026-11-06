"""E2 selection (E2_PLAN.md "Selection rule"), run locally on copied results.

  python -I e2_select.py

Reads e1/results/eval/{floor,real,E1b_ep2} and e2/results/eval/{E2_ep1,E2_ep2}; writes
e2/results/selection_e2.{json,md}.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

E1R = Path(__file__).resolve().parents[1] / "e1" / "results" / "eval"
E2R = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1"))
import e1_select as S  # noqa: E402  (stats(): the E1 guard metrics, unchanged)

PAIRS = S.PAIRS


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def extra(rows: list[dict], problem: list[str], floor_rows: list[dict] | None = None) -> dict:
    zh = [r for r in rows if r["lang"] == "zh"]
    en = [r for r in rows if r["lang"] == "en"]
    by = collections.defaultdict(list)
    for r in zh:
        by[r["row"]].append(r["pred"] != r["row"])
    leak_v = {k: sum(v) / len(v) for k, v in by.items()}
    en_lid = collections.defaultdict(list)
    for r in en:
        if r.get("lid_ratio") is not None:
            en_lid[r["row"]].append(r["lid_ratio"])
    def m(ks):
        v = [x for k in ks for x in en_lid.get(k, [])]
        return sum(v) / len(v) if v else None
    zl = collections.defaultdict(list)
    for r in zh:
        if r.get("lid_ratio") is not None:
            zl[r["row"]].append(r["lid_ratio"])
    return {"zh_leak": sum(r["pred"] != r["row"] for r in zh) / len(zh),
            "voices_zh_leak_le_0.15": sum(v <= 0.15 for v in leak_v.values()),
            "zh_leak_problem": {k: round(leak_v[k], 3) for k in problem},
            "zh_lid_problem": {k: round(sum(zl[k]) / len(zl[k]), 2) for k in problem},
            "en_lid": m(list(en_lid)), "en_lid_problem": m(problem), "en_lid_others": m([k for k in en_lid if k not in problem])}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bl = json.loads((Path(__file__).resolve().parents[1] / "e1" / "bundle" / "e2_baselines.json").read_text(encoding="utf-8"))
    problem = bl["problem_voices"]
    floor_rows = read(E1R / "floor" / "scored.jsonl")
    floor = S.stats(floor_rows)
    ref_rows = read(E1R / "E1b_ep2" / "scored.jsonl")
    ref = {**S.stats(ref_rows), **extra(ref_rows, problem)}
    out = {"floor": {**{k: v for k, v in floor.items() if k != "voice_ja_leak"}, **extra(floor_rows, problem)},
           "E1b_ep2": {k: v for k, v in ref.items() if k != "voice_ja_leak"}, "candidates": {}}
    for c in ("E2_ep1", "E2_ep2"):
        p = E2R / "eval" / c / "scored.jsonl"
        if not p.exists():
            continue
        rows = read(p)
        s = {**S.stats(rows), **extra(rows, problem)}
        worse = {k: round(v - floor["voice_ja_leak"].get(k, 0), 3) for k, v in s["voice_ja_leak"].items()
                 if v - floor["voice_ja_leak"].get(k, 0) > S.GUARDS["voice_ja_leak_over_floor"]}
        checks = {"zh_ser": s["zh_ser"] <= S.GUARDS["zh_ser"], "zh_lid": s["zh_lid"] <= S.GUARDS["zh_lid"],
                  "zh_ser_voice_max": s["zh_ser_voice_max"] <= S.GUARDS["zh_ser_voice_max"],
                  "ja_cer": s["ja_cer"] <= S.GUARDS["ja_cer"], "en_wer": s["en_wer"] <= S.GUARDS["en_wer"],
                  "runaway_share": s["runaway_share"] <= S.GUARDS["runaway_share"], "no_voice_worse": not worse,
                  "ja_leak_le_0.03": s["ja_leak"] <= 0.03, "en_lid_no_regression": s["en_lid"] <= -9.3}
        conf = collections.Counter((r["row"], r["pred"]) for r in rows if r["lang"] == "ja" and r["pred"] != r["row"])
        out["candidates"][c] = {**{k: v for k, v in s.items() if k != "voice_ja_leak"}, "checks": checks,
                                "eligible": all(checks.values()), "voices_worse_than_floor": worse,
                                "pair_confusions_ja": {f"{a}/{b}": conf[(a, b)] + conf[(b, a)] for a, b in PAIRS}}
    elig = [c for c, v in out["candidates"].items() if v["eligible"]]
    pick = min(elig, key=lambda c: out["candidates"][c]["zh_leak"]) if elig else None
    adopt = bool(pick) and out["candidates"][pick]["zh_leak"] <= 0.10 and out["candidates"][pick]["voices_zh_leak_le_0.15"] >= 40
    out["pick"], out["adopt_over_E1b_ep2"] = pick, adopt
    (E2R / "selection_e2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    lines = ["| cond | ja leak | ja p_true | ja CER | zh SER | zh lid | zh leak | voices zh leak ≤ 0.15 | en lid (problem / others) | en WER | eligible |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in [("floor", out["floor"]), ("E1b_ep2", out["E1b_ep2"])] + list(out["candidates"].items()):
        lines.append(f"| {name} | {f(s['ja_leak'])} | {f(s['ja_p_true'])} | {f(s['ja_cer'])} | {f(s['zh_ser'])} | {f(s['zh_lid'])} | "
                     f"{f(s['zh_leak'])} | {s['voices_zh_leak_le_0.15']}/49 | {f(s['en_lid'])} ({f(s['en_lid_problem'])} / {f(s['en_lid_others'])}) | "
                     f"{f(s['en_wer'])} | {s.get('eligible', '')} |")
    lines.append(f"\npick: {pick}; adopt over E1b_ep2: {adopt}")
    (E2R / "selection_e2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for c, v in out["candidates"].items():
        print(c, "failed:", [k for k, ok in v["checks"].items() if not ok])
        print("  zh leak problem voices:", v["zh_leak_problem"])
        print("  persona/pairs:", v["pair_confusions_ja"])
    print("E1b_ep2 zh leak problem voices:", out["E1b_ep2"]["zh_leak_problem"])


if __name__ == "__main__":
    main()
