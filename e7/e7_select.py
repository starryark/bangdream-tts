"""E7 (pod or local): apply E7_PLAN.md's pre-registered rule to the scored conditions.

  python e7/e7_select.py            # reads eval/<cond>/scored.jsonl for C0, E7a_ep1/2, E7d_ep1/2

C0 is E1b_ep2 as scored in E6 (same checkpoint, probe, bundle, texts and seed bases; uploaded, not re-rendered).
Guards are E6's, unchanged (e6/e6_select.py: metrics, guards, ACCENT, the G3 exemption for mygo_raana and the
Mutsumi<->Mortis / Sakiko<->Sakiko-tsuki pair rule). The one difference: G3's ja-mid leak checks (pooled and per
row) use every render of the Our Notes rows, including E7's extra seed bases (16000, 21000; 80 per row). Every
other metric uses only E6's seed bases (6000, 11000), so G1, G2 and the C0 comparison are on E6's footing.

Rule (E7_PLAN.md section 5): the first E7a candidate (ep1, then ep2) passing G1-G3 is released. If none passes but
one passes G3 and fails only accent bounds (E6's set), E7d runs and is judged the same way in E7a's place.
Otherwise v2 is not adopted and v1 stays. Writes eval/selection.json and eval/selection.md; "run_E7d" is true
when the trigger fires and E7d is not scored yet (the queue then runs E7d and calls this again).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e6"))
import e6_select as S  # noqa: E402

BASE_SEEDS = {6000, 11000}
CONDS = ("C0", "E7a_ep1", "E7a_ep2", "E7d_ep1", "E7d_ep2")


def evaluate(scored: dict[str, list[dict]], sets: dict) -> dict:
    on, v1 = set(sets["on"]), set(sets["v1"])
    allr = on | v1
    base_routed = on - set(S.VC4)

    def M(rows):
        b = [r for r in rows if r.get("seed_base") in BASE_SEEDS]
        return {"v1": S.metrics(b, v1), "all": S.metrics(b, allr), "on": S.metrics(b, on, pairs_ok=True),
                "on_all_seeds": S.metrics(rows, on, pairs_ok=True),
                "base_routed": S.metrics(b, base_routed, pairs_ok=True), "vc4": S.metrics(b, set(S.VC4), pairs_ok=True),
                "not_vc4": S.metrics(b, allr - set(S.VC4), pairs_ok=True),
                "replaced": S.metrics(b, set(sets["replaced"]), pairs_ok=True)}
    m = {c: M(rows) for c, rows in scored.items()}
    c0 = m["C0"]
    res = {}
    for c in m:
        if c == "C0":
            continue
        ch = S.guards({**m[c], "on": m[c]["on_all_seeds"]}, c0, sets)   # G3 ja-mid checks on every seed
        failed = [k for k, ok in ch.items() if not ok]
        res[c] = {"checks": ch, "failed": failed, "pass": not failed,
                  "g3_pass": all(ok for k, ok in ch.items() if k.startswith("G3")),
                  "accent_only": bool(failed) and set(failed) <= S.ACCENT}
    order = lambda arm: [f"{arm}_ep{i}" for i in (1, 2) if f"{arm}_ep{i}" in res]   # noqa: E731
    out = {"candidates": res, "run_E7d": False, "e7d_trigger": False, "release": None, "notes": []}
    pick = next((c for c in order("E7a") if res[c]["pass"]), None)
    if pick is None:
        trig = any(res[c]["g3_pass"] and res[c]["accent_only"] for c in order("E7a"))
        out["e7d_trigger"] = trig
        if trig and not order("E7d"):
            out["run_E7d"] = True
            out["notes"].append("E7a: no candidate passes; a candidate passes G3 and fails only accent bounds -> run E7d")
            return {**out, "metrics": m}
        if trig:
            pick = next((c for c in order("E7d") if res[c]["pass"]), None)
            out["notes"].append(f"E7d judged in E7a's place: {pick or 'none passes'}")
    if pick is None:
        out["notes"].append("no candidate passes: v2 not adopted, v1 stays")
        return {**out, "metrics": m}
    out["release"] = pick
    c1 = m[pick]["replaced"]["ja_short_leak"], c0["replaced"]["ja_short_leak"]
    out["C1_short_line_claim"] = {"release": c1[0], "C0": c1[1], "holds": S.le(c1[0], c1[1] - 0.10)}
    return {**out, "metrics": m}


def main() -> None:
    E = ROOT / "eval"
    sel = json.loads((ROOT / "bundle" / "rows_e6.json").read_text())
    sets = {"v1": sel["v1_kept"], "on": sel["replaced"] + sel["new"], "replaced": sel["replaced"]}
    scored = {c: S.read(E / c / "scored.jsonl") for c in CONDS if (E / c / "summary.json").exists()}
    if "C0" not in scored:
        raise SystemExit("C0 (E1b_ep2, from E6) missing")
    out = evaluate(scored, sets)
    (E / "selection.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
    lines = ["| cond | v1 ja-mid | v1 ja-short | v1 zh leak | v1 zh SER | v1 zh lid | v1 en lid | all ja CER | ON ja-mid (all seeds) | ON zh (base-routed) | VC4 zh | replaced ja-short | pass | failed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, mm in out["metrics"].items():
        r = out["candidates"].get(c, {})
        lines.append(f"| {c} | {f(mm['v1']['ja_mid_leak'])} | {f(mm['v1']['ja_short_leak'])} | {f(mm['v1']['zh_leak'])} | {f(mm['v1']['zh_ser'])} | "
                     f"{f(mm['v1']['zh_lid'])} | {f(mm['v1']['en_lid'])} | {f(mm['all']['ja_cer'])} | {f(mm['on_all_seeds']['ja_mid_leak'])} | "
                     f"{f(mm['base_routed']['zh_leak'])} | {f(mm['vc4']['zh_leak'])} | {f(mm['replaced']['ja_short_leak'])} | "
                     f"{r.get('pass', '')} | {', '.join(r.get('failed', []))} |")
    for c, r in out["candidates"].items():
        bad = {k: round(p["ja_mid_leak"], 3) for k, p in out["metrics"][c]["on_all_seeds"]["per_row"].items()
               if k not in S.EXEMPT_G3 and p["ja_mid_leak"] is not None and p["ja_mid_leak"] > 0.20}
        if bad:
            lines.append(f"- {c}: G3 per-row over 0.20 (all seeds): {bad}")
    lines.append(f"\nrelease: {out['release']}  e7d_trigger: {out['e7d_trigger']}  run_E7d: {out['run_E7d']}")
    lines += [f"- {n}" for n in out["notes"]]
    if "C1_short_line_claim" in out:
        lines.append(f"- C1 short-line claim: {out['C1_short_line_claim']}")
    (E / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"run_E7d": out["run_E7d"], "release": out["release"]}))


if __name__ == "__main__":
    main()
