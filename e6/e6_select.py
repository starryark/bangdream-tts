"""E6 (pod or local): apply the pre-registered selection rule (E6_PLAN.md §5-6) to scored conditions.

  python e6/e6_select.py            # reads eval/<cond>/scored.jsonl for every condition present

Conditions: C0 (E1b_ep2), E6a_ep1/2, E6b_ep1/2, E6c_ep1/2, E6d_ep1/2 (if run), real, floor_on.
Writes eval/selection.json and eval/selection.md. The pick is "release" (a condition name or None);
"run_E6d" is true when the E6d trigger fires and E6d has not been scored yet (the queue then runs E6d
and calls this again).

Leak = pred != row. In G3 and C1 the pair confusions Mutsumi<->Mortis and Sakiko<->Sakiko-tsuki are
not leak (they are reported). Pooled values are means over renders, as E1.
Accent bounds (the E6d trigger): G1 zh_ser, zh_lid, en_lid and G2 zh_ser, zh_lid, zh_ser_row_max
(the per-row form of G2's zh SER bound; pre-run clarification 2026-11-07).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
PAIRS = [("mujica_mutsumi", "mujica_mortis"), ("mujica_sakiko", "mujica_sakiko_tsuki")]
VC4 = ["mewtype_arale", "mewtype_nonoka", "millsage_hotaru", "ikka_yomogi"]
VC_V1 = ["hhw_hagumi", "hhw_kanon", "hhw_kaoru", "mygo_soyo_tsuki", "pp_aya", "pp_eve", "ppp_kasumi", "ppp_rimi",
         "ras_chuchu", "ras_lock", "ras_masking", "ro_ako_maou", "ro_rinko"]
ACCENT = {"G1.zh_ser", "G1.zh_lid", "G1.en_lid", "G2.zh_ser", "G2.zh_lid", "G2.zh_ser_row_max"}
EXEMPT_G3 = {"mygo_raana"}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def is_pair(row: str, pred: str) -> bool:
    return any({row, pred} == {a, b} for a, b in PAIRS)


def leak(r: dict, pairs_ok: bool) -> bool:
    return r["pred"] != r["row"] and not (pairs_ok and is_pair(r["row"], r["pred"]))


def metrics(rows: list[dict], keep: set[str], pairs_ok: bool = False) -> dict:
    rs = [r for r in rows if r["row"] in keep]
    mid = [r for r in rs if r["kind"] in ("ja_heldout", "ja_novel")]
    short = [r for r in rs if r["kind"] == "ja_short"]
    zh = [r for r in rs if r["kind"] == "zh"]
    en = [r for r in rs if r["kind"] == "en"]
    ser = lambda rr: [r["ser_shared"] for r in rr if r.get("ser_shared") is not None and r.get("m23_valid", True)]  # noqa: E731
    per = {}
    for k in sorted({r["row"] for r in rs}):
        m = [r for r in mid if r["row"] == k]
        z = [r for r in zh if r["row"] == k]
        per[k] = {"ja_mid_leak": mean([leak(r, pairs_ok) for r in m]), "zh_leak": mean([leak(r, pairs_ok) for r in z]),
                  "zh_ser": mean(ser(z)), "ja_short_leak": mean([leak(r, pairs_ok) for r in short if r["row"] == k])}
    return {"n": len(rs), "ja_mid_leak": mean([leak(r, pairs_ok) for r in mid]), "ja_short_leak": mean([leak(r, pairs_ok) for r in short]),
            "zh_leak": mean([leak(r, pairs_ok) for r in zh]), "zh_ser": mean(ser(zh)), "zh_lid": mean([r.get("lid_ratio") for r in zh]),
            "en_lid": mean([r.get("lid_ratio") for r in en]), "en_wer": mean([r["cer"] for r in en]),
            "ja_cer": mean([r["cer"] for r in rs if r["kind"] == "ja_heldout"]),
            "runaway": mean([bool(r.get("runaway")) for r in rs]), "per_row": per,
            "pair_confusions": sum(is_pair(r["row"], r["pred"]) for r in mid + short + zh)}


def le(a, b) -> bool:
    return a is not None and b is not None and a <= b + 1e-12


def guards(c: dict, c0: dict, sets: dict) -> dict:
    """c, c0: metrics dicts keyed by set name ('v1', 'all', 'on', 'base_routed')."""
    v, v0, a, on, br = c["v1"], c0["v1"], c["all"], c["on"], c["base_routed"]
    ch = {
        "G1.ja_mid_leak": le(v["ja_mid_leak"], v0["ja_mid_leak"] + 0.02),
        "G1.ja_short_leak": le(v["ja_short_leak"], v0["ja_short_leak"] + 0.05),
        "G1.zh_leak": le(v["zh_leak"], v0["zh_leak"] + 0.03),
        "G1.zh_ser": le(v["zh_ser"], v0["zh_ser"] + 0.010),
        "G1.zh_lid": le(v["zh_lid"], v0["zh_lid"] + 1.0),
        "G1.en_lid": le(v["en_lid"], v0["en_lid"] + 1.0),
        "G1.per_row": all(le(p["ja_mid_leak"], v0["per_row"][k]["ja_mid_leak"] + 0.20) and le(p["zh_leak"], v0["per_row"][k]["zh_leak"] + 0.20)
                          for k, p in v["per_row"].items()),
        "G2.zh_ser": le(a["zh_ser"], 0.087), "G2.zh_lid": le(a["zh_lid"], -18.0),
        "G2.zh_ser_row_max": all(le(p["zh_ser"], 0.15) for p in a["per_row"].values() if p["zh_ser"] is not None),
        "G2.ja_cer": le(a["ja_cer"], 0.035), "G2.en_wer": le(a["en_wer"], 0.02), "G2.runaway": le(a["runaway"], 0.01),
        "G3.ja_mid_leak": le(on["ja_mid_leak"], 0.05),
        "G3.per_row": all(le(p["ja_mid_leak"], 0.20) for k, p in on["per_row"].items() if k not in EXEMPT_G3),
        "G3.zh_leak_base_routed": le(br["zh_leak"], 0.15),
    }
    return ch


def evaluate(scored: dict[str, list[dict]], sets: dict) -> dict:
    """scored: {cond: rows}. sets: {'v1': [...37], 'on': [...28], 'replaced': [...12]}. Returns the decision record."""
    on, v1 = set(sets["on"]), set(sets["v1"])
    allr = on | v1
    base_routed = on - set(VC4)

    def M(rows):
        return {"v1": metrics(rows, v1), "all": metrics(rows, allr), "on": metrics(rows, on, pairs_ok=True),
                "base_routed": metrics(rows, base_routed, pairs_ok=True), "vc4": metrics(rows, set(VC4), pairs_ok=True),
                "not_vc4": metrics(rows, allr - set(VC4), pairs_ok=True),
                "replaced": metrics(rows, set(sets["replaced"]), pairs_ok=True)}
    m = {c: M(rows) for c, rows in scored.items() if c not in ("real", "floor_on")}
    c0 = m["C0"]
    res = {}
    for c in m:
        if c == "C0":
            continue
        ch = guards(m[c], c0, sets)
        failed = [k for k, ok in ch.items() if not ok]
        res[c] = {"checks": ch, "failed": failed, "pass": not failed,
                  "g3_pass": all(ok for k, ok in ch.items() if k.startswith("G3")),
                  "accent_only": bool(failed) and set(failed) <= ACCENT}
    order = lambda arm: [f"{arm}_ep{i}" for i in (1, 2) if f"{arm}_ep{i}" in res]   # noqa: E731
    out = {"candidates": res, "run_E6d": False, "base": None, "release": None, "notes": []}

    base = next((c for c in order("E6a") if res[c]["pass"]), None)
    if base is None:
        trig = any(res[c]["g3_pass"] and res[c]["accent_only"] for c in order("E6a"))
        out["e6d_trigger"] = trig
        if trig and not order("E6d"):
            out["run_E6d"] = True
            out["notes"].append("E6a: no candidate passes; a candidate passes G3 and fails only accent bounds -> run E6d")
            return {**out, "metrics": m}
        if trig:
            base = next((c for c in order("E6d") if res[c]["pass"]), None)
            out["notes"].append(f"E6d judged in E6a's place: {base or 'none passes'}")
    if base is None:
        out["notes"].append("no baseline candidate passes: v2 not adopted, v1 stays")
        return {**out, "metrics": m}
    out["base"] = base
    release = base

    # E6b over the baseline
    b_pick = None
    for c in order("E6b"):
        if not res[c]["pass"]:
            continue
        vc_c, vc_b = m[c]["vc4"]["zh_leak"], m[base]["vc4"]["zh_leak"]
        nv_c, nv_b = m[c]["not_vc4"]["zh_leak"], m[base]["not_vc4"]["zh_leak"]
        ok = le(vc_c, 0.30) and le(vc_c, vc_b - 0.20) and le(nv_c, nv_b + 0.02)
        res[c]["b_rule"] = {"vc4_zh_leak": vc_c, "base_vc4_zh_leak": vc_b, "other_zh_leak": nv_c, "base_other_zh_leak": nv_b, "ok": ok}
        if ok:
            b_pick = c
        break   # the first passing candidate is the one judged
    # E6c only if E6b is not adopted
    c_pick = None
    for c in order("E6c"):
        if not res[c]["pass"]:
            continue
        ep = c.rsplit("_", 1)[1]
        ref = f"{base.split('_')[0]}_{ep}"
        if ref not in m:
            res[c]["c_rule"] = {"ok": False, "why": f"{ref} not scored"}
            break
        nv = [r for r in scored[c] if r["kind"] == "nv"]
        kana_ok = mean([r.get("nv_kana", 99) <= 2 for r in nv])
        dur_ok = mean([r.get("dur_trim") is not None and 0.3 <= r["dur_trim"] <= 2.0 for r in nv])
        run = sum(bool(r.get("runaway")) for r in nv)
        ok = (le(m[c]["all"]["ja_mid_leak"], m[ref]["all"]["ja_mid_leak"] + 0.01) and le(m[c]["all"]["ja_cer"], m[ref]["all"]["ja_cer"] + 0.003)
              and kana_ok is not None and kana_ok >= 0.8 and dur_ok is not None and dur_ok >= 0.8 and run == 0)
        res[c]["c_rule"] = {"vs": ref, "nv_kana_le2": kana_ok, "nv_dur_ok": dur_ok, "nv_runaway": run, "ok": ok}
        if ok:
            c_pick = c
        break
    if b_pick:
        release = b_pick
        if c_pick:
            out["notes"].append(f"{c_pick} also passes; [nv] goes to a later E6b+[nv] run (no untested combination ships)")
    elif c_pick:
        release = c_pick
    out["release"] = release
    c1 = m[release]["replaced"]["ja_short_leak"], c0["replaced"]["ja_short_leak"]
    out["C1_short_line_claim"] = {"release": c1[0], "C0": c1[1], "holds": le(c1[0], c1[1] - 0.10)}
    return {**out, "metrics": m}


def main() -> None:
    E = ROOT / "eval"
    sel = json.loads((ROOT / "bundle" / "rows_e6.json").read_text())
    sets = {"v1": sel["v1_kept"], "on": sel["replaced"] + sel["new"], "replaced": sel["replaced"]}
    scored = {d.name: read(d / "scored.jsonl") for d in sorted(E.iterdir()) if (d / "summary.json").exists()}
    if "C0" not in scored:
        raise SystemExit("C0 (E1b_ep2) not scored")
    out = evaluate(scored, sets)
    (E / "selection.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
    lines = ["| cond | v1 ja-mid | v1 ja-short | v1 zh leak | v1 zh SER | v1 zh lid | v1 en lid | all ja CER | ON ja-mid | ON zh (base-routed) | VC4 zh | replaced ja-short | pass | failed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, mm in out["metrics"].items():
        r = out["candidates"].get(c, {})
        lines.append(f"| {c} | {f(mm['v1']['ja_mid_leak'])} | {f(mm['v1']['ja_short_leak'])} | {f(mm['v1']['zh_leak'])} | {f(mm['v1']['zh_ser'])} | "
                     f"{f(mm['v1']['zh_lid'])} | {f(mm['v1']['en_lid'])} | {f(mm['all']['ja_cer'])} | {f(mm['on']['ja_mid_leak'])} | "
                     f"{f(mm['base_routed']['zh_leak'])} | {f(mm['vc4']['zh_leak'])} | {f(mm['replaced']['ja_short_leak'])} | "
                     f"{r.get('pass', '')} | {', '.join(r.get('failed', []))} |")
    lines.append(f"\nbase: {out['base']}  release: {out['release']}  run_E6d: {out['run_E6d']}")
    lines += [f"- {n}" for n in out["notes"]]
    if "C1_short_line_claim" in out:
        lines.append(f"- C1 short-line claim: {out['C1_short_line_claim']}")
    (E / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"run_E6d": out["run_E6d"], "release": out["release"]}))


if __name__ == "__main__":
    main()
