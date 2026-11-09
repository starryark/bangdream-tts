"""E9 (pod or local): catastrophe screens for one scored condition (E9_PLAN.md §5).

  python e9/e9_screens.py COND        # reads eval/COND/scored.jsonl and eval/C0/scored.jsonl; writes eval/COND/screens.json
  exit status 0 = all screens pass, 2 = a screen failed

E8's screens, adapted to E9's voices (fixed before the run):
- probeE6's classes for the MyGO!!!!! / Ave Mujica characters were fitted on Our Notes clips. So the probe-based
  identity screens S1/S2 cover the 16 new voices and the Our-Notes-era ids (`X_ournotes` is scored against probe
  class X; Sakiko-tsuki predictions count as Sakiko, Soyo-tsuki as Soyo), plus the two single-id characters
  (mygo_raana, mujica_nyamu).
- The 9 v1-era ids of two-era characters are NOT gated by the probe (for them it measures "how Our-Notes-like");
  the share of their renders the probe assigns to their own character (either era) is reported, and listening
  judges them against v1.
- S3-S6 as in E8, on the 36 unchanged voices; C0 comparisons are pooled, not seed-paired (the eval layout changes).
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
REAL_ON_CER = 0.042
PERSONA = {"mygo_soyo_tsuki": "mygo_soyo", "mujica_sakiko_tsuki": "mujica_sakiko"}
OURN = "_ournotes"


def probe_name(row: str) -> str:
    return row[: -len(OURN)] if row.endswith(OURN) else row


def screens(rows: list[dict], c0_rows: list[dict], sets: dict) -> dict:
    new, gated_on, v1, v1_era = set(sets["new"]), set(sets["replaced"]), set(sets["v1_kept"]), set(sets.get("v1_era", []))
    gated = new | gated_on
    # rows of gated ids, renamed to probe classes; persona predictions folded
    g = [{**r, "row": probe_name(r["row"]), "pred": PERSONA.get(r["pred"], r["pred"])} for r in rows if r["row"] in gated]
    gnames = {probe_name(n) for n in gated}
    b = [r for r in rows if r.get("seed_base") in BASE_SEEDS]
    c0 = [r for r in c0_rows if r.get("seed_base") in BASE_SEEDS]
    m_g = S.metrics(g, gnames, pairs_ok=True)
    m_on_cer = S.metrics([r for r in b if r["row"] in gated], gated)
    m_v1, c_v1 = S.metrics(b, v1), S.metrics(c0, v1)
    m_all = S.metrics(b, gated | v1 | v1_era)
    worst = {k: round(p["ja_mid_leak"], 3) for k, p in m_g["per_row"].items() if p["ja_mid_leak"] is not None and p["ja_mid_leak"] > 0.40}
    le = S.le
    chk = {
        "S1_gated_ja_mid_leak": (round(m_g["ja_mid_leak"], 4), "<= 0.06 (new + Our-Notes-era + single ids, all seeds)", le(m_g["ja_mid_leak"], 0.06)),
        "S2_gated_voice_ja_mid_leak": (worst, "no gated voice > 0.40", not worst),
        "S3_v1_ja_mid_leak": (round(m_v1["ja_mid_leak"], 4), f"<= C0 {c_v1['ja_mid_leak']:.4f} + 0.03", le(m_v1["ja_mid_leak"], c_v1["ja_mid_leak"] + 0.03)),
        "S3_v1_zh_leak": (round(m_v1["zh_leak"], 4), f"<= C0 {c_v1['zh_leak']:.4f} + 0.10", le(m_v1["zh_leak"], c_v1["zh_leak"] + 0.10)),
        "S4_v1_ja_cer": (round(m_v1["ja_cer"], 4), f"<= C0 {c_v1['ja_cer']:.4f} + 0.01", le(m_v1["ja_cer"], c_v1["ja_cer"] + 0.01)),
        "S4_on_ja_cer": (round(m_on_cer["ja_cer"], 4), f"<= real {REAL_ON_CER} + 0.02", le(m_on_cer["ja_cer"], REAL_ON_CER + 0.02)),
        "S5_v1_zh_ser": (round(m_v1["zh_ser"], 4), f"<= C0 {c_v1['zh_ser']:.4f} + 0.02", le(m_v1["zh_ser"], c_v1["zh_ser"] + 0.02)),
        "S5_v1_zh_lid": (round(m_v1["zh_lid"], 3), f"<= C0 {c_v1['zh_lid']:.3f} + 2.0", le(m_v1["zh_lid"], c_v1["zh_lid"] + 2.0)),
        "S5_v1_en_lid": (round(m_v1["en_lid"], 3), f"<= C0 {c_v1['en_lid']:.3f} + 2.0", le(m_v1["en_lid"], c_v1["en_lid"] + 2.0)),
        "S6_runaway": (round(m_all["runaway"], 4), "<= 0.01", le(m_all["runaway"], 0.01)),
    }
    v1e = {}
    for n in sorted(v1_era):
        rs = [r for r in rows if r["row"] == n and r["kind"] in ("ja_heldout", "ja_novel")]
        if rs:
            v1e[n] = round(sum(PERSONA.get(r["pred"], r["pred"]) == n for r in rs) / len(rs), 3)
    return {"pass": all(v[2] for v in chk.values()), "checks": {k: {"value": v[0], "bound": v[1], "ok": v[2]} for k, v in chk.items()},
            "gated_ja_mid_leak_per_voice_all_seeds": {k: round(p["ja_mid_leak"], 3) for k, p in sorted(m_g["per_row"].items()) if p["ja_mid_leak"] is not None},
            "v1_era_share_assigned_to_own_character_reported": v1e}


def main() -> None:
    cond = sys.argv[1]
    E = ROOT / "eval"
    sel = json.loads((ROOT / "bundle" / "rows_e6.json").read_text())
    out = screens(S.read(E / cond / "scored.jsonl"), S.read(E / "C0" / "scored.jsonl"), sel)
    (E / cond / "screens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in out["checks"].items():
        print(f"[screens] {cond} {k}: {v['value']} ({v['bound']}) {'ok' if v['ok'] else 'FAIL'}")
    print(f"[screens] {cond}: {'PASS' if out['pass'] else 'FAIL'}")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
