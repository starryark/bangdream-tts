"""E8 (pod or local): catastrophe screens for one scored condition (E8_PLAN.md §5).

  python e8/e8_screens.py COND          # reads eval/COND/scored.jsonl and eval/C0/scored.jsonl; writes eval/COND/screens.json
  exit status 0 = all screens pass, 2 = a screen failed

Metrics are e6_select.metrics (pair rule, CER definition, leak = pred != row). Our Notes ja-mid leak uses every
render (E7's extra seed bases included); everything else uses E6's seed bases (6000, 11000), matching C0.
Real-audio anchors from E6's `real` condition (codec resynth): Our Notes ja-mid leak 0.027, Our Notes ja CER 0.042.
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
EXEMPT = {"mygo_raana"}


def screens(rows: list[dict], c0_rows: list[dict], sets: dict) -> dict:
    on, v1 = set(sets["on"]), set(sets["v1"])
    b = [r for r in rows if r.get("seed_base") in BASE_SEEDS]
    c0 = [r for r in c0_rows if r.get("seed_base") in BASE_SEEDS]
    m_on_all = S.metrics(rows, on, pairs_ok=True)
    m_on = S.metrics(b, on, pairs_ok=True)
    m_v1, c_v1 = S.metrics(b, v1), S.metrics(c0, v1)
    m_all = S.metrics(b, on | v1)
    worst = {k: round(p["ja_mid_leak"], 3) for k, p in m_on_all["per_row"].items()
             if k not in EXEMPT and p["ja_mid_leak"] is not None and p["ja_mid_leak"] > 0.40}
    le = S.le
    chk = {
        "S1_on_ja_mid_leak": (round(m_on_all["ja_mid_leak"], 4), "<= 0.06", le(m_on_all["ja_mid_leak"], 0.06)),
        "S2_on_voice_ja_mid_leak": (worst, "no voice > 0.40 (mygo_raana reported)", not worst),
        "S3_v1_ja_mid_leak": (round(m_v1["ja_mid_leak"], 4), f"<= C0 {c_v1['ja_mid_leak']:.4f} + 0.03", le(m_v1["ja_mid_leak"], c_v1["ja_mid_leak"] + 0.03)),
        "S3_v1_zh_leak": (round(m_v1["zh_leak"], 4), f"<= C0 {c_v1['zh_leak']:.4f} + 0.10", le(m_v1["zh_leak"], c_v1["zh_leak"] + 0.10)),
        "S4_v1_ja_cer": (round(m_v1["ja_cer"], 4), f"<= C0 {c_v1['ja_cer']:.4f} + 0.01", le(m_v1["ja_cer"], c_v1["ja_cer"] + 0.01)),
        "S4_on_ja_cer": (round(m_on["ja_cer"], 4), f"<= real {REAL_ON_CER} + 0.02", le(m_on["ja_cer"], REAL_ON_CER + 0.02)),
        "S5_v1_zh_ser": (round(m_v1["zh_ser"], 4), f"<= C0 {c_v1['zh_ser']:.4f} + 0.02", le(m_v1["zh_ser"], c_v1["zh_ser"] + 0.02)),
        "S5_v1_zh_lid": (round(m_v1["zh_lid"], 3), f"<= C0 {c_v1['zh_lid']:.3f} + 2.0", le(m_v1["zh_lid"], c_v1["zh_lid"] + 2.0)),
        "S5_v1_en_lid": (round(m_v1["en_lid"], 3), f"<= C0 {c_v1['en_lid']:.3f} + 2.0", le(m_v1["en_lid"], c_v1["en_lid"] + 2.0)),
        "S6_runaway": (round(m_all["runaway"], 4), "<= 0.01", le(m_all["runaway"], 0.01)),
    }
    per_voice = {k: round(p["ja_mid_leak"], 3) for k, p in sorted(m_on_all["per_row"].items()) if p["ja_mid_leak"] is not None}
    return {"pass": all(v[2] for v in chk.values()), "checks": {k: {"value": v[0], "bound": v[1], "ok": v[2]} for k, v in chk.items()},
            "on_ja_mid_leak_per_voice_all_seeds": per_voice}


def main() -> None:
    cond = sys.argv[1]
    E = ROOT / "eval"
    sel = json.loads((ROOT / "bundle" / "rows_e6.json").read_text())
    sets = {"v1": sel["v1_kept"], "on": sel["replaced"] + sel["new"]}
    out = screens(S.read(E / cond / "scored.jsonl"), S.read(E / "C0" / "scored.jsonl"), sets)
    (E / cond / "screens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in out["checks"].items():
        print(f"[screens] {cond} {k}: {v['value']} ({v['bound']}) {'ok' if v['ok'] else 'FAIL'}")
    print(f"[screens] {cond}: {'PASS' if out['pass'] else 'FAIL'}")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
