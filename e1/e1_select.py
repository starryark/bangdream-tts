"""E1 (pod or local): apply the pre-registered selection rule (E1_PLAN.md) to scored conditions.

  python e1_select.py --cands E1a_ep1,E1a_ep2,E1a_ep3,E1b_ep1,E1b_ep2,E1b_ep3

Reads eval/<cond>/scored.jsonl for the candidates, eval/floor and eval/real. Prints and writes
eval/selection.json + eval/selection.md: pooled metrics, guards, the per-voice Japanese leak
(floor vs candidate), the named-pair confusions, and the pick (or "none eligible").
Pooled CIs bootstrap over voices (renders of one voice are not independent).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BD_POD", "/workspace/bd"))
PAIRS = [("pp_eve", "ras_lock"), ("ppp_rimi", "ras_lock"), ("hhw_hagumi", "ro_ako"), ("ag_himari", "ppp_kasumi"),
         ("mujica_sakiko", "mujica_sakiko_tsuki"), ("mygo_soyo", "mygo_soyo_tsuki"), ("ro_ako", "ro_ako_maou"),
         ("mujica_mutsumi", "mujica_mortis"), ("mujica_uika", "mujica_umiri")]
GUARDS = {"zh_ser": 0.087, "zh_lid": -18.0, "zh_ser_voice_max": 0.15, "ja_cer": 0.035, "en_wer": 0.02,
          "runaway_share": 0.01, "voice_ja_leak_over_floor": 0.2}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def stats(rows: list[dict]) -> dict:
    import numpy as np
    ja = [r for r in rows if r["lang"] == "ja"]
    zh = [r for r in rows if r["lang"] == "zh"]
    en = [r for r in rows if r["lang"] == "en"]
    by = {}
    for r in ja:
        by.setdefault(r["row"], []).append(r["pred"] != r["row"])
    keys = sorted(by)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys))
        boot.append(sum(sum(by[keys[i]]) for i in pick) / sum(len(by[keys[i]]) for i in pick))
    zser = {}
    for r in zh:
        if r.get("ser_shared") is not None and r.get("m23_valid", True):
            zser.setdefault(r["row"], []).append(r["ser_shared"])
    return {"ja_leak": mean([r["pred"] != r["row"] for r in ja]),
            "ja_leak_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ja_p_true": mean([r["p_true"] for r in ja]),
            # the guard is on held-out lines (what E0's floor CER 0.031 / real 0.023 measured);
            # novel-sentence CER is reported alongside
            "ja_cer": mean([r["cer"] for r in ja if r.get("kind") == "ja_heldout"]),
            "ja_cer_novel": mean([r["cer"] for r in ja if r.get("kind") == "ja_novel"]),
            "zh_leak": mean([r["pred"] != r["row"] for r in zh]), "en_leak": mean([r["pred"] != r["row"] for r in en]),
            "zh_ser": mean([x for v in zser.values() for x in v]), "zh_ser_voice_max": max(mean(v) for v in zser.values()) if zser else None,
            "zh_lid": mean([r.get("lid_ratio") for r in zh]), "zh_cer": mean([r["cer"] for r in zh]),
            "en_wer": mean([r["cer"] for r in en]), "runaway_share": mean([bool(r.get("runaway")) for r in rows]),
            "voice_ja_leak": {k: sum(v) / len(v) for k, v in by.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    args = ap.parse_args()
    E = ROOT / "eval"
    floor = stats(read(E / "floor" / "scored.jsonl"))
    real = [r for r in read(E / "real" / "scored.jsonl") if r["cond"] == "real_resynth"]
    real_s = {"ja_leak": mean([r["pred"] != r["row"] for r in real]), "ja_p_true": mean([r["p_true"] for r in real])}
    real_raw = [r for r in read(E / "real" / "scored.jsonl") if r["cond"] == "real_raw"]
    real_s["ja_cer_raw"] = mean([r["cer"] for r in real_raw])
    res, lines = {}, []
    for c in args.cands.split(","):
        rows = read(E / c / "scored.jsonl")
        s = stats(rows)
        worse = {k: round(v - floor["voice_ja_leak"].get(k, 0), 3) for k, v in s["voice_ja_leak"].items()
                 if v - floor["voice_ja_leak"].get(k, 0) > GUARDS["voice_ja_leak_over_floor"]}
        checks = {"zh_ser": s["zh_ser"] is not None and s["zh_ser"] <= GUARDS["zh_ser"],
                  "zh_lid": s["zh_lid"] is not None and s["zh_lid"] <= GUARDS["zh_lid"],
                  "zh_ser_voice_max": s["zh_ser_voice_max"] is not None and s["zh_ser_voice_max"] <= GUARDS["zh_ser_voice_max"],
                  "ja_cer": s["ja_cer"] <= GUARDS["ja_cer"], "en_wer": s["en_wer"] <= GUARDS["en_wer"],
                  "runaway_share": s["runaway_share"] <= GUARDS["runaway_share"], "no_voice_worse": not worse}
        conf = Counter((r["row"], r["pred"]) for r in rows if r["lang"] == "ja" and r["pred"] != r["row"])
        pair = {f"{a}/{b}": conf[(a, b)] + conf[(b, a)] for a, b in PAIRS}
        muj_to_game = sum(v for (a, p), v in conf.items() if a.startswith("mujica_") and not p.startswith("mujica_"))
        res[c] = {**{k: v for k, v in s.items() if k != "voice_ja_leak"}, "checks": checks, "eligible": all(checks.values()),
                  "voices_worse_than_floor": worse, "pair_confusions_ja": pair, "mujica_to_game_ja": muj_to_game,
                  "voice_ja_leak": s["voice_ja_leak"]}
    elig = [c for c in res if res[c]["eligible"]]
    pick = min(elig, key=lambda c: (res[c]["ja_leak"], -res[c]["ja_p_true"])) if elig else None
    out = {"floor": {k: v for k, v in floor.items() if k != "voice_ja_leak"}, "real_resynth": real_s,
           "guards": GUARDS, "candidates": res, "pick": pick}
    (E / "selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    lines.append("| cond | ja leak (CI) | ja p_true | ja CER | zh SER | zh lid | zh leak | en WER | runaway | eligible |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in [("real (codec)", {**real_s}), ("floor", floor)] + list(res.items()):
        ci = s.get("ja_leak_ci")
        lines.append(f"| {name} | {f(s.get('ja_leak'))}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "") +
                     f" | {f(s.get('ja_p_true'))} | {f(s.get('ja_cer', s.get('ja_cer_raw')))} | {f(s.get('zh_ser'))} | "
                     f"{f(s.get('zh_lid'))} | {f(s.get('zh_leak'))} | {f(s.get('en_wer'))} | {f(s.get('runaway_share'))} | "
                     f"{s.get('eligible', '')} |")
    lines.append(f"\npick: {pick}")
    (E / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for c, s in res.items():
        print(c, "failed:", [k for k, v in s["checks"].items() if not v], "worse voices:", s["voices_worse_than_floor"],
              "pairs:", s["pair_confusions_ja"], "mujica->game:", s["mujica_to_game_ja"])


if __name__ == "__main__":
    main()
