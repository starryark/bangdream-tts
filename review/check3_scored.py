"""Read-only checks on e0/floor/*/scored.jsonl, e0/real, e0/resynth, floor_table.json. CPU only, no models."""
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
POD = Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402  (patches p03_build.READINGS; imports pyopenjtalk lazily)
import metrics  # noqa: E402
import accent_metrics as AM  # noqa: E402


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


floor = []
for d in sorted((BD / "e0" / "floor").iterdir()):
    if (d / "scored.jsonl").exists():
        floor += rj(d / "scored.jsonl")
real = rj(BD / "e0" / "real" / "scored.jsonl")
res = rj(BD / "e0" / "resynth" / "scored.jsonl")
print("floor", len(floor), collections.Counter(r["lang"] for r in floor), "real", len(real), "resynth", len(res))
print("pid unique within floor:", len({r["pid"] for r in floor}) == len(floor))
print("floor ja pids == real pids:", {r["pid"] for r in floor if r["lang"] == "ja"} == {r["pid"] for r in real})

# --- 5. READINGS patch visible through metrics' `from p03_build import reading`
import p03_build
print("p03_build.READINGS is BD table:", "有咲" in p03_build.READINGS and "紅莉栖" not in p03_build.READINGS)
print("metrics._norm_text('有咲','ja') ->", metrics._norm_text("有咲", "ja", "v3"))
# lower() before reading(): Latin keys in BD_READINGS never match
for s in ("Roselia", "MyGO", "Poppin'Party", "RAISE A SUILEN", "CiRCLE", "Chu²"):
    print(f"   {s!r}: C.reading -> {C.reading(s)}   metrics._norm_text -> {metrics._norm_text(s, 'ja', 'v3')}")

# Latin in ja ASR hypotheses (where the lower() bug bites)
LAT = re.compile(r"[A-Za-z]")
ja_all = [r for r in floor + real if r["lang"] == "ja"]
lat = [r for r in ja_all if LAT.search(r.get("asr", ""))]
print("ja rows (floor+real) whose ASR hyp has Latin:", len(lat), "of", len(ja_all))
for r in lat[:12]:
    ref = metrics._norm_text(r["text"], "ja", "v3")
    hyp_nolower = C.reading(r["asr"])
    print(f"   {r['cond']} {r['pid']}: cer {r['cer']} -> without lower(): {round(C.cer(ref, hyp_nolower), 4)} | {r['asr'][:50]}")
# recompute ja CER exactly as b05 for a sample, to confirm stored values
mism = 0
for r in ja_all[:200]:
    ref = metrics._norm_text(r["text"], "ja", "v3"); hyp = metrics._norm_text(r.get("asr", ""), "ja", "v3")
    mism += abs(metrics._err(ref, hyp, "ja") - r["cer"]) > 1e-3
print("ja CER recompute mismatches (first 200):", mism)

# --- 6. accent: m23_valid on the 16 zh texts; SER pooled vs gated ---------------------------------
zh = [r for r in floor if r["lang"] == "zh"]
valid = {}
for t in sorted({(r["tid"], r["text"]) for r in zh}):
    ref, info = AM.zh_reference(t[1])
    valid[t[0]] = bool(ref) and not info["has_latin"] and info["n_unknown"] == 0
print("zh texts:", len(valid), "m23 invalid:", [k for k, v in valid.items() if not v])
print("zh rows with shared_positions provided:", sum("shared_positions" in r for r in zh), "/", len(zh))
print("zh rows missing ser_shared / lid:", sum(r.get("ser_shared") is None for r in zh), sum(r.get("lid_ratio") is None for r in zh))
en = [r for r in floor if r["lang"] == "en"]
print("en rows with lid:", sum("lid_ratio" in r for r in en), "/", len(en))
ss = np.array([r["ser_shared"] for r in zh if r.get("ser_shared") is not None])
ss_nr = np.array([r["ser_shared"] for r in zh if r.get("ser_shared") is not None and not r["runaway"]])
print("zh ser_shared macro all %.4f, excl runaway %.4f, median %.4f" % (ss.mean(), ss_nr.mean(), np.median(ss)))
print("zh runaway:", [(r["pid"], r["dur"], r["ser_shared"]) for r in zh if r["runaway"]])
# runaway rule check: cap 400 frames @12.5 Hz = 32 s
print("max floor dur:", max(r["dur"] for r in floor), " #dur>=31:", sum(r["dur"] >= 31 for r in floor))

# --- render sample rates
srs = collections.Counter(sf.info(r["path"]).samplerate for r in floor[::25])
print("render sample rates (every 25th):", srs, " real:", collections.Counter(sf.info(r["path"]).samplerate for r in real[::25]))

# --- 7/9. leak per row: ja only vs all languages, with Wilson CIs --------------------------------
tab = json.loads((BD / "e0" / "floor_table.json").read_text(encoding="utf-8"))
mins = {k: v["min"] for k, v in json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))["per_row"].items()}
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in floor:
    by[r["row"]][r["lang"]].append(r["pred"] != r["row"])
rowsN = sorted(by, key=lambda k: mins[k])
print("\nrow | ja k/n (CI) | zh k/n | en k/n | all k/n (CI)")
for k in sorted(rowsN, key=lambda k: -np.mean(sum(by[k].values(), []))):
    a = sum(by[k].values(), [])
    if np.mean(a) < 0.2 and np.mean(by[k]["ja"]) < 0.2:
        continue
    j = by[k]["ja"]
    cj, ca = wilson(sum(j), len(j)), wilson(sum(a), len(a))
    print(f"{k:20s} ja {sum(j)}/{len(j)} ({cj[0]:.2f}-{cj[1]:.2f}) zh {sum(by[k]['zh'])}/{len(by[k]['zh'])} "
          f"en {sum(by[k]['en'])}/{len(by[k]['en'])} all {sum(a)}/{len(a)} ({ca[0]:.2f}-{ca[1]:.2f})")
B = [("<10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-60", 30, 99)]
print("\nbucket | rows | leak all-lang (b06) | ja-only row-mean | ja pooled k/n (CI) | zh pooled | en pooled")
for name, lo, hi in B:
    ks = [k for k in rowsN if lo <= mins[k] < hi]
    la = [np.mean(sum(by[k].values(), [])) for k in ks]
    lj = [np.mean(by[k]["ja"]) for k in ks]
    kj = sum(sum(by[k]["ja"]) for k in ks); nj = sum(len(by[k]["ja"]) for k in ks)
    kz = sum(sum(by[k]["zh"]) for k in ks); nz = sum(len(by[k]["zh"]) for k in ks)
    ke = sum(sum(by[k]["en"]) for k in ks); ne = sum(len(by[k]["en"]) for k in ks)
    c = wilson(kj, nj)
    print(f"{name:6s} {len(ks):2d}  {np.mean(la):.3f}  {np.mean(lj):.3f}  {kj}/{nj} ({c[0]:.2f}-{c[1]:.2f})  {kz}/{nz}={kz/nz:.3f}  {ke}/{ne}={ke/ne:.3f}")
# b06 leak recompute
for k in rowsN:
    t = tab[k]
    n_all = t["floor_ja"]["n"] + t["floor_zh"]["n"] + t["floor_en"]["n"]
    l06 = sum(t[f"floor_{l}"].get("leak", 0) * t[f"floor_{l}"]["n"] for l in ("ja", "zh", "en")) / n_all
    assert abs(l06 - np.mean(sum(by[k].values(), []))) < 0.01, k
print("b06 leak == exact all-language share for every row (within rounding): OK")
# Spearman rho minutes vs leak (ja-only and all) over rows
from scipy.stats import spearmanr
print("spearman(min, leak_all) =", spearmanr([mins[k] for k in rowsN], [np.mean(sum(by[k].values(), [])) for k in rowsN]))
print("spearman(min, leak_ja)  =", spearmanr([mins[k] for k in rowsN], [np.mean(by[k]["ja"]) for k in rowsN]))

# --- sink class analysis --------------------------------------------------------------------------
leaked = [r for r in floor if r["pred"] != r["row"]]
print("\nfloor leaked:", len(leaked), "dest:", collections.Counter(r["pred"] for r in leaked).most_common(5))
print("   ja leaked dest:", collections.Counter(r["pred"] for r in leaked if r["lang"] == "ja").most_common(5))
rl = [r for r in res if r["pred"] != r["row"]]
print("resynth leaked:", len(rl), "/", len(res), collections.Counter(r["pred"] for r in rl).most_common(5))
print("resynth p_true mean %.3f vs real %.3f; per-row worst drop:" % (np.mean([r["p_true"] for r in res]), np.mean([r["real_p_true"] for r in res])))
drop = collections.defaultdict(list)
for r in res:
    drop[r["row"]].append(r["real_p_true"] - r["p_true"])
print("   ", sorted(((k, round(float(np.mean(v)), 2)) for k, v in drop.items()), key=lambda t: -t[1])[:8])
# p_true of collapsed rows: real vs resynth vs floor
for k in ("ppp_kasumi", "ras_lock", "pp_eve", "ppp_rimi", "mujica_sakiko_tsuki"):
    print(f"   {k}: real p_true {np.mean([r['real_p_true'] for r in res if r['row']==k]):.2f} "
          f"resynth {np.mean([r['p_true'] for r in res if r['row']==k]):.2f} "
          f"floor ja {np.mean([r['p_true'] for r in floor if r['row']==k and r['lang']=='ja']):.2f}")
# stale fields in resynth
print("resynth rows whose 'sim' equals the real clip's sim (stale, not recomputed):",
      sum(1 for r, q in zip(res, real) if r["pid"] == q["pid"] and r["sim"] == q["sim"]), "/", len(res))

# --- sim bias: floor ja sim minus real sim per row
gap = {k: tab[k]["floor_ja"]["sim"] - tab[k]["real_ja"]["sim"] for k in rowsN}
print("\nrows with floor ja sim > real ja sim:", sorted(((k, round(v, 3)) for k, v in gap.items() if v > 0), key=lambda t: -t[1]))
# within-row: does sim track p_true across floor clips?
from scipy.stats import spearmanr as sp
fj = [r for r in floor if r["lang"] == "ja"]
print("floor ja: spearman(sim, p_true) =", sp([r["sim"] for r in fj], [r["p_true"] for r in fj]))
