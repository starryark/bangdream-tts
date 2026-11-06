"""(a) hand softmax vs sklearn predict_proba; (b) b03 report p_true (predict_proba) vs b05 p_true (hand softmax);
(c) codec-induced p_true drop vs floor leak per row; (d) row-cluster bootstrap of pooled ja leak. CPU only."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def nrm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# (a) small multinomial fit, same settings as b03
cv = np.load(BD / "vectors" / "clipvec.npz")
ids, V = list(cv["ids"]), cv["V"].astype(np.float64)
pr = np.load(BD / "vectors" / "probe.npz")
mu, classes = pr["mu"].astype(np.float64), list(pr["classes"])
lab = np.array([classes.index(i.rsplit("_", 1)[0]) for i in ids])
rng = np.random.default_rng(0)
sub_cls = rng.choice(len(classes), 12, replace=False)
m = np.isin(lab, sub_cls)
X, y = nrm(V[m] - mu), lab[m]
clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X, y)
P = clf.predict_proba(X)
Z = X @ clf.coef_.T + clf.intercept_
H = np.exp(Z - Z.max(1, keepdims=True)); H /= H.sum(1, keepdims=True)
print("sklearn", sklearn.__version__, "| max |predict_proba - hand softmax| =", float(np.abs(P - H).max()),
      "| n_iter", clf.n_iter_)

# (b) b03 report (predict_proba) vs b05 real scored (hand softmax), per row
rep = json.loads((BD / "e0" / "vectors_report.json").read_text(encoding="utf-8"))["probe_real_heldout"]["per_row"]
real = rj(BD / "e0" / "real" / "scored.jsonl")
b05 = collections.defaultdict(list)
for r in real:
    b05[r["row"]].append(r["p_true"])
d = [abs(rep[k]["p_true_mean"] - np.mean(v)) for k, v in b05.items()]
print("b03 p_true_mean (predict_proba) vs b05 mean p_true (hand softmax): max |diff| per row", round(max(d), 4))

# (c) codec-induced drop vs floor leak / floor p_true per row
res = rj(BD / "e0" / "resynth" / "scored.jsonl")
drop = collections.defaultdict(list)
for r in res:
    drop[r["row"]].append(r["real_p_true"] - r["p_true"])
floor = []
for dd in sorted((BD / "e0" / "floor").iterdir()):
    if (dd / "scored.jsonl").exists():
        floor += rj(dd / "scored.jsonl")
fl_ja = collections.defaultdict(list)
fl_all = collections.defaultdict(list)
fp_ja = collections.defaultdict(list)
for r in floor:
    fl_all[r["row"]].append(r["pred"] != r["row"])
    if r["lang"] == "ja":
        fl_ja[r["row"]].append(r["pred"] != r["row"])
        fp_ja[r["row"]].append(r["p_true"])
ks = sorted(drop)
dv = [np.mean(drop[k]) for k in ks]
print("spearman(codec p_true drop, floor ja leak) =", spearmanr(dv, [np.mean(fl_ja[k]) for k in ks]))
print("spearman(codec p_true drop, floor all-lang leak) =", spearmanr(dv, [np.mean(fl_all[k]) for k in ks]))
print("spearman(codec p_true drop, floor ja p_true) =", spearmanr(dv, [np.mean(fp_ja[k]) for k in ks]))
# resynth margins: how many real clips sit near the boundary? p_true distribution of resynth by row group
coll = [k for k in ks if np.mean(fl_ja[k]) >= 0.5]
print("collapsed rows (ja leak>=0.5):", coll, " mean codec drop %.3f vs others %.3f" % (
    np.mean([np.mean(drop[k]) for k in coll]), np.mean([np.mean(drop[k]) for k in ks if k not in coll])))
# resynth p_true for the 'collapsed' rows vs floor
for k in coll:
    print(f"   {k}: real {np.mean([r['real_p_true'] for r in res if r['row']==k]):.2f} -> resynth "
          f"{np.mean([r['p_true'] for r in res if r['row']==k]):.2f} -> floor ja {np.mean(fp_ja[k]):.2f}")

# (d) pooled ja leak: binomial CI vs row-cluster bootstrap
k_tot = sum(sum(v) for v in fl_ja.values()); n_tot = sum(len(v) for v in fl_ja.values())
rows = sorted(fl_ja)
B = []
for _ in range(20000):
    s = rng.choice(rows, len(rows), replace=True)
    B.append(sum(sum(fl_ja[k]) for k in s) / sum(len(fl_ja[k]) for k in s))
print(f"pooled ja leak {k_tot}/{n_tot} = {k_tot/n_tot:.3f}; row-cluster bootstrap 95% CI "
      f"{np.percentile(B, 2.5):.3f}-{np.percentile(B, 97.5):.3f}")
# same for the bucket difference <10 vs 30-60 in all-language leak (b06's headline), row bootstrap
mins = {k: v["min"] for k, v in json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))["per_row"].items()}
lo = [k for k in rows if mins[k] < 10]; hi = [k for k in rows if mins[k] >= 30]
diffs = []
for _ in range(20000):
    a = rng.choice(lo, len(lo)); b = rng.choice(hi, len(hi))
    diffs.append(np.mean([np.mean(fl_all[k]) for k in b]) - np.mean([np.mean(fl_all[k]) for k in a]))
print("all-lang leak, 30-60 minus <10 bucket (row means): %.3f, bootstrap 95%% CI %.3f..%.3f" % (
    np.mean([np.mean(fl_all[k]) for k in hi]) - np.mean([np.mean(fl_all[k]) for k in lo]),
    np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)))
diffs = []
for _ in range(20000):
    a = rng.choice(lo, len(lo)); b = rng.choice(hi, len(hi))
    diffs.append(np.mean([np.mean(fl_ja[k]) for k in b]) - np.mean([np.mean(fl_ja[k]) for k in a]))
print("ja leak, 30-60 minus <10 bucket (row means): %.3f, bootstrap 95%% CI %.3f..%.3f" % (
    np.mean([np.mean(fl_ja[k]) for k in hi]) - np.mean([np.mean(fl_ja[k]) for k in lo]),
    np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)))
