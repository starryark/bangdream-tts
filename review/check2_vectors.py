"""Read-only checks on vectors/{clipvec.npz, locked.pt, probe.npz} and e0/real/scored.jsonl. CPU only."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def nrm(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


man = rj(BD / "data" / "manifest.jsonl")
mid = {r["id"]: r for r in man}
split = {r["id"]: r["split"] for r in rj(BD / "data" / "split.jsonl")}
cv = np.load(BD / "vectors" / "clipvec.npz")
ids, V = list(cv["ids"]), cv["V"].astype(np.float64)
idx = {i: n for n, i in enumerate(ids)}
pr = np.load(BD / "vectors" / "probe.npz")
mu, coef, icpt, classes = pr["mu"].astype(np.float64), pr["coef"], pr["intercept"], list(pr["classes"])
lk = torch.load(BD / "vectors" / "locked.pt", map_location="cpu")
names, T = lk["names"], lk["table"].float().numpy().astype(np.float64)
rows = sorted({r["row"] for r in man})
print("V", V.shape, "coef", coef.shape, "T", T.shape, "classes==rows", classes == rows, "names==rows", names == rows)

train = [i for i in ids if split[i] == "train"]
test = [i for i in ids if split[i] == "test"]
mu_chk = V[[idx[i] for i in train]].mean(0)
mu_all = V.mean(0)
print("|mu - mean(train V)| =", np.abs(mu - mu_chk).max(), " |mu - mean(all V)| =", np.abs(mu - mu_all).max())

# --- locked vector = mean of refs rescaled to mean norm? and its centred direction -------
norm_mu = np.linalg.norm(mu)
print("|mu| =", round(norm_mu, 3), " mean |v| =", round(np.linalg.norm(V, axis=1).mean(), 3))
s_list, cos_T_vs_mean, rows_info = [], [], {}
Lc_T, Lc_M = [], []
for j, k in enumerate(rows):
    refs = (BD / "refs" / f"ref_{k}.txt").read_text(encoding="utf-8").split()
    a2i = {mid_i["audio"]: mid_i["id"] for mid_i in man if mid_i["row"] == k}
    R = V[[idx[a2i[p]] for p in refs]]
    m = R.mean(0)
    s = np.linalg.norm(R, axis=1).mean() / np.linalg.norm(m)
    recon = m * s
    c_recon = float(nrm(recon) @ nrm(T[j]))
    s_list.append(s)
    cos_T_vs_mean.append(c_recon)
    Lc_T.append(nrm(T[j] - mu))
    Lc_M.append(nrm(m - mu))
    rows_info[k] = (s, c_recon, float(np.linalg.norm(T[j]) / np.linalg.norm(recon)))
print("recon(locked) vs saved locked: min cos", round(min(cos_T_vs_mean), 5),
      " norm ratio range", round(min(v[2] for v in rows_info.values()), 4), round(max(v[2] for v in rows_info.values()), 4))
s_arr = np.array(s_list)
print("rescale factor s = mean|v|/|mean v|: min %.3f median %.3f max %.3f" % (s_arr.min(), np.median(s_arr), s_arr.max()))
Lc_T, Lc_M = np.stack(Lc_T), np.stack(Lc_M)
dir_cos = (Lc_T * Lc_M).sum(1)
print("cos( nrm(T-mu), nrm(mean_refs-mu) ): min %.3f median %.3f" % (dir_cos.min(), np.median(dir_cos)))
# projection of centred locked onto mu direction
u = mu / norm_mu
print("component of nrm(T-mu) along mu: median %.3f range %.3f..%.3f" % (np.median(Lc_T @ u), (Lc_T @ u).min(), (Lc_T @ u).max()))
print("component of nrm(mean_refs-mu) along mu: median %.3f" % np.median(Lc_M @ u))
for name, L in (("T-mu (as b03/b05)", Lc_T), ("mean_refs-mu", Lc_M)):
    S = L @ L.T
    off = S[~np.eye(len(rows), dtype=bool)]
    print(f"separability {name}: median {np.median(off):.3f} p95 {np.percentile(off, 95):.3f} max {off.max():.3f}")

# --- real held-out sim recomputed from clipvec vs scored -----------------------------------
real = rj(BD / "e0" / "real" / "scored.jsonl")
d_sim, d_pt, alt_sim = [], [], collections.defaultdict(list)
pred_ok = 0
for r in real:
    i = f"{r['row']}_{r['tid'].split('_', 1)[1]}"
    x = nrm(V[idx[i]] - mu)
    j = rows.index(r["row"])
    d_sim.append(abs(float(x @ Lc_T[j]) - r["sim"]))
    z = coef @ x + icpt
    p = np.exp(z - z.max()); p /= p.sum()
    d_pt.append(abs(float(p[j]) - r["p_true"]))
    pred_ok += classes[int(p.argmax())] == r["pred"]
    alt_sim[r["row"]].append((r["sim"], float(x @ Lc_M[j])))
print(f"real clips: |sim(clipvec) - sim(b05)| max {max(d_sim):.4f}; |p_true diff| max {max(d_pt):.4f}; pred agree {pred_ok}/{len(real)}")
a = np.array([v for vs in alt_sim.values() for v in vs])
print("real sim mean: vs nrm(T-mu) %.3f   vs nrm(mean_refs-mu) %.3f" % (a[:, 0].mean(), a[:, 1].mean()))

# --- near-duplicate audio across train/test (raw cosine) ----------------------------------
Vn = nrm(V)
Xc = nrm(V - mu)
nn_test, nn_train = [], []
by_row = collections.defaultdict(list)
for i in ids:
    by_row[mid[i]["row"]].append(i)
for k in rows:
    tr = [idx[i] for i in by_row[k] if split[i] == "train"]
    te = [idx[i] for i in by_row[k] if split[i] == "test"]
    S = Xc[te] @ Xc[tr].T
    nn_test += list(S.max(1))
    St = Xc[tr] @ Xc[tr].T
    np.fill_diagonal(St, -1)
    nn_train += list(St.max(1))
nn_test, nn_train = np.array(nn_test), np.array(nn_train)
print("centred NN cosine test->train(same row): median %.3f p99 %.3f max %.3f; >0.95: %d" % (
    np.median(nn_test), np.percentile(nn_test, 99), nn_test.max(), (nn_test > 0.95).sum()))
print("centred NN cosine train->train(same row): median %.3f p99 %.3f max %.3f; >0.95: %d" % (
    np.median(nn_train), np.percentile(nn_train, 99), nn_train.max(), (nn_train > 0.95).sum()))

# --- probe on its own train set and on test (from saved coef) -------------------------------
def probs(X):
    Z = X @ coef.T + icpt
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)
Xtr = Xc[[idx[i] for i in train]]
ytr = np.array([rows.index(i.rsplit("_", 1)[0]) for i in train])
Xte = Xc[[idx[i] for i in test]]
yte = np.array([rows.index(i.rsplit("_", 1)[0]) for i in test])
Ptr, Pte = probs(Xtr), probs(Xte)
print("probe train acc %.4f  test acc %.4f (n=%d, errors %d)" % ((Ptr.argmax(1) == ytr).mean(), (Pte.argmax(1) == yte).mean(),
                                                               len(yte), (Pte.argmax(1) != yte).sum()))
print("p_true mean train %.3f test %.3f" % (Ptr[np.arange(len(ytr)), ytr].mean(), Pte[np.arange(len(yte)), yte].mean()))
# test-eligible-like train clips (same filter as test) vs all train: CV-free proxy for selection bias
asr = {r["id"]: r for r in rj(BD / "data" / "asr.jsonl")}
import re
LAT = re.compile(r"[A-Za-z]")
elig = np.array([asr[i]["verdict"] == "ok" and 2.0 <= mid[i]["dur_src"] <= 12.0 and not LAT.search(mid[i]["text"])
                 and asr[i]["ref_kana"] >= 8 for i in train])
print("train p_true mean: test-like clips %.3f, other clips %.3f" % (Ptr[np.arange(len(ytr)), ytr][elig].mean(),
                                                                  Ptr[np.arange(len(ytr)), ytr][~elig].mean()))
cnt = collections.Counter(ytr)
print("train class sizes min %d max %d" % (min(cnt.values()), max(cnt.values())))
print("intercept range %.3f..%.3f" % (icpt.min(), icpt.max()))
