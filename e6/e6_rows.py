"""E6 step 3 (local, TTS venv): gates, row selection, reference lists, eval sets and the E6 bundle.

  python -I -X utf8 e6_rows.py [--pod_root /workspace/bd]

E6_PLAN.md §3.1-3.3, 3.7 and §4. Inputs: data_on/{manifest,asr}.jsonl (e6_onbuild, e6_asr),
e6/work/on_vec.npz (Qwen speaker-encoder vectors of every clip, v1 audio policy), e1/bundle (v1 rows).

Gates on Our Notes clips (§3.3):
  clip probe  27-role logistic regression on the vectors, out-of-fold by episode (5 folds); a clip
              is dropped if the probe gives another role p > 0.5
  >= 3 kana   data_on/asr.jsonl verdict ok (either ASR within v1's cer_bar)
  1-2 kana    the probe's argmax is the clip's role (or its pair, Mutsumi/Mortis), trimmed
              duration (policy duration - 0.37 s of padding) 0.25-2.0 s
Selection: training episodes, kind "line", gates passed; cap 300 per row, stratified proportionally
over (duration bin < 1 / 1-3 / 3-8 / >= 8 s of source audio) x (expression class) cells by largest
remainder, seed 61 within cells. Val = max(2, round(1%)) of the selected, never a reference clip.
References (v1 recipe): gated training-episode lines, >= 2 s, not monologue, not val; centred cosine
to the row centroid, keep the top 80%, shuffle (seed 1), at most 64.
Eval sets from the held-out lines (split "test" in e6_onbuild; gated, ASR ok, no Latin or digits): ja_heldout 8 lines of 3-8 s with
>= 5 kana; ja_short 16 lines of 0.7-1.6 s with >= 3 kana; plus one shared ja_short set of 16 lines
for the 37 v1 rows (no Our Notes character names), round-robin over Our Notes rows.
[nv] rows (E6c): training-episode kind "nv" clips, 0.6-3.0 s source, probe argmax own role or pair,
text "[nv]", at most 120 per row (seed 67).
Writes e6/bundle/ (ja_rows.jsonl, refs/, speakers.txt, eval_rows.jsonl, eval_short.jsonl,
nv_rows.jsonl, probes_ja_novel.jsonl, rows_e6.json, selection_report.json) and
data_on/gates.jsonl (per-clip gate results).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402

DATA = BD / "data_on"
E1B = BD / "e1" / "bundle"
OUT = HERE / "bundle"
CAP, NV_CAP = 300, 120
REPLACED = ["mygo_tomori", "mygo_anon", "mygo_raana", "mygo_soyo", "mygo_taki", "mujica_uika", "mujica_mutsumi",
            "mujica_mortis", "mujica_umiri", "mujica_nyamu", "mujica_sakiko", "mujica_sakiko_tsuki"]
DROPPED_V1 = REPLACED                 # v1 rows whose data is replaced (mygo_soyo_tsuki stays on v1 data)
PAIR = {"mutsumi": "mortis", "mortis": "mutsumi"}
CAST_KANA = ["トモリ", "アノン", "ラーナ", "ソヨ", "タキ", "ウイカ", "ムツミ", "ウミリ", "ニャム", "サキ", "アラレ", "ノノカ",
             "リツ", "ミヤコ", "ユノ", "ホタル", "ナツメ", "ナギ", "マホロ", "ホウカ", "ライカ", "ミク", "ヨモギ", "チエリ",
             "シズク", "カナタ", "モテイス", "モーティス", "ハツネ", "クライシツク"]


def dbin(d: float) -> str:
    return "a<1" if d < 1 else "b1-3" if d < 3 else "c3-8" if d < 8 else "d8+"


def seeded(tag: str) -> random.Random:
    return random.Random(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))


def stratified(pool: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if len(pool) <= cap:
        return list(pool)
    cells = defaultdict(list)
    for r in pool:
        cells[(dbin(r["dur_src"]), r["exp"])].append(r)
    quota = {k: cap * len(v) / len(pool) for k, v in cells.items()}
    take = {k: int(q) for k, q in quota.items()}
    rest = cap - sum(take.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), str(k)))[:rest]:
        take[k] += 1
    out = []
    for k in sorted(cells, key=str):
        v = sorted(cells[k], key=lambda r: r["id"])
        rng.shuffle(v)
        out += v[:take[k]]
    return out


def main() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod_root", default="/workspace/bd")
    args = ap.parse_args()
    pod = lambda p: f"{args.pod_root}/" + Path(p).resolve().relative_to(BD.resolve()).as_posix()   # noqa: E731

    man = C.read_jsonl(DATA / "manifest.jsonl")
    asr = {r["id"]: r for r in C.read_jsonl(DATA / "asr.jsonl")}
    V = np.load(HERE / "work" / "on_vec.npz")["V"]
    roles = sorted({r["role"] for r in man})
    assert len(roles) == 27, roles

    # ---- clip probe, out-of-fold by episode
    X = V[[r["vec"] for r in man]].astype("float32")
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array([roles.index(r["role"]) for r in man])
    groups = np.array([r["episode"] for r in man])
    P = np.zeros((len(man), len(roles)), dtype="float32")
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])
        print(f"[rows] probe fold {k + 1}/5 acc {float((P[te].argmax(1) == y[te]).mean()):.3f}", flush=True)
    print(f"[rows] clip probe oof acc {float((P.argmax(1) == y).mean()):.3f}", flush=True)

    gates = []
    for i, r in enumerate(man):
        top = int(P[i].argmax())
        pred = roles[top]
        own = pred == r["role"] or PAIR.get(r["role"]) == pred
        g = {"probe": not (pred != r["role"] and P[i, top] > 0.5), "argmax_own": own}
        if r["kind"] == "line":
            if r["kana"] >= 3:
                g["text"] = asr.get(r["id"], {}).get("verdict") == "ok"
            else:
                g["text"] = own and 0.25 <= r["dur"] - 0.37 <= 2.0
        else:
            g["text"] = own and 0.6 <= r["dur_src"] <= 3.0
        r["gates"] = g
        r["ok"] = g["probe"] and g["text"] and (r["kind"] == "line" or own)
        gates.append({"id": r["id"], "row": r["row"], "pred": pred, "p_pred": round(float(P[i, top]), 3),
                      "p_own": round(float(P[i, roles.index(r["role"])]), 3), **g, "ok": r["ok"]})
    C.write_jsonl(DATA / "gates.jsonl", gates)

    rows_new = sorted({r["row"] for r in man})
    assert len(rows_new) == 28, rows_new
    removal = defaultdict(lambda: [0, 0])
    for r in man:
        if r["split"] == "train" and r["kind"] == "line":
            removal[r["exp"]][0] += 1
            removal[r["exp"]][1] += not r["ok"]

    # ---- selection, val, refs, eval sets, nv rows
    mu = np.load(BD / "vectors" / "probe.npz")["mu"]
    ja_new, refs, ev_held, ev_short, nv, report = [], {}, [], {}, [], {}
    latin = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]")
    for row in rows_new:
        mine = [r for r in man if r["row"] == row]
        pool = [r for r in mine if r["split"] == "train" and r["kind"] == "line" and r["ok"]]
        sel = stratified(pool, CAP, seeded(f"e6sel:{row}:61"))
        # references: v1 recipe over the gated training pool (not only the selected rows)
        cand = [r for r in pool if r["dur_src"] >= 2.0 and not r["mono"]]
        Xr = V[[r["vec"] for r in cand]] - mu
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        c = Xr.mean(0)
        c /= np.linalg.norm(c)
        order = np.argsort(-(Xr @ c))[: max(1, int(round(len(cand) * 0.8)))]
        keep = [cand[j] for j in order]
        random.Random(1).shuffle(keep)
        ref = keep[:64]
        ref_ids = {r["id"] for r in ref}
        refs[row] = [pod(r["audio"]) for r in ref]
        vpool = sorted([r for r in sel if r["id"] not in ref_ids], key=lambda r: r["id"])
        val = {r["id"] for r in seeded(f"val:{row}").sample(vpool, max(2, round(0.01 * len(sel))))}
        for r in sel:
            ja_new.append({"id": r["id"], "audio": pod(r["audio"]), "text": r["text"], "language": "japanese", "speaker": row,
                           "split": "val" if r["id"] in val else "train", "dur": r["dur"], "source": "ja_on", "exp": r["exp"],
                           "kana": r["kana"]})
        test = [r for r in mine if r["split"] == "test" and r["kind"] == "line" and r["ok"] and r["kana"] >= 3
                and asr.get(r["id"], {}).get("verdict") == "ok" and not latin.search(r["text"])]
        held = sorted([r for r in test if 3.0 <= r["dur_src"] <= 8.0 and r["kana"] >= 5], key=lambda r: r["id"])
        short = sorted([r for r in test if 0.7 <= r["dur_src"] <= 1.6], key=lambda r: r["id"])
        held = seeded(f"held:{row}").sample(held, min(8, len(held)))
        short = seeded(f"short:{row}").sample(short, min(16, len(short)))
        for r in held:
            ev_held.append({"real_id": r["id"], "row": row, "lang": "ja", "text": r["text"], "tid": f"ja_{r['id']}",
                            "real_audio": pod(r["audio"])})
        ev_short[row] = [{"tid": f"sh_{r['id']}", "text": r["text"], "real_id": r["id"], "real_audio": pod(r["audio"]),
                          "real_dur": r["dur_src"]} for r in short]
        nvp = sorted([r for r in mine if r["split"] == "train" and r["kind"] == "nv" and r["ok"]], key=lambda r: r["id"])
        nvs = seeded(f"nv:{row}:67").sample(nvp, min(NV_CAP, len(nvp)))
        nv += [{"id": f"nv_{r['id']}", "audio": pod(r["audio"]), "text": "[nv]", "language": "japanese", "speaker": row,
                "split": "train", "dur": r["dur"], "source": "nv", "ja_raw": r["ja_raw"]} for r in nvs]
        report[row] = {"clips": len(mine), "train_pool_gated": len(pool), "selected": len(sel), "val": len(val),
                       "refs": len(ref), "ref_candidates": len(cand), "eval_held": len(held), "eval_short": len(short),
                       "nv": len(nvs), "dur_bins": dict(Counter(dbin(r["dur_src"]) for r in sel)),
                       "under_1s": sum(r["dur_src"] < 1 for r in sel)}
        print(f"[rows] {row:22s} pool {len(pool):5d} sel {len(sel):3d} refs {len(ref):2d} held {len(held)} short {len(short):2d} "
              f"nv {len(nvs):3d}  bins {report[row]['dur_bins']}", flush=True)

    # shared short set for the 37 v1 rows: round-robin over Our Notes rows' short test lines without cast names
    rr = {row: [s for s in v if not any(k in C.reading(s["text"]) for k in CAST_KANA)] for row, v in ev_short.items()}
    shared, i = [], 0
    while len(shared) < 16 and any(rr.values()):
        row = rows_new[i % len(rows_new)]
        if rr[row]:
            s = rr[row].pop(0)
            shared.append({"tid": f"shv1_{s['real_id']}", "text": s["text"], "src_row": row})
        i += 1

    # ---- v1 rows (E1b recipe, unchanged) and the bundle
    v1_names = [s.split("=")[0] for s in (E1B / "speakers.txt").read_text(encoding="utf-8").strip().split(",")]
    keep_v1 = [k for k in v1_names if k not in DROPPED_V1]
    assert len(keep_v1) == 37, len(keep_v1)
    names = keep_v1 + [k for k in REPLACED] + [k for k in rows_new if k not in REPLACED]
    assert len(names) == 65 == len(set(names)) and set(names) - set(keep_v1) == set(rows_new), names
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "refs").mkdir(parents=True)
    ja_v1 = [r for r in C.read_jsonl(E1B / "ja_rows.jsonl") if r["speaker"] in keep_v1]
    for k in keep_v1:
        shutil.copy(E1B / "refs" / f"ref_{k}.txt", OUT / "refs" / f"ref_{k}.txt")
    for k, paths in refs.items():
        (OUT / "refs" / f"ref_{k}.txt").write_text("".join(p + "\n" for p in paths), encoding="utf-8", newline="\n")
    C.write_jsonl(OUT / "ja_rows.jsonl", ja_v1 + ja_new)
    ev_v1 = [r for r in C.read_jsonl(E1B / "eval_rows.jsonl") if r["row"] in keep_v1]
    C.write_jsonl(OUT / "eval_rows.jsonl", ev_v1 + ev_held)
    short_rows = [{"row": row, **s} for row, v in ev_short.items() for s in v] + \
                 [{"row": row, **s, "real_audio": None} for row in keep_v1 for s in shared]
    C.write_jsonl(OUT / "eval_short.jsonl", short_rows)
    C.write_jsonl(OUT / "nv_rows.jsonl", nv)
    (OUT / "speakers.txt").write_text(",".join(f"{k}={args.pod_root}/bundle/refs/ref_{k}.txt" for k in names), encoding="utf-8")
    shutil.copy(E1B / "probes_ja_novel.jsonl", OUT / "probes_ja_novel.jsonl")
    (OUT / "rows_e6.json").write_text(json.dumps({"v1_kept": keep_v1, "replaced": REPLACED,
                                                  "new": [k for k in rows_new if k not in REPLACED]}, indent=1), encoding="utf-8")
    # pseudo-zh texts (§3.4): official zh-Hans of the row's own training-episode lines, 8-30 hanzi,
    # <= 1 numeral, no Latin, not an eval probe; 60 per row (seed 31), disjoint across rows
    probes = {json.loads(l)["text"].rstrip("。？！") for l in (Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod") / "probes_zh_v3.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()}
    zh_texts, used = {}, set()
    for row in rows_new:
        cand = []
        for r in sorted([r for r in man if r["row"] == row and r["split"] == "train" and r["kind"] == "line" and r.get("zh")],
                        key=lambda r: r["id"]):
            t = re.sub(r"<[^>]+>", "", r["zh"]).replace("\n", "").replace("\r", "").strip().strip("「」『』")
            if t.startswith("（") and t.endswith("）"):
                t = t[1:-1]
            n_han = len(re.findall(r"[\u4e00-\u9fff]", t))
            if (8 <= n_han <= 30 and len(re.findall(r"[0-9０-９]+", t)) <= 1 and not re.search(r"[A-Za-zＡ-Ｚａ-ｚ]", t)
                    and t.rstrip("。？！") not in probes and t not in used):
                cand.append((r["id"], t))
        random.Random(31).shuffle(cand)
        zh_texts[row] = cand[:60]
        used |= {t for _, t in zh_texts[row]}
    (OUT / "zh_texts.json").write_text(json.dumps(zh_texts, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[rows] zh pseudo texts per row: { {k: len(v) for k, v in zh_texts.items()} }", flush=True)

    rem = {k: {"clips": a, "removed": b, "share": round(b / a, 3)} for k, (a, b) in sorted(removal.items(), key=lambda t: -t[1][0]) if a >= 100}
    rep = {"rows": report, "gate_removal_by_expression": rem, "shared_short": shared,
           "counts": {"ja_v1": len(ja_v1), "ja_new": len(ja_new), "nv": len(nv), "eval_held_new": len(ev_held),
                      "eval_held_v1": len(ev_v1), "eval_short": len(short_rows)}}
    (OUT / "selection_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[rows] gate removal by expression (train lines): {json.dumps({k: v['share'] for k, v in rem.items()})}", flush=True)
    print(f"[rows] bundle: {json.dumps(rep['counts'])}; shared short set {len(shared)}: {[s['text'] for s in shared]}", flush=True)
    short_low = {k: v["eval_short"] for k, v in report.items() if v["eval_short"] < 16}
    held_low = {k: v["eval_held"] for k, v in report.items() if v["eval_held"] < 8}
    print(f"[rows] rows below the eval-set sizes: short {short_low or 'none'}; held {held_low or 'none'}", flush=True)


if __name__ == "__main__":
    main()
