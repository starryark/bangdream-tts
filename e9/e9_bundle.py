"""E9 (local): build the 72-voice bundle and row sources (E9_PLAN.md §2-3), then lint them.

  python -I -X utf8 e9/e9_bundle.py        (TTS venv: numpy; reads e6/bundle, e1/bundle, data*, e5/e6 VC rows, E8's pseudo rows)

Voices (user, 2026-11-09):
- 36 unchanged v1 voices (E6's 37 kept minus mygo_soyo_tsuki).
- 9 MyGO!!!!! / Ave Mujica characters with TWO ids: the v1-era voice under the usual name (v1 data, E1 refs) and the
  Our Notes voice as <name>_ournotes (Our Notes data, E6 refs). Soyo + Soyo-tsuki and Sakiko + Sakiko-tsuki are one
  voice per era; the old persona ids become aliases of the v1-era voice (e9_aliases.py).
- mygo_raana and mujica_nyamu: one id each with both eras (the codec table has 72 speaker rows, 3000-3071; Rana's
  Our Notes data is 4.8 min and Nyamu's two eras are the closest, centroid cosine 0.74).
- 16 new Our Notes voices.
Pruning (curation_2026-11-09.md §1.4): training clips >= 2 s with robust z < -3.5 against their own voice's centroid
AND nearer another voice's centroid, per era (per-source centring). Writes e9/bundle/* (pod paths), e9/rows/*.jsonl
(pseudo_v1, pseudo_on, vc, pod paths) and e9/bundle/e9_bundle_report.json.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
from pathlib import Path

import numpy as np

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
E6B, E1B, OUT = BD / "e6" / "bundle", BD / "e1" / "bundle", BD / "e9" / "bundle"
POD = "/workspace/bd/"
TWO_ERA = ["mygo_tomori", "mygo_anon", "mygo_soyo", "mygo_taki", "mujica_uika", "mujica_mutsumi", "mujica_mortis",
           "mujica_umiri", "mujica_sakiko"]
SINGLE = ["mygo_raana", "mujica_nyamu"]
PERSONA = {"mygo_soyo_tsuki": "mygo_soyo", "mujica_sakiko_tsuki": "mujica_sakiko"}
REPLACED12 = TWO_ERA + SINGLE + ["mujica_sakiko_tsuki"]
OURN = "_ournotes"


def rd(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def wj(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


def base(s: str) -> str:
    return PERSONA.get(s, s)


def local(p: str) -> Path:
    p = p.replace("\\", "/")
    if p.startswith(POD + "pseudo_on/"):                    # E8's download of the Our Notes pseudo rows
        return BD / "e8" / "results" / "pseudo_on" / p[len(POD + "pseudo_on/"):]
    if p.startswith(POD):
        return BD / p[len(POD):]
    if p.startswith("/workspace/bd1/pseudo/"):
        return BD / "e8" / "results" / "pseudo_v1_e1recipe" / p[len("/workspace/bd1/pseudo/"):]
    return Path(p)


def podpath(p: str) -> str:
    p = p.replace("\\", "/")
    root = str(BD).replace("\\", "/") + "/"
    return POD + p[len(root):] if p.startswith(root) else p


def main() -> None:
    sel6 = json.loads((E6B / "rows_e6.json").read_text(encoding="utf-8"))
    unch = [r for r in sel6["v1_kept"] if r != "mygo_soyo_tsuki"]
    new = list(sel6["new"])
    on_era = [b + OURN for b in TWO_ERA]
    names = unch + TWO_ERA + SINGLE + on_era + new
    assert len(names) == 72 and len(set(names)) == 72, len(names)
    on_ids = {r["id"] for r in rd(BD / "data_on" / "manifest.jsonl")}

    def label_e6(s: str, rid: str) -> str:        # rows of E6's bundle
        if s in unch or s in new:
            return s
        if s == "mygo_soyo_tsuki":                 # v1 GBP Tsukinomori Soyo: v1-era Soyo
            return "mygo_soyo"
        b = base(s)
        assert rid in on_ids, (s, rid)             # E6's 12 replaced rows hold Our Notes data
        return b if b in SINGLE else b + OURN

    def label_e1(s: str) -> str:                   # v1 data of the 12 replaced characters
        return base(s)

    # ---- training rows
    rows, seen = [], set()
    for r in rd(E6B / "ja_rows.jsonl"):
        if r["id"] in seen:
            continue
        seen.add(r["id"]); rows.append({**r, "speaker": label_e6(r["speaker"], r["id"])})
    for r in rd(E1B / "ja_rows.jsonl"):
        if r["speaker"] in REPLACED12 and r["id"] not in seen:
            seen.add(r["id"]); rows.append({**r, "speaker": label_e1(r["speaker"])})
    assert {r["speaker"] for r in rows} == set(names), sorted({r["speaker"] for r in rows} ^ set(names))

    # ---- pruning: per era, per-source centring
    vec = {}
    d = np.load(BD / "vectors" / "clipvec.npz", allow_pickle=True)
    vec.update(zip(d["ids"].tolist(), d["V"]))
    e = np.load(BD / "e6" / "work" / "on_vec.npz", allow_pickle=True)
    EV = e["V"]
    for m in rd(BD / "data_on" / "manifest.jsonl"):
        if m.get("vec") not in (None, "None", ""):
            vec[m["id"]] = EV[int(m["vec"])]
    drop = set()
    for era in ("v1", "on"):
        sub = [r for r in rows if r["split"] == "train" and r["id"] in vec and ((r["id"] in on_ids) == (era == "on"))]
        X = np.stack([vec[r["id"]] for r in sub]).astype(np.float64); X -= X.mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
        lab = np.array([r["speaker"] for r in sub]); ns = sorted(set(lab)); li = np.array([ns.index(x) for x in lab])
        C = np.stack([X[li == j].mean(0) for j in range(len(ns))]); C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = X @ C.T; own = S[np.arange(len(sub)), li].copy(); S[np.arange(len(sub)), li] = -9; other = S.max(1)
        for j in range(len(ns)):
            idx = np.where(li == j)[0]; o = own[idx]; med = np.median(o); mad = np.median(np.abs(o - med)) + 1e-9
            for k, z in zip(idx, (o - med) / (1.4826 * mad)):
                if z < -3.5 and other[k] > own[k] and sub[k]["dur"] >= 2.0:
                    drop.add(sub[k]["id"])
    rows = [r for r in rows if r["id"] not in drop]

    # ---- eval rows, short lines, refs
    ev6, ev1 = rd(E6B / "eval_rows.jsonl"), rd(E1B / "eval_rows.jsonl")
    evals, seen_t = [], set()

    def add_eval(r, row):
        key = (row, r["tid"])
        if key not in seen_t:
            seen_t.add(key); evals.append({**r, "row": row})
    for r in ev6:
        s = r["row"]
        if s in unch or s in new:
            add_eval(r, s)
        elif s == "mygo_soyo_tsuki":
            add_eval(r, "mygo_soyo")
        else:
            b = base(s); add_eval(r, b if b in SINGLE else b + OURN)
    for r in ev1:
        if r["row"] in REPLACED12:
            add_eval(r, base(r["row"]))
    sh6 = rd(E6B / "eval_short.jsonl")
    shared = [r for r in sh6 if r["row"] == "mygo_soyo_tsuki"]          # the shared 16-line set given to v1 rows
    assert len(shared) == 16 and all(r["tid"].startswith("shv1_") for r in shared)
    shorts = []
    for r in sh6:
        s = r["row"]
        if s in unch or s in new:
            shorts.append(r)
        elif s in REPLACED12:
            b = base(s); shorts.append({**r, "row": b if b in SINGLE else b + OURN})
    for b in TWO_ERA:
        shorts += [{**r, "row": b} for r in shared]
    dedup, seen_s = [], set()
    for r in shorts:
        if (r["row"], r["tid"]) not in seen_s:
            seen_s.add((r["row"], r["tid"])); dedup.append(r)
    shorts = dedup

    def refs(p: Path) -> list[str]:
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "refs").mkdir(parents=True)
    for n in names:
        if n in unch or n in new:
            lst = refs(E6B / "refs" / f"ref_{n}.txt")
        elif n.endswith(OURN):
            b = n[: -len(OURN)]
            lst = refs(E6B / "refs" / f"ref_{b}.txt") + (refs(E6B / "refs" / "ref_mujica_sakiko_tsuki.txt") if b == "mujica_sakiko" else [])
        elif n in SINGLE:
            lst = refs(E1B / "refs" / f"ref_{n}.txt") + refs(E6B / "refs" / f"ref_{n}.txt")
        else:   # v1-era of a two-era character
            lst = refs(E1B / "refs" / f"ref_{n}.txt") + [x for p in PERSONA if PERSONA[p] == n for x in refs(E1B / "refs" / f"ref_{p}.txt")]
        lst = list(dict.fromkeys(lst))
        assert lst, n
        (OUT / "refs" / f"ref_{n}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8", newline="\n")
    (OUT / "speakers.txt").write_text(",".join(f"{n}={POD}bundle/refs/ref_{n}.txt" for n in names), encoding="utf-8", newline="\n")
    wj(OUT / "ja_rows.jsonl", rows); wj(OUT / "eval_rows.jsonl", evals); wj(OUT / "eval_short.jsonl", shorts)
    wj(OUT / "nv_rows.jsonl", [])
    for f in ("probe_e6.npz", "probe_e6_calib.json", "zh_texts.json", "probes_ja_novel.jsonl"):
        shutil.copy(E6B / f, OUT / f)
    sel = {"v1_kept": unch, "replaced": on_era + SINGLE, "new": new, "v1_era": TWO_ERA, "aliases": PERSONA,
           "note": "E9: 'replaced' = the Our-Notes-era ids (+ the two single-id characters); 'v1_era' = v1-era ids of the 9 two-era characters"}
    (OUT / "rows_e6.json").write_text(json.dumps(sel, indent=1), encoding="utf-8", newline="\n")

    # ---- pseudo and VC rows (pod paths)
    R = BD / "e9" / "rows"
    if R.exists():
        shutil.rmtree(R)
    R.mkdir(parents=True)

    def relabel_v1(s):                     # rows made in a v1 voice (E1-recipe pseudo, E5 VC)
        return base(s)

    def relabel_on(s):                     # rows made in an Our Notes voice (pseudo_on)
        if s in new:
            return s
        b = base(s)
        return b if b in SINGLE else b + OURN
    # the pseudo wavs are uploaded at the same relative path they have here (e8/results/...), so one resumable
    # upload list covers everything; their rows are pointed there
    def mv(a, old, new):
        a = a.replace("\\", "/")
        assert a.startswith(old), a
        return new + a[len(old):]
    pv1 = [{**r, "speaker": relabel_v1(r["speaker"]), "audio": mv(r["audio"], "/workspace/bd1/pseudo/", POD + "e8/results/pseudo_v1_e1recipe/")}
           for r in rd(BD / "e8/results/pseudo_v1_e1recipe/pseudo_rows.jsonl")]
    pon = [{**r, "speaker": relabel_on(r["speaker"]), "audio": mv(r["audio"], POD + "pseudo_on/", POD + "e8/results/pseudo_on/")}
           for r in rd(BD / "e8/results/pseudo_on/pseudo_rows.jsonl")]
    vc5 = [{**r, "speaker": relabel_v1(r["speaker"]), "audio": podpath(r["audio"])} for r in rd(BD / "e5/vc/rvc_rows.jsonl")]
    vc6 = [{**r, "speaker": r["speaker"], "audio": podpath(r["audio"])} for r in rd(BD / "e6/vc/rvc_rows.jsonl")]
    wj(R / "pseudo_v1_rows.jsonl", pv1); wj(R / "pseudo_on_rows.jsonl", pon); wj(R / "vc_rows.jsonl", vc5 + vc6)

    # ---- lint
    problems = []
    for nm, rs in (("ja", rows), ("pseudo_v1", pv1), ("pseudo_on", pon), ("vc", vc5 + vc6)):
        bad = {r["speaker"] for r in rs} - set(names)
        if bad:
            problems.append(f"{nm}: unknown speakers {sorted(bad)}")
        ids = collections.Counter(r["id"] for r in rs)
        if any(v > 1 for v in ids.values()):
            problems.append(f"{nm}: duplicate ids")
        for k in ("id", "audio", "text", "speaker", "split", "language"):
            if any(k not in r for r in rs):
                problems.append(f"{nm}: rows without '{k}'")
        miss = [r["audio"] for r in rs if not local(r["audio"]).is_file()]
        if miss:
            problems.append(f"{nm}: {len(miss)} audio files missing locally, e.g. {miss[:2]}")
    test_ids = {r["real_id"] for r in evals} | {r["real_id"] for r in shorts if r.get("real_id")}
    if test_ids & {r["id"] for r in rows}:
        problems.append(f"test clips in training: {len(test_ids & {r['id'] for r in rows})}")
    allids = collections.Counter(r["id"] for r in rows + pv1 + pon + vc5 + vc6)
    if any(v > 1 for v in allids.values()):
        problems.append("ids collide across sources")
    for n in names:
        if not any(r["row"] == n for r in evals):
            problems.append(f"no eval rows for {n}")
        if not any(r["row"] == n for r in shorts):
            problems.append(f"no short lines for {n}")
    for p in sorted((OUT / "refs").glob("*.txt")):
        miss = [x for x in p.read_text(encoding="utf-8").split() if not local(x).is_file()]
        if miss:
            problems.append(f"{p.name}: {len(miss)} ref clips missing")
    tr = collections.Counter(r["speaker"] for r in rows if r["split"] == "train")
    rep = {"voices": len(names), "train_rows": sum(tr.values()), "pruned": len(drop), "min_train_rows": min(tr.values()),
           "per_voice_train": dict(sorted(tr.items())), "eval_rows": len(evals), "eval_short": len(shorts),
           "pseudo_v1": len(pv1), "pseudo_on": len(pon), "vc": len(vc5 + vc6),
           "pseudo_per_voice": dict(collections.Counter(r["speaker"] for r in pv1 + pon)), "problems": problems}
    (OUT / "e9_bundle_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_voice_train", "pseudo_per_voice")}, ensure_ascii=False, indent=1))
    print("two-era train rows:", {n: (tr[n], tr[n + OURN]) for n in TWO_ERA}, "| single:", {n: tr[n] for n in SINGLE})
    if problems:
        raise SystemExit("LINT FAILED")
    print("LINT OK")


if __name__ == "__main__":
    main()
