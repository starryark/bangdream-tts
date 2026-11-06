"""Read-only checks: labels, split, refs, dedup, minutes. No models, CPU only."""
import collections
import json
import os
import re
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


man = rj(BD / "data" / "manifest.jsonl")
split = {r["id"]: r for r in rj(BD / "data" / "split.jsonl")}
asr = {r["id"]: r for r in rj(BD / "data" / "asr.jsonl")}
rep = json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))
rows_json = json.loads((BD / "pipeline" / "rows.json").read_text(encoding="utf-8"))
rows_json.pop("_note", None)
rows = sorted({r["row"] for r in man})
print("manifest", len(man), "rows", len(rows), "split", len(split), "asr", len(asr))

# --- 2. label parsing --------------------------------------------------------------
ids = [r["id"] for r in man]
print("dup ids:", len(ids) - len(set(ids)))
bad = [r["id"] for r in man if r["id"].rsplit("_", 1)[0] != r["row"]]
print("rsplit label != manifest row:", len(bad), bad[:3])
print("ids whose sha part has '_':", sum("_" in r["id"].rsplit("_", 1)[1] for r in man))
print("sha10 prefix collisions inside a row:",
      sum(c - 1 for c in collections.Counter((r["row"], r["sha256"][:10]) for r in man).values() if c > 1))
print("row values in rows.json unique:", len({v["row"] for v in rows_json.values()}) == len(rows_json))
# is any row name a prefix such that rsplit would be ambiguous? (row names end in non-hex?)
print("rows:", rows)

# --- 8. dedup -----------------------------------------------------------------------
shas = collections.Counter(r["sha256"] for r in man)
print("dup sha in manifest:", sum(c > 1 for c in shas.values()))
missing = [r["audio"] for r in man if not os.path.isfile(r["audio"])]
print("manifest audio missing on disk:", len(missing))
on_disk = {str(p) for p in (BD / "data" / "wav24").rglob("*.wav")}
orphans = on_disk - {r["audio"] for r in man}
print("wav files on disk not in manifest (orphans):", len(orphans), list(orphans)[:3])
dups = rep["duplicates"]
mid = {r["id"]: r for r in man}
within = [d for d in dups if d["id"] == d["dup_of"]]
across = [d for d in dups if d["id"] != d["dup_of"]]
print("dups within set:", len(within), " across sets:", len(across))
for d in across[:10]:
    print("   across:", d)
# same (row, text) appearing more than once with different bytes (near-dup candidates)
by_rt = collections.defaultdict(list)
for r in man:
    by_rt[(r["row"], r["text"])].append(r)
multi = {k: v for k, v in by_rt.items() if len(v) > 1}
print("(row,text) groups with >1 distinct clip:", len(multi), "clips involved:", sum(len(v) for v in multi.values()))
fmts = collections.Counter(tuple(sorted(x["fmt"] for x in v)) for v in multi.values())
print("   fmt combos:", fmts.most_common(6))
# cross-row identical text
by_t = collections.defaultdict(set)
for r in man:
    by_t[r["text"]].add(r["row"])
print("texts present in >1 row:", sum(len(v) > 1 for v in by_t.values()))
# subfolders per set
subs = collections.Counter((r["row"], r["sub"]) for r in man)
print("rows with >1 sub-folder:", sorted({k[0] for k, v in subs.items() if k[1]} ))

# --- 1. split leakage ------------------------------------------------------------------
test = {i for i, s in split.items() if s["split"] == "test"}
print("test clips:", len(test))
n_test = collections.Counter(split[i]["row"] for i in test)
print("test per row min/max:", min(n_test.values()), max(n_test.values()), "rows w/o test:", set(rows) - set(n_test))
ref_paths = {}
for k in rows:
    ref_paths[k] = (BD / "refs" / f"ref_{k}.txt").read_text(encoding="utf-8").split()
audio2id = {r["audio"]: r["id"] for r in man}
leak_ref = [(k, p) for k, v in ref_paths.items() for p in v if audio2id[p] in test]
print("test clips inside ref lists:", len(leak_ref))
wrong_row = [(k, p) for k, v in ref_paths.items() for p in v if mid[audio2id[p]]["row"] != k]
print("ref clips from another row:", len(wrong_row))
bad_ref = [(k, p) for k, v in ref_paths.items() for p in v
           if asr[audio2id[p]]["verdict"] != "ok" or mid[audio2id[p]]["dur_src"] < 2.0]
print("ref clips failing ok/>=2s:", len(bad_ref))
# near-duplicate by text: test clip text equal to a train clip text in same row / any row
tt_same, tt_any = [], []
for i in test:
    r = mid[i]
    same = [x for x in by_rt[(r["row"], r["text"])] if x["id"] != i]
    if same:
        tt_same.append((i, [x["id"] for x in same]))
    if len(by_t[r["text"]] - {r["row"]}):
        tt_any.append((i, sorted(by_t[r["text"]] - {r["row"]})))
print("test clips whose exact text also exists as a train clip in the SAME row:", len(tt_same), tt_same[:5])
print("test clips whose exact text exists in ANOTHER row:", len(tt_any), tt_any[:5])
in_refs = {audio2id[p] for v in ref_paths.values() for p in v}
print("   ... of same-row text twins, twin is in the ref list:",
      sum(any(t in in_refs for t in tw) for _, tw in tt_same))

# --- 7. minutes / buckets ------------------------------------------------------------
mins_exact = collections.defaultdict(float)
mins_src = collections.defaultdict(float)
mins_ok_train = collections.defaultdict(float)
for r in man:
    mins_exact[r["row"]] += r["dur"] / 60
    mins_src[r["row"]] += r["dur_src"] / 60
    if split[r["id"]]["split"] == "train" and asr[r["id"]]["verdict"] == "ok":
        mins_ok_train[r["row"]] += r["dur"] / 60
diff = {k: round(rep["per_row"][k]["min"] - mins_exact[k], 3) for k in rows}
print("build_report min - exact sum(dur)/60: max abs", max(abs(v) for v in diff.values()))
print("rows >= 99 min (dropped by b06 BUCKETS):", [k for k in rows if rep["per_row"][k]["min"] >= 99])
B = [("<10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-60", 30, 99)]
def bucket(m):
    for n, lo, hi in B:
        if lo <= m < hi:
            return n
moved = [(k, round(rep["per_row"][k]["min"], 2), round(mins_src[k], 2), round(mins_ok_train[k], 2))
         for k in rows if bucket(rep["per_row"][k]["min"]) != bucket(mins_ok_train[k])]
print("rows whose bucket changes using train+ok minutes (padded min, src min, train-ok min):", moved)
print("bucket counts padded:", collections.Counter(bucket(rep["per_row"][k]["min"]) for k in rows))
print("bucket counts dur_src:", collections.Counter(bucket(mins_src[k]) for k in rows))
print("bucket counts train-ok:", collections.Counter(bucket(mins_ok_train[k]) for k in rows))
print("total pad minutes:", round(sum(mins_exact.values()) - sum(mins_src.values()), 1),
      "of", round(sum(mins_exact.values()), 1))

# --- ASR / echo ------------------------------------------------------------------------
print("verdict counts:", collections.Counter(a["verdict"] for a in asr.values()))
print("qwen None (echo):", sum(a.get("qwen") is None for a in asr.values()))
# ok clips with cer_anime None? best uses ca always
print("dur_src < 1 among ok:", sum(1 for r in man if asr[r["id"]]["verdict"] == "ok" and r["dur_src"] < 1.0))
