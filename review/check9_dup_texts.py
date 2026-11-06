"""Within-set byte-identical duplicates: do their filename texts differ lexically (reading space) or only in punctuation?"""
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402

rep = json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))
man = {json.loads(l)["id"]: json.loads(l) for l in (BD / "data" / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l}
by_set = collections.defaultdict(set)
for d in rep["duplicates"]:
    by_set[d["set"]].add(d["id"])
lex, punct = [], 0
for key, dids in by_set.items():
    band, stem = key.split("/", 1)
    row = man[next(iter(dids))]["row"]
    names = collections.defaultdict(list)
    with zipfile.ZipFile(C.SRC / band / f"{stem}.zip") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                rid = f"{row}_{hashlib.sha256(zf.read(info)).hexdigest()[:10]}"
                if rid in dids:
                    names[rid].append(info.filename)
    for rid, fns in names.items():
        texts = {C.clean_text(Path(f).stem) for f in fns}
        if len(texts) < 2:
            continue
        reads = {C.reading(t) for t in texts}
        if len(reads) > 1:
            lex.append((rid, man[rid]["text"], sorted(texts)))
        else:
            punct += 1
print("text-differing dup groups: punctuation/space only", punct, "| lexical", len(lex))
for x in lex:
    print("   kept:", x[1], "| variants:", x[2])
