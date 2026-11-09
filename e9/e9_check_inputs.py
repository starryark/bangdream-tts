"""E9 (pod): count the input files bootstrap needs that are missing; list them in logs/missing.txt. Prints the count."""
import glob
import json
import os

W = "/workspace/bd"


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


paths = [r["audio"] for f in ("bundle/ja_rows.jsonl", "pseudo_v1/rows.jsonl", "pseudo_on/pseudo_rows.jsonl", "vc/vc_rows.jsonl")
         for r in rd(f"{W}/{f}")]
paths += [r["real_audio"] for r in rd(f"{W}/bundle/eval_rows.jsonl")]
paths += [r["real_audio"] for r in rd(f"{W}/bundle/eval_short.jsonl") if r.get("real_audio") not in (None, "None", "")]
paths += [p.strip() for f in glob.glob(f"{W}/bundle/refs/*.txt") for p in open(f) if p.strip()]
paths += [f"{W}/models/E1b_ep2/model.safetensors", f"{W}/bundle/probe_e6.npz", f"{W}/eval/C0/scored.jsonl", f"{W}/e9/listen_items.json"]
miss = [p for p in paths if not os.path.exists(p)]
print(len(miss))
if miss:
    open(f"{W}/logs/missing.txt", "w").write("\n".join(miss))
