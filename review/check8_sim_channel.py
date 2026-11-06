"""Channel bias on 'sim': real vs codec-resynth sim to the locked vector (b07 never recomputed it), and floor
sim with the mean codec shift removed. Also: do the 46 within-set duplicates carry different filename texts?
CPU-only ECAPA (speaker_encoder weights only), read-only."""
import collections
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open

torch.set_num_threads(8)
BD = Path(r"E:\STEINS;GATEVoice\bangdream")
QB = Path(r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --- within-set duplicate texts (manifest has the kept text; compare with all filenames in the zip by sha)
import hashlib
rep = json.loads((BD / "data" / "build_report.json").read_text(encoding="utf-8"))
man = {r["id"]: r for r in rj(BD / "data" / "manifest.jsonl")}
by_set = collections.defaultdict(set)
for d in rep["duplicates"]:
    by_set[d["set"]].add(d["id"])
print("within-set dup sets:", {k: len(v) for k, v in by_set.items()})
SRC = Path(r"E:\BaiduNetdiskDownload\bangdreamai\角色语音")
diff_text = []
for key, dids in by_set.items():
    band, stem = key.split("/", 1)
    names = collections.defaultdict(list)
    with zipfile.ZipFile(SRC / band / f"{stem}.zip") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            sha = hashlib.sha256(zf.read(info)).hexdigest()
            rid = f"{man[next(iter(dids))]['row']}_{sha[:10]}"
            if rid in dids:
                names[rid].append(info.filename)
    for rid, fns in names.items():
        stems = {Path(f).stem for f in fns}
        if len(stems) > 1:
            diff_text.append((rid, sorted(fns)))
print("within-set dups whose filenames (texts) differ:", len(diff_text))
for x in diff_text[:6]:
    print("   ", x)
sub_pairs = collections.Counter(tuple(sorted(Path(f).parent.name for f in fns)) for fns in
                                [n for n in names.values()])
print("   sub-folder pattern of last set's dup groups:", sub_pairs.most_common(3))

# --- sim channel bias
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram  # noqa: E402
cfg = json.loads((QB / "config.json").read_text(encoding="utf-8"))["speaker_encoder_config"]
enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**cfg)).eval()
with safe_open(str(QB / "model.safetensors"), "pt") as f:
    enc.load_state_dict({k[16:]: f.get_tensor(k).float() for k in f.keys() if k.startswith("speaker_encoder.")})


def cvec(path, mu):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    with torch.inference_mode():
        m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                            hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        v = enc(m)[0].numpy().astype(np.float64) - mu
    return v / np.linalg.norm(v)


import torch as _t
pr = np.load(BD / "vectors" / "probe.npz")
mu = pr["mu"].astype(np.float64)
lk = _t.load(BD / "vectors" / "locked.pt", map_location="cpu")
names = lk["names"]
L = lk["table"].float().numpy().astype(np.float64) - mu
L /= np.linalg.norm(L, axis=1, keepdims=True)
delta = np.load(Path(__file__).with_name("_codec_delta.npz"))["delta"]
real = rj(BD / "e0" / "real" / "scored.jsonl")
res = {r["pid"]: r for r in rj(BD / "e0" / "resynth" / "scored.jsonl")}
per = collections.defaultdict(lambda: {"real": [], "res": [], "fl": [], "fl_c": []})
for r in real:
    j = names.index(r["row"])
    per[r["row"]]["real"].append(r["sim"])
    per[r["row"]]["res"].append(float(cvec(res[r["pid"]]["path"], mu) @ L[j]))
for d in sorted((BD / "e0" / "floor").iterdir()):
    if not (d / "scored.jsonl").exists():
        continue
    for r in rj(d / "scored.jsonl"):
        if r["lang"] != "ja":
            continue
        j = names.index(r["row"])
        x = cvec(r["path"], mu)
        xc = (x - delta) / np.linalg.norm(x - delta)
        per[r["row"]]["fl"].append(float(x @ L[j]))
        per[r["row"]]["fl_c"].append(float(xc @ L[j]))
M = {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in per.items()}
a = {q: np.mean([m[q] for m in M.values()]) for q in ("real", "res", "fl", "fl_c")}
print("row-mean sim: real %.3f | codec-resynth %.3f | floor ja %.3f | floor ja, codec shift removed %.3f" % (
    a["real"], a["res"], a["fl"], a["fl_c"]))
drops = sorted(((k, round(m["real"] - m["res"], 3)) for k, m in M.items()), key=lambda t: -t[1])
print("codec-only sim drop per row: max", drops[:4], " min", drops[-3:])
above = [(k, round(m["fl"] - m["res"], 3)) for k, m in M.items() if m["fl"] > m["res"]]
print("rows where floor ja sim exceeds the codec-resynth target:", sorted(above, key=lambda t: -t[1]))
