"""Centred Qwen speaker-encoder cosine between 祐天寺若麥's nyamu0/ and nyamu1/ sub-folders,
using the global mean saved by spk_centered.py (scripts/spk_vecs.npz)."""
import sys, io, zipfile, numpy as np, soundfile as sf, torch
sys.stdout.reconfigure(encoding='utf-8')
from scipy.signal import resample_poly
zpath, npz = sys.argv[1], sys.argv[2]
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
w = Qwen3TTSModel.from_pretrained('C:/Users/lyang/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3', dtype=torch.bfloat16, attn_implementation='sdpa')
enc = w.model.speaker_encoder.to('cuda').eval()
d = np.load(npz); names = list(d['names']); allv = np.concatenate([d[f'v{i}'] for i in range(len(names))]); mu = allv.mean(0)
def nrm(a): return a / np.linalg.norm(a, axis=-1, keepdims=True)
zf = zipfile.ZipFile(zpath); groups = {}
for i in zf.infolist():
    if i.is_dir(): continue
    x, sr = sf.read(io.BytesIO(zf.read(i)), dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    if len(x) < sr: continue
    g = np.gcd(sr, 24000); x = resample_poly(x, 24000 // g, sr // g).astype('float32')
    with torch.inference_mode():
        m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        groups.setdefault(i.filename.split('/')[0], []).append(enc(m.to('cuda').to(torch.bfloat16))[:1].float().cpu().numpy()[0])
G = {k: np.stack(v) - mu for k, v in groups.items()}
for k, v in G.items():
    idx = np.random.RandomState(0).permutation(len(v))
    print(k, len(v), 'split-half', round(float(nrm(v[idx[::2]].mean(0)) @ nrm(v[idx[1::2]].mean(0))), 3))
a, b = [nrm(v.mean(0)) for v in G.values()]
print('nyamu0 ~ nyamu1 centred cosine', round(float(a @ b), 3))
C = {n: nrm(d[f'v{i}'].mean(0) - mu) for i, n in enumerate(names)}
for k, v in G.items():
    c = nrm(v.mean(0)); near = sorted(((float(c @ C[n]), n) for n in names if n != '祐天寺若麥'), reverse=True)[:3]
    print(k, 'nearest other sets', [(n, round(s, 3)) for s, n in near])
