"""Qwen3-TTS speaker-encoder clip vectors (all clips >=1 s, up to 80 per set) -> mean-centred analysis."""
import sys, io, zipfile, glob, os, random, numpy as np, soundfile as sf, torch
sys.stdout.reconfigure(encoding='utf-8')
from scipy.signal import resample_poly
root, out = sys.argv[1], sys.argv[2]; K = 80; random.seed(1); dev = 'cuda'
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
w = Qwen3TTSModel.from_pretrained('C:/Users/lyang/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3', dtype=torch.bfloat16, attn_implementation='sdpa')
enc = w.model.speaker_encoder.to(dev).eval()
names, vecs = [], {}
for z in sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True)):
    spk = os.path.splitext(os.path.basename(z))[0]
    if spk == '梦中情祥': continue
    zf = zipfile.ZipFile(z); items = [i for i in zf.infolist() if not i.is_dir()]
    vs = []
    for i in random.sample(items, min(K, len(items))):
        x, sr = sf.read(io.BytesIO(zf.read(i)), dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if len(x) < sr: continue
        g = np.gcd(sr, 24000); x = resample_poly(x, 24000 // g, sr // g).astype('float32')
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(x).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            vs.append(enc(m.to(dev).to(torch.bfloat16))[:1].float().cpu().numpy()[0])
    names.append(spk); vecs[spk] = np.stack(vs)
np.savez(out, names=np.array(names), **{f'v{i}': vecs[n] for i, n in enumerate(names)})
mu = np.concatenate([vecs[n] for n in names]).mean(0)
def nrm(a): return a / np.linalg.norm(a, axis=-1, keepdims=True)
# split-half: centroid of half A vs half B within a set (same-speaker ceiling), vs between sets
cent = {}; half_same = {}
for n in names:
    v = vecs[n] - mu; idx = np.random.RandomState(0).permutation(len(v)); a, b = v[idx[::2]].mean(0), v[idx[1::2]].mean(0)
    half_same[n] = float(nrm(a) @ nrm(b)); cent[n] = nrm(v.mean(0))
C = np.stack([cent[n] for n in names]); S = C @ C.T; N = len(names)
off = S[~np.eye(N, dtype=bool)]
print(f'centred centroid cosine between sets: median {np.median(off):.3f}, p95 {np.percentile(off,95):.3f}, max {off.max():.3f}')
print(f'split-half same-set cosine: median {np.median(list(half_same.values())):.3f}, min {min(half_same.values()):.3f} ({min(half_same, key=half_same.get)})')
pairs = sorted(((S[i, j], names[i], names[j]) for i in range(N) for j in range(i + 1, N)), reverse=True)[:12]
print('closest pairs (centred):'); [print(f'  {s:.3f} {a} ~ {b}') for s, a, b in pairs]
ix = {n: i for i, n in enumerate(names)}
for a, b in [('豐川祥子', '豐川祥子（月ノ森限定）'), ('若葉睦', '墨提斯'), ('宇田川亞子', '大魔姬亞子'), ('長崎素世', '長崎素世（月ノ森限定）'), ('冰川紗夜', '冰川日菜')]:
    print(f'persona/related pair {a} ~ {b}: centred {S[ix[a], ix[b]]:.3f} (split-half {half_same[a]:.3f} / {half_same[b]:.3f})')
# centred clip nearest-centroid accuracy
acc = {n: float(((nrm(vecs[n] - mu) @ C.T).argmax(1) == ix[n]).mean()) for n in names}
print(f'centred clip->centroid acc mean {np.mean(list(acc.values())):.1%}; worst', sorted(acc.items(), key=lambda t: t[1])[:6])
