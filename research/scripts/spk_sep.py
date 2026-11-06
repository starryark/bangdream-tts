"""Per-set speaker vectors from Qwen3-TTS-Base's speaker encoder (same recipe as train_sft.speaker_embedding)
and from WavLM-base-plus-sv; prints nearest neighbours and within/between-set cosine stats."""
import sys, io, zipfile, glob, os, random, json, numpy as np, soundfile as sf, torch
sys.stdout.reconfigure(encoding='utf-8')
from scipy.signal import resample_poly
root, out = sys.argv[1], sys.argv[2]; K = 40
random.seed(0); dev = 'cuda'
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
w = Qwen3TTSModel.from_pretrained('C:/Users/lyang/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3', dtype=torch.bfloat16, attn_implementation='sdpa')
enc = w.model.speaker_encoder.to(dev).eval()
from transformers import AutoFeatureExtractor, WavLMForXVector
fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv'); sv = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(dev).eval()
def load(b):
    x, sr = sf.read(io.BytesIO(b), dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    return x, sr
names, Q, W, clipsQ, clipsW = [], [], [], {}, {}
for z in sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True)):
    spk = os.path.splitext(os.path.basename(z))[0]
    if spk == '梦中情祥': continue
    zf = zipfile.ZipFile(z); items = [i for i in zf.infolist() if not i.is_dir()]
    qs, ws = [], []
    for i in random.sample(items, min(K, len(items))):
        x, sr = load(zf.read(i))
        if len(x) < sr * 1.0: continue
        g = np.gcd(sr, 24000); x24 = resample_poly(x, 24000 // g, sr // g).astype('float32')
        g = np.gcd(sr, 16000); x16 = resample_poly(x, 16000 // g, sr // g).astype('float32')
        with torch.inference_mode():
            m = mel_spectrogram(torch.from_numpy(x24).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000,
                                hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
            qs.append(enc(m.to(dev).to(torch.bfloat16))[:1].float().cpu())
            f = fe(x16, sampling_rate=16000, return_tensors='pt').to(dev)
            ws.append(sv(**f).embeddings.float().cpu())
    qs = torch.cat(qs); ws = torch.cat(ws)
    names.append(spk); clipsQ[spk] = torch.nn.functional.normalize(qs, dim=-1); clipsW[spk] = torch.nn.functional.normalize(ws, dim=-1)
    Q.append(qs.mean(0)); W.append(ws.mean(0))
    print('done', spk, len(qs), flush=True)
def report(tag, cents, clips):
    C = torch.nn.functional.normalize(torch.stack(cents), dim=-1); S = C @ C.T
    n = len(names); off = S[~torch.eye(n, dtype=bool)]
    print(f'\n== {tag}: centroid cosine between sets: median {off.median():.3f}, max {off.max():.3f}')
    pairs = sorted(((float(S[i, j]), names[i], names[j]) for i in range(n) for j in range(i + 1, n)), reverse=True)[:15]
    for s, a, b in pairs: print(f'  {s:.3f}  {a} ~ {b}')
    # clip-level nearest-centroid accuracy (how often a clip's nearest centroid is its own set)
    acc = []; conf = {}
    for k, s in enumerate(names):
        sims = clips[s] @ C.T; pred = sims.argmax(1)
        acc.append(float((pred == k).float().mean()))
        wrong = [names[p] for p in pred.tolist() if p != k]
        if wrong: conf[s] = max(set(wrong), key=wrong.count)
    print(f'  clip->nearest-centroid accuracy: mean {np.mean(acc):.2%}; worst: ' + ', '.join(f'{names[i]} {acc[i]:.0%}->{conf.get(names[i],"")}' for i in np.argsort(acc)[:10]))
    return S, acc
SQ, aQ = report('Qwen3-TTS speaker encoder', Q, clipsQ)
SW, aW = report('WavLM-base-plus-sv', W, clipsW)
json.dump({'names': names, 'qwen_cos': SQ.tolist(), 'wavlm_cos': SW.tolist(), 'qwen_acc': aQ, 'wavlm_acc': aW}, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
