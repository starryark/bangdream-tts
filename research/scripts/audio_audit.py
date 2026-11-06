import sys, io, zipfile, glob, os, random, numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8')
root = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 25
random.seed(0)
def stats(x, sr):
    if x.ndim > 1: x = x.mean(1)
    n = 2048; hop = 512
    if len(x) < n * 2: return None
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    rms = 10 * np.log10(np.mean(fr ** 2, 1) + 1e-12)
    loud = rms > rms.max() - 30
    spec = 10 * np.log10(S[loud].mean(0) + 1e-20)
    ref = spec[(f > 300) & (f < 3000)].max()
    above = f[spec > ref - 70]
    bw = above.max() if len(above) else 0
    # silence at edges (frames below max-40 dB)
    act = np.where(rms > rms.max() - 40)[0]
    lead = act[0] * hop / sr; trail = (len(rms) - 1 - act[-1]) * hop / sr
    floor = np.percentile(rms, 5) - rms.max()   # noise floor relative to peak frame
    peak = 20 * np.log10(np.abs(x).max() + 1e-9)
    return bw, lead, trail, floor, peak, np.mean(rms[loud])
print('speaker\tsr\tbw_kHz(med)\tlead_s\ttrail_s(med/max)\tfloor_dB(med)\tpeak_dBFS(max)\tactive_rms_dB(med)\tclipped')
for z in sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True)):
    zf = zipfile.ZipFile(z); items = [i for i in zf.infolist() if not i.is_dir()]
    res = []; srs = set(); clip = 0
    for i in random.sample(items, min(K, len(items))):
        try:
            x, sr = sf.read(io.BytesIO(zf.read(i)), dtype='float32')
        except Exception as e:
            continue
        srs.add(sr); s = stats(x, sr)
        if s: res.append(s); clip += (np.abs(x) > 0.999).sum() > 3
    a = np.array(res)
    print(f"{os.path.splitext(os.path.basename(z))[0]}\t{sorted(srs)}\t{np.median(a[:,0])/1000:.1f}\t{np.median(a[:,1]):.2f}\t{np.median(a[:,2]):.2f}/{a[:,2].max():.2f}\t{np.median(a[:,3]):.0f}\t{a[:,4].max():.1f}\t{np.median(a[:,5]):.0f}\t{clip}/{len(a)}")
