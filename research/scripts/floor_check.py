import sys, io, zipfile, glob, os, numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8')
root = sys.argv[1]; names = sys.argv[2].split(',')
for z in sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True)):
    spk = os.path.splitext(os.path.basename(z))[0]
    if spk not in names: continue
    zf = zipfile.ZipFile(z); fl = []; hf = []
    for i in [i for i in zf.infolist() if not i.is_dir()]:
        x, sr = sf.read(io.BytesIO(zf.read(i)), dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        n = 1024; fr = np.lib.stride_tricks.sliding_window_view(x, n)[::256]
        if len(fr) < 10: continue
        rms = 10 * np.log10(np.mean(fr ** 2, 1) + 1e-12)
        fl.append(np.percentile(rms, 5) - rms.max())
        # spectral flatness of quietest 10% frames (noise ~1, tonal/BGM low, silence n/a)
        q = fr[rms <= np.percentile(rms, 10)] * np.hanning(n)
        S = np.abs(np.fft.rfft(q, axis=1)) ** 2 + 1e-20
        hf.append(np.mean(np.exp(np.mean(np.log(S), 1)) / np.mean(S, 1)))
    fl = np.array(fl); hf = np.array(hf)
    print(f"{spk}\tn={len(fl)}\tfloor>-50: {np.mean(fl>-50):.0%}\tfloor>-60: {np.mean(fl>-60):.0%}\tmedian floor {np.median(fl):.0f}\tflatness(quiet frames) med {np.median(hf):.3f}")
