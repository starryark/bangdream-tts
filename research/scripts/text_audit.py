import sys, io, zipfile, glob, os, re, json, unicodedata, collections
sys.stdout.reconfigure(encoding='utf-8')
root = sys.argv[1]
rows = []
for z in sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True)):
    band = os.path.basename(os.path.dirname(z)); spk = os.path.splitext(os.path.basename(z))[0]
    for i in zipfile.ZipFile(z).infolist():
        if i.is_dir(): continue
        sub = os.path.dirname(i.filename); stem = os.path.splitext(os.path.basename(i.filename))[0]
        rows.append(dict(band=band, spk=spk, sub=sub, stem=stem, bytes=len(os.path.basename(i.filename).encode('utf-8'))))
def norm(t):
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(r'^\([A-Za-z0-9]\)', '', t)
    return re.sub(r'[\s、。，,．.！!？?…‥・〜~ー―\-「」『』（）()♪☆★]+', '', t)
by = collections.defaultdict(list)
for r in rows: by[(r['band'], r['spk'])].append(r)
print('speaker\tn\tdup_norm\tprefix(A)\tlatin\tlong>=200B\tdigits\tmax_bytes\tsubdirs')
for (b, s), rs in by.items():
    c = collections.Counter(norm(r['stem']) for r in rs)
    dup = sum(v - 1 for v in c.values() if v > 1)
    pre = sum(bool(re.match(r'^\([A-Za-z0-9]\)', r['stem'])) for r in rs)
    lat = sum(bool(re.search(r'[A-Za-z]{2,}', re.sub(r'^\([A-Za-z0-9]\)', '', r['stem']))) for r in rs)
    lng = sum(r['bytes'] >= 200 for r in rs)
    dig = sum(bool(re.search(r'[0-9０-９]', r['stem'])) for r in rs)
    subs = sorted(set(r['sub'] for r in rs))
    print(f"{s}\t{len(rs)}\t{dup}\t{pre}\t{lat}\t{lng}\t{dig}\t{max(r['bytes'] for r in rs)}\t{subs if subs!=[''] else ''}")
# cross-speaker overlap of identical normalized lines
idx = collections.defaultdict(set)
for r in rows: idx[norm(r['stem'])].add(r['spk'])
pairs = collections.Counter()
for k, v in idx.items():
    if len(v) > 1 and len(k) > 4:
        v = sorted(v)
        for a in range(len(v)):
            for b2 in range(a + 1, len(v)): pairs[(v[a], v[b2])] += 1
print('\ncross-speaker identical lines (>4 chars):', pairs.most_common(15))
# examples of latin / prefix / digit
ex = [r['stem'] for r in rows if re.search(r'[A-Za-z]{2,}', r['stem'])][:12]
print('\nlatin examples:', ex)
print('\ndigit examples:', [r['stem'] for r in rows if re.search(r'[0-9０-９]', r['stem'])][:10])
print('\nlongest names:', sorted(rows, key=lambda r: -r['bytes'])[:3])
chars = sum(len(norm(r['stem'])) for r in rows); print('\ntotal normalized chars', chars, 'rows', len(rows))
