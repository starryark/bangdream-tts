"""E0 step 1: zips -> deduplicated 24 kHz mono WAVs + manifest.

  python -I b01_build.py

Reads every zip listed in rows.json straight from the read-only source folder (no
extraction to disk), drops byte-identical duplicates (first occurrence in rows.json order
wins; 梦中情祥 is not in rows.json at all), resamples to 24 kHz and applies the Kurisu v3
audio policy (bdcommon.normalise_audio). Writes
  data/wav24/<row>/<id>.wav          PCM16, id = <row>_<sha256[:10]>
  data/manifest.jsonl                one row per kept clip
  data/build_report.json             counts, duplicates, rejects
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdcommon as C  # noqa: E402


def process_zip(key: str, meta: dict) -> tuple[list[dict], list[dict]]:
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    from scipy.signal import resample_poly

    meter = pyloudnorm.Meter(C.SR)
    band, stem = key.split("/", 1)
    out_dir = C.DATA / "wav24" / meta["row"]
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, rejects = [], []
    with zipfile.ZipFile(C.SRC / band / f"{stem}.zip") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            blob = zf.read(info)
            sha = hashlib.sha256(blob).hexdigest()
            name = info.filename.rsplit("/", 1)[-1]
            sub = info.filename.rsplit("/", 1)[0] if "/" in info.filename else ""
            text_raw = os.path.splitext(name)[0]
            base = {"sha256": sha, "row": meta["row"], "set": key, "sub": sub, "text_raw": text_raw}
            try:
                wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
            except Exception as e:  # pragma: no cover - reported, not fatal
                rejects.append({**base, "reason": f"decode:{type(e).__name__}"})
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            dur_src = len(wav) / sr
            if sr != C.SR:
                g = gcd(sr, C.SR)
                wav = resample_poly(wav, C.SR // g, sr // g).astype("float32")
            new, gain_db = C.normalise_audio(wav, meter)
            if new is None:
                rejects.append({**base, "reason": "no_audible_speech"})
                continue
            rid = f"{meta['row']}_{sha[:10]}"
            path = out_dir / f"{rid}.wav"
            sf.write(str(path), new, C.SR, subtype="PCM_16")
            kept.append({
                "id": rid, **base, "character": meta["character"], "band": meta["band"],
                "persona": meta.get("persona"), "source": meta["source"],
                "fmt": os.path.splitext(name)[1].lower().lstrip("."), "sr_src": sr,
                "text": C.clean_text(text_raw), "dur_src": round(dur_src, 3),
                "dur": round(len(new) / C.SR, 3), "gain_db": round(gain_db, 2),
                "audio": str(path),
            })
    return kept, rejects


def main() -> None:
    rows = C.load_rows()
    C.DATA.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        results = list(ex.map(process_zip, rows.keys(), rows.values()))
    seen, manifest, dups, rejects = {}, [], [], []
    for (kept, rej), key in zip(results, rows.keys()):
        rejects += rej
        for r in kept:
            if r["sha256"] in seen:
                kept_id, kept_audio = seen[r["sha256"]]
                dups.append({"id": r["id"], "dup_of": kept_id, "set": key})
                if r["audio"] != kept_audio:      # a dup inside one set has the same id and path
                    os.remove(r["audio"])
                continue
            seen[r["sha256"]] = (r["id"], r["audio"])
            manifest.append(r)
    C.write_jsonl(C.DATA / "manifest.jsonl", manifest)
    per_row = {}
    for r in manifest:
        d = per_row.setdefault(r["row"], {"n": 0, "min": 0.0})
        d["n"] += 1
        d["min"] = round(d["min"] + r["dur"] / 60, 2)
    report = {"kept": len(manifest), "duplicates_dropped": len(dups), "rejects": rejects,
              "hours": round(sum(r["dur"] for r in manifest) / 3600, 2), "per_row": per_row,
              "duplicates": dups}
    (C.DATA / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"kept {len(manifest)}  dups {len(dups)}  rejects {len(rejects)}  "
          f"{report['hours']} h  rows {len(per_row)}")


if __name__ == "__main__":
    main()
