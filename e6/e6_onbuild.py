"""E6 step 1 (local, TTS venv): the Our Notes clip manifest and 24 kHz audio (E6_PLAN.md §3.3).

  python -I -X utf8 e6_onbuild.py [--workers 8]

Inputs: the extraction (E:\\BangDream_ournotes\\extracted), and e6/work/{on_meta.jsonl, on_vec.npz,
s04_lines.jsonl} from the 2026-11-06 measurement scripts (e6/prep/s01-s05). Story clips only
(cue-sheet group adv_voice), single speaker, role from the story's TargetName, 0.6-15 s source length.

Rows: 28 Our Notes rows from 27 roles (Sakiko split by arc: lines in adv_script_mygo_* ->
mujica_sakiko_tsuki, all other Sakiko lines -> mujica_sakiko). Amoris folds into Nyamu and Timoris
into Umiri; every other persona or side role is excluded (facts §7.2).

Text (facts §5): <ruby=R>B</ruby> and <r=R>B</r> -> R (the reading the audio uses; B when R is only
emphasis dots, 傍点), other tags
stripped, newlines removed, a line wrapped in （） is unwrapped and flagged mono, then the Kurisu v3
normalise_text. kind = "nv" if nothing speakable is left once marks, っ/ッ, ー and brackets are
removed (ellipsis-only and nonverbal-only lines: E6c material), else "line".

Split (pre-run clarification 3b): line-level holdout, split_lines(): per row 24 short-eligible and 12
held-eligible lines (sha256 order, distinct texts) are test; same-row clips sharing a test text never
train ("held_text"). An episode split gave 124 units and left 12 rows with no test lines; a scene split
needed ~25% of all clips and still found 2-6 short lines for some rows. The clip probe's group folds
use episode = <arc>_<chapter>_<part>.

Audio: the v1 policy (bdcommon.normalise_audio: 24 kHz, trim to speech, -23 LUFS, gain clamped,
0.12 s lead / 0.25 s tail pad) -> bangdream/data_on/wav24/<row>/<cue>.wav. Existing files are kept.
Writes bangdream/data_on/manifest.jsonl and prints counts per row and split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BD = HERE.parent
sys.path.insert(0, str(BD / "pipeline"))
import bdcommon as C  # noqa: E402

EXT = Path(r"E:\BangDream_ournotes\extracted")
SA = EXT / "server_assets"
OUT = BD / "data_on"
WORK = HERE / "work"

ROLE_ROW = {
    "tomori": "mygo_tomori", "anon": "mygo_anon", "rana": "mygo_raana", "soyo": "mygo_soyo", "taki": "mygo_taki",
    "uika": "mujica_uika", "mutsumi": "mujica_mutsumi", "mortis": "mujica_mortis", "umiri": "mujica_umiri",
    "timoris": "mujica_umiri", "nyamu": "mujica_nyamu", "amoris": "mujica_nyamu",
    "sakiko": None,  # split by arc below
    "arale": "mewtype_arale", "nonoka": "mewtype_nonoka", "ritsu": "mewtype_ritsu", "miyako": "mewtype_miyako",
    "yuno": "mewtype_yuno",
    "hotaru": "millsage_hotaru", "natsume": "millsage_natsume", "nagi": "millsage_nagi", "mahoro": "millsage_mahoro",
    "houka": "millsage_houka",
    "raika": "ikka_raika", "miku": "ikka_miku", "yomogi": "ikka_yomogi", "chieri": "ikka_chieri",
    "shizuku": "ikka_shizuku",
    "kanata": "ournotes_kanata",
}
ROLE_OF = {"timoris": "umiri", "amoris": "nyamu"}          # the probe role a folded persona counts as
ROWS = sorted({v for v in ROLE_ROW.values() if v} | {"mujica_sakiko", "mujica_sakiko_tsuki"})
MARKS = r"[…‥・。、，．！？!?〜～ー―\-っッ「」『』（）()\s♪☆★♡]"


def _ruby(m: re.Match) -> str:
    """A ruby reading replaces its base text, except emphasis marks (傍点: a reading of only dots)."""
    reading = re.sub(r"[\s　]", "", m.group(1))
    base = re.sub(r"</?[a-zA-Z]+(=[^>]*)?>", "", m.group(2))
    return base if not reading or set(reading) <= set("・.．·•") else reading


def clean(t: str) -> tuple[str, bool]:
    t = re.sub(r"<ruby=([^>]*)>(.*?)</ruby>", _ruby, t or "")
    t = re.sub(r"<r=([^>]*)>(.*?)</r>", _ruby, t)
    t = re.sub(r"</?[a-zA-Z]+(=[^>]*)?>", "", t)
    t = t.replace("\r", "").replace("\n", "").strip()
    mono = t.startswith("（") and t.endswith("）")
    if mono:
        t = t[1:-1]
    return C.p03_build.normalise_text(t), mono


def episode(asset: str) -> str:
    p = asset.split("_")          # adv_script_<arc>_<chapter>_<part>_<scene>
    return "_".join(p[2:5])


LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]")
SHORT_T, HELD_T = 24, 12      # drawn with margin; e6_rows keeps up to 16 / 8 that pass the gates


def split_lines(rows: list[dict]) -> None:
    """Line-level holdout (as v1's E0 test split): per row, the first SHORT_T short-eligible (0.7-1.6 s,
    >= 3 kana) and HELD_T held-eligible (3-8 s, >= 5 kana) lines in sha256(id) order, no Latin or digits,
    distinct texts -> "test". Any other clip of the same row with a test line's text -> "held_text" (never
    trains). Everything else -> "train"."""
    h = lambda r: int(hashlib.sha256(f"e6:{r['id']}".encode()).hexdigest(), 16)   # noqa: E731
    ok = lambda r: r["kind"] == "line" and not LATIN.search(r["text"])             # noqa: E731
    for r in rows:
        r["split"] = "train"
    for row in sorted({r["row"] for r in rows}):
        mine = sorted([r for r in rows if r["row"] == row], key=h)
        texts = set()
        for cond, n in ((lambda r: r["kana"] >= 3 and 0.7 <= r["dur_src"] <= 1.6, SHORT_T),
                        (lambda r: r["kana"] >= 5 and 3.0 <= r["dur_src"] <= 8.0, HELD_T)):
            k = 0
            for r in mine:
                if k >= n:
                    break
                if r["split"] == "train" and ok(r) and cond(r) and r["text"] not in texts:
                    r["split"] = "test"
                    texts.add(r["text"])
                    k += 1
        for r in mine:
            if r["split"] == "train" and r["text"] in texts:
                r["split"] = "held_text"


def exp_class(name: str | None) -> str:
    if not name:
        return "none"
    return re.sub(r"\d+$", "", name.replace("exp_", "")).rstrip("_") or "none"


def audio_job(job: tuple[str, str]) -> tuple[str, float | None]:
    import numpy as np  # noqa: F401
    import pyloudnorm
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    src, dst = job
    d = Path(dst)
    if d.exists():
        return dst, sf.info(dst).duration
    wav, sr = sf.read(src, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != C.SR:
        g = gcd(sr, C.SR)
        wav = resample_poly(wav, C.SR // g, sr // g).astype("float32")
    new, _ = C.normalise_audio(wav, pyloudnorm.Meter(C.SR))
    if new is None:
        return dst, None
    d.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, new, C.SR, subtype="PCM_16")
    return dst, len(new) / C.SR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import numpy as np
    meta_all = C.read_jsonl(WORK / "on_meta.jsonl")
    vidx = {int(i): k for k, i in enumerate(np.load(WORK / "on_vec.npz")["idx"])}   # meta index -> vector row
    lines = {r["cue"]: r for r in C.read_jsonl(WORK / "s04_lines.jsonl")}
    rows, skip = [], Counter()
    for mi, m in enumerate(meta_all):
        if m["group"] != "adv_voice":
            continue
        L = lines.get(m["cue"])
        role = m.get("target")
        if L is None or role not in ROLE_ROW:
            skip["role"] += 1
            continue
        if len(L.get("speakers") or []) != 1 or L.get("n_cues", 1) != 1:
            skip["multi"] += 1
            continue
        if not (0.6 <= m["dur"] <= 15.0):
            skip["length"] += 1
            continue
        if mi not in vidx:
            skip["no_vector"] += 1
            continue
        asset = L["asset"]
        arc = asset.split("_")[2]
        row = ROLE_ROW[role] or ("mujica_sakiko_tsuki" if arc == "mygo" else "mujica_sakiko")
        text, mono = clean(m["ja"])
        core = re.sub(MARKS, "", text)
        ep = episode(asset)
        rows.append({"id": m["cue"], "row": row, "role": ROLE_OF.get(role, role), "target": role, "arc": arc,
                     "episode": ep, "scene": asset, "split": None, "kind": "line" if core else "nv", "mono": mono,
                     "exp": exp_class(L.get("exp")), "exp_raw": L.get("exp"), "stale": L.get("stale"),
                     "text": text, "ja_raw": m["ja"], "zh": L.get("zh"), "en": L.get("en"),
                     "kana": len(C.reading(text)) if core else 0, "dur_src": m["dur"],
                     "src": str(SA / m["path"]), "audio": str(OUT / "wav24" / row / f"{m['cue']}.wav"),
                     "vec": vidx[mi]})
    split_lines(rows)
    print(f"[onbuild] {len(rows)} clips; skipped {dict(skip)}; split {dict(Counter(r['split'] for r in rows))}", flush=True)
    ruby = [r for r in rows if "<r" in (r["ja_raw"] or "")]
    print("[onbuild] ruby/r examples:", [(r["ja_raw"][:40], r["text"][:30]) for r in ruby[:6]], flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(audio_job, [(r["src"], r["audio"]) for r in rows], chunksize=64))
    keep = []
    for r in rows:
        d = res[r["audio"]]
        if d is None:
            skip["no_audible_speech"] += 1
            continue
        r["dur"] = round(d, 3)
        keep.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(OUT / "manifest.jsonl", keep)
    per = Counter((r["row"], r["split"], r["kind"]) for r in keep)
    for row in ROWS:
        print(f"[onbuild] {row:22s} train {per[(row, 'train', 'line')]:5d} (+{per[(row, 'train', 'nv')]} nv)  "
              f"test {per[(row, 'test', 'line')]:4d} (+{per[(row, 'test', 'nv')]} nv)", flush=True)
    eps = Counter(r["split"] for r in {r["episode"]: r for r in keep}.values())
    print(f"[onbuild] kept {len(keep)}; episodes {dict(eps)}; skipped {dict(skip)}", flush=True)


if __name__ == "__main__":
    main()
