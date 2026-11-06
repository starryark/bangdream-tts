"""Shared helpers for the BanG Dream pipeline.

Reuses the Kurisu v3 text and audio policy from kurisu_v2/pod/p03_build.py (reading-space
CER, cer_bar, normalise_audio: 24 kHz, trim to speech, -23 LUFS with gain clamped to
-6..+9 dB, 0.12 s lead / 0.25 s tail pad), with the OpenJTalk readings table swapped for
BanG Dream names.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# BD_ROOT / BD_DIR / BD_KPOD / BD_BASE let the same module run on the pod (/workspace/bd)
ROOT = Path(os.environ.get("BD_ROOT", r"E:\STEINS;GATEVoice"))
BD = Path(os.environ.get("BD_DIR", ROOT / "bangdream"))
KPOD = Path(os.environ.get("BD_KPOD", ROOT / "kurisu_v2" / "pod"))
DATA = Path(os.environ.get("BD_DATA", BD / "data"))   # BD_DATA / BD_OUT: smoke-test overrides
OUT = Path(os.environ.get("BD_OUT", BD))
SRC = Path(r"E:\BaiduNetdiskDownload\bangdreamai\角色语音")
QWEN_BASE = os.environ.get("BD_BASE", r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
                                      r"\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
for p in (KPOD, BD / ".pydeps", BD / ".pystubs"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import p03_build  # noqa: E402  (kurisu_v2/pod)

SR = p03_build.SR
cer, cer_bar, normalise_audio = p03_build.cer, p03_build.cer_bar, p03_build.normalise_audio

# OpenJTalk misreads most of the cast (有咲 -> ユーサキ, 燈 -> アカリ, はぐみ -> ワグミ).
# Both the script and the ASR output go through this table, so they meet in katakana.
# Replacement is sequential str.replace in this order: longer / guard keys first.
BD_READINGS = {
    "彩り": "イロドリ", "色彩": "シキサイ", "睦まじ": "ムツマジ",
    "要楽奈": "カナメラーナ", "三角初華": "ミスミウイカ", "三角さん": "ミスミサン",
    "白金燐子": "シロカネリンコ", "白金さん": "シロカネサン", "祐天寺": "ユウテンジ",
    "有咲": "アリサ", "沙綾": "サアヤ", "日菜": "ヒナ", "千聖": "チサト", "友希那": "ユキナ",
    "紗夜": "サヨ", "はぐみ": "ハグミ", "七深": "ナナミ", "瑠唯": "ルイ", "令王那": "レオナ",
    "愛音": "アノン", "楽奈": "ラーナ", "立希": "タキ", "初華": "ウイカ", "海鈴": "ウミリ",
    "祥子": "サキコ", "祥ちゃん": "サキチャン", "若麦": "ニャム", "美竹": "ミタケ",
    "和奏": "ワカナ", "珠手": "タマデ", "豊川": "トガワ", "鳰原": "ニュウバラ",
    "花女": "ハナジョ", "蔵練": "クラレン", "燈": "トモリ", "睦": "ムツミ", "彩": "アヤ",
    "Poppin'Party": "ポッピンパーティー", "Afterglow": "アフターグロウ",
    "Pastel*Palettes": "パステルパレット", "Roselia": "ロゼリア", "Morfonica": "モルフォニカ",
    "RAISE A SUILEN": "レイズアスイレン", "MyGO": "マイゴ", "Ave Mujica": "アヴェムジカ",
    "CiRCLE": "サークル", "RiNG": "リング", "Chu²": "チュチュ",
}
# metrics._norm_text lowercases before reading(): add lowercase twins of Latin-script keys
BD_READINGS.update({k.lower(): v for k, v in list(BD_READINGS.items()) if k != k.lower()})
p03_build.READINGS.clear()
p03_build.READINGS.update(BD_READINGS)
reading = p03_build.reading

PREFIX_RE = re.compile(r"^\([A-Za-z0-9]\)")


def clean_text(stem: str) -> str:
    """Training text from a filename stem: drop the `(A)` take prefix, NFC, trim spaces."""
    t = PREFIX_RE.sub("", unicodedata.normalize("NFC", stem)).strip()
    return p03_build.normalise_text(t)


def load_rows() -> dict:
    rows = json.loads((BD / "pipeline" / "rows.json").read_text(encoding="utf-8"))
    rows.pop("_note", None)
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8", newline="\n")
    tmp.replace(path)
