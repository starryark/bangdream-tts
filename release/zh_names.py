"""Chinese names for the voice tables, voices.json files and the Chinese README (data: release/zh_names.json).

  python -I -X utf8 release/zh_names.py card IN.md OUT.md       # add a 中文名 column to a card's voice table
  python -I -X utf8 release/zh_names.py voices IN.json OUT.json # add "zh" (official name) to each voice entry
  python -I -X utf8 release/zh_names.py lists CARD.md [...]     # Mandarin own-voice / generic lists, in Chinese

zh_names.json holds, per speaker id: {"zh": official simplified name, "nick": [...], "confidence", "source"}.
Persona and era ids get a suffix: ro_ako_maou (大魔姬), *_tsuki (月之森), *_ournotes (Our Notes).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BD = Path(__file__).resolve().parent.parent
NAMES = json.loads((BD / "release" / "zh_names.json").read_text(encoding="utf-8"))
ROW = re.compile(r"^\| `([a-z0-9_]+)` \|")


def zh(sid: str, nick: bool = True) -> str:
    suffix = {"_ournotes": "（Our Notes）", "_tsuki": "（月之森）"}
    end = next((k for k in suffix if sid.endswith(k)), "")
    base = sid[:-len(end)] if end else sid
    e = NAMES.get(sid) if sid in NAMES else NAMES.get(base, {})
    name = e.get("zh", "") if isinstance(e, dict) else ""
    if not name:
        return ""
    if nick and e.get("nick"):
        name += f"〔{e['nick'][0]}〕"
    return name + suffix.get(end, "")


def card(src: str, dst: str) -> None:
    lines = Path(src).read_text(encoding="utf-8").split("\n")
    out, n = [], 0
    for l in lines:
        if l.startswith("| `speaker` | Character |"):
            l = l.replace("| `speaker` | Character |", "| `speaker` | Character | 中文名 |", 1)
        elif out and out[-1].startswith("| `speaker` | Character | 中文名 |") and l.startswith("|---|---|"):
            l = "|---|---|---|" + l[len("|---|---|"):]
        elif ROW.match(l) and l.count("|") >= 5:
            sid = ROW.match(l).group(1)
            parts = l.split(" | ")
            parts.insert(2, zh(sid) or "—")
            l = " | ".join(parts)
            n += 1
        out.append(l)
    Path(dst).write_text("\n".join(out), encoding="utf-8")
    print(f"[zh] {n} table rows -> {dst}")


def voices(src: str, dst: str) -> None:
    v = json.loads(Path(src).read_text(encoding="utf-8"))
    for sid, e in v.items():
        if isinstance(e, dict) and zh(sid, nick=False):
            e["zh"] = zh(sid, nick=False)
    Path(dst).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[zh] {sum(isinstance(e, dict) and 'zh' in e for e in v.values())} voices with zh -> {dst}")


def lists(cards: list[str]) -> dict[str, list[str]]:
    out = {"own voice": [], "partly generic": [], "**generic**": []}
    for c in cards:
        for l in Path(c).read_text(encoding="utf-8").split("\n"):
            m = ROW.match(l)
            if m:
                kind = l.rstrip(" |").split(" | ")[-1].strip()
                if kind in out:
                    out[kind].append(f"{zh(m.group(1), nick=False)}（`{m.group(1)}`）")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "card":
        card(sys.argv[2], sys.argv[3])
    elif cmd == "voices":
        voices(sys.argv[2], sys.argv[3])
    elif cmd == "lists":
        for k, v in lists(sys.argv[2:]).items():
            print(f"{k} ({len(v)}): " + "、".join(v))
    else:
        raise SystemExit(__doc__)
