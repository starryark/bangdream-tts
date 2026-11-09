"""Discovery pass on the two HF model cards (2026-11-09): canonical description, aliases, a Chinese section with the
Mandarin own-voice / generic lists, an FAQ, a 中文名 column, and the item-10 + Chinese AI-labelling wording.

  python -I -X utf8 release/patch_cards_zh.py MAIN_IN.md MAIN_OUT.md BRANCH_IN.md BRANCH_OUT.md

Line endings of each input are kept (the live main card is CRLF). A one-time pass: its outputs are now release/MODEL_CARD.md and
release/MODEL_CARD_ournotes.md, and rerunning it on them fails its asserts by design.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zh_names as Z  # noqa: E402

CANON = ("**BanG Dream! (邦邦) character-voice TTS: 49 voices + 25 opt-in Our Notes voices (MyGO!!!!!, Ave Mujica, "
         "Girls Band Party, Our Notes). They speak Japanese, Mandarin and English with no reference audio. A Qwen3-TTS fine-tune. "
         "Unofficial fan project, not affiliated with Bushiroad, Craft Egg, the BanG Dream! project, the official "
         "夢ノ結唱 voicebanks or the voice cast.**")
AKA = ("**Also known as:** 邦邦 · 邦多利 · BanG Dream! 少女乐团派对! (Girls Band Party) · 《迷途之子!!!!!》 (MyGO!!!!!) · "
       "《颂乐人偶》 (Ave Mujica) · BanG Dream! 交织的乐章 (Our Notes) · バンドリ！ ガールズバンドパーティ！ · ガルパ.")
RIGHTS_OLD = ("- The official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) restricts derivative works "
              "that use the franchise's audio. Read it before using outputs anywhere public.")
RIGHTS_NEW = ("- **The official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) prohibits derivative works "
              "that use the franchise's audio.** Item 10 reads 「本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開」. "
              "This model is an unofficial fan work made despite that. Read the guideline before using outputs anywhere public.\n"
              "- **AI labels:** in mainland China, AI-generated audio must carry an explicit AI label (《人工智能生成合成内容标识办法》, "
              "in force since 2025-09-01), and bilibili asks uploaders to set its AI标识 when uploading.")


def faq(branch: bool) -> str:
    rev = 'revision="ournotes", ' if branch else ""
    spk = "ikka_raika" if branch else "mygo_anon"
    return f"""## FAQ

**Does it need reference audio?**
No. Every voice is a built-in speaker row, called by its id (`speaker="{spk}"`).

**Can the characters speak Chinese?**
Yes. Pass `language="Chinese"`. Every voice speaks Mandarin with a native accent. Most voices also keep their own timbre in Mandarin; those that drift toward a generic timbre are marked in the table and listed in the Chinese section.

**Which revision should I load?**
- `main` for the 49 Girls Band Party / MyGO!!!!! / Ave Mujica voices.
- `revision="ournotes"` for the 16 Our Notes characters and the `*_ournotes` voices.
- The [demo](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS) picks the right one automatically.

**How is it different from GPT-SoVITS, RVC or Bert-VITS2 character models?**
- It is one checkpoint with every voice built in, so it needs no reference clip and no source recording.
- It was trained so that its Mandarin is not Japanese-accented.
- Its checkpoints were chosen by rules fixed before training, and the later ones by blind listening.

**What hardware does it need?**
It is a fine-tune of Qwen3-TTS-12Hz-1.7B-Base; the bf16 weights are 3.8 GB, and inference runs on a CUDA GPU (`pip install qwen-tts`). Without a GPU, use the demo.

**Can I download it from mainland China?**
Set `HF_ENDPOINT=https://hf-mirror.com` before `from_pretrained("starrydark/BanG_Dream_Qwen3_TTS", {rev}...)`.

**Can I post what I make?**
Only as a labelled, non-commercial fan work. The BanG Dream! fan-work guideline prohibits derivative works that use the franchise's audio (item 10), and China requires an explicit AI label on generated audio. Never impersonate the cast or present output as official. See [Rights and responsible use](#rights-and-responsible-use).

**Is this official, or related to 夢ノ結唱 (梦的结唱)?**
No. This is an unofficial fan project. 夢ノ結唱 is Bushiroad Music's official AI singing product and is unrelated.

"""


def zh_section(card_path: str, branch: bool) -> str:
    L = Z.lists([card_path])
    own, part, gen = L["own voice"], L["partly generic"], L["**generic**"]
    scope = ("本分支（`ournotes`）是 25 个《BanG Dream! Our Notes》（交织的乐章）声线：一家Dumb Rock!、梦限大MewType、millsage 的 15 位新角色、"
             "泽海奏多，以及 MyGO!!!!!／Ave Mujica 9 位角色的 Our Notes 版本（`*_ournotes`）。其余 49 个声线请使用 `main`。"
             if branch else
             "本模型（`main`）包含少女乐团派对 7 个乐队、MyGO!!!!!（《迷途之子!!!!!》）和 Ave Mujica（《颂乐人偶》）共 49 个声线；"
             "《BanG Dream! Our Notes》的 25 个声线在可选分支 `ournotes`。")
    rev = 'revision="ournotes", ' if branch else ""
    spk = "ikka_raika" if branch else "mygo_anon"
    return f"""## 中文说明

非官方 BanG Dream!（邦邦）角色语音合成模型，基于 Qwen3-TTS 微调。按说话人 ID 直接合成，**无需参考音频**；会说日语、普通话和英语。粉丝自制，与 Bushiroad、Craft Egg、BanG Dream! 官方、官方 AI 声库“梦的结唱”及声优均无关。

{scope}在线体验：[演示 Space](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS)（74 个声线，可直接输入中文名搜索）。

```python
import os; os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 中国大陆网络
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts
tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS", {rev}device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(text="今天的排练真开心，明天也一起加油吧！", speaker="{spk}", language="Chinese")
sf.write("out.wav", wavs[0], sr)
```

**普通话音色（说中文时像不像本人）：**

| 程度 | 数量 | 声线 |
|---|---:|---|
| 保持角色音色 | {len(own)} | 下表中标为 own voice 的声线 |
| 部分偏通用 | {len(part)} | {"、".join(part) or "—"} |
| 偏通用音色 | {len(gen)} | {"、".join(gen) or "—"} |

“偏通用”的意思是：口音仍是标准普通话，但音色更接近基础模型的默认声音，而不是角色本人。

**使用规则：**
- 生成的音频均为 AI 合成，发布时须明确标注“AI生成”（《人工智能生成合成内容标识办法》，2025-09-01 起施行）。在 bilibili 投稿时，请在上传页勾选 AI 标识。
- 禁止商用。禁止冒充声优或官方，禁止用于损害角色及声优形象的内容。
- BanG Dream! 官方二创指南（<https://bang-dream.com/bdp-guideline/>）第 10 条禁止使用作品音声制作、公开二次创作。本模型是在此前提下的非官方粉丝作品。权利方或声优提出要求时，模型与演示将立即下架。

"""


def patch(src: str, dst: str, branch: bool) -> None:
    raw = open(src, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    s = raw.replace("\r\n", "\n")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "c.md"
        tmp.write_text(s, encoding="utf-8")
        Z.card(str(tmp), str(tmp))
        s = tmp.read_text(encoding="utf-8")
        zh = zh_section(str(tmp), branch)
    assert s.count(RIGHTS_OLD) == 1, "rights line"
    s = s.replace(RIGHTS_OLD, RIGHTS_NEW)
    if branch:
        h1 = "# BanG Dream! character voices, `ournotes` branch: 25 BanG Dream! Our Notes voices (Qwen3-TTS)"
        assert s.count(h1) == 1
        s = s.replace(h1, "# BanG Dream! (邦邦) character voices, `ournotes` branch: 25 BanG Dream! Our Notes (交织的乐章) voices on Qwen3-TTS")
        anchor = "| | |\n|---|---|\n| **Voices recommended here**"
        assert s.count(anchor) == 1
        s = s.replace(anchor, CANON + "\n\n" + AKA + "\n\n" + anchor)
        before = "## How this branch was evaluated"
    else:
        h1 = "# BanG Dream! character voices: Qwen3-TTS text-to-speech in Japanese, Mandarin and English"
        assert s.count(h1) == 1
        s = s.replace(h1, "# BanG Dream! (邦邦 / バンドリ) character voices: Qwen3-TTS text-to-speech in Japanese, Mandarin and English")
        intro = s[s.index("An unofficial, fan-made **text-to-speech model with 49"):]
        intro = intro[:intro.index("\n")]
        s = s.replace(intro, CANON + "\n\nThis revision (`main`) holds the 49 Girls Band Party, MyGO!!!!! and Ave Mujica voices. "
                      "Persona variants such as 大魔姫あこ, 月ノ森 Soyo / Sakiko and Mortis have their own rows.")
        old_aka = "**Also known as:** BanG Dream! Girls Band Party · バンドリ！ ガールズバンドパーティ！ · ガルパ · 邦邦 · 少女乐团派对."
        assert s.count(old_aka) == 1
        s = s.replace(old_aka, AKA)
        before = "## Evaluation"
    assert s.count("\n## Voices\n") == 1 and s.count(f"\n{before}\n") == 1
    s = s.replace("\n## Voices\n", "\n" + zh + "## Voices\n")
    s = s.replace(f"\n{before}\n", "\n" + faq(branch) + f"{before}\n")
    Path(dst).write_text(s.replace("\n", nl), encoding="utf-8", newline="")
    print(f"[cards] {dst}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    patch(sys.argv[1], sys.argv[2], branch=False)
    patch(sys.argv[3], sys.argv[4], branch=True)
