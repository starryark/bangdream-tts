---
license: other
license_name: unofficial-fan-model
license_link: https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS#rights-and-responsible-use
language:
- ja
- zh
- en
base_model:
- Qwen/Qwen3-TTS-12Hz-1.7B-Base
base_model_relation: finetune
pipeline_tag: text-to-speech
library_name: qwen-tts
tags:
- text-to-speech
- tts
- qwen3-tts
- custom-voice
- character-voice
- multi-speaker
- cross-lingual-tts
- japanese
- mandarin
- english
- bang-dream
- bandori
- mygo
- ave-mujica
- speech-synthesis
- voice-model
- game-voice
---

# BanG Dream! (邦邦) character voices, `ournotes` branch: 25 BanG Dream! Our Notes (交织的乐章) voices on Qwen3-TTS

This branch of the unofficial, fan-made BanG Dream! TTS model is a separate checkpoint (experiment E9). It adds **25 voices trained on BanG Dream! Our Notes story audio:**
- **16 new characters:** 一家Dumb Rock!, 夢限大みゅーたいぷ, millsage and 沢海奏多;
- **the Our Notes era of nine MyGO!!!!! / Ave Mujica characters:** ids ending in `_ournotes`.

**Use this branch only for those 25 voices.** It also contains the 49 voices of the main model (v1), but in a blind listening test its versions of the MyGO!!!!! / Ave Mujica v1 voices were judged worse than v1. For all 49 of those, use `main`.

- **Each voice is called by its speaker id.** No reference audio is needed.
- **Japanese** is the voices' own language. Mandarin and English work, but were only checked automatically (see the table).
- **Try it online:** [demo Space](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS). It serves these 25 voices from this branch and the other 49 from `main`. Its output is AI-generated, under the same terms as this model.

日本語：BanG Dream! Our Notes のストーリー音声で学習した 25 ボイス（新キャラクター 16 名と、MyGO!!!!!／Ave Mujica 9 名の Our Notes 版 `*_ournotes`）を含む非公式ブランチです。その他の 49 ボイスは `main` をお使いください。公式・キャスト・運営とは無関係です。

中文：本分支（非官方）包含以 BanG Dream! Our Notes 剧情语音训练的 25 个声线：16 位新角色，以及 MyGO!!!!!／Ave Mujica 9 位角色的 Our Notes 版本（`*_ournotes`）。其余 49 个声线请使用 `main`。与官方、运营及声优无关。

**BanG Dream! (邦邦) character-voice TTS: 49 voices + 25 opt-in Our Notes voices (MyGO!!!!!, Ave Mujica, Girls Band Party, Our Notes). They speak Japanese, Mandarin and English with no reference audio. A Qwen3-TTS fine-tune. Unofficial fan project, not affiliated with Bushiroad, Craft Egg, the BanG Dream! project, the official 夢ノ結唱 voicebanks or the voice cast.**

**Also known as:** 邦邦 · 邦多利 · BanG Dream! 少女乐团派对! (Girls Band Party) · 《迷途之子!!!!!》 (MyGO!!!!!) · 《颂乐人偶》 (Ave Mujica) · BanG Dream! 交织的乐章 (Our Notes) · バンドリ！ ガールズバンドパーティ！ · ガルパ.

| | |
|---|---|
| **Voices recommended here** | 25 (table below); the checkpoint holds 72 speaker rows and 74 ids |
| **Languages** | Japanese, Mandarin Chinese, English (`language="Japanese" / "Chinese" / "English"`) |
| **Type** | Qwen3-TTS `custom_voice` checkpoint: text in, speech out, no reference audio |
| **Base model** | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), full fine-tune |
| **Version** | E9 `avg` (the average of the epoch 1.0, 1.5 and 2.0 checkpoints), trained 2026-11-09 |
| **Default model** | `main` (v1, 49 voices) stays the default; this branch is opt-in |
| **License** | Unofficial fan model, no rights granted ([details](#rights-and-responsible-use)) |
| **Code** | [starryark/bangdream-tts](https://github.com/starryark/bangdream-tts) (`e9/`) |

## Use (Python)

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS", revision="ournotes",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(
    text="今日の練習、思ったより早く終わったね。",
    speaker="ikka_raika",            # one of the 25 voices below
    language="Japanese",             # or "Chinese" / "English"
)
sf.write("raika.wav", wavs[0], sr)
```

**Usage notes:**
- **Always pass `language`.**
- **Keep the default sampling** (temperature 0.9, top-k 50).
- Generated audio is synthetic: **label it as AI-generated** wherever you share it.

## 中文说明

非官方 BanG Dream!（邦邦）角色语音合成模型，基于 Qwen3-TTS 微调。按说话人 ID 直接合成，**无需参考音频**；会说日语、普通话和英语。粉丝自制，与 Bushiroad、Craft Egg、BanG Dream! 官方、官方 AI 声库“梦的结唱”及声优均无关。

本分支（`ournotes`）是 25 个《BanG Dream! Our Notes》（交织的乐章）声线：一家Dumb Rock!、梦限大MewType、millsage 的 15 位新角色、泽海奏多，以及 MyGO!!!!!／Ave Mujica 9 位角色的 Our Notes 版本（`*_ournotes`）。其余 49 个声线请使用 `main`。在线体验：[演示 Space](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS)（74 个声线，可直接输入中文名搜索）。

```python
import os; os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 中国大陆网络
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts
tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS", revision="ournotes", device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(text="今天的排练真开心，明天也一起加油吧！", speaker="ikka_raika", language="Chinese")
sf.write("out.wav", wavs[0], sr)
```

**普通话音色（说中文时像不像本人）：**

| 程度 | 数量 | 声线 |
|---|---:|---|
| 保持角色音色 | 19 | 下表中标为 own voice 的声线 |
| 部分偏通用 | 4 | 高松灯（Our Notes）（`mygo_tomori_ournotes`）、Mortis（Our Notes）（`mujica_mortis_ournotes`）、仲町阿拉蕾（`mewtype_arale`）、宫永野乃花（`mewtype_nonoka`） |
| 偏通用音色 | 2 | 矢仓蓬咲（`ikka_yomogi`）、汐见萤（`millsage_hotaru`） |

“偏通用”的意思是：口音仍是标准普通话，但音色更接近基础模型的默认声音，而不是角色本人。

**使用规则：**
- 生成的音频均为 AI 合成，发布时须明确标注“AI生成”（《人工智能生成合成内容标识办法》，2025-09-01 起施行）。在 bilibili 投稿时，请在上传页勾选 AI 标识。
- 禁止商用。禁止冒充声优或官方，禁止用于损害角色及声优形象的内容。
- BanG Dream! 官方二创指南（<https://bang-dream.com/bdp-guideline/>）第 10 条禁止使用作品音声制作、公开二次创作。本模型是在此前提下的非官方粉丝作品。权利方或声优提出要求时，模型与演示将立即下架。

## Voices

**Columns:**
- **Listening:** the answers of a blind check in which each generated clip was played next to a real clip of the character, answering "does it sound like this character?" (yes / partly / no).
- **Japanese identity / Mandarin identity:** the share of test renders (80 Japanese / 32 Mandarin) that an independent speaker probe attributes to the right character.
- **Mandarin voice:**
  - **generic:** the voice speaks Mandarin with a native accent, but in a timbre closer to the base model's default than to the character;
  - *partly generic:* that happens on some seeds.

| `speaker` | Character | 中文名 | Band | Japanese data | Listening (2 answers) | Japanese identity | Mandarin identity | Mandarin voice |
|---|---|---|---|---:|---|---:|---:|---|
| `mygo_tomori_ournotes` | 高松燈 | 高松灯（Our Notes） | MyGO!!!!! | 17 min | yes, yes | 1.00 | 0.84 | partly generic |
| `mygo_anon_ournotes` | 千早愛音 | 千早爱音（Our Notes） | MyGO!!!!! | 23 min | yes, yes | 1.00 | 1.00 | own voice |
| `mygo_soyo_ournotes` | 長崎そよ | 长崎爽世〔素世〕（Our Notes） | MyGO!!!!! | 20 min | yes, yes | 1.00 | 1.00 | own voice |
| `mygo_taki_ournotes` | 椎名立希 | 椎名立希（Our Notes） | MyGO!!!!! | 18 min | yes, partly | 1.00 | 1.00 | own voice |
| `mujica_uika_ournotes` | 三角初華 | 三角初华（Our Notes） | Ave Mujica | 19 min | yes, partly | 0.99 | 1.00 | own voice |
| `mujica_mutsumi_ournotes` | 若葉睦 | 若叶睦（Our Notes） | Ave Mujica | 13 min | yes, yes | 1.00 | 0.88 | own voice |
| `mujica_mortis_ournotes` | 若葉睦 | Mortis〔墨缇丝〕（Our Notes） | Ave Mujica | 12 min | yes, yes | 1.00 | 0.84 | partly generic |
| `mujica_umiri_ournotes` | 八幡海鈴 | 八幡海铃（Our Notes） | Ave Mujica | 20 min | yes, yes | 1.00 | 1.00 | own voice |
| `mujica_sakiko_ournotes` | 豊川祥子 | 丰川祥子〔小祥〕（Our Notes） | Ave Mujica | 30 min | yes, yes | 1.00 | 1.00 | own voice |
| `ikka_chieri` | 梅里ちえり | 梅里千樱梨 | 一家Dumb Rock! | 25 min | yes, partly | 1.00 | 1.00 | own voice |
| `ikka_miku` | 馬橋心玖 | 马桥心玖 | 一家Dumb Rock! | 24 min | yes, partly | 0.95 | 0.97 | own voice |
| `ikka_raika` | 須賀蕾叶 | 须贺蕾叶 | 一家Dumb Rock! | 21 min | yes, yes | 0.99 | 1.00 | own voice |
| `ikka_shizuku` | 四宮寧月 | 四宫宁月 | 一家Dumb Rock! | 25 min | yes, yes | 1.00 | 1.00 | own voice |
| `ikka_yomogi` | 矢倉蓬咲 | 矢仓蓬咲 | 一家Dumb Rock! | 28 min | partly, yes | 0.82 | 0.19 | **generic** |
| `mewtype_arale` | 仲町あられ | 仲町阿拉蕾 | 夢限大みゅーたいぷ | 19 min | yes, yes | 0.94 | 0.66 | partly generic |
| `mewtype_miyako` | 藤都子 | 藤都子 | 夢限大みゅーたいぷ | 19 min | partly, partly | 1.00 | 0.97 | own voice |
| `mewtype_nonoka` | 宮永ののか | 宫永野乃花 | 夢限大みゅーたいぷ | 17 min | yes, no | 0.99 | 0.78 | partly generic |
| `mewtype_ritsu` | 峰月律 | 峰月律 | 夢限大みゅーたいぷ | 15 min | yes, yes | 1.00 | 1.00 | own voice |
| `mewtype_yuno` | 千石ユノ | 千石由乃 | 夢限大みゅーたいぷ | 15 min | yes, yes | 1.00 | 0.97 | own voice |
| `millsage_hotaru` | 汐見蛍 | 汐见萤 | millsage | 20 min | yes, yes | 1.00 | 0.53 | **generic** |
| `millsage_houka` | 和泉朋花 | 和泉朋花 | millsage | 25 min | yes, yes | 1.00 | 0.94 | own voice |
| `millsage_mahoro` | 浜崎まほろ | 滨崎茉幌 | millsage | 23 min | yes, no | 1.00 | 0.88 | own voice |
| `millsage_nagi` | 琴平凪 | 琴平凪 | millsage | 18 min | yes, yes | 1.00 | 1.00 | own voice |
| `millsage_natsume` | 伊沢なつめ | 伊泽枣 | millsage | 21 min | yes, yes | 0.99 | 1.00 | own voice |
| `ournotes_kanata` | 沢海奏多 | 泽海奏多 | — | 18 min | yes, partly | 0.95 | 0.97 | own voice |

**Two eras of one character.** `mygo_anon_ournotes` and `mygo_anon` are the same character recorded years apart. Their real recordings are 94–100% separable by a speaker classifier, partly because of the recording channel and partly the performance. The model therefore keeps them as separate voices:
- on this branch, `*_ournotes` is the Our Notes voice;
- on `main`, the plain id is the original (Girls Band Party / anime) voice.

## FAQ

**Does it need reference audio?**
No. Every voice is a built-in speaker row, called by its id (`speaker="ikka_raika"`).

**Can the characters speak Chinese?**
Yes. Pass `language="Chinese"`. Every voice speaks Mandarin with a native accent. Most voices also keep their own timbre in Mandarin; those that drift toward a generic timbre are marked in the table and listed in the Chinese section.

**Which revision should I load?**
- `main` for the 49 Girls Band Party / MyGO!!!!! / Ave Mujica voices.
- `revision="ournotes"` for the 16 Our Notes characters and the `*_ournotes` voices.
- The [demo](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS) picks the right one automatically.

**How is it different from GPT-SoVITS, RVC or Bert-VITS2 character models?**
- It is one checkpoint with every voice built in, so it needs no reference clip (GPT-SoVITS) and no source recording (RVC).
- Bert-VITS2 character models also call built-in speakers; against those, the difference is Mandarin and evaluation.
- It was trained so that its Mandarin is not Japanese-accented.
- Its checkpoints were chosen by rules fixed before training, and the later ones by blind listening.

**What hardware does it need?**
It is a fine-tune of Qwen3-TTS-12Hz-1.7B-Base; the bf16 weights are 3.8 GB, and inference runs on a CUDA GPU (`pip install qwen-tts`). Without a GPU, use the demo.

**Can I download it from mainland China?**
Set `HF_ENDPOINT=https://hf-mirror.com` before `from_pretrained("starrydark/BanG_Dream_Qwen3_TTS", revision="ournotes", ...)`.

**Can I post what I make?**
Only as a labelled, non-commercial fan work. The BanG Dream! fan-work guideline prohibits derivative works that use the franchise's audio (item 10), and China requires an explicit AI label on generated audio. Never impersonate the cast or present output as official. See [Rights and responsible use](#rights-and-responsible-use).

**Is this official, or related to 夢ノ結唱 (梦的结唱)?**
No. This is an unofficial fan project. 夢ノ結唱 is Bushiroad Music's official AI singing product and is unrelated.

## How this branch was evaluated

**Automatic screens** (catastrophe level; all passed):

| Screen | E9 | Bound |
|---|---:|---:|
| Japanese identity errors on the 25 voices + Rāna and Nyamu (all seeds) | 0.016 | ≤ 0.06 |
| Any of those voices above 0.40 errors | none (yomogi 0.18) | none |
| Japanese CER on Our Notes lines (real recordings through the codec: 0.042) | 0.054 | ≤ 0.062 |
| Mandarin identity errors on the 36 unchanged v1 voices (v1: 0.155) | 0.138 | ≤ 0.255 |
| Mandarin accent SER on the 36 unchanged v1 voices (v1: 0.073) | 0.071 | ≤ 0.093 |
| Japanese CER on the 36 unchanged v1 voices (v1: 0.025) | 0.026 | ≤ 0.035 |

**Blind listening (one listener, decisive by a rule fixed before training):**
- **Absolute check:** all 25 voices were judged in character. 40 of 50 answers were "yes", 8 "partly" and 2 "no" (Nonoka and Mahoro, once each).
- **Paired comparison against v1 on the 49 shared ids:**
  - the unchanged voices, the persona ids and Mandarin were not worse (pooled);
  - the MyGO!!!!! / Ave Mujica v1-era voices were worse: mean −0.58 on a −3…+3 scale (p = 0.046), mainly on held-out game lines, with Uika at −7 over three items.
- **Outcome:** under its pre-registered rule this checkpoint did **not** replace v1. The author then chose to publish it as this opt-in branch for the 25 voices that passed, with `main` unchanged.
- **Records:** the plan, results and listening analysis are in `eval/`.

## Training details

- **Data:**
  - Japanese voice lines of 72 voices: 16,790 training clips, after removing 244 speaker outliers (1.5%).
    - The 36 other v1 voices (Girls Band Party) keep v1's data.
    - The nine MyGO!!!!! / Ave Mujica characters train on both eras, as two ids each.
    - Rāna and Nyamu train both eras under one id, and the 16 new characters train on Our Notes story audio.
  - Plus about 13% native Mandarin/English replay, 1,008 gated in-voice pseudo-Mandarin rows and 242 voice-converted Mandarin rows.
  - In total 20,773 training rows, 34.4 h.
- **Speaker rows:** each voice has a fixed vector (the mean speaker-encoder embedding of screened reference clips), written into codec-embedding rows 3000–3071.
  - Unlike v1, a small learnable offset per row is trained with its own learning rate (1e-4), which moves crowded voices slightly apart.
  - `mygo_soyo_tsuki` and `mujica_sakiko_tsuki` are aliases of `mygo_soyo` and `mujica_sakiko` on this branch.
- **Optimisation:** as v1 (full fine-tune, fp32 masters, AdamW lr 5e-6, effective batch 32, language ids). This checkpoint is the weight average of the epoch 1.0, 1.5 and 2.0 checkpoints.
- **Compute:** one H100 80GB, about 2.4 h, about $9.5.

## Limitations

- **Use `main` for the other 49 voices.** On this branch, the MyGO!!!!! / Ave Mujica v1-era voices (`mygo_tomori` … `mujica_sakiko`, `mygo_raana`, `mujica_nyamu`) were judged worse than v1.
- **Mandarin in a generic timbre:**
  - `ikka_yomogi` and `millsage_hotaru` (generic);
  - `mewtype_arale`, `mewtype_nonoka`, `mygo_tomori_ournotes` and `mujica_mortis_ournotes` (partly generic).
- **Weakest Japanese voices:**
  - `ikka_yomogi` has the lowest Japanese identity (0.82).
  - `mewtype_miyako` was rated "partly" twice.
  - `mewtype_nonoka` and `millsage_mahoro` each got one "no".
- **Mandarin and English were not judged by ear** for these voices, only by the automatic probe and accent metrics.
- **Listening was one listener**, on two clips per voice for the absolute check.

## Rights and responsible use

This is an unofficial fan project, with no affiliation to Bushiroad, Craft Egg, the BanG Dream! project, or the voice cast.
- The characters, the source audio and all related marks belong to their respective owners.
- **No copyright, performer, publicity or voice rights are granted** by this repository.
- No game audio, dataset, transcripts or audio samples are distributed here; only model weights and evaluation numbers.
- **The official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) prohibits derivative works that use the franchise's audio.** Item 10 reads 「本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開。」. This model is an unofficial fan work made despite that. Read the guideline before using outputs anywhere public.
- **AI labels:** in mainland China, AI-generated audio must carry an explicit AI label (《人工智能生成合成内容标识办法》, in force since 2025-09-01), and bilibili asks uploaders to set its AI标识 when uploading.

Do not use it to:
- impersonate the voice actors or anyone else;
- mislead people about authentic or official speech, or make anything that could be mistaken for official material;
- harm the image of the characters or the cast;
- suggest endorsement, harass, or make commercial use.

Disclose synthetic audio as synthetic. Users are responsible for lawful use in their jurisdiction. **Rights holders and cast members may contact the repository owner (Hugging Face: starrydark; GitHub: starryark, issues) for removal, and the repository will be taken down on request.**

Model weights derive from Qwen3-TTS (Apache-2.0).

## Citation

```bibtex
@misc{bangdream_qwen3_tts_2026,
  title  = {BanG Dream! character voices: a cross-lingual Qwen3-TTS fine-tune},
  author = {starrydark},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS}},
  note   = {Code: \url{https://github.com/starryark/bangdream-tts}}
}
@article{qwen3tts2026,
  title   = {Qwen3-TTS Technical Report},
  journal = {arXiv preprint arXiv:2601.15621},
  year    = {2026}
}
```
