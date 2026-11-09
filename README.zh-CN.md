# BanG Dream!（邦邦）角色语音合成：74 个声线，说日语、普通话和英语（Qwen3-TTS）

[English](README.md) · **简体中文**

**非官方 BanG Dream!（邦邦）角色语音合成（TTS）模型：主模型 49 个声线，另有 25 个可选的《BanG Dream! Our Notes》声线（MyGO!!!!!、Ave Mujica、少女乐团派对、Our Notes）。按说话人 ID 直接合成，无需参考音频，会说日语、普通话和英语。基于 Qwen3-TTS 微调。粉丝自制，与 Bushiroad、Craft Egg、BanG Dream! 官方、官方 AI 声库“梦的结唱”及声优均无关。**

- **在线体验：** [演示 Space](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS)。选择角色，输入中文、日语或英语即可合成；声线列表可以直接输入中文名搜索（如 千早爱音、丰川祥子）。
- **模型下载：** [starrydark/BanG_Dream_Qwen3_TTS](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS)（`main`，49 个声线）；分支 [`ournotes`](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS/tree/ournotes)（25 个 Our Notes 声线）。中国大陆网络请用 hf-mirror，见下文。
- **规则：** 生成的音频是 AI 合成，发布时须标注“AI生成”；禁止商用，禁止冒充声优或官方。BanG Dream! 官方二创指南第 10 条禁止使用作品音声进行二次创作；本项目是在此前提下的非官方粉丝作品，权利方或声优要求时将立即下架。详见[使用规则](#使用规则)。

这个模型是为邦邦二创准备的：MyGO二创、Ave Mujica 二创（也就是常说的“少女乐队二创”）里的 AI 配音、手书和 MAD 配音、WebGAL 同人剧、桌宠语音，还有“让角色说中文”。比如让千早爱音、丰川祥子、高松灯用普通话说一段新台词。

## 声线

| 来源 | 数量 | 乐队与角色 |
|---|---:|---|
| `main`（v1） | 49 | 少女乐团派对：Poppin'Party、Afterglow、Pastel＊Palettes、Roselia、Hello, Happy World！、Morfonica、RAISE A SUILEN；MyGO!!!!!（《迷途之子!!!!!》）；Ave Mujica（《颂乐人偶》）。另有大魔姬亚子、月之森时期的爽世和祥子、Mortis 等人格声线 |
| `ournotes` 分支（E9） | 25 | 《BanG Dream! Our Notes》（交织的乐章）：一家Dumb Rock!、梦限大MewType、millsage 的 15 位新角色和泽海奏多；MyGO!!!!!／Ave Mujica 9 位角色的 Our Notes 版本（`*_ournotes`） |

每个声线的中文名、数据量和质量见模型卡：[main](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS#voices)、[ournotes](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS/blob/ournotes/README.md#voices)。

**说普通话时像不像本人：** 所有声线都说标准普通话（没有日语口音），但音色不一样。

| 程度 | 数量 | 声线 |
|---|---:|---|
| 保持角色音色 | 53 | 其余全部声线 |
| 部分偏通用 | 12 | 丸山彩、若宫伊芙、濑田薰、北泽育美、松原花音、佐藤益木、珠手知由、丰川祥子（月之森）、高松灯（Our Notes）、Mortis（Our Notes）、仲町阿拉蕾、宫永野乃花 |
| 偏通用音色 | 9 | 户山香澄、牛込里美、宇田川亚子（大魔姬）、白金燐子、朝日六花、高松灯、长崎爽世（月之森）、矢仓蓬咲、汐见萤 |

“偏通用”指口音仍是标准普通话，但音色更接近基础模型的默认声音，而不是角色本人。这些数字来自独立说话人识别模型的自动评测；Our Notes 的 25 个声线另外经过了盲听测试。

## 快速使用

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 中国大陆网络：通过 hf-mirror 下载
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(text="今天的排练真开心，明天也一起加油吧！",
                                     speaker="mygo_anon",      # 千早爱音
                                     language="Chinese")       # 或 "Japanese" / "English"
sf.write("anon_zh.wav", wavs[0], sr)
```

- **Our Notes 声线：** 加 `revision="ournotes"`，例如 `speaker="ikka_raika"`（须贺蕾叶）。
- **始终传入 `language`：** 这是普通话不带日语口音的关键。
- **保持默认采样参数：** 普通话不满意时可以换几个随机种子再试。

## 常见问题

**需要参考音频吗？**
不需要。每个声线都内置在模型里，用说话人 ID 调用（如 `mygo_anon`）。GPT-SoVITS 类模型每次合成都要准备参考音频，这个模型不用。

**和 GPT-SoVITS、RVC、Bert-VITS2 的角色模型有什么不同？**
- GPT-SoVITS 每次合成都要参考音频，RVC 需要一段现成的录音来变声，这个模型两者都不需要。
- Bert-VITS2 角色模型同样用内置说话人；和它们相比，区别在于普通话口音处理和评测方式。
- 这里是一个模型包含全部 74 个声线。训练时专门处理了普通话口音（语言 ID、母语回放数据、角色声线的伪普通话数据）。
- 每个发布版本都按训练前写好的规则选出，后几轮由盲听决定。

**需要什么显卡？**
基于 Qwen3-TTS-12Hz-1.7B-Base，bf16 权重 3.8 GB，用 NVIDIA 显卡（CUDA）推理。没有显卡可以用在线演示。

**中国大陆能下载吗？**
huggingface.co 在大陆无法直接访问。如上文代码所示，设置 `HF_ENDPOINT=https://hf-mirror.com` 即可通过镜像下载。

**能发到 bilibili 吗？**
只能作为标注了 AI 的非商用粉丝作品，并且请先了解下面的规则。

**这是官方的吗？和“梦的结唱”有关吗？**
都不是。本项目是非官方粉丝作品。梦的结唱（夢ノ結唱）是 Bushiroad Music 的官方 AI 歌声合成产品，与本项目无关。

## 使用规则

- **标注 AI：** 生成的音频均为 AI 合成。依据《人工智能生成合成内容标识办法》（2025-09-01 起施行），发布时必须明确标注“AI生成”。在 bilibili 投稿时，请在上传页勾选 AI 标识。
- **禁止的用途：**
  - 商用；
  - 冒充声优本人或官方；
  - 把生成内容当作官方台词或真实录音发布；
  - 损害角色及声优形象；
  - 骚扰他人。
- **官方二创指南：** BanG Dream! 官方二创指南（<https://bang-dream.com/bdp-guideline/>）第 10 条禁止使用作品中的音声进行二次创作的制作、展示、发布与公开。本项目是在此前提下的非官方粉丝作品，使用者需自行承担责任。
- **下架：** 权利方或声优提出要求时，模型与演示将立即下架（Hugging Face：starrydark；GitHub：starryark，可提 issue）。
- **不公开的内容：** 不公开任何游戏音频、台词文本或数据集。

## 本仓库包含什么

本仓库是完整、可复现的研究记录：
- 数据构建与校验；
- 经过编解码器校准的说话人识别探针；
- 多说话人微调（语言 ID、母语回放、角色声线伪普通话、逐角色 RVC 转换）；
- 每个实验在训练前写好的计划和结果（E0–E9）。

详见 [English README](README.md) 与 [`research/chinese_fandom_discovery_2026-11-09.md`](research/chinese_fandom_discovery_2026-11-09.md)。后者介绍邦邦中文圈的用语、AI 语音生态和相关规则。

**相关项目：**
- [Mahiruoshi/BangDream-Bert-VITS2](https://huggingface.co/spaces/Mahiruoshi/BangDream-Bert-VITS2)：2023 年的邦邦 Bert-VITS2 演示；
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：社区角色模型分享的主流工具；
- [BANDORI-PET-REV](https://github.com/HELPMEEADICE/BANDORI-PET-REV)：邦邦桌宠。
