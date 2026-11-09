---
title: BanG Dream! AI Voices (unofficial) — Qwen3-TTS
emoji: 🎸
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
python_version: 3.12.12
app_file: app.py
suggested_hardware: zero-a10g
short_description: Unofficial BanG Dream! 邦邦 voice TTS, 74 voices, ja/zh/en
models:
  - starrydark/BanG_Dream_Qwen3_TTS
preload_from_hub:
  - starrydark/BanG_Dream_Qwen3_TTS
tags:
  - text-to-speech
  - tts
  - qwen3-tts
  - bang-dream
  - character-voice
  - japanese
  - chinese
  - mandarin
  - english
  - bandori
  - mygo
  - ave-mujica
  - audio
  - zerogpu
license: other
---

# BanG Dream! AI Voices — unofficial Qwen3-TTS demo

Type text and hear it in a BanG Dream! character's voice, in Japanese, Mandarin Chinese or English. It is an
unofficial fan demo of the [BanG Dream! Qwen3-TTS model](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS).
Every voice is a built-in speaker row of the model, so no reference audio is used or shipped. Japanese is
the language the voices were trained in; Mandarin and English quality varies by voice (see the model card's
limitations).

- **Voices:** 74 in total.
  - 49 come from the model's `main` revision: Girls Band Party, MyGO!!!!! and Ave Mujica.
  - 25 come from its opt-in `ournotes` branch: BanG Dream! Our Notes.
  - Each clip's status line names the checkpoint it came from.
- **Search by name:** the voice list can be searched by Chinese or Japanese name (e.g. 千早爱音 / 千早愛音).

**中文：** 非官方 BanG Dream!（邦邦）角色语音合成演示，共 74 个声线，说中文、日语或英语，无需参考音频。生成的音频为 AI 合成，发布时须标注“AI生成”，禁止商用及冒充声优。

## Rights and responsible use

This is an unofficial fan project, with no affiliation to Bushiroad, Craft Egg, the BanG Dream! project, or the
voice cast. The characters, the source audio and all related marks belong to their respective owners, and **no
copyright, performer, publicity or voice rights are granted**. No game audio, transcripts or dataset are
distributed here. The BanG Dream! fan-work guideline prohibits derivative works that use the franchise's audio
(item 10); this demo is an unofficial fan work made despite that.

- All output is **AI-generated**. Label it as AI-generated wherever you share it.
- Do not use it to impersonate the voice actors or anyone else, to mislead people about authentic or official
  speech, to harm the image of the characters or the cast, to suggest endorsement, to harass, or for commercial use.
- Read the official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) before using outputs
  anywhere public.

Rights holders and cast members may contact the owner (Hugging Face: starrydark; GitHub: starryark, issues) for
removal, and the demo and model will be taken down on request.
