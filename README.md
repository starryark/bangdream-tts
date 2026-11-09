# BanG Dream! character-voice TTS (邦邦 / バンドリ): 74 voices in Japanese, Mandarin and English on Qwen3-TTS

**English** · [简体中文](README.zh-CN.md)

**BanG Dream! (邦邦) character-voice TTS: 49 voices + 25 opt-in Our Notes voices (MyGO!!!!!, Ave Mujica, Girls Band Party, Our Notes). They speak Japanese, Mandarin and English with no reference audio. A Qwen3-TTS fine-tune. Unofficial fan project, not affiliated with Bushiroad, Craft Egg, the BanG Dream! project, the official 夢ノ結唱 voicebanks or the voice cast.**

- **Try it:** [online demo](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS). Pick a voice and type Japanese, Chinese or English. The output is AI-generated; label it as such.
- **Use it:** `pip install qwen-tts`, load `starrydark/BanG_Dream_Qwen3_TTS` and call a voice by its speaker id ([example](#use-the-model)).
- **Rules:** BanG Dream!'s fan-work guideline prohibits derivative works that use the franchise's voice audio (item 10). This model exists despite that and will be taken down at a rights holder's or cast member's request. Don't use it commercially or to impersonate anyone ([details](#rights-and-responsible-use)).

This repository is the complete, reproducible research record, from extracted voice lines to the selected checkpoints:
- data building and verification (`pipeline/`);
- a codec-calibrated speaker-identity probe and the zero-shot floor (`e0`);
- one shared 49-voice fine-tune with language ids, native replay and in-voice pseudo-Mandarin (`e1`);
- two negative results kept on record: more self-generated rows (`e2`) and zero-shot voice conversion (`e3`);
- per-voice RVC converters that put native Mandarin into the 15 voices whose Mandarin was generic (`e4`, `e5`);
- BanG Dream! Our Notes data, learnable speaker rows and blind-listening selection (`e6`–`e9`);
- every experiment's plan, written before its results, and the results themselves.

| | |
|---|---|
| **Model** | [starrydark/BanG_Dream_Qwen3_TTS](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS): `main` (v1, 49 voices); branch [`ournotes`](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS/tree/ournotes) (25 opt-in BanG Dream! Our Notes voices, E9) |
| **Demo** | [Hugging Face Space](https://huggingface.co/spaces/starrydark/BanG_Dream_Qwen3_TTS) (ZeroGPU, all 74 voices; app in `release/space/`) |
| **Results** | [E1 (released model)](e1/E1_RESULTS.md) · [E2](e2/E2_RESULTS.md) · [E4](e4/E4_RESULTS.md) · [E5](e5/E5_RESULTS.md) · [E6](e6/E6_RESULTS.md) · [E7](e7/E7_RESULTS.md) · [E8](e8/E8_RESULTS.md) · [E9 (ournotes branch)](e9/E9_RESULTS.md) · [E0 data and probe](e0/e0_results.md) |
| **Sibling projects** | [makise-kurisu-tts](https://github.com/starryark/makise-kurisu-tts) (where the trainer, language-id recipe and accent metrics started) · [tricolour-tts](https://github.com/starryark/tricolour-tts) |

**Bands and voices:**
- **Girls Band Party:** Poppin'Party, Afterglow, Pastel\*Palettes, Roselia, Hello, Happy World!, Morfonica and RAISE A SUILEN.
- **MyGO!!!!! and Ave Mujica.**
- Together these are 49 speaker rows on `main`: 45 characters plus the persona rows 大魔姫あこ, 月ノ森 Soyo, 月ノ森 Sakiko and Mortis.
- **From BanG Dream! Our Notes (25 voices on the `ournotes` branch):** 一家Dumb Rock!, 夢限大みゅーたいぷ, millsage and 沢海奏多, plus the Our Notes era of nine MyGO!!!!! / Ave Mujica characters.

Per-voice tables, with Chinese names, are on the model cards ([main](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS#voices), [ournotes](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS/blob/ournotes/README.md#voices)).

**Also known as:** 邦邦 · 邦多利 · BanG Dream! 少女乐团派对! · 《迷途之子!!!!!》 (MyGO!!!!!) · 《颂乐人偶》 (Ave Mujica) · BanG Dream! 交织的乐章 (Our Notes) · バンドリ！ ガールズバンドパーティ！ · ガルパ.

## 中文 / 日本語

**中文说明见 [README.zh-CN.md](README.zh-CN.md)。** 非官方 BanG Dream!（邦邦）角色语音合成模型：主模型 49 个声线，另有 `ournotes` 分支的 25 个《BanG Dream! Our Notes》声线。会说日语、普通话和英语，无需参考音频。粉丝自制，与官方及声优无关；生成的音频须标注“AI生成”。

『BanG Dream!』キャラクターボイスの非公式ファンメイド TTS（Qwen3-TTS 微調整）。話者 ID で呼び出し、参照音声は不要。日本語・中国語・英語に対応。公式・キャストとは無関係です。

## Use the model

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(text="今天的练习比想象中结束得早呢。",
                                     speaker="ppp_kasumi", language="Chinese")
sf.write("kasumi_zh.wav", wavs[0], sr)
```

Speaker ids are in `pipeline/rows.json` (`row`) and `e1/bundle/speakers.txt`. Always pass `language`.

For the 25 Our Notes voices, add `revision="ournotes"`, e.g. `speaker="ikka_raika"` (ids in `e9/bundle/speakers.txt`). From mainland China, set `HF_ENDPOINT=https://hf-mirror.com` first.

## FAQ

**Does it need reference audio?**
No. Every voice is a built-in speaker row, called by its id (`speaker="mygo_anon"`). GPT-SoVITS-style models need a reference clip for each generation; this one doesn't.

**Can the MyGO!!!!! / Ave Mujica characters speak Chinese?**
Yes. Every voice speaks Mandarin with a native accent (`language="Chinese"`). In 34 of the 49 `main` voices and 19 of the 25 `ournotes` voices, the Mandarin also keeps the character's timbre. The others drift toward a generic timbre in Mandarin; they are marked on the model cards.

**Which checkpoint should I load?**
- `main` for the 49 Girls Band Party / MyGO!!!!! / Ave Mujica voices.
- `revision="ournotes"` for the 16 Our Notes characters and the `*_ournotes` voices.
- The demo picks the right one automatically.

**How is it different from GPT-SoVITS, RVC or Bert-VITS2 character models?**
GPT-SoVITS clones from a reference clip, and RVC converts an existing recording; this checkpoint needs neither. Bert-VITS2 character models also call built-in speakers, so against those the difference is Mandarin and evaluation. This model was trained with language ids, native replay and in-voice pseudo-Mandarin, so its Mandarin is not Japanese-accented. Each released checkpoint was chosen by a rule fixed before training, and the later ones by blind listening.

**What hardware does it need?**
It is a fine-tune of Qwen3-TTS-12Hz-1.7B-Base; the bf16 weights are 3.8 GB, and inference runs on a CUDA GPU through `qwen-tts`. Without a GPU, use the online demo.

**Can I download it from mainland China?**
huggingface.co is not reachable there. Set `HF_ENDPOINT=https://hf-mirror.com` before loading to download through the mirror.

**Can I post what I make on bilibili, YouTube or X?**
Only as a labelled, non-commercial fan work, and know the rules:
- BanG Dream!'s fan-work guideline prohibits derivative works that use the franchise's voice audio (item 10).
- China's 《人工智能生成合成内容标识办法》 (in force since 2025-09-01) requires an explicit AI label on generated audio, and bilibili asks uploaders to set its AI标识 when uploading.
- Don't impersonate the cast, and don't present output as official dialogue.

**Is this official, or related to 夢ノ結唱 (梦的结唱)?**
No. This is an unofficial fan project. 夢ノ結唱 is Bushiroad Music's official AI singing product and is unrelated.

## What was learned

The released model is **E1b epoch 2**, chosen by a rule fixed before training (`e1/E1_PLAN.md`).

| | ja leak ↓ | zh SER ↓ | zh leak ↓ | Own-voice Mandarin | en lid ↓ |
|---|---:|---:|---:|---:|---:|
| Real recordings, through the codec | 0.012 | — | — | — | — |
| Floor: untrained Base, same speaker vectors | 0.100 | 0.077 | 0.157 | 36/49 | −12.2 |
| E1a: Japanese + language ids + native replay | 0.009 | 0.177 | 0.008 | — | — |
| **E1b epoch 2 (released)** | **0.014** | **0.069** | **0.155** | **34/49** | **−9.3** |
| E5 epoch 2: + 218 RVC Mandarin rows (not adopted) | 0.014 | 0.078 | 0.088 | 40/49 | −9.5 |

1. **One shared fine-tune fixes Japanese identity for all 49 voices** (leak 0.10 → 0.014, the real-recording level). A codec-calibrated probe was needed to measure this: the Qwen3-TTS codec shifts every speaker vector by a shared amount, and a probe fitted on raw recordings misreads renders (`e0/`).
2. **Language ids plus native replay are not enough.** With only Japanese rows in the characters' voices (E1a), their Mandarin and English come out Japanese-accented (Mandarin SER 0.177). Adding 5% gated in-voice pseudo-Mandarin brings the accent back to the floor (E1b: 0.069), and that fix generalises to voices that got no pseudo rows.
3. **Mandarin identity does not generalise.** Voices without in-voice Mandarin rows speak Mandarin in the base model's timbre: 15 of 49.
4. **Sampling more from our own models does not fix that** (E2: identity diluted), and **zero-shot voice conversion brings the accent back** (E3: Seed-VC copies the Japanese reference's articulation).
5. **Per-voice RVC converters do** (E4, E5): content from ContentVec, no index, F0 from the native source plus a per-voice shift. The converted clips passed a listening check as native and in character for all 15 voices. Retraining with them took Mandarin identity from 34 to 40 voices (pooled leak 0.155 → 0.088) with Japanese unchanged. It still missed the pre-registered adoption bar (≤ 0.08, ≥ 43/49), so the released model stays E1b. Kasumi, Ako/大魔姫, Tomori and Rimi remain generic in Mandarin.
6. **Some accent metrics fail at a character's own pitch.** E4's pitch-shift-only control (native Mandarin only shifted to Kasumi's and Eve's 450–500 Hz) scored SER 0.14–0.21 against 0.04–0.05 unshifted, so for high-pitched voices the ear decides (`e4/E4_RESULTS.md`).
7. **A v2 on BanG Dream! Our Notes data was not released** (E6, E7). The MyGO!!!!! / Ave Mujica voices were retrained on the new story audio and 16 new characters added (65 voices). Coverage of the Our Notes voices reached near the real-recording level (Japanese leak ≈ 0.02). The 16 new voices cost the weakest v1 voices their Mandarin identity unless they had own-voice Mandarin rows, so VC rows belong in the base recipe (E7). No candidate passed its pre-registered guards. The blockers were one new voice's Japanese identity (yomogi), Japanese CER on the new text (real recordings of it score 0.042 through the codec, so the absolute 0.035 bound leaves almost no room) and per-voice Mandarin accent at epoch 2 (`e6/E6_RESULTS.md`, `e7/E7_RESULTS.md`).
8. **From E8 on, a blind listening test against v1 decides; the automatic screens only catch failures. A v2 is still not released** (E8, E9).
   - **E8:** learnable speaker rows and averaged checkpoints passed every screen. But replacing the MyGO!!!!! / Ave Mujica training data with Our Notes made two of those voices clearly worse to the ear.
   - **E9:** both eras were trained, one id per era.
     - All 16 new voices and all 9 Our-Notes-era voices were judged in character.
     - The v1-era voices of those characters were judged worse than v1, mainly on held-out game lines (mean −0.58 on a −3…+3 scale, p = 0.046).
     - So v1 stays the default. E9 is published only as the opt-in `ournotes` branch for its 25 Our Notes voices, and the demo Space serves each voice from v1 if v1 has it, otherwise from that branch.
   - Details: `e8/E8_RESULTS.md`, `e9/E9_RESULTS.md`.

## Layout

```
pipeline/  E0 data build and probe: b01 build (24 kHz, levelling), b02 ASR check, b03 vectors,
           b04 Base zero-shot floor, b05 scoring, b08 codec-calibrated identity probe; bdcommon.py
e0/        E0 results: data floor, codec probe, vector reports
e1/        E1 plan/results; e1_build (bundle), e1_pseudo (gated in-voice pseudo-zh), e1_codes,
           e1_mix, e1_eval + bdscore (identity, CER, SER, lid), e1_select (pre-registered rule),
           pod bootstrap and queues
e2/        E2 plan/results: round-2 pseudo rows from E1b (not adopted)
e3/        E3 plan: zero-shot Seed-VC feasibility (failed its gate)
e4/        E4 plan/results: per-voice RVC (Applio) feasibility for two voices; train, convert,
           pitch shift, score
e5/        E5 plan/results: 13 more converters, sources, gates, listening sets, VC rows, pod
           bootstrap/queue, selection; vc/ numeric reports
e6/        E6 plan/results: Our Notes data build, 16 new rows, probeE6, four more converters (two
           trained on a rented GPU), VC/[nv] arms, pod scripts, selection (not adopted)
e7/        E7 plan/results: E6b's data as the base recipe, extra-seed G3 renders, selection
           (not adopted)
e8/        E8 plan/results: learnable speaker rows (--spk_lr), checkpoint averaging, catastrophe
           screens, blind listening pages and analysis (not adopted)
e9/        E9 plan/results: both eras per MyGO!!!!! / Ave Mujica character (two ids), persona aliases,
           outlier pruning, listening (not adopted)
kurisu_pod/  the Kurisu project's pod code this pipeline imports (train_sft.py trainer, p03 audio
           policy, p05 native replay, accent_metrics, probes), vendored at the version used
research/  approach research, data facts, rights notes and the recommendation behind the plan
review/    independent code reviews and their check scripts
release/   publish_hf.py (model upload), stage_github.sh (builds this repo), MODEL_CARD.md,
           space/ (the demo Space's app), publish_hf_v2.py (unused: no v2 was released),
           publish_hf_ournotes.py + MODEL_CARD_ournotes.md (the opt-in ournotes branch, E9)
```

**Not in this repository:** audio of any kind, model weights, speaker vectors, game transcripts, reference-clip lists and per-clip evaluation files. The scripts default to the author's local folders (`bdcommon.py`; `BD_DIR`, `BD_DATA`, `BD_POD`, `BD_KPOD` override them), and the pod scripts expect `/workspace/bd`.

## Reproduce

You need the 49 voices' Japanese clips yourself; they are not distributed.

1. **Build and probe (local):** `pipeline/b01_build.py` (rows from `pipeline/rows.json`), `b02_asr.py`, `b03_vectors.py`, `b04_floor.py`, `b05_score.py`, `b08_codec_probe.py` (`pipeline/run_e0.sh`).
2. **Bundle (local):** `e1/e1_build.py` writes `e1/bundle/` (rows, refs, eval rows, probe).
3. **Train and evaluate (one H100, ~5 h):** upload `data/wav24`, `e1/bundle` → `bundle/`, `kurisu_pod/` → `code/`, `e1/`, `pipeline/` to `/workspace/bd` over SSH, then `e1/bootstrap_e1.sh` and `e1/queue_e1.sh smoke E1a refs E1b select`.
4. **Pick:** `e1/e1_select.py` applies the rule in `e1/E1_PLAN.md`.
5. **E5 (optional):** `e4/e4_rvc_train.sh` per voice (Applio 59c1131, 200 epochs, local GPU), `e5/e5_rows.sh` (sources → conversion → gates → listening sets), `e5/e5_collect.py rows`, then `e5/bootstrap_e5.sh` and `e5/queue_e5.sh` on a pod and `e5/e5_select.py` locally.

## Rights and responsible use

- Unofficial fan project; characters, source audio and marks belong to their owners. No copyright, performer, publicity or voice rights are granted.
- **The BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) prohibits derivative works that use the franchise's audio.** Item 10 reads 「本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開。」.
  - This project is an unofficial fan work made despite that.
  - `research/community_and_legal_2026-10-29.md` summarises the guideline and the 2026 Japanese guidance on voice rights.
  - `research/chinese_fandom_discovery_2026-11-09.md` covers the Chinese rules.
- **Label generated audio as AI-generated.** In China this is required by 《人工智能生成合成内容标识办法》 (2025-09-01), and bilibili has an AI标识 setting at upload.
- Do not use the model to impersonate the cast or anyone else, to make content that could be mistaken for official material, to harm the image of the characters or cast, or commercially.
- Rights holders and cast members can open an issue or contact the owner for removal; the model will be taken down on request.

Code: Apache-2.0 (`LICENSE`). The Qwen3-TTS weights are Apache-2.0; the fine-tuned voices carry no grant of rights.
