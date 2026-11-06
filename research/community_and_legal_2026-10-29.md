# BanG Dream! AI voices: community practice, data sources, legal and platform status (2026-10-29)

**Scope.** This report covers four things:
- what the community has already built for BanG Dream! and comparable multi-character sets;
- where the voice data comes from and at what quality;
- the 2026 Chinese and Japanese toolchains;
- the legal and platform risk of publishing on Hugging Face (HF).

It does not repeat the model landscape; see `kurisu_v2/research/sota_models_2026-10.md` §0–§2 for that.

**Tags:**
- **[P]** Primary source I read: a repo, the HF/ModelScope/GitHub API, a model card, an official page or PDF, or an MP3 header I parsed.
- **[S]** Secondary: news, a blog or a search snippet.
- **[I]** My inference.
- **[U]** Unverified.

All sources were accessed on 2026-10-29 unless noted. The cast are never named; I write "a cast member of <band>" or "a voice actor".

---

## 0. Answer in brief

1. **The pack is a small, curated subset of freely mirrored game audio, and it has not been degraded.**
   - Bestdori serves the GBP (Girls Band Party, the BanG Dream! mobile game) JP voice files as **32 kHz mono VBR MP3 encoded with LAME 3.99r**, at about 83 kbps [P, header probe].
   - The pack's clips have exactly the same format (32 kHz / mono / LAME 3.99r / 65–87 kbps) [P, local read].
   - The Bestdori index lists 366,642 entries under JP `sound/voice`. That is **~361k MP3s** once the ~5.6k per-folder `.asset` descriptors are removed, and ~355k of them are story-scenario lines [P]. That is roughly 375 h in total and roughly 5–10 h per main character [I].
   - The pack holds 21.6 h across 52 sets, so it uses about 5% of what exists. Example clip counts: 89 for Anon, 196 for Taki, 334 for Saaya [P].
   - Re-extracting from Bestdori would give 10× or more data per character at the same quality. Whether the game client stores anything better than 32 kHz MP3 is [U].
2. **Prior BanG Dream TTS is all Japanese-only and community-grade.**
   - Mahiruoshi's Bert-VITS2 Spaces (2023–25) had 101 likes. The author now says GPT-SoVITS has made them obsolete [P].
   - Wanlau has 47 RVC v2 voice-conversion models [P].
   - KitsuneX07 published 40-character GPT-SoVITS v1/v2 weights plus a dataset with 963 downloads [P].
   - The closest precedent to the user's plan is **`HELPMEEADICE/BanGDream-Qwen3TTS-LoRA` on ModelScope (2026-05-21)** [P]:
     - a single 40-speaker LoRA on Qwen3-TTS-12Hz-1.7B-Base, rank 16, talker-only;
     - a 5-speaker MyGO version trained for only 110 steps;
     - GPL-3.0, an empty model card, and no published samples or evaluation.
   - Apart from Mahiruoshi's 2023 Bert-VITS2 "中日" (zh/ja) site, which now shows a runtime error, no public BanG Dream model claims multilingual output, and none claims English.
   - Chinese users run GPT-SoVITS v2ProPlus with a Japanese reference clip and Chinese target text [P, TTZW1001].
3. **The pack's origin is not identified.**
   - The checkpoints are dated 2025-06-06 to 2025-10-15 and sit inside the official `GPT-SoVITS-v2pro-20250604` integration package [P].
   - Band prefixes mix Japanese and Simplified Chinese spellings (e.g. `HHW_北泽育美`, `ppp_戸山香澄`), which suggests several trainers [I].
   - A similar 9-band v2ProPlus set was circulating on Quark Netdisk by 2026-06 [P].
4. **GPT-SoVITS V5 is on `main` (2026-10-05) but has no release tag.**
   - The latest tag is still `20250606v2pro` [P].
   - Its claims: "significantly improved voice similarity without SoVITS fine-tuning" and a new vocoder.
   - It makes no cross-lingual claim, and the pretrained folder contains only the S2 (SoVITS) stage and the vocoder [P].
5. **Legal position: stricter than for Kurisu, because the rights holder has written explicit rules.**
   - The official BanG Dream! guideline prohibits "creating, displaying, distributing or publishing derivative works **that use the content's audio**" [P].
   - The GBP streaming guideline (dated 2026-09-24) prohibits videos made by altering game voices, "**including inputting voices into generative AI**" [P].
   - The GBP terms of service ban reverse engineering [P].
   - Japan's Ministry of Justice (MOJ) published interpretive guidance on 2026-08-07 (v1.1 on 2026-10-22). It says **voice is protected by publicity rights and by the right not to have one's voice used without permission**, and **non-profit posting does not negate the latter** [P].
   - A Tokyo District Court ruling (2026-09-30) first held that voice falls within publicity rights [P/S].
   - Nippairen (日俳連, the Japan Actors' Union) published its **voice-AI guideline on 2026-10-15**. It requires stage-by-stage consent and treats **multilingual use as needing new permission** [P].
   - HF itself has no voice-specific rule. No BanG Dream repo on HF has been disabled; the only removal route is DMCA [P].
   - The user's plan (public weights; Japanese characters speaking zh/en) is squarely in the category the 2026 Japanese documents target [I].
6. **Correction to the Kurisu report §5 item 6.** I found no Nippairen guideline dated 2024-10-15.
   - The document with the "consent, compensation, control" pillars is the **2026-10-15** guideline [P].
   - Nippairen's 2024-10 items are different things: the #NOMORE launch (2024-10-15, a voice-actor volunteer group), an open questionnaire to political parties (one party's reply, dated 2024-10-22, is on nippairen.com [P]), and the 2024-11-13 statement by three industry bodies [S].
   - The prior date conflates these [I].

---

## 1. Existing BanG Dream TTS / VC work

### 1.1 Hugging Face

**Method.** I swept the HF API for models, Spaces and datasets matching `bangdream`, `bang-dream`, `bandori`, `mygo`, `mujica`, `roselia`, `poppin`, `morfonica`, `bestdori` and `bandream`.

| Repo | Base / type | Characters | Langs | Data source | Created / modified | Status, reception | Tag |
|---|---|---|---|---|---|---|---|
| `spaces/Mahiruoshi/BangDream-Bert-VITS2-OUTDATE` ("BangDream-ShojoKageki Bert VITS2") | Bert-VITS2 multi-speaker | ~50: 7 bands, MyGO, plus Revue Starlight | ja (zh not verified) | `filelists/Scenarioband4-046.asset` (GBP scenario files) | 2023-10-01 / 2025-02-16 | **RUNTIME_ERROR**; 101 likes, the most-liked BanG Dream voice repo on HF | [P] HF API |
| `spaces/Mahiruoshi/BangDream-Bert-VITS2` | Wrapper that iframes `Bangstarlight-VITS2` (Docker) | ~60 images incl. Ave Mujica | "日语特化" (Japanese-specialised) tab live; "中日" (zh/ja) tab commented out | — | 2025-02-16 / 2025-09-14 | RUNNING. Banner (2025-05-18): "鉴于GPT-Sovits已相当成熟，本网站仅供爱好者进行便捷的在线推理" (GPT-SoVITS is now mature enough that this site is only for hobbyist online inference) + Baidu Netdisk local package + QQ group | [P] app.py |
| `spaces/Mahiruoshi/MyGO_VIts-bert` | Bert-VITS2 (`G_400000`, `BangDreamV22/G_630000`) | 7 bands + MyGO | zh + ja (the wrapper Space labels it the "中日" site) | **`Data/Bushiroad/classifedSample/<char>/classN/*.wav`**: raw game clips named `band5-008-017`, `event43-05-032`, `main005-032` | 2023-09-30 / 2024-05-14 | RUNTIME_ERROR but public; game audio still downloadable | [P] |
| `spaces/Plana-Archive/BanG-Dream-VITS` | Bert-VITS2 fork | MyGO, Ave Mujica, 7 bands | ja | not stated | 2026-02-13 / 2026-08-30 | RUNNING, 8 likes; Indonesian README; no disclaimer | [P] |
| `Wanlau/RVC_BanGDream` | **RVC v2 48 kHz** voice conversion | 47 models: PPP, AG, HHW, PasPale, Roselia, Morfonica, RAS, MyGO. Soyo in two modes; singing ("chanter") variants for Ako, Rinko, Sayo | VC | "voice data from bestdori" | 2024-01-10 / 2025-07-18 | live, 62 likes; disclaimer quoted in §5.6 | [P] README |
| `Wanlau/GPT-SoVITS_BanGDream` | GPT-SoVITS **v2ProPlus** | 1 (Rinko); ships its own training MP3s and `.list` | ja | Bestdori | 2025-08-18 | live, 0 likes | [P] |
| `KitsuneX07/BanG_Dream_TTS` | GPT-SoVITS v1 + v2 | 40; IDs 1–40 follow Bestdori character IDs; no Ave Mujica | ja | dataset below | 2024-08-04 / 2024-08-19 | live, `cc-by-nc-sa-4.0` | [P] |
| `datasets/KitsuneX07/Datasets_for_BanG_Dream` | 40 per-character zips | 40; e.g. Kasumi 1.97 GB, Arisa 1.79 GB, Lisa 1.72 GB, Moca 1.66 GB | ja | made with `KitsuneX07/Dataset_Maker_for_Galgames` | 2024-06-27 | live, **963 downloads**, `cc-by-nc-sa-4.0` | [P] HF API |
| `spaces/dabao1955/BanGDreamAI-light` (title "BanGDreamAI") | Flask Live2D chat UI; calls TTS at `127.0.0.1:8000` | Live2D for 7+ bands | — | Bestdori Live2D | 2025-05-22 | SLEEPING; same layout as GitHub `Paraworks/BangDreamAi` | [P] / [I] |

**Excluded (not voice):**
- CyberHarem and BangumiBase image sets (2023–25).
- `HELPMEEADICE/BanG-Dream-All-in-One`, an image LoRA (2026-04-18).
- `KomeijiForce/*`, text and role-play benchmarks.
- `Takaharadesu/BanG_Dream_150k` / `270k`, zh–ja text.

**Status.** Every voice repo above reports `disabled:false` and `gated:false` [P HF API]. I found no BanG Dream voice repo that was taken down.

### 1.2 ModelScope (dolphin search API)

| Repo | What | Date | Tag |
|---|---|---|---|
| **`HELPMEEADICE/BanGDream-Qwen3TTS-LoRA`** | Details below | 2026-05-21 | [P] |
| `HELPMEEADICE/MyGO-Qwen3TTS-LoRA` | Same recipe, 5 MyGO speakers (`spk_id` 3000–3004) | 2026-05-21 | [P] |
| `HELPMEEADICE/GPT-SoVITS-BanG-Dream-PPP` | PPP ×5, one ~296 MB `.7z` per character; GPL-3.0; empty card | 2025-06-13 | [P] |
| `jiey502/bangdream-tsugu`, `YetGirt/MygoTTS_test`, `zijia233/mygo_Anon` | Single-character tests | 2025-01 … 2025-03 | [P] listing only |

**`HELPMEEADICE/BanGDream-Qwen3TTS-LoRA` in detail** [P]:
- PEFT LoRA on **Qwen3-TTS-12Hz-1.7B-Base**: r=16, α=32, `talker_only`, targeting q/v/o projections.
- `tts_model_type: custom_voice`, with **40 speakers** in one adapter: AG, HHW, Morfonica, MyGO, PasPale, PPP, RAS, Roselia.
- Plus a `speaker_embedding.safetensors`.
- GPL-3.0 licence and an empty model card.
- 15 downloads at the time of checking.

**The MyGO run's own `metrics.json`** [P]:
- 3 epochs and **110 steps** at batch 16×4, lr 3e-5.
- Final talker loss 5.70.
- No validation set and no samples.

This is the only public Qwen3-TTS BanG Dream model I found. It is a very short run with no evidence of quality and no multilingual claim [I].

### 1.3 GitHub

| Repo | What | Dates | Tag |
|---|---|---|---|
| `zyf722/bestdori-voice-extractor` (MIT, 24★) | Downloads Bestdori `.asset` scripts and `.mp3` voices and writes a GPT-SoVITS `.list` (`file\|chara-id\|locale\|text`). README: "nearly 14 GB of voice files". Single-character mode added 2026-01-08 | 2024-02-27 / 2026-01-11 | [P] |
| `Paraworks/BangDreamAi` (58★) | Live2D "AI vtuber" whose TTS backend is `Mahiruoshi/MyGO_VIts-bert`. **Likely the source of the "BanGDreamAI" name** | 2023-10-20 / 2024-05-16 | [P] / [I] |
| `TTZW1001/bangdream-voice-tts` (0★) | Next.js + FastAPI inference UI for pre-trained GPT-SoVITS **v2ProPlus** models, 9 bands. **`prompt_lang` fixed to `ja`; target `zh`/`ja`; default `zh`**. Importer reads `G:\QuarkDownload\bangdream` with band folders "Hallo Happy World / pastel⋆palettes / Mujica / RAS …" and `参考音频` sub-folders; character variants live in sub-folders | 2026-06-12 / 06-13 | [P] |
| `HELPMEEADICE/Bestdori_voice_download`, `CodyNotFound/BestdoriVoice`, `SubaruAkuru/BandoriStoryAssetAndVoiceDownload`, `zhaomaoniu/BanGDream-vits-dataset-downloader` | More Bestdori story-voice downloaders | 2023-07 … 2026-07 | [P] search API |

### 1.4 Bilibili / Tieba

- **渐展工作室, BV1UkLxzCEV9 (2025-04-23).** Introduces bestdori-voice-extractor and shares a MyGO five-member training set on Baidu Netdisk [P].
- **Same uploader, BV11XJnzKEAp (2025-05-22, ~14k views).** MyGO GPT-SoVITS **V4** models on Baidu Netdisk.
  - Notes: "AI synthesis" and "no reposting".
  - Tip: "效果不好可以多换几个参考音频" (if results are poor, try other reference clips).
  - No data-source statement and no visible quality comments [P].
- **Search snippets** mention two Bilibili v2ProPlus model shares (8 characters incl. MyGO members; a 6-character Ave Mujica set). I could not open them [S].
- **Tieba thread 9670908957**, "【资源导航？】bangdream语音模型相关资源传送门整合". Both the desktop and mobile pages return HTTP 403 [U].

### 1.5 Origin of the local "bangdreamai" pack (`E:\BaiduNetdiskDownload\bangdreamai`, read-only inspection)

- **Layout** [P]:
  - `GPT-SoVITS-v2pro-20250604/` is the official integration package; its matching release tag is `20250606v2pro`.
  - It contains 52 GPT and 52 SoVITS v2ProPlus weights, with file mtimes from **2025-06-06 to 2025-10-15**.
  - `角色语音/<band>/<char>.zip` holds the MP3 clips, named by transcript.
- **Naming** [P]: Band prefixes `ppp_`, `Afterglow_`, `HHW_`, `パスパレ_`, `モニカ_`, `RAS_`, `Roselia_`, `MyGO_`, `Mujica_`.
- **Persona variants** [P]:
  - `豊川祥子_白/黒`
  - `祐天寺若麥_乖猫/哈气`
  - `长崎素世_夹`
  - `大魔姬亚子`
  - `有咲_吐槽特化`
  - `ykn神人版`
  - `Mortis`
- **Mixed scripts** [P/I]: HHW, MyGO and Roselia use Simplified Chinese names (北泽育美, 千早爱音, 冰川纱夜), while the other bands use Japanese. This points to several contributors or a repack [I].
- **Not found** [U]: no readme or credits file in the pack, and no web hit for "bangdreamai" as a pack name.
- **Closest public analogues** [P/I]:
  - TTZW1001's Quark-sourced set (§1.3): same 9 bands, v2ProPlus, variant sub-folders, but different folder names. It is a related distribution, not provably the same one.
  - Mahiruoshi's 2025-05-18 Baidu "整合包" link. It predates the pack's latest checkpoints, so it is at most a predecessor [I].

---

## 2. Where BanG Dream voice data comes from

| Source | What / scale | Quality | Tag |
|---|---|---|---|
| **Bestdori asset mirror** (`bestdori.com/assets/jp/sound/voice/scenario/<dir>_rip/<id>.mp3`; index `api/explorer/jp/assets/_info.json`) | **366,642 entries under `sound/voice`**, ≈361k MP3s after removing ~5.6k per-folder `SoundBundleBuildData.asset` files. Scenario entries 360,327 (≈355k MP3s): resourceset (card stories) 150.5k, eventstory 142.2k, bandstory 26.3k, actionset 22.2k, birthdaystory 6.4k, afterlivetalk 4.2k, mainstory 3.7k, loginstory 3.6k. Plus system 2.5k, gacha 1.4k, backstage 1.1k. Transcripts and speaker IDs are in the scenario `.asset` JSON | Two sample files probed with a 4 KB range request: **MPEG-1 Layer III, 32 kHz, mono, Xing VBR, LAME 3.99r**. Durations 2.3 s and 0.83 s → **~83 kbps**. `Last-Modified: 2025-10-22`, served via Cloudflare | [P] |
| Volume estimate | 14 GB (extractor README) at ~83 kbps ≈ **~375 h** in total. ÷ ~45 main characters ≈ **5–10 h each**. MyGO and Ave Mujica have fewer because they joined the game later | — | [I] |
| **The user's pack** | 52 sets, 21.6 h (≈5–6% of the estimate above). Sampled zips: Anon 89 clips, Taki 196, Saaya 334, mostly 6–10 s lines | Same container as Bestdori: 32 kHz mono LAME 3.99r, **65–87 kbps**. It is the Bestdori file itself, not a re-encode | [P] local headers |
| Game client (Unity asset bundles; Bestdori's `_rip` folders include `SoundBundleBuildData.asset`) | — | Native codec unknown. The LAME/Xing header suggests Bestdori or Craft Egg encoded with LAME, not a raw FMOD (Unity's audio engine) extract. Whether the bundles hold anything above 32 kHz is unknown. **The GBP terms ban reverse engineering (Art. 6(17))** | [U] / [I] / [P] ToS |
| Anime (BanG Dream! 1–3, It's MyGO!!!!!, Ave Mujica) | Hours per main character for the anime casts | Broadcast/BD audio is 48 kHz but mixed with BGM and SFX, so it needs source separation. BangumiBase only has image sets on HF | [I] |
| Already-made HF corpora | KitsuneX07 (40 characters, 1.7–2.0 GB per top character; format not checked); Wanlau's per-character MP3s | as Bestdori | [P] / [U] |

**Implication** [I]: data quantity is not the bottleneck. Ceiling quality is: 32 kHz lossy mono. Any 44.1/48 kHz codec will be trained on band-limited input. The hard ceiling is the 16 kHz Nyquist limit of a 32 kHz sample rate, and LAME's VBR low-pass at ~80 kbps sits below that.

---

## 3. GPT-SoVITS V5 and Chinese community practice (2026)

### 3.1 GPT-SoVITS V5

| Item | Finding | Tag |
|---|---|---|
| Release tag | **None.** Latest release is `20250606v2pro` (2025-06-06). V5 merged to `main` 2026-10-06 ("Integrate v5 training and inference updates with single-GPU training support"). Later commits: 2026-10-18 "为 API v2 增加 V5 和 V2PP 语义流式加速" (V5/V2PP semantic streaming for API v2) and a 2026-10-19 streaming fix | [P] GitHub API |
| Claims (README "V5 Release Notes") | (1) "Significantly improved voice similarity without SoVITS fine-tuning". (2) The updated vocoder "significantly reduces high-frequency spectral mirroring and aliasing". (3) `cuda_graph` / `flash_attention` acceleration. README carries a SeedTTS-eval CN speaker-similarity chart ("v5pro-seedtts-img-v1"). **No cross-lingual or few-shot claim specific to V5.** Generic README still says cross-lingual inference across en/ja/ko/yue/zh | [P] README |
| Weights | `lj1995/GPT-SoVITS/gsv-v5-pretrained/`: `s2Gv5dev.pth` (766 MB), `s2Gv5turbo.pth` (766 MB), `vocoder.pth` (58 MB). **No new S1/GPT model in that folder**, so the AR stage is presumably unchanged from v2Pro. Repo licence MIT | [P] HF API / [I] |
| Changelog_CN | "2026.10.05 … V5新版本 … 类型: 优化" (V5 new version; type: optimisation) | [P] |
| Note relevant to game-MP3 data | v2Pro notes: "For training sets with average audio quality, v1/v2/v2Pro can deliver decent results, but v3/v4 cannot". This is why community packs stay on v2ProPlus | [P] README |

### 3.2 What creators use and how Japanese characters are made to speak Chinese

- **The dominant pattern is unchanged** [P]: train a per-character GPT-SoVITS v2ProPlus model on Japanese game lines, then synthesise with a Japanese reference clip and Chinese target text.
  - `TTZW1001` hard-codes `prompt_lang=ja` and target `zh` (default) or `ja`. Its stated reason: "当前主要合成中文" (we mainly synthesise Chinese).
  - Bilibili model shares give the same advice: swap reference clips when results are poor.
- **Movement since 2025** [P / I]:
  - Mahiruoshi abandoned Bert-VITS2 for GPT-SoVITS (2025-05-18).
  - Bilibili shares moved to V4 (2025-05), then to v2ProPlus (the pack, 2025-06–10).
  - In 2026-05 the first **Qwen3-TTS LoRA** multi-speaker BanG Dream adapters appeared (§1.2).
  - I found no BanG Dream model on IndexTTS2, CosyVoice3, Fish S2 or VoxCPM2 [U: absence on HF, ModelScope and GitHub search only].
- **Generic 2026 Chinese comparisons are zero-shot only** [S]. Liudon's blog (2026-05-18, updated 2026-09-09) compares IndexTTS-2, CosyVoice3, GPT-SoVITS, Fish S2 Pro and VoxCPM.
  - It rates IndexTTS-2 the most natural but slow.
  - It recommends VoxCPM "用于快速上手" (for getting started quickly).
  - It does not test Japanese reference → Chinese text.
- **No measured accent results** [U]: I found no Bilibili or Zhihu source that measures the Japanese accent in Chinese output for anime voices.

---

## 4. Japanese-side practice

| Item | Finding | Tag |
|---|---|---|
| Style-Bert-VITS2 | Still at v2.7.0 (2025-08-24); no commits since | [P] GitHub API |
| Irodori speaker inversion, `zenn.dev/platina` (2026-05-19, upd. 05-23) | Irodori-TTS-500M-v3. 16 learned tokens ≈ 12k parameters, **49 KB file** (vs LoRA r4: 17M params, 69 MB). 3000 steps, ~5–14 min on an RTX 4070 Ti Super. Listening only, no metrics. Where reference cloning failed (whisper style, 100 clips), both inversion and LoRA "かなり再現できている" (reproduce it quite well). Author expects inversion to be less expressive (untested) | [P] |
| Irodori LoRA on small character data, note "かみもと" (sepiablue) | ~100 clips × ~3 s (≈5 min) of a free character voice gave "比較的良い精度" (relatively good accuracy), with somewhat suppressed emotion | [S] snippet |
| Irodori-TTS-Client guide | "Broad" preset overfits on little data. Emoji-TTS fork suggests 30 min to several hours per character | [S] snippet |
| Anime character deployment | `spaces/kawai321/Umamusume-Irodori-TTS` (2026-08-23, RUNNING, ZeroGPU): Irodori v4.1-Small with **9 Umamusume characters as `.speaker.safetensors` inversion files**, `lora_adapter=None`, no disclaimer in README | [P] |
| **`phasefield-audio/Irodori-TTS-v4.1-Anime`** | 2026-09-04, 136 likes. Fine-tune of `Aratako/Irodori-TTS-v4.1-Small` "using anime-style speech data"; data not named. Independently re-annotated captions, so emoji/caption control "may behave differently". **MIT**, "follows the same … ethical restrictions as the base model". Base restriction 1: "**No Impersonation:** Do not use this model to clone or impersonate the voice of any individual (e.g., voice actors…) without their explicit consent." Quantised int8/int4/fp8 variants | [P] card |
| Irodori language coverage | Japanese only (prior report). Irrelevant for zh/en except as a Japanese-quality benchmark | [P prior] |

---

## 5. Legal and platform risk

### 5.1 Bushiroad / BanG Dream! official rules

| Document | Clause | Tag |
|---|---|---|
| **「BanG Dream!（バンドリ！）著作物利用に関するガイドライン」** (BanG Dream! copyrighted-works usage guideline), https://bang-dream.com/bdp-guideline/ (undated; "随時更新", updated from time to time) | Covers "キャラクターの名称…映像、**音声**…" (character names … video, **audio** …). Derivative works allowed only "個人により営利を目的としない場合に限り" (only by individuals, non-commercially). **Prohibited item 10: "本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開。"** (creating, displaying, distributing or publishing derivative works that use the content's audio), with "※夢ノ結唱の音声利用については別途定める" (Yume no Kesshou audio use is governed separately). Also prohibited: item 4, damaging the image of "出演キャスト" (the cast); item 8, anything that could be mistaken for official material. No AI clause | [P] |
| **GBP streaming guideline**, https://bang-dream.bushimo.jp/rule/ ("制定日：2026年9月24日", enacted 2026-09-24) | Prohibits posting game elements "（キャラクターイラスト、音楽、ボイス等）をゲーム映像から切り離して個別に投稿" (posting illustrations, music, voices etc. separated from gameplay footage). Also prohibits "…を加工・改変して作成した動画等を投稿・配信するもの（**加工・改変することを目的として生成AIにボイス等を入力する行為を含みます。**）" (videos made by altering them, **including inputting voices into generative AI for that purpose**) | [P] |
| GBP Terms of Service (rev. 2024-01-17), same page | Art. 6(17): no "改変、毀損し、または逆アセンブル、逆コンパイル、リバースエンジニアリング" (modifying, damaging, disassembling, decompiling or reverse engineering) | [P] |
| **Official AI voice product: 夢ノ結唱 (Yume no Kesshou)** — CeVIO AI / Synthesizer V voicebanks POPY and ROSE, based on the Kasumi and Yukina character voices; CeVIO release 2022-12 | Synthesizer V terms (enacted 2023-12-01) Art. 7-23 forbid using its output to make "歌声合成や音声合成モデル" (singing- or speech-synthesis models). Bushiroad sells licensed AI voices for two of these characters, so a free fan model of the same voices competes with a product [I] | [P] terms / [S] dates |

**Reading** [I]: a public HF model trained on GBP audio is, on its face, a "derivative work that uses the content's audio" (item 10), and the 2026-09 streaming rule names generative-AI input explicitly. The project's non-commercial status does not help, because item 10 is listed among uses prohibited *even within* permitted fan activity.

### 5.2 Cast and agencies

- **No BanG Dream-specific statement found** [U]. Searches in Japanese, Chinese and English turned up no statement on AI voice cloning by any cast member of PPP, Afterglow, PasPale, Roselia, HHW, Morfonica, RAS, MyGO!!!!! or Ave Mujica, and none by 響 (Hibiki), the Bushiroad-group agency that represents part of the cast [S/U for the affiliation]. I did not check individual X accounts.
- **Industry bodies that cover these casts' agencies** [S]:
  - The 3-body joint statement (2024-11-13): 日俳連 (Nippairen), マネ協 (Japan Entertainment Management Association) and 声事協 (Japan Voice-Acting Business Council). It asks for consent before training on or using voices.
  - The 13-organisation statement (2025-02): asks for AI labelling and consent.
  - #NOMORE無断生成AI (launched 2024-10-15, 26 voice actors).
  - A volunteer voice-actor group's submission to the MOJ panel (2026-05-28).
- **What the MOJ submission says** (2026-05-28, 資料7) [P]:
  - It lists "⑤モデルの公開 — 特定の声優の声を学習させたAIモデルを…プラットフォームで無償公開する類型" (publishing models: free release on platforms of AI models trained on a specific voice actor's voice) as a harm category.
  - It cites cases where a developer stated that a model was trained on "特定のゲームに収録された特定キャラクター" (a specific character recorded in a specific game).
  - It asks for a platform takedown framework.

### 5.3 Japanese law and policy, 2025–2026

| Date | Item | Content relevant here | Tag |
|---|---|---|---|
| 2025-05 | METI, 「肖像と声のパブリシティ価値に係る現行の不正競争防止法における考え方の整理」 (how the current Unfair Competition Prevention Act applies to the publicity value of likeness and voice) | Voice may fall under the Unfair Competition Prevention Act (UCPA) Art. 2(1)(i)/(ii)/(xx)/(xxi). A **voice + characteristic lines or character combination** can be a source indicator | [S] Yuhikaku / [P] via MOJ summary |
| 2026-04-17 → 08-07 → 10-22 | **MOJ 「肖像、声等の無断利用による民事責任の在り方に関する検討会 取りまとめ報告書―生成AIによるパブリシティ権侵害等に関する解釈指針―」** (study group on civil liability for unauthorised use of likeness and voice: final report and interpretive guidance on generative-AI publicity-right infringement), https://www.moj.go.jp/MINJI/minji07_00400.html. Report `001468286.pdf`, summary `001468506.pdf`; v1.1 adds the 2026-09-30 ruling | (a) "保護の対象である「肖像等」には…人の「声」も含まれる" (the protected "likeness etc." includes a person's voice), both for publicity rights and for 声をみだりに利用されない権利 (the right not to have one's voice used without permission). (b) "特定のキャラクターを演じる声優の声自体に…効力がある場合には、その声に顧客吸引力が認められる" (when the voice of an actor playing a character itself drives sales, that voice has commercial pull). (c) Case 2-2, an AI cover in the voice of the actor's **anime character** with the character named: the character name "本人と結び付く情報" (links to the person) and supports infringement. (d) Providing a generative AI that makes a celebrity's voice easy to produce, **offered for a fee with that ease as the selling point**, is infringement and can be enjoined. (e) A **dataset specialised to a specific voice actor** "第1類型に該当し…得る" (may fall under the first infringement type). (f) Training itself: no consensus, but a **preventive injunction** is possible. (g) Case 4, **no profit motive**: not a publicity-right infringement as such, but may infringe the right not to have one's voice used without permission; "収益目的がないことは…受忍の限度を超えるとの判断を否定する要素となるとはいえない" (the absence of a profit motive is not a reason to find the use tolerable). Not legally binding | [P] PDFs |
| 2026-09-30 | Tokyo District Court, 令和7年(ワ)第70635号 (case number): a male voice actor vs. a short-video platform over 188 AI-narration videos | **First ruling that voice falls within publicity rights** ("肖像と同様に個人の人格の象徴", a symbol of personality like a likeness). Claim dismissed because the videos had already been deleted | [P] case number on MOJ page; [S] Jiji, Nikkei, Bengo4 |
| 2026 | 知的財産推進計画2026 (Intellectual Property Strategic Program 2026) | Continue discussing "ハードロー整備（不正競争防止法改正を含む）" (binding legislation, including amending the UCPA). **No voice-protection bill has been enacted** | [P] quoted in MOJ summary; [S] |
| **2026-10-15** | **Nippairen 「音声に関する生成AI利用ガイドライン」** (voice generative-AI usage guideline), PR TIMES release by 協同組合日本俳優連合 | Prior consent "いかなる段階においても" (at every stage: training, generation, use). Compensation equivalent to a performance. **"多言語化される場合でも、新たな利用許諾と…追加の対価"** (multilingual versions require new permission and extra payment). Penalty of "売上相当額" (an amount equal to sales); model deletion on demand. Scope includes overseas use. Personal fan expression using legitimately bought products is excluded [S]. Endorsed by マネ協 and 声事協. Private standard, not binding | [P] PR TIMES; [S] ben54, Kumanichi |
| 2024-03 (unchanged) | Agency for Cultural Affairs (ACA), 「AIと著作権に関する考え方について」 (General understanding on AI and copyright) | Copyright Act Art. 30-4 (non-enjoyment use) allows training unless enjoyment purposes coexist or the rights holder's interests are unjustly harmed. **I found no 2025–26 revision** | [S] / [U] |

**China, for Bilibili distribution of Mandarin output** [S/U]:
- Chinese voice actors protested AI dubbing in 2026-03, and creators removed videos (OECD AI incident log, 2026-03-16) [S].
- Civil Code Art. 1023 extends portrait-right protection to voice [U, from memory].

### 5.4 Hugging Face policy and takedowns

- **Content Policy** (effective **2025-04-10**, no later revision) [P]:
  - No clause on voice cloning, impersonation or synthetic voices.
  - Relevant clauses: IP infringement; "intentionally deceptive Content"; "consent as a core value"; and the right to "moderate other types of Content in response to evolving challenges".
  - Actions available: gating, unranking, NFAA tag (Not For All Audiences), disabling, suspension.
  - Takedowns go through `dmca@huggingface.co` with a counter-notice process.
- **Observed enforcement** [P]:
  - BanG Dream: none. Mahiruoshi's Space still hosts `Data/Bushiroad/*.wav` (since 2023), and KitsuneX07's dataset (963 downloads) is live.
  - Comparable sets are also live and ungated: `simon3000/genshin-voice` (654k wavs, 16.5k downloads, updated 2026-08) and `joujiboi/japanese-anime-speech-v2`.
  - `litagin/moe-speech` (Japanese game-voice corpus; controversy 2024-01 [S Automaton]) is now **gated manual** (modified 2026-04-17).
- **Takedown precedent** [S]: the AnimeSessions RVC collection was taken down after "many takedown requests, and even a potential legal dispute". Its README says so, but the date is not shown.
- **Gap** [U]: I found no documented 2025–26 HF takedown of an anime-character voice model, and no BanG Dream takedown on any platform.

### 5.5 How comparable public projects word their disclaimers (verbatim) [P]

| Project | Licence label | Disclaimer |
|---|---|---|
| `Wanlau/RVC_BanGDream` | none | "These models are aiming to facilitate communication and learning. Engaging in illegal activities, is strictly prohibited." / "仅供学习交流试用，不可用于非法用途。" (for learning and exchange only; not for illegal use) |
| `zomehwh/vits-uma-genshin-honkai` (375 likes, PAUSED) | apache-2.0 (sic) | "严禁将模型用于任何商业项目，否则后果自负" (commercial use strictly prohibited; you bear the consequences) |
| `Genius-Society/hoyoTTS` | cc-by-nc-nd-4.0 | none |
| `KitsuneX07/*`, `ORI-Muchim/BlueArchiveTTS`, `simon3000/genshin-voice`, `kawai321/Umamusume-Irodori-TTS` | NC-SA / MIT / none / none | none |
| 渐展工作室 (Bilibili) | — | "该视频使用人工智能合成技术" (this video uses AI synthesis) / "未经作者授权，禁止转载" (no reposting without permission) |
| Irodori base (contrast) | MIT | "No Impersonation … without their explicit consent" |

None of them names the rights holder or the cast, cites the official guideline, or provides an opt-out or takedown contact [P]. In 2026 these are exactly the elements the MOJ report and the Nippairen guideline make relevant [I].

### 5.6 What this means for the planned HF release [I]

**Risk, from highest to lowest:**
1. **Redistributing the game audio itself.** This means datasets, reference clips in Spaces, or training clips in repos. It is a straight copy of Bushiroad content, violates guideline item 10, and is the easiest DMCA target. Mahiruoshi, Wanlau and KitsuneX07 do this today.
2. **A public multi-character model plus a Space that generates named-character speech in zh/en.**
   - This matches MOJ case 2-2: character name plus voice similarity.
   - It matches the volunteer group's "⑤ 無償公開" (free release) category.
   - It matches Nippairen's multilingual clause.
   - Non-commercial status reduces publicity-right exposure but, per MOJ case 4, not exposure under the right not to have one's voice used without permission.
3. **Weights only, gated, no hosted demo, no audio.** This is still within item 10 [I], but it is the least visible option.

**Mitigations the 2026 documents point to:**
- Gate access.
- Ship no game audio.
- Mark every output as AI-generated.
- Avoid content that harms the cast's image (guideline item 4).
- Give a clear takedown contact.
- State that the project is unofficial (item 8).

None of these removes the item-10 conflict. Only a licence from Bushiroad would.

---

## 6. Corrections to earlier reports

1. **`kurisu_v2/research/final_report.md` §5 item 6.** It dates the Nippairen guideline to "2024-10-15".
   - The guideline titled 「音声に関する生成AI利用ガイドライン」 was announced on **2026-10-15** [P PR TIMES].
   - 2024-10-15 is when #NOMORE無断生成AI launched [S].
   - The 2024/10 PDF on nippairen.com is a political party's reply, dated 2024-10-22, to Nippairen's open questionnaire on generative AI, not a guideline [P].
   - Nippairen also has an earlier 「生成系AI技術の活用に関する提言」 (proposal on using generative AI); I did not check its date [U].
   - Verdict: the prior report is imprecise. It conflates the 2024 campaign and statements with the 2026 guideline.
2. **`sota_models_2026-10.md` §1.2, GPT-SoVITS V5 row.** It is still correct that there is no tag. Add these facts:
   - V5 was merged to `main` on 2026-10-06.
   - The weights consist only of the S2 stage and the vocoder.
   - Streaming V5 support arrived in API v2 on 2026-10-18 [P].

---

## 7. Could not verify

- **Pack origin:** the uploader of the "bangdreamai" Baidu pack and the trainer of the 52 checkpoints. The Tieba resource thread returns 403; Bilibili v2ProPlus share pages were seen only as snippets.
- **Game-client audio:** whether the GBP client stores voice above 32 kHz or in a lossless codec. I did not examine any asset bundle, by design and because of ToS Art. 6(17).
- **Per-character hours:** I did not compute these from the `.asset` speaker IDs. The 5–10 h figure is an estimate. I also did not check the audio format inside KitsuneX07's zips.
- **Quality reports:** user-reported quality for any BanG Dream model. No comments or metrics were accessible, and the ModelScope Qwen3-TTS LoRAs have no samples.
- **Cast statements:** any BanG Dream cast member's or 響 (Hibiki)'s statement on AI voice. None found; individual X accounts not checked. I also did not check whether any BanG Dream cast member is among #NOMORE's 26 participants or the volunteer group behind the 2026-05-28 MOJ hearing.
- **Full texts:** the Tokyo District Court judgment (seen only via news and the MOJ citation); the full 143-page MOJ report beyond the summary PDF; and the full Nippairen guideline text, including the fan-use exclusion (only the release and news summaries were read).
- **HF takedowns:** any 2025–26 takedown of an anime voice model, and the date of the AnimeSessions takedown.
- **Other details:**
  - The access conditions of gated `litagin/moe-speech`.
  - Any ACA AI-copyright revision after 2024-03.
  - The text of the CeVIO AI 夢ノ結唱 terms (only the Synthesizer V terms were read).
  - The China Civil Code voice clause (from memory).
