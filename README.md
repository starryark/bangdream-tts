# BanG Dream! character voice TTS: 49 voices in Japanese, Mandarin and English with Qwen3-TTS

**Reference-free text-to-speech in the voices of 49 BanG Dream! characters, made by fine-tuning Qwen3-TTS on their Japanese voice lines, with native-accented Mandarin and English.**

This repository is the complete, reproducible research record, from extracted voice lines to a selected checkpoint:
- data building and verification (`pipeline/`);
- a codec-calibrated speaker-identity probe and the zero-shot floor (`e0`);
- one shared 49-voice fine-tune with language ids, native replay and in-voice pseudo-Mandarin (`e1`);
- two negative results kept on record: more self-generated rows (`e2`) and zero-shot voice conversion (`e3`);
- per-voice RVC converters that put native Mandarin into the 15 voices whose Mandarin was generic (`e4`, `e5`);
- every experiment's plan, written before its results, and the results themselves.

It is an unofficial fan project, with no affiliation to Bushiroad, Craft Egg, the BanG Dream! project or the voice cast.

| | |
|---|---|
| **Model** | [starrydark/BanG_Dream_Qwen3_TTS](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS) |
| **Results** | [E1 (released model)](e1/E1_RESULTS.md) · [E2](e2/E2_RESULTS.md) · [E4](e4/E4_RESULTS.md) · [E5](e5/E5_RESULTS.md) · [E0 data and probe](e0/e0_results.md) |
| **Sibling projects** | [makise-kurisu-tts](https://github.com/starryark/makise-kurisu-tts) (where the trainer, language-id recipe and accent metrics started) · [tricolour-tts](https://github.com/starryark/tricolour-tts) |

**Bands and voices:** Poppin'Party, Afterglow, Pastel\*Palettes, Roselia, Hello, Happy World!, Morfonica, RAISE A SUILEN, MyGO!!!!!, Ave Mujica: 45 characters, 49 speaker rows (大魔姫あこ, 月ノ森 Soyo, 月ノ森 Sakiko and Mortis are separate rows). The full table with per-voice quality is on the [model card](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS#voices).

## 中文简介 / 日本語

《BanG Dream!》49 位角色的跨语言语音合成（TTS）：用日语原声台词微调 Qwen3-TTS，按说话人 ID 直接合成（无需参考音频），支持日语、普通话和英语。普通话口音问题的处理：语言 ID、母语回放数据、角色声线伪数据，以及针对 15 个声线的逐角色 RVC 转换。仓库包含完整流程、每个实验的预先登记计划与结果。非官方粉丝项目，与官方及声优无关。

『BanG Dream!』49 キャラクターの多言語 TTS（Qwen3-TTS 微調整）。話者 ID で呼び出し、日本語・中国語・英語に対応。非公式ファンプロジェクトで、公式・キャストとは無関係です。

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
kurisu_pod/  the Kurisu project's pod code this pipeline imports (train_sft.py trainer, p03 audio
           policy, p05 native replay, accent_metrics, probes), vendored at the version used
research/  approach research, data facts, rights notes and the recommendation behind the plan
review/    independent code reviews and their check scripts
release/   publish_hf.py (model upload), stage_github.sh (builds this repo), MODEL_CARD.md
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
- The official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) restricts derivative works that use the franchise's audio; `research/community_and_legal_2026-10-29.md` summarises it and the 2026 Japanese guidance on voice rights.
- Do not use the model to impersonate the cast or anyone else, to make content that could be mistaken for official material, to harm the image of the characters or cast, or commercially. Label generated audio as AI-generated.
- Rights holders and cast members can open an issue or contact the owner for removal; the model will be taken down on request.

Code: Apache-2.0 (`LICENSE`). The Qwen3-TTS weights are Apache-2.0; the fine-tuned voices carry no grant of rights.
