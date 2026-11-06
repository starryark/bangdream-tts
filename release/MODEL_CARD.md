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
- girls-band-party
- mygo
- ave-mujica
- speech-synthesis
- voice-model
- game-voice
- anime
---

# BanG Dream! character voices: Qwen3-TTS text-to-speech in Japanese, Mandarin and English

An unofficial, fan-made **text-to-speech model with 49 BanG Dream! character voices** across eight bands: Poppin'Party, Afterglow, Pastel\*Palettes, Roselia, Hello, Happy World!, Morfonica, RAISE A SUILEN, MyGO!!!!! and Ave Mujica (persona variants such as 大魔姫あこ, 月ノ森 Soyo / Sakiko and Mortis have their own rows).

- **Each voice is called by its speaker id.** No reference audio is needed.
- **Three languages:** Japanese (the characters' own language) plus Mandarin and English. Language ids, native replay and in-voice pseudo-Mandarin keep the Japanese accent out of Mandarin.
- **Built on** Qwen3-TTS-12Hz-1.7B-Base, full fine-tune.

**Code:** [github.com/starryark/bangdream-tts](https://github.com/starryark/bangdream-tts): data pipeline, fixed trainer, per-voice RVC route for native Mandarin, and every pre-registered plan and result.

**Also known as:** BanG Dream! Girls Band Party · バンドリ！ ガールズバンドパーティ！ · ガルパ · 邦邦 · 少女乐团派对.

日本語：『BanG Dream!』49 キャラクターボイスの非公式ファンメイド TTS（Qwen3-TTS 微調整）。話者 ID で呼び出し、参照音声は不要。日本語・中国語・英語に対応。公式・キャスト・運営とは無関係です。

中文：《BanG Dream!》49 位角色的非官方粉丝向语音合成（TTS）模型，基于 Qwen3-TTS 微调。按说话人 ID 直接合成，无需参考音频；支持日语、普通话和英语。与官方、运营及声优无关。

| | |
|---|---|
| **Voices** | 49 speaker rows (45 characters; 4 persona variants), table below |
| **Languages** | Japanese, Mandarin Chinese, English (`language="Japanese" / "Chinese" / "English"`) |
| **Type** | Qwen3-TTS `custom_voice` checkpoint: text in, speech out, no reference audio |
| **Base model** | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), full fine-tune |
| **Training data** | 9,268 Japanese voice lines of the 49 voices (from 22.1 h kept), plus 1,509 native Mandarin/English replay clips and 564 gated in-voice pseudo-Mandarin rows: 11,341 training rows, 23.4 h |
| **Version** | E1b epoch 2 (trained 2026-10-30, released 2026-11-06) |
| **License** | Unofficial fan model, no rights granted ([details](#rights-and-responsible-use)) |
| **Code** | [starryark/bangdream-tts](https://github.com/starryark/bangdream-tts) |

## Use (Python)

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/BanG_Dream_Qwen3_TTS",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(
    text="今日の練習、思ったより早く終わったね。",
    speaker="ppp_kasumi",            # see the table below
    language="Japanese",             # or "Chinese" / "English"
)
sf.write("kasumi.wav", wavs[0], sr)
```

**Usage notes:**
- **Always pass `language`.** Training used a language id on every row; it is what keeps Mandarin and English free of the Japanese accent.
- **Keep the default sampling** (temperature 0.9, top-k 50). Greedy decoding makes codec language models loop.
- **Try a few seeds** for Mandarin and English on the voices marked "generic" below.
- Generated audio is synthetic: **label it as AI-generated** wherever you share it.

## Voices

`speaker` ids are `<band>_<name>`. "Mandarin identity" is the share of 32 Mandarin test renders that an independent speaker probe attributes to the right voice among all 49 ("Japanese identity" is the same over the Japanese test renders. **generic** means the voice speaks Mandarin with a native accent but in a timbre closer to the base model's default voice than to the character; *partly generic* means that happens on some seeds).

| `speaker` | Character | Band | Japanese data | Japanese identity | Mandarin identity | Mandarin voice |
|---|---|---|---:|---:|---:|---|
| `ppp_kasumi` | 戸山香澄 | Poppin'Party | 35 min | 0.84 | 0.25 | **generic** |
| `ppp_tae` | 花園たえ | Poppin'Party | 31 min | 1.00 | 0.94 | own voice |
| `ppp_rimi` | 牛込りみ | Poppin'Party | 24 min | 1.00 | 0.25 | **generic** |
| `ppp_saaya` | 山吹沙綾 | Poppin'Party | 47 min | 1.00 | 1.00 | own voice |
| `ppp_arisa` | 市ヶ谷有咲 | Poppin'Party | 61 min | 0.98 | 0.94 | own voice |
| `ag_ran` | 美竹蘭 | Afterglow | 26 min | 1.00 | 0.91 | own voice |
| `ag_moca` | 青葉モカ | Afterglow | 59 min | 0.98 | 1.00 | own voice |
| `ag_himari` | 上原ひまり | Afterglow | 45 min | 1.00 | 0.88 | own voice |
| `ag_tomoe` | 宇田川巴 | Afterglow | 31 min | 1.00 | 0.97 | own voice |
| `ag_tsugumi` | 羽沢つぐみ | Afterglow | 30 min | 1.00 | 0.88 | own voice |
| `pp_aya` | 丸山彩 | Pastel*Palettes | 52 min | 0.98 | 0.84 | partly generic |
| `pp_hina` | 氷川日菜 | Pastel*Palettes | 34 min | 1.00 | 0.97 | own voice |
| `pp_chisato` | 白鷺千聖 | Pastel*Palettes | 26 min | 1.00 | 1.00 | own voice |
| `pp_maya` | 大和麻弥 | Pastel*Palettes | 45 min | 1.00 | 0.94 | own voice |
| `pp_eve` | 若宮イヴ | Pastel*Palettes | 46 min | 1.00 | 0.62 | partly generic |
| `ro_yukina` | 湊友希那 | Roselia | 19 min | 1.00 | 1.00 | own voice |
| `ro_sayo` | 氷川紗夜 | Roselia | 31 min | 1.00 | 1.00 | own voice |
| `ro_lisa` | 今井リサ | Roselia | 39 min | 1.00 | 1.00 | own voice |
| `ro_ako` | 宇田川あこ | Roselia | 8 min | 1.00 | 0.88 | own voice |
| `ro_ako_maou` | 宇田川あこ | Roselia | 5 min | 0.88 | 0.31 | **generic** |
| `ro_rinko` | 白金燐子 | Roselia | 42 min | 1.00 | 0.53 | **generic** |
| `hhw_kokoro` | 弦巻こころ | Hello, Happy World! | 29 min | 1.00 | 0.94 | own voice |
| `hhw_kaoru` | 瀬田薫 | Hello, Happy World! | 30 min | 1.00 | 0.84 | partly generic |
| `hhw_hagumi` | 北沢はぐみ | Hello, Happy World! | 23 min | 1.00 | 0.81 | partly generic |
| `hhw_kanon` | 松原花音 | Hello, Happy World! | 36 min | 1.00 | 0.66 | partly generic |
| `hhw_misaki` | 奥沢美咲 | Hello, Happy World! | 37 min | 1.00 | 0.97 | own voice |
| `mo_mashiro` | 倉田ましろ | Morfonica | 23 min | 1.00 | 0.97 | own voice |
| `mo_touko` | 桐ヶ谷透子 | Morfonica | 22 min | 1.00 | 0.91 | own voice |
| `mo_nanami` | 広町七深 | Morfonica | 34 min | 1.00 | 0.94 | own voice |
| `mo_tsukushi` | 二葉つくし | Morfonica | 25 min | 1.00 | 1.00 | own voice |
| `mo_rui` | 八潮瑠唯 | Morfonica | 26 min | 1.00 | 1.00 | own voice |
| `ras_layer` | 和奏レイ | RAISE A SUILEN | 33 min | 1.00 | 1.00 | own voice |
| `ras_lock` | 朝日六花 | RAISE A SUILEN | 34 min | 1.00 | 0.50 | **generic** |
| `ras_masking` | 佐藤ますき | RAISE A SUILEN | 23 min | 1.00 | 0.81 | partly generic |
| `ras_pareo` | 鳰原令王那 | RAISE A SUILEN | 22 min | 1.00 | 0.97 | own voice |
| `ras_chuchu` | 珠手ちゆ | RAISE A SUILEN | 42 min | 0.93 | 0.75 | partly generic |
| `mygo_tomori` | 高松燈 | MyGO!!!!! | 30 min | 1.00 | 0.44 | **generic** |
| `mygo_anon` | 千早愛音 | MyGO!!!!! | 11 min | 1.00 | 0.94 | own voice |
| `mygo_raana` | 要楽奈 | MyGO!!!!! | 13 min | 1.00 | 0.91 | own voice |
| `mygo_soyo` | 長崎そよ | MyGO!!!!! | 10 min | 0.88 | 1.00 | own voice |
| `mygo_soyo_tsuki` | 長崎そよ | MyGO!!!!! | 16 min | 1.00 | 0.44 | **generic** |
| `mygo_taki` | 椎名立希 | MyGO!!!!! | 27 min | 1.00 | 0.97 | own voice |
| `mujica_uika` | 三角初華 | Ave Mujica | 6 min | 1.00 | 1.00 | own voice |
| `mujica_mutsumi` | 若葉睦 | Ave Mujica | 3 min | 1.00 | 0.91 | own voice |
| `mujica_mortis` | 若葉睦 | Ave Mujica | 5 min | 1.00 | 0.94 | own voice |
| `mujica_umiri` | 八幡海鈴 | Ave Mujica | 3 min | 0.97 | 1.00 | own voice |
| `mujica_nyamu` | 祐天寺にゃむ | Ave Mujica | 6 min | 1.00 | 1.00 | own voice |
| `mujica_sakiko` | 豊川祥子 | Ave Mujica | 6 min | 0.83 | 0.94 | own voice |
| `mujica_sakiko_tsuki` | 豊川祥子 | Ave Mujica | 13 min | 1.00 | 0.78 | partly generic |

**Summary:** Japanese is in-voice for all 49 (pooled identity 0.986, real recordings 0.988). Mandarin is in-voice for 34 voices; the 15 marked *generic* or *partly generic* fall back toward a generic timbre in Mandarin and English.

**Persona rows.** `ro_ako_maou` (Ako's 大魔姫 persona), `mygo_soyo_tsuki` and `mujica_sakiko_tsuki` (the 月ノ森 versions) and `mujica_mortis` (Mortis) are separate rows. Their real recordings are confused with the base character 89–97% of the time by the same probe, so expect them to be close to their base row.

## Evaluation

Checkpoints were chosen by selection rules written down before each training run (`eval/E1_PLAN.md`, `eval/E5_PLAN.md`). Per checkpoint: 4,356 renders, i.e. 49 voices × (held-out game lines + 12 new Japanese sentences, 16 Mandarin, 8 English) × 2 seeds.

**Measures:**
- **Identity (leak):** the share of renders a codec-calibrated speaker probe (logistic regression on Qwen3-TTS speaker-encoder vectors, fitted on resynthesised real clips) assigns to a different voice. Lower is better.
- **ja CER:** Qwen3-ASR character error rate on held-out Japanese lines.
- **Mandarin accent:** SER (shared-character syllable error rate, human-validated on the sibling Kurisu project) and lid (a language-id log-ratio; more negative = more native).
- **en WER / lid:** the same for English.

| Model | ja leak ↓ | ja p_true ↑ | ja CER ↓ | zh SER ↓ | zh lid ↓ | zh leak ↓ | Own-voice Mandarin (zh leak ≤ 0.15) | en lid ↓ | en WER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real recordings, through the codec | 0.012 | 0.777 | 0.023 | — | — | — | — | — | — |
| Floor: untrained Base, same speaker vectors | 0.100 | 0.567 | 0.032 | 0.077 | −19.5 | 0.157 | 36/49 | −12.2 | 0.001 |
| E1a epoch 1: Japanese + language ids + replay, no pseudo-Mandarin (rejected: accented Mandarin) | 0.009 | 0.810 | 0.027 | 0.177 | −12.6 | 0.008 | — | — | 0.006 |
| **This model: E1b epoch 2** | **0.014** | **0.802** | **0.028** | **0.069** | **−18.6** | **0.155** | **34/49** | **−9.3** | **0.003** |
| E2 epoch 2: + more self-generated rows (not adopted) | 0.021 | 0.786 | — | 0.063 | −19.7 | 0.186 | 34/49 | −10.1 | — |
| E5 epoch 2: + 218 RVC-converted Mandarin rows (not adopted) | 0.014 | 0.815 | 0.025 | 0.078 | −18.1 | 0.088 | 40/49 | −9.5 | 0.003 |

**How to read the table:**
- **Japanese identity is at the real-recording level** (leak 0.014 against 0.012 for real clips through the same codec; the untrained floor is 0.100). The remaining errors are mostly inside persona pairs, plus `ppp_kasumi` (0.84).
- **Mandarin accent stays at the floor** (SER 0.069 vs 0.077). With language ids and 13% native replay but only Japanese in the characters' voices (E1a), Mandarin comes out Japanese-accented (SER 0.177); adding 5% gated in-voice pseudo-Mandarin (this model) removes it.
- **Mandarin identity is the open problem.** Voices that had no usable in-voice Mandarin rows fall back to the base model's timbre.
- **Two follow-ups did not replace this model.**
  - E2 (doubling the self-generated rows) diluted identity.
  - E5 trained per-voice RVC converters on each voice's Japanese clips, converted native Mandarin into 15 voices, and retrained. Mandarin identity improved (0.155 → 0.088, 34 → 40 voices; Eve, Kanon, Hagumi, Aya, Kaoru fixed), but it missed the adoption bar fixed before training (≤ 0.08, ≥ 43/49, targets ≤ 0.15). Following that rule, E5 is not released.
  - Details: `eval/E5_RESULTS.md`.

## Training details

The code is in [the GitHub repository](https://github.com/starryark/bangdream-tts). Training records (data counts, mix, reports) are in `training_data/`; per-clip transcripts are deliberately not published.

**Data**
- **Japanese:** 9,804 voice lines (22.1 h) of the 49 voices, re-cut and levelled per voice; transcripts checked with Qwen3-ASR; held-out test lines and validation are split off; 9,268 lines are in training. Poppin'Party to MyGO!!!!! come from the game *Girls Band Party*; Ave Mujica's (3–13 min per voice) mostly from the anime.
- **Native replay:** about 13% of training rows are native Mandarin (AISHELL-3) and English (LibriTTS-R) clips from 55 auxiliary speakers, so the model keeps native articulation in those languages.
- **Pseudo-Mandarin (in-voice):** gated Base-model renders in each voice's locked speaker vector (accent, text and identity gates), capped per voice.

**Speaker vectors**
- Each voice has a fixed vector: the mean Qwen3-TTS speaker-encoder embedding of up to 64 screened reference clips.
- It is written into codec-embedding rows 3000–3048 and never trained.

**Optimisation**
- Full fine-tune, fp32 master weights, bf16 autocast; AdamW, lr 5e-6, weight decay 0.01, cosine over a 3-epoch schedule with checkpoints at epochs 1, 2 and 3; this is the epoch-2 checkpoint (picked by the pre-registered rule); effective batch 32.
- Language id on every row, 25% of rows on the Auto prefix.
- **Trainer:** the fixed multi-speaker trainer from the [Kurisu project](https://github.com/starryark/makise-kurisu-tts) (corrects the official `sft_12hz.py`'s label shift, sub-talker frame and `text_projection` bugs).

**Compute:** one H100 80GB. The whole E1 study (smoke test, two arms × three epochs, floor and real-clip references, evaluation) took about 5 h, about $20; this model's own training took about 10 minutes. The follow-ups (E2, E5) are documented in the code repository.

## Limitations

- **Mandarin in a generic timbre for 15 voices:** `ppp_kasumi`, `ppp_rimi`, `ro_ako_maou`, `ro_rinko`, `ras_lock`, `mygo_tomori`, `mygo_soyo_tsuki` (generic), and `pp_eve`, `hhw_kanon`, `ras_chuchu`, `mujica_sakiko_tsuki`, `hhw_hagumi`, `ras_masking`, `pp_aya`, `hhw_kaoru` (partly). Their Mandarin is native-accented but may not sound like the character.
- **English is intelligible but mildly Japanese-accented** for most voices (en lid −9.3; the untrained floor is −12.2), and clearly accented for the generic group. There were no English in-voice rows.
- **Persona pairs are close.** `mujica_sakiko` / `mujica_sakiko_tsuki`, `mygo_soyo` / `mygo_soyo_tsuki` and `ro_ako` / `ro_ako_maou` sometimes swap (Japanese identity 0.83–0.88 for the weaker side). `ppp_kasumi` has the lowest Japanese identity among the game voices (0.84).
- **Ave Mujica voices have little data** (3–13 min, mostly anime audio), so they are less stable than the game voices.
- **Evaluation is automatic** (speaker probe, ASR, accent metrics). Listening checks were used for intermediate data, not for a full blind test of this checkpoint.

## Rights and responsible use

This is an unofficial fan project, with no affiliation to Bushiroad, Craft Egg, the BanG Dream! project, or the voice cast.
- The characters, the source audio and all related marks belong to their respective owners.
- **No copyright, performer, publicity or voice rights are granted** by this repository.
- No game audio, dataset, transcripts or audio samples are distributed here; only model weights and evaluation numbers.
- The official BanG Dream! fan-work guideline (<https://bang-dream.com/bdp-guideline/>) restricts derivative works that use the franchise's audio. Read it before using outputs anywhere public.

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
