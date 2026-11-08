# BanG Dream voice pack: data facts (measured 2026-10-29)

Source: `E:\BaiduNetdiskDownload\bangdreamai\角色语音\<band>\<character>.zip` (**50 zips** = 49 distinct sets after dropping the 梦中情祥 duplicate = **45 characters** once persona sets are merged; the bundled GPT-SoVITS pack has 52 models because it also splits styles such as ykn神人版 and 有咲_吐槽特化; `Poppin\ksm\` is an unzipped copy of 戶山香澄.zip, 258 files). Scripts: `scripts/{text_audit,audio_audit,floor_check}.py`. All numbers below are [L] (measured locally).

## 1. Size

- 9,919 files, **9,804 unique audio files** (byte-identical duplicates removed), **21.6 h** before dedup.
- Transcripts: the filename stem *is* the transcript (385k normalised characters). No `.list` / `.lab` files.
- Languages: Japanese. 珠手知由 (Chu²) has 59 lines with Latin-script English words (her character code-switches); elsewhere Latin appears only in loanwords (SNS, MV, VIP).

## 2. Two very different sources

| | Game sets (42 zips) | Anime-style sets (Mujica, 8 zips) |
|---|---|---|
| Format | mp3, 32 kHz, 52–82 kbps | WAV 44.1 kHz PCM (豐川祥子 is mixed: 60 mp3 32 kHz + 57 wav) |
| Clip length | median 7–9 s; most sets are **length-filtered to ~7–10 s** (e.g. 上原緋瑪麗 min 8.5 / max 9.5 s). Exceptions: 要樂奈 (median 2.5 s), 宇田川亞子 (3.2 s) | median 2.3–3.2 s, 0.6–14 s |
| Effective bandwidth | **15.3 kHz** (mp3 low-pass) | **15.3–15.8 kHz**: the WAVs were decoded from a lossy source too |
| Level | peak ≈ −3 dBFS, active RMS ≈ −20 dB (normalised) | active RMS ≈ −28 to −29 dB (≈ 8 dB quieter) |
| Edge silence | lead 0.02 s, trail ~0.06 s (tightly trimmed) | lead 0.06–0.19 s, trail ~0.08 s |
| Inter-word floor | mostly digital silence (−100 dB) | mostly −80 to −107 dB: looks already separated or gated |

**Implications** [I]:
- The length filter means the pack has almost no short interjections (except 要樂奈 and 宇田川亞子) and nothing over ~10–15 s. Short-utterance and long-utterance behaviour must come from the base model or replay data.
- A 15.3 kHz bandwidth is **not** a ceiling for a 24 kHz-output model (Qwen3-TTS: Nyquist 12 kHz). It *is* a ceiling for 44.1/48 kHz-output models (Irodori, GPT-SoVITS v4, VoxCPM2 at 48 kHz), which would learn the missing top octave as silence.
- Loudness must be normalised to one target across both sources, or the Mujica speakers will be learned as "quiet".
- Tight edge trimming (20 ms lead) is shorter than the 100–200 ms lead the Kurisu report recommends; pad with silence at build time.

## 3. Minutes per set and buckets (unique audio)

| Bucket | Sets (minutes) |
|---|---|
| **< 10 min (10 sets; 9 after dropping 梦中情祥)** | 若葉睦 2.7 · 八幡海鈴 3.1 · 梦中情祥 3.1 (*duplicate, see §4*) · 墨提斯 5.0 · 大魔姬亞子 5.2 · 三角初華 5.5 · 豐川祥子 6.1 · 祐天寺若麥 6.5 (two modes) · 宇田川亞子 7.9 · 長崎素世 9.5 |
| **10–20 min (5)** | 千早愛音 10.8 · 豐川祥子（月ノ森限定） 12.5 · 要樂奈 12.6 · 長崎素世（月ノ森限定） 16.0 · 湊友希那 18.4 |
| **20–30 min (16)** | 牛込里美 23.8, 二葉筑紫 24.2, 白鷺千聖 24.9, 美竹蘭 25.5, 八潮瑠唯 25.6, 椎名立希 26.4, 高松燈 29.1, 弦卷心 29.2, 羽澤鶇 29.4, 瀨田薰 29.5, 宇田川巴 29.8, 倉田真白 22.3, 桐谷透子 21.4, 佐藤益木 22.0, 鳰原令王那 21.7, 北澤育美 21.8 |
| **30–60 min (19)** | 花園多惠 30.1, 冰川紗夜 30.7, 和奏瑞依 32.4, 冰川日菜 32.7, 朝日六花 33.2, 廣町七深 33.6, 戶山香澄 34.5, 松原花音 35.4, 奧澤美咲 35.8, 今井莉莎 38.5, 白金燐子 40.4, 珠手知由 41.2, 上原緋瑪麗 43.6, 大和麻彌 44.1, 若宮伊芙 44.7, 山吹沙綾 46.2, 丸山彩 50.4, 青葉摩卡 57.8, 市ヶ谷有咲 59.2 |

Every Ave Mujica member is in the < 10 min bucket, and they are also the anime-sourced, 1–3 s-clip sets: the hardest speakers have the least and the shortest data.

## 4. Variants, duplicates and merge decisions

| Item | Finding | Decision needed |
|---|---|---|
| 梦中情祥 (69 clips) | **All 69 are byte-identical to clips in 豐川祥子（月ノ森限定）** | Drop as a dataset; at most keep it as a style label on those 69 rows |
| 豐川祥子 vs 豐川祥子（月ノ森限定） | 4 shared lines. GPT-SoVITS pack calls them 黒 / 白 (Ave Mujica vs Tsukinomori delivery) | One character with two styles, or two speaker ids |
| 祐天寺若麥 `nyamu0/` (74) vs `nyamu1/` (63) | Two delivery modes (pack: 乖猫 / 哈气); 4 rows start with `(A)`; 13 byte-identical dup groups | Strip `(A)`; one speaker with a style label, or two ids |
| 若葉睦 (61) vs 墨提斯 (95) | Same character, alternate persona (Mortis) | Same: style vs id |
| 宇田川亞子 (104) vs 大魔姬亞子 (44) | Persona ("great demon princess") | Same |
| 長崎素世 vs （月ノ森限定） | Same character, sweeter delivery (pack: 夹) | Same |
| 要樂奈 | 19 byte-identical dup groups (`…` and `…。` variants of one take) | Dedup on audio hash |
| Others | 1–4 dup groups each in 弦卷心, 豐川祥子, 宇田川亞子, 二葉筑紫, 廣町七深, 千早愛音, 高松燈, 冰川日菜, 大和麻彌 | Dedup on audio hash |

After merging personas into characters, the smallest characters are: 八幡海鈴 3.1, 三角初華 5.5, 祐天寺若麥 6.5, 若葉睦+墨提斯 7.7, 宇田川亞子+大魔姬 13.1, 豐川祥子 (both) 18.6 min.

## 5. Transcript quality flags

- Filenames up to 247 UTF-8 bytes; no sign of Windows truncation, but this is unproven until ASR comparison [U].
- Full-width digits (`５人`, `３０分後`, `２対１`) in ~600 rows need reading normalisation (ご にん / さんじゅっぷん / に たい いち).
- `(A)` prefixes (4 rows), `『』`, `――`, `〜`, `♪`, and `…` / `……` are frequent; decide a normalisation policy once for all speakers.
- Duplicate-text-different-take pairs exist (e.g. one 要樂奈 line recorded twice, its text differing only by a final `。`).

## 6. Sets to check by ear or with a quality model before training

Low inter-word floor on many clips (5th-percentile frame RMS above −50 dB re peak, low spectral flatness), which means either continuous speech with no pauses or a bed of reverb/BGM [U]:

| Set | Clips with floor > −50 dB |
|---|---|
| 大魔姬亞子 | 45% |
| 珠手知由 | 44% |
| 宇田川亞子 | 27% |
| 桐谷透子 | 20% |
| 市ヶ谷有咲 | 16% |
| 千早愛音, 鳰原令王那 | 11% |

大魔姬亞子 is a theatrical persona and may carry an effect; Chu² lines may have BGM. Gate these with DNSMOS/UTMOS and spot-listen before training.

## 7. Speaker separability (local GPU run, 40 random clips ≥ 1 s per set; 梦中情祥 excluded)

Script: `scripts/spk_sep.py`; raw matrix in `scripts/spk_sep.json`. The Qwen vectors use exactly the recipe in `kurisu_v2/pod/train_sft.py::speaker_embedding` (24 kHz mel, mean of clips).

| Encoder | Centroid cosine between sets (median / max) | Clip → nearest-centroid accuracy (49 sets) | Worst sets (accuracy → confused with) |
|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base speaker encoder** | **0.984 / 0.997** | **93.9%** | 墨提斯 56% → 三角初華; 豐川祥子 62% → 三角初華; 豐川祥子（月ノ森限定） 62% → 豐川祥子; 鳰原令王那 85% → 朝日六花; 宇田川亞子 88% → 北澤育美; 丸山彩 88% → 市ヶ谷有咲 |
| WavLM-base-plus-sv | 0.939 / 0.993 | 55.6% | 豐川祥子（月ノ森限定） 15%; 墨提斯 18%; 要樂奈 22%; 上原緋瑪麗 22%; 三角初華 25% |

**Mean-centred check** (second run, up to 80 clips ≥ 1 s per set, global mean of all clip vectors subtracted; script `scripts/spk_centered.py`, vectors in `scripts/spk_vecs.npz`):

| Measure | Value |
|---|---|
| Centred centroid cosine between sets | median **−0.05**, p95 0.57, max 0.86 |
| Same-set split-half cosine (ceiling) | median **0.97**, min 0.92 (丸山彩) |
| Centred clip → nearest-centroid accuracy | 93.6% |
| Closest pairs | 豐川祥子 ~ 月ノ森 0.857; 若宮伊芙 ~ 朝日六花 0.818; 三角初華 ~ 八幡海鈴 0.814; 上原緋瑪麗 ~ 戶山香澄 0.760; 三角初華 ~ 祐天寺若麥 0.754 |

| Persona / related pair | Centred cosine | Verdict [I] |
|---|---|---|
| 豐川祥子 ~ 豐川祥子（月ノ森限定） | 0.857 (ceiling 0.97) | Closest pair in the pack; separate rows are possible but at leakage risk |
| 祐天寺若麥 nyamu0 ~ nyamu1 | 0.941 (split-half 0.98 / 0.962) | Indistinguishable to the encoder; merge into one row (`scripts/nyamu_check.py`) |
| 長崎素世 ~ 長崎素世（月ノ森限定） | 0.593 | Separable |
| 若葉睦 ~ 墨提斯 | 0.237 | Acoustically distinct persona; separate rows |
| 宇田川亞子 ~ 大魔姬亞子 | 0.197 | Distinct; separate rows |
| 冰川紗夜 ~ 冰川日菜 (twins, different performers) | −0.541 | Clearly distinct |

What this means [I]:
- **The raw cosines of ~0.98 are a shared offset, not missing information.** After centring, different characters sit near 0 while same-set halves sit at 0.97. The Base's zero-shot cloning already works with this geometry (the Kurisu log shows 0.93–0.95 raw between unrelated women), so locked raw vectors are not by themselves a blending risk. The risk is concentrated in the handful of pairs above ~0.75 centred.
- **The persona variants mostly separate**, except Sakiko's two sets. Mortis and the Ako persona are further from their base character than many unrelated characters are from each other, so they can be separate speaker rows.
- **The anime-sourced Mujica sets cluster together** (三角初華 ~ 八幡海鈴 0.814, ~ 祐天寺若麥 0.754): the shared anime channel and mix are part of what the encoder hears. Clean them (separation, loudness) before building their vectors, and expect leakage among them first.
- **WavLM-base-plus-sv cannot tell these characters apart** (56% clip accuracy), so it must not be the leakage metric. Candidates: litagin `anime_speaker_embedding` (`char` variant, see `vc_and_data_2026-10-29.md` §8), or a linear probe on centred Qwen speaker-encoder vectors (≈94% clip accuracy on real clips, so the probe's real-clip confusion matrix is the reference to compare generated clips against).
