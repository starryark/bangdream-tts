# BanG Dream! Our Notes extraction: data facts (measured 2026-11-06 to 07)

The recommendation built on these facts is `recommendation_2026-11-07.md`; the draft pre-registered experiment is `../e6/E6_PLAN.md`.

Source: `E:\BangDream_ournotes\extracted\` (EN server, resource version 1.1.0.301, README in that folder). Scripts and raw output: the session scratchpad listed in §11. All numbers are [L] unless tagged.

Tags: **[L]** measured locally, **[I]** inference, **[U]** unverified.

## 1. What this is

A **different game** from the pack behind v1 (Girls Band Party + anime). It is not more of the same data.

| | v1 pack (`bangdreamai`) | Our Notes |
|---|---|---|
| Characters | 45 (49 rows) | 25 main + Kanata (id 1001) + side roles |
| Overlap with v1 rows | | **10 characters**: MyGO 5, Ave Mujica 5 (13 v1 rows counting `soyo_tsuki`, `sakiko_tsuki`, `mortis`) |
| New characters | | **15 + Kanata** |
| No new data | | **36 v1 rows**: PPP, Afterglow, Pastel\*Palettes, Roselia, HHW, Morfonica, RAS |
| Voice audio | 21.6 h | 29.3 h, 30,021 distinct files |
| Format | 32 kHz mp3 low-passed at 15.3 kHz (game); 44.1 kHz WAV decoded from lossy, also ~15.4 kHz (Mujica) | **48 kHz mono FLAC decoded from HCA, no low-pass** |
| Clip length | mostly length-filtered to 7–10 s (game); 1–3 s (Mujica) | natural: median 3.2 s, 14% < 1 s, 33% 1–3 s, 49% 3–8 s, 4% 8–15 s |
| Transcript | filename stem | script text, five languages |
| Labels | none | speaker, story arc, **expression/motion cue on most lines** |

## 2. Minutes per character (distinct single-speaker files)

From `voice_index.csv`, all sources. "v1" is the unique minutes in the old pack.

| Band | Character | CV (game data) | Our Notes min | Clips | v1 min |
|---|---|---|---:|---:|---:|
| MyGO!!!!! | 高松燈 Tomori | 羊宮妃那 | 47.5 | 1,005 | 29.1 |
| | 千早愛音 Anon | 立石凛 | 82.9 | 1,272 | 10.8 |
| | 要楽奈 Rāna | 青木陽菜 | **10.3** | 450 | 12.6 |
| | 長崎そよ Soyo | 小日向美香 | 59.8 | 1,047 | 9.5 + 16.0 (月ノ森) |
| | 椎名立希 Taki | 林鼓子 | 50.2 | 1,025 | 26.4 |
| Ave Mujica | 三角初華 Uika | 佐々木李子 | 42.9 (+ 初音 20.3) | 742 | 5.5 |
| | 若葉睦 Mutsumi / Mortis | 渡瀬結月 | 44.0 (睦 22.4, モーティス 19.6) | 955 | 2.7 / 5.0 |
| | 八幡海鈴 Umiri | 岡田夢以 | 39.9 | 700 | 3.1 |
| | 祐天寺にゃむ Nyamu | 米澤茜 | 54.0 | 875 | 6.5 |
| | 豊川祥子 Sakiko | 高尾奏音 | 87.6 | 1,512 | 6.1 + 12.5 (月ノ森) |
| Band 3 (`yumemita`) | 仲町あられ Arale | *empty in data* | 89.6 | 1,683 | — |
| | 宮永ののか Nonoka | *empty* | 50.6 | 1,004 | — |
| | 峰月律 Ritsu | *empty* | 40.1 | 977 | — |
| | 藤都子 Miyako | *empty* | 54.3 | 1,092 | — |
| | 千石ユノ Yuno | *empty* | 42.7 | 1,028 | — |
| Band 4 | 汐見蛍 Hotaru | 薬師寺李有 | 74.4 | 1,368 | — |
| | 伊沢なつめ Natsume | 千春 | 45.3 | 755 | — |
| | 琴平凪 Nagi | 結川あさき | 48.7 | 981 | — |
| | 浜崎まほろ Mahoro | 伊駒ゆりえ | 78.6 | 1,203 | — |
| | 和泉朋花 Houka | 咲川ひなの | 51.1 | 718 | — |
| Band 5 | 須賀蕾叶 Raika | 橘めい | 103.0 | 1,454 | — |
| | 馬橋心玖 Miku | 涼泉桜花 | 86.1 | 1,267 | — |
| | 矢倉蓬咲 Yomogi | 花宮初奈 | 71.7 | 874 | — |
| | 梅里ちえり Chieri | 菱川花菜 | 70.0 | 880 | — |
| | 四宮寧月 Shizuku | 遠野ひかる | 58.7 | 795 | — |
| (id 1001) | 沢海奏多 Kanata | *empty* | 19.4 | 270 | — |

Side roles with speech (Story, `character_source = none`): 初音 Hatsune 20.3, ビオラ Viola 19.4, マネージャー 13.7, 召使 11.1, 蕾叶の祖母 6.3, 鈴木 5.5. Identity of these voices is checked in §7.

## 3. Audio [L]

Sample: 400 story clips and up to 150 per other group.

| Measure | Our Notes `adv_voice` | v1 game mp3 | v1 Mujica WAV |
|---|---|---|---|
| Sample rate | 48 kHz mono | 32 kHz | 44.1 kHz |
| Top of spectrum (−70 dB from speech max) | 24 kHz (no sharp edge) | 15.3 kHz | container 22 kHz, edge 15.4 kHz |
| Energy 12–16 kHz (dB re total) | −27.9 | −31 to −38 | −25 to −29 |
| Active RMS | −18.7 dB | −17 to −18 | −24 to −25 |
| Peak | −0.9 dB | | |
| Lead silence | **0.00 s** | 0.02–0.03 s | 0.05–0.14 s |
| Trail | 0.11 s | ~0.06 s | ~0.08 s |
| Floor (p5 frames re peak) | −118 dB, p95 −60 dB | −70 to −116 | −88 to −109 |

- The 16–20 kHz band is populated (−41.6 dB), so there is no MP3-style cut-off. The `VoiceLocation` files are the exception, with a real edge at 14.9 kHz and a 15 dB drop; they are also multi-speaker and excluded.
- At the 24 kHz output of Qwen3-TTS (Nyquist 12 kHz), both sources are full-band, so mixing them is not a bandwidth mismatch [I]. The extra bandwidth pays only with a 44.1/48 kHz model or vocoder, or for VC converters.
- Inter-word floors are near digital silence: the audio is dry and gated, with no BGM in the voice sheets.

## 4. Sources

| Cue-sheet group | Clips | Hours | Median s | Use |
|---|---:|---:|---:|---|
| `adv_voice` (story) | 26,844 | 26.57 | 3.18 | main |
| `VoiceSystem` | ~2,200 | ~1 | | single speaker, stated label |
| `VoiceLive` | 800 | 0.47 | 2.03 | short shouts; stated label |
| `spot` | 628 | 0.46 | 2.38 | home conversation; rows are multi-speaker groups, needs per-cue check |
| `VoiceGacha`, `VoiceRealLive`, `InitialVoice` | 118 | 0.06 | | short, stated label |
| `VoiceLocation` | 6 | 0.03 | 18 | multi-speaker, low-passed: exclude |

Songs (95, full mixes) are out of scope.

## 5. Script text: policies this corpus forces [L]

Over the 28,053 story rows:

| Feature | Rows | Minutes | Note |
|---|---:|---:|---|
| Newline inside the line | 9,807 | 964 | One third of the data. Needs one policy |
| Monologue `（…）` | 1,966 | 140 | Strip brackets, keep audio. Check for reverb |
| **Ellipsis-only** (`…………`) | 1,898 | 31 | Audio is a short reaction: median 0.84 s; ASR hears 1–2 kana (`うん`, `はい`, `はは`) in 139 of 143 sampled (§9.6). `……` is Qwen's native pause token (id 14053), so training on it as text would corrupt pausing |
| Nonverbal-only (`……！`, `？`, `……っ`) | 870 | | Gasps and reactions; v1 reads `……っ` as `うん` (§9.5). With the row above, ~2,770 lines of nonverbal audio: material for an `[nv]` tag, not plain text |
| 1–3 speakable characters (`え`, `うん`, `はい`, `はぁ`) | 3,129 | | Short lines v1 never trained on (§9.5) |
| Full-width digits | 584 | 52 | Reading normalisation (as v1) |
| Latin | 951 | 67 | CRYCHIC etc. Readings table needs new terms |
| `<size=…>`, `<ruby=…>` | 383 | 18 | Strip tags. Ruby may supply readings |
| Multi-speaker labels (`燈 & そよ`) | 338 | 7.4 | Exclude |
| No voice cue | ~6,200 of 34,227 | | Not relevant |

Also: new-cast names (蕾叶, 寧月, 蓬咲, 心玖, 朋花, 凪, 蛍 …) need entries in `BD_READINGS`, since OpenJTalk misreads names.

## 6. Labels [L]

**Speaker.** `TargetName` on each `Talk` gives a role-level id that is finer than `characters`: `mutsumi` / `mortis` / `mutsumi_maybe` / `mutsumi_role2`, `sakiko` / `oblivionis` / `sakiko_role`, `uika` / `doloris` / `hatsune`, `nagi` / `nagi_child`, `hotaru_child`, `kanata_child`.

**Expression.** Story scripts carry a `Motion` or `Expression` command per character, with `ExpressionName` (Live2D face state) and `MotionName` (gesture).
- 28,038 voiced lines. 25,844 (92%) have a current expression; 22,570 (80%) have one set since the previous line.
- 110 expression names, about 20 meaningful classes.
- Counts by class: smile 6,098; idle 5,143; sad 2,162; serious 2,005; surprised 1,763; angry 1,466; kime 1,418; bsmile 1,169; question 709; cry 678; thinking 583; panic 381; pale 319; shy 314; shadow 274; puzzled 245; kirakira 237.
- These are art-direction labels for a face, so they are a proxy for the voice [I]. Whether they predict audio is measured in §7.

**Translation.** Every voiced story line has official English, Korean, Traditional and Simplified Chinese text.

## 7. Speaker and label checks [L]

Qwen speaker-encoder vectors for all 29,445 clips (5 rejected as no audible speech), v1 audio policy, scripts `s05_vectors.py`, `s06_analyze.py`, `s13_channel.py`. Same-source split-half ceiling: 0.98–0.999.

### 7.1 The two games differ by a channel offset as large as the gap between speakers

- Distance between source means: Our Notes–GBP **2.38**, Our Notes–anime 1.56, anime–GBP 2.32. Between-speaker spread inside a source: about 2.0–2.1.
- **Centring on the v1 mean is wrong for this corpus.** It leaves the offset in, which makes new characters look falsely close to each other (Raika~Chieri 0.88 → 0.74 after correct centring) and makes same-actress pairs look unrelated (Anon 0.46, Soyo 0.30). Any v2 probe or pair list must account for this.

**Same voice actor across games, per-source centring** (different characters: median −0.04, p95 0.38, max 0.74):

| Old row | New role | Cos | Rank among new roles | Clips: old→new probe | Clips: new→old probe |
|---|---|---:|---:|---:|---:|
| `mygo_tomori` | tomori | 0.82 | 1st | 0.84 | 1.00 |
| `mygo_anon` | anon | 0.68 | 1st | 0.94 | 0.95 |
| `mygo_raana` | rana | 0.84 | 1st | 0.98 | 0.99 |
| `mygo_soyo` | soyo | 0.57 | 1st | 0.85 (either Soyo row) | 1.00 |
| `mygo_soyo_tsuki` | soyo | 0.70 | 1st | | 1.00 |
| `mygo_taki` | taki | 0.81 | 1st | 0.93 | 0.99 |
| `mujica_uika` | uika | 0.48 | 1st | 0.89 | 0.92 |
| `mujica_mutsumi` | mutsumi | 0.81 | 1st | 0.88 | 0.97 |
| `mujica_mortis` | mortis | 0.60 | **5th** | 0.69 | 0.82 |
| `mujica_umiri` | umiri | 0.70 | 1st | 0.99 | 0.99 |
| `mujica_nyamu` | nyamu | 0.70 | 1st | 0.89 | 0.98 |
| `mujica_sakiko` | sakiko | 0.59 | 1st | 0.95 (either Sakiko row) | 0.71 |
| `mujica_sakiko_tsuki` | sakiko | 0.58 | 1st | | 0.81 |

**Reading.** It is the same voice (right character for 12 of 13), but mean 0.68 against a 0.99 ceiling, so game and session move the vector a long way. **Do not merge old and new clips into one row:** a locked vector would be an average of two channels [I]. Replace old data where the new corpus covers the voice.

### 7.2 Personas and aliases (Our Notes, new-corpus centring)

| Pair | Cos | Decision [I] |
|---|---:|---|
| mutsumi ~ mortis | **0.26** (old pack −0.32) | Two rows. Clip probe 0.88 / 0.81, mostly confused with each other |
| uika ~ 初音 hatsune (19 min) | 0.52 | Same voice actor, distinct delivery. Do not fold into `uika`; candidate extra row |
| uika ~ doloris (1.5 min) | 0.51 | Stage persona; drop |
| sakiko ~ oblivionis (3.2 min) | 0.51 | Stage persona; drop |
| sakiko ~ 召使 `sakiko_role` (11 min) | **0.23**, nearest houka 0.36 | A role, possibly another voice; drop |
| mutsumi ~ `mutsumi_maybe` 睦？ (2.5 min) | 0.10; ~ mortis 0.49 | Ambiguous by design; drop |
| nyamu ~ amoris (1.6 min) / umiri ~ timoris (0.8 min) | 0.77 / 0.89 | Fold in or drop; negligible |
| hotaru ~ ビオラ viola (18 min) | 0.64; ~ arale 0.58 | Unknown voice; needs ears. Exclude until checked |
| `*_child` | nagi −0.08, miku −0.08, others 0.60–0.68 | Exclude; some are other voice actors |
| manager (13.5 min) | ≤ 0.35 to all | Different person; exclude |

**Story arc is a real persona signal for Sakiko.** Inside Our Notes, Sakiko's MyGO-arc and Mujica-arc centroids are at 0.69, as far apart as the old 白/黒 pair (0.70) and well below other characters' arcs (anon 0.94, taki 0.88, tomori 0.84). MyGO-arc Sakiko sits closer to the old `sakiko_tsuki` (0.68) than to `mujica_sakiko` (0.54). The old `mujica_sakiko_tsuki` is itself anime-sourced (v1 manifest: 203 clips, all `anime`, median 3.24 s), while `mygo_soyo_tsuki` is GBP game audio (126 clips, median 7.3 s). Soyo's arcs are 0.81 with no clean split, though MyGO-arc Soyo is much closer to old `soyo_tsuki` (0.66) than to `mygo_soyo` (0.35).

### 7.3 Pairs and label quality

- Closest pairs (per-source centring): raika~chieri 0.74, arale~nonoka 0.70, nonoka~raika 0.70, arale~yomogi 0.67, yuno~nagi 0.64; across sets arale~`ppp_kasumi` **0.70**, arale~`hhw_hagumi` 0.68, yuno~`ag_ran` 0.68, nagi~`ag_ran` 0.66, chieri~`ag_himari` 0.66. v1 handled eve~lock at 0.83, so these are watch items, not blockers.
- **Speaker labels are reliable.** A 27-role probe, 5-fold by episode: 95.9% overall; 89.7% at 1–1.5 s rising to 97.7% at 4–7 s. Outside the Mutsumi/Mortis pair, every role is 93–99%. A probe-disagreement gate would remove roughly 2–4% of clips [I].
- **Monologues are dry.** Floor is lower than dialogue (p25 −62.8 vs −48.5 dB), so no added reverb. They are breathier (8–12 kHz +2.8 dB) and further from the centroid (0.71 vs 0.76). Keep for training; exclude from reference lists.

### 7.4 Are expression labels audible? Yes

Within-character centred x-vectors, 14,975 clips, 13 classes, group k-fold by episode:
- **Balanced accuracy 0.315**, against chance 0.077 and shuffled 0.081: **4× chance**, from an encoder built to ignore such things.
- Pairs: panic/idle 0.86, cry/smile 0.83, surprised/idle 0.81, smile/sad 0.80, angry/sad 0.75, serious/smile 0.76, shy/angry 0.75.
- Prosody moves in the expected directions (within-character z-scores): F0 panic +1.04, cry +0.77, shy +0.74, surprised +0.41, thinking −0.62, idle −0.32; level sad −0.45, thinking −0.39, panic +0.39; rate cry −0.41, sad −0.34, surprised −0.34, smile +0.15.
- Prosody features alone are weak (0.111 vs 0.077), so the signal is mostly in spectral and voice quality [I].
- Expression also moves the x-vector off the character centroid: kime 0.78, smile 0.78, idle 0.77, sad 0.74, angry 0.71, cry 0.67, shadow 0.62, pale 0.62. Reference lists should prefer neutral and smile classes [I].
- **Why this matters:** Kurisu v3 found a tag needs about 150 verified examples to act as a control (`kurisu_v2/research/v3_results.md`). This corpus has smile 6,098, sad 2,162, angry 1,466, surprised 1,763, cry 678, panic 381, shy 314. `train_sft.py --new_tokens` already supports atomic tags.

### 7.5 Usable pool

Not ellipsis-only, ≥ 0.6 s, single speaker, role-labelled: **23,238 clips, 24.7 h** over 27 rows. Largest arale 1,498 clips; smallest mortis 260, kanata 260, rana 332. **Rāna is the only thin voice: 8.9 min, median 1.15 s, only 89 clips ≥ 2 s.**

## 8. What the v1 evals say more data can and cannot do [L]

Existing scores only (`e1/results/eval/E1b_ep2`, `e5/results/eval/E5_ep2`, `e0/floor_codecprobe.jsonl`, script `s12_dose.py`), 49 voices.

| Across voices, E1b epoch 2 | Spearman ρ | p |
|---|---:|---:|
| Japanese minutes vs ja leak | +0.03 | 0.83 |
| Japanese minutes vs zh leak | +0.08 | 0.58 |
| Japanese minutes vs zh SER | +0.12 | 0.42 |
| Japanese minutes vs en lid | 0.00 | 1.00 |
| Japanese minutes vs zh lid | +0.27 (more data, slightly *worse*) | 0.06 |
| **Base floor zh leak vs zh leak** | **+0.68** | < 0.001 |
| **In-voice zh rows vs zh leak** | **−0.66** | < 0.001 |

| Group | zh leak | zh lid | en lid | ja p_true |
|---|---:|---:|---:|---:|
| < 15 min (12 voices) | 0.117 | −20.19 | −9.15 | 0.780 |
| ≥ 30 min (22 voices) | 0.163 | −18.01 | −9.66 | 0.803 |

**Reading.**
- On every metric we have, **Japanese minutes buy nothing** in the 3–60 min range. A 3-minute anime-sourced voice (`mujica_umiri`: ja 0.97, zh 1.00) matches a 60-minute game voice.
- These metrics are near saturation for Japanese, so this does not prove more data is useless. It means *what extra data buys is invisible to the current metrics* (expressive range, short lines, stability). v2 needs a measure that can see it [I].
- Mandarin identity is decided by how well the Base clones the vector and whether in-voice Mandarin exists, not by Japanese data.
- Among E5's 15 targets, the number of VC rows added does not predict the improvement (ρ = +0.15, p = 0.60): `ppp_rimi` +2 rows 0.75 → 0.34, `hhw_kanon` +6 rows 0.34 → 0.06, `ppp_kasumi` +15 rows 0.75 → 0.69. This corrects E5's reading that Rimi's gap is a row-count problem [I].
- **Kasumi is partly a nearest-neighbour case.** The E5 Mandarin renders of `ppp_kasumi` go to `ag_himari` (10 of 32), the row's closest pair (centred 0.76), and the row's real-audio confidence is low (Japanese p_true 0.53 against about 0.85 for its peers). Whether listeners hear Himari is untested.

## 9. Pre-checks on the new audio (local RTX 5060 Ti, `s07_precheck.py`, `s14_fix.py`) [L]

Sample: 120 clips per role over the 27 roles, plus 200 ellipsis-only clips, v1 audio policy (24 kHz). Total run 71 min.

### 9.1 Script text matches the audio

Qwen3-ASR (one system; v1 took the better of two, so these are lower bounds) against the cleaned script, scored with v1's own `cer_bar`:

| Clips | n | Mean CER | Pass v1 bar |
|---|---:|---:|---:|
| All non-ellipsis | 3,240 | 0.085 (median 0.000) | **94.4%** |
| 1–1.5 s | 315 | 0.176 | 88.3% |
| 1.5–3 s | 946 | 0.107 | 92.0% |
| 3–6 s | 1,396 | 0.067 | 95.8% |
| 6–30 s | 583 | 0.046 | 98.1% |
| Monologue / dialogue | | 0.039 / 0.089 | |

- Lowest pass rates: yomogi 0.84, arale 0.86, mortis 0.88, kanata 0.90; soyo 1.00.
- **Under 1 s** (`s16_checks.py`, 399 clips at 0.6–1.0 s): pass 0.68. For 1–2 kana lines the pass rate is 0.40 and means nothing: ASR maps `へぇ`→`え`, `ふふん`→`うん`, `……はぁ`→`はい`. At 3–4 kana it is 0.81, at 5+ kana 0.86. **A CER gate can't verify 1–2 kana lines**, so they need a different gate (probe agreement and duration), fixed in advance.
- **The failures are mostly expressive, not mislabelled.** The longest failures are crying, laughing, panting and singing (`ぐすっ……うわぁあ〜〜ん`, `フフフフ……`, `今はまだ～♪`), and names or coined words the ASR doesn't know (寧月 → 静く, millsage). A strict ASR gate would remove expressive lines more than neutral ones [I]. Keep v1's gate for consistency, but report per expression class what it removes.

### 9.2 Speaker probe with the new rows

A 63-class codec-calibrated probe (27 Our Notes roles + the 36 old rows that get no new data), trained on tokenizer-resynthesised clips ≥ 1.5 s, cv5 accuracy **0.973**.
- New roles 0.88–0.99, except **mortis 0.74** (16 of its errors go to mutsumi, the same voice actor). Mahoro 0.88, mutsumi 0.90.
- No old row drops below 0.92, so the 36 unchanged rows stay separable with the new roles present.
- Use: the v2 evaluation probe works with 63+ classes. Mutsumi/Mortis needs pair-aware scoring (count a Mortis→Mutsumi confusion separately from a leak to another character).
- **Caveat:** probe63 centres on one global mean over two channels (§7.1). Logistic regression can use the channel offset as a shortcut, which inflates separability between the old and new sets. A render that "stays on its own class" partly means "stays inside its set". E6 should report leak within the set and across sets separately. Re-rendering E1b under the same probe keeps the regression comparison fair either way.
- **Gate calibration** (5-fold, real resynthesised clips): under E1's 49-class probe, 91.0% of real clips have `p_true ≥ 0.5`. Under probe63, the bar that passes the same 91.0% is **0.39** (an argmax margin of 0.27 is equivalent). Median real `p_true` drops from 0.81 to 0.78, and new roles are less confident than old (0.74 vs 0.90 pass at 0.5) because the new set holds the close pairs.

### 9.3 Do E5's VC Mandarin rows sit on the right speaker?

The mean vector of each voice's RVC-converted Mandarin clips was compared with all 49 v1 locked rows. For **15 of 15 voices the nearest row is the voice's own**, at cos 0.44–0.75 (lowest `pp_aya` 0.44, `mujica_sakiko_tsuki` 0.48). No evidence for separate per-language rows; VC rows can share the voice's row.

### 9.4 Base zero-shot from the new locked vectors (the floor for each new row)

v1 recipe for locked vectors (≥ 2 s, not monologue, not ellipsis, drop the 20% least typical, cap 64 clips). Base model, 8 Japanese + 16 Mandarin (`probes_zh_v3` shared texts) per role, scored with the 63-class probe and the v1 accent scorer.

| | ja leak | zh leak | zh lid | zh SER | E1 gate pass (p ≥ 0.5 + accent + text) | Same, identity = argmax only |
|---|---:|---:|---:|---:|---:|---:|
| Pooled, 27 roles | 0.069 | **0.113** | −19.8 | 0.061 | 0.22 | 0.49 |
| v1 floor, 49 rows (49-class probe) | 0.100 | 0.157 | −19.5 | 0.077 | | |

The two rows are not directly comparable (27 roles / 63 classes vs 49 rows / 49 classes). The pooled 0.113 is 22 roles near zero plus 4 roles at 0.56–0.81.

- **Game-channel MyGO and Ave Mujica Mandarin renders stay on their own class:** zh leak 0.00 for all 11 overlap roles except mutsumi 0.25 (all to mortis, the same voice actor). For several the argmax is right at low confidence (p_true tomori 0.28, rana 0.30, mutsumi 0.36, sakiko 0.37), so this is "not lost", not "cloned well". The worry that game-channel vectors would clone worse than v1's anime vectors does not hold.
- **Four new roles fail at the floor, all into one attractor:** arale 0.69, nonoka 0.81, hotaru 0.56, yomogi 0.56 zh leak, mostly predicted as **chieri** (arale 11/16, nonoka 9/16). Yomogi and arale also leak in Japanese (0.88, 0.50). This matches the close pairs in §7.3 (arale~nonoka 0.70, raika~chieri 0.74, arale~yomogi 0.67).
- Identity confidence is low for several roles even when argmax is right (p_true 0.25–0.37: tomori, rana, sakiko, mahoro, houka, mutsumi). Under E1's `p_true ≥ 0.5` bar they yield almost no pseudo rows. With 63+ classes, probability is spread thinner than in E1's 49, so the bar needs re-deciding before E6 runs, not after [I].
- Accent at the floor is as good as v1's: SER 0.061, lid −19.8. The accent gate passes 0.57 of renders; nagi (lid −15.7), soyo (−16.1) and arale (−13.8) are weakest.
- **Full E1 gate per render, with the matched identity bar 0.39** (§9.2; accent `p_target ≥ 0.9`, `lid ≤ −19`; text CER ≤ 0.05). Pooled 0.22 at bar 0.5, 0.28 at 0.39:

| Pass | Roles |
|---|---|
| ≥ 0.4 | taki 0.75, uika 0.69, umiri 0.62, nyamu 0.62, natsume 0.56, ritsu 0.50, anon 0.44, mortis 0.44 |
| 0.2–0.4 | raika 0.38, miku 0.38, kanata 0.38, yuno 0.31, chieri 0.31, sakiko 0.25, nagi 0.25, shizuku 0.25 |
| < 0.2 (accent or identity) | miyako 0.19, tomori 0.06, mahoro 0.06, rana 0.00, soyo 0.00, houka 0.00, mutsumi 0.00 |
| 0 (identity fails) | arale, nonoka, hotaru, yomogi |

  Rana, soyo and houka fail on accent (lid −17.7, −16.1, −18.4), not identity. More samples per text raise their yield; E1 used 2.
- **Chieri is not a generic-Base attractor** (`s16_checks.py`, the advisor's check). Chieri's Mandarin renders are nearest Chieri's real centroid (0.66, next raika 0.50). The real centroids nearest Base's average output are nyamu, umiri and kanata, not chieri. **What the four failures share is that their real voices are the furthest from Base's average output** (cos to the grand render mean: arale −0.09, nonoka 0.00, yomogi 0.09, hotaru 0.12; others 0.15–0.42). Base can't reach them and lands on the nearest neighbour it can reach (chieri, raika). By render-to-real cosine, miku (nearest umiri), mortis (nearest nyamu) and tomori (nearly tied with taki) are borderline as well.

**Routing that follows [I]:**
- Base pseudo-Mandarin with the matched bar for the roles that pass.
- More samples per text where the accent gate is the bottleneck (rana, soyo, houka, tomori, mahoro, miyako).
- A non-Base source (VC) for arale, nonoka, hotaru and yomogi.
- Pair-aware gating for mutsumi/mortis.

### 9.5 How v1 handles Our Notes lines (the baseline v2 has to beat)

E1b_ep2 renders, Japanese, 12 short lines (0.7–1.6 s of real audio) and 12 mid lines (3–5.5 s) per row, for the 13 overlap rows and 3 controls (`ppp_kasumi`, `ag_ran`, `ro_yukina`: rows with no Our Notes data, scored against their own class). Real reference: the same real lines through the codec, same probe. n is small (156 overlap clips and 36 control clips per cell), so read it as direction, not precision.

| Leak (63-class probe) | Short lines | Mid lines |
|---|---:|---:|
| Real audio through the codec (overlap roles) | 0.13 | 0.06 |
| v1, overlap rows | **0.37** | 0.12 |
| v1, controls (no channel difference) | **0.44** | 0.17 |

- **Short lines are v1's measurable weakness.** On a 1 s line, v1's identity leak is about 3× real audio's. It shows on the controls too, which have no channel mismatch with their probe class, so it is not just the game-vs-anime offset. v1's game data was length-filtered to 7–10 s, so it never trained on short lines. Our Notes has 14% of clips under 1 s and 33% at 1–3 s.
- **Short renders are on time once silence is trimmed.** The raw ratio of 1.29 (render 1.36 s vs real 1.10 s) is padding: renders carry model-emitted lead and trail silence, and Our Notes clips have 0.00 s lead. Trimmed at −40 dB, the median ratio is 1.02 for short lines (10% over 1.5×) and 0.94 for mid lines. Duration metrics must be computed on trimmed audio.
- **Nonverbal lines are read as words:** `……っ`, `っ！！` and `……っ！` come out as `うん`, `はい`, `ふん` (CER 2.0). v1 has no way to make a gasp.
- **Anime-trained Ave Mujica rows have flatter intonation than the game delivery:** F0 range (10th–90th percentile, semitones) render/real umiri 5.6/9.2 and 6.1/10.6, mortis 9.0/12.9 and 7.8/14.6, sakiko 8.2/13.1, uika 8.0/10.3. Pooled, short-line range matches (12.0 vs 12.3).
- Mid-line Japanese is fine: CER 0.081 against 0.067 for real audio at 3–6 s; no runaways.
- The real-audio reference may include a few clips that were in probe63's training sample (about 12% chance per clip), which flatters the real row slightly. The direction stands.
- **This is the metric family v2 should be judged on:** short-line identity, nonverbal handling and F0 range, with duration on trimmed audio as a secondary check. §8 explains why the v1 metrics (mid-length lines) are saturated.

### 9.6 What the "ellipsis-only" audio is

The sampled 200 mixed two things:
- **Story lines with `……` as text** (143): median 0.84 s, and ASR hears 1–2 kana in 139 (`うん`, `はい`, `はは`). These are short reactions, hums and breaths. Dropping them as text is right; they are candidates for an `[nv]` tag.
- **Lines with no text in the index** (57: `spot` 46, `InitialVoice` 8, `VoiceSystem` 3): real sentences (`すべて混ぜて飲み干しますわ。…`, `バンドリ、アワーノーツ`). They could be ASR-transcribed, but `spot` rows are multi-speaker groups and the total is under 0.5 h. Exclude.

## 10. Codec artifact (Qwen3-TTS issue #386) [L]

The issue reports a periodic 40 ms echo in all 12 Hz output. Reproduced on our audio (`s10_echo.py`, `s11_echo_fix.py`): cepstral peak at 40 ms as a multiple of the 15–120 ms background.

| Audio | 40 ms | 80 ms | n |
|---|---:|---:|---:|
| Real v1 clips (24 kHz) | 0.08 | −0.26 | 120 |
| Our Notes raw (48 kHz) | −0.18 | −2.11 | 59 |
| **Same real clips through the codec** | **19.04** | 6.03 | 120 |
| E1b epoch 2 renders | 11.19 | 3.44 | 24 |
| Base zero-shot | 9.95 | 2.38 | 120 |

- It is entirely a decoder property. The decoder has `upsampling_ratios [2, 2]` built from `ConvTranspose1d(kernel = stride = factor)`, so the first ×2 from 12.5 Hz has a 40 ms period and no overlap. That is the likely cause [I]. No training data on the TTS side can change it.
- **But it is small.** Absolute echo gain ≈ 0.011, **−39 dB** re the direct sound. The 25 Hz comb in the log spectrum is 0.31 dB peak-to-peak against 0.21 dB in real audio. Frame-locked envelope modulation is the same as real audio (0.55 vs 0.55). The large ratio reflects a tiny background.
- A single-tap inverse comb only halves the ratio (18.5 → 10.6), and log-spectral distance to the real clip does not improve (10.42 → 10.42 dB). So it is not a simple LTI echo, and DSP is not a fix [I].
- **Verdict:** likely inaudible on its own, not a priority [I]. One ear test decides whether it matters: A/B clips (real, codec, codec + comb) in `<scratchpad>\echo\listen_echo\`.
- **If you hear the codec as a problem:** the decoder is a separate module (`speech_tokenizer/` in the checkpoint) and the LM only ever sees codes, so a decoder fine-tuned on clean audio is a drop-in that needs no TTS retrain. There is no public training code for it, so this would be a new experiment.

## 11. Scripts

The session scratchpad (local, not published): `s01_index.py`, `s02_audio.py` (and `s02_audio.json`), `s03_story.py`, `s04_labels.py` (and `s04_lines.jsonl`), `s05_vectors.py`, `s06_analyze.py`, `s07_precheck.py` (log `vec\s07.log`, outputs `vec\precheck\`), `s10_echo.py`, `s11_echo_fix.py`, `s12_dose.py`, `s13_channel.py`, `s14_fix.py` (log `vec\s14.log`), `s15_count.py`, `s16_checks.py` (log `vec\s16.log`, gate calibration in `vec\precheck\gate_calib.json`). Run with the Kurisu venv as `python -I -X utf8 <script> <args>`; each script's docstring gives its arguments.
