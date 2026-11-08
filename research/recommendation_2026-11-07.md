# BanG Dream voice v2: how to use the Our Notes data (2026-11-07)

Built on `data_facts_2026-11-06.md` (measured facts, §-references below) and the v1 results (E1, E5). The draft pre-registered experiment is `../e6/E6_PLAN.md`. Nothing has been trained and no pod has been started.

Tags: **[L]** measured locally, **[I]** inference, **[W]** from the web (sources at the end).

## 1. The answer

**Keep the v1 engine and recipe, and change what goes into it.** v2 is one shared full fine-tune of Qwen3-TTS-12Hz-1.7B-Base, as v1 is, with these changes:

1. **Add 16 new characters and replace the MyGO / Ave Mujica data with Our Notes audio.** That gives 28 Our Notes rows (12 replacing v1 rows, 16 new) plus 37 v1 rows on their v1 data: **65 rows**, inside the 72 trained slots. Sakiko's two rows split by story arc.
2. **Don't merge old and new clips of a character into one row.** The two games differ by a channel offset as large as the gap between speakers (§7.1).
3. **Don't use all 24.7 hours.** Cap each new row at about 300 clips (the upper quartile of v1's rows), sampled across duration and expression. More Japanese minutes bought nothing measurable in v1 (§8), and more steps put Mandarin and English accent at risk.
4. **Aim v2 at what v1 gets wrong, which the new data can fix:**
   - **short lines**: v1's identity leak on 1 s lines is about 3× that of real audio (§9.5);
   - **nonverbal reactions**: v1 reads `……っ` as `うん`;
   - **flat intonation on the Ave Mujica rows**.
   Judge v2 on those, alongside the v1 guards.
5. **Mandarin by route, not by volume.** Use Base pseudo-Mandarin where the Base model can reach the voice. Use RVC rows for the four new voices it can't reach (arale, nonoka, hotaru, yomogi) (§9.4).
6. **Run it as one pre-registered experiment (E6)** with a regression guard on the 37 unchanged rows, on one chained pod run, about **$40–55**.

## 2. What the new data is worth, and what it isn't

| | Evidence | Use in v2 |
|---|---|---|
| **16 new voices** (bands 3–5 and Kanata), 19–103 min each | §2, §7.5 | New rows |
| **In-game delivery for MyGO and Ave Mujica.** v1's Ave Mujica rows were 2.7–6.5 min of 1–3 s anime clips | §2. v1 renders of umiri, mortis, sakiko and uika have 20–47% less F0 range than the game delivery (§9.5) | Replace their data |
| **Natural short lines.** 14% under 1 s, 3,129 lines of 1–3 characters. v1's game data was cut to 7–10 s | §1, §5, §9.5 | Duration-stratified rows |
| **Nonverbal audio.** ~2,770 lines whose text is only `……`, `……っ`, `！` | §5, §9.6 | `[nv]` tag arm |
| **Expression labels on 92% of lines**, audible at 4× chance | §6, §7.4 | Later arm, not baseline |
| **Official zh-Hans / English text of each character's own lines** | §6 | Texts for pseudo-Mandarin (in character, not news sentences) |
| **Clean, aligned scripts.** 94% pass v1's ASR gate; failures are mostly crying, laughter and names | §9.1 | Same gate, with an exception for 1–2 kana lines |
| More minutes per voice | Zero correlation with any v1 metric (§8) | Cap, don't maximise |
| 48 kHz audio | No gain at the model's 24 kHz output (§3) | Not a reason by itself |

## 3. Recommended design

### 3.1 Engine: unchanged

- **No better base exists today [W].** Qwen3-TTS-12Hz-1.7B-Base has had no new weights since 2026-01-23, and `qwen-tts` is still 0.1.1.
- Audio8-TTS-Realistic-Air (Apache-2.0, 44.1 kHz, official SFT script) is the only new candidate. It conditions only on a reference clip and transcript, with no speaker table or language id, so it would copy the Japanese accent into Mandarin and English. That is the problem v1's design solves.
- Keep the v1 setup: speaker rows at 3000+, a language id on every row, `p_auto_lang` 0.25, lr 5e-6, effective batch 32, fp32 master weights, 3-epoch cosine.

### 3.2 Rows

| Set | Rows | Data |
|---|---:|---|
| v1 rows with no new data (PPP, Afterglow, Pastel\*Palettes, Roselia, HHW, Morfonica, RAS) | 36 | E1b recipe, unchanged |
| `mygo_soyo_tsuki` (GBP game audio, 126 clips) | 1 | E1b recipe, unchanged. Soyo's Our Notes arcs don't split cleanly (0.81, §7.2) |
| MyGO 5 + Ave Mujica 5 + Mortis (v1 names kept, so existing code keeps working) | 11 | **Our Notes only.** `mujica_sakiko` takes Sakiko's Mujica-arc and other non-MyGO lines |
| `mujica_sakiko_tsuki` | 1 | **Our Notes MyGO-arc Sakiko** (D2) |
| New characters | 16 | Our Notes |
| *Optional:* 初音 Hatsune (same voice actor as Uika, cos 0.52, 19 min) | (1) | Our Notes. Decision D2 |

Why Sakiko splits by arc:
- v1's `mujica_sakiko_tsuki` is anime-sourced (203 clips, median 3.2 s; checked in v1's manifest). It has the same short-clip, flat-intonation channel as the other anime rows being replaced.
- Inside Our Notes, Sakiko's MyGO-arc and Mujica-arc centroids are 0.69 apart, as far as v1's 白/黒 pair (0.70).
- MyGO-arc Sakiko sits closer to the old Tsukinomori row (0.68) than to `mujica_sakiko` (0.54).
- Splitting puts both Sakiko rows on one channel and keeps a persona distinction the audio actually shows.
- The conservative alternative (keep the anime row unchanged) is listed under D2.

Persona handling (§7.2):
- Mutsumi and Mortis stay two rows.
- Doloris, Oblivionis, the 召使 role, `mutsumi_maybe`, `*_child`, Viola and the manager are excluded.
- Amoris and Timoris fold into Nyamu and Umiri.

Rāna is the one thin voice: 8.9 min, median clip 1.15 s, 89 clips ≥ 2 s. Use Our Notes only. Fall back to v1 clips only under the rule in E6_PLAN, never merged by hand.

### 3.3 Japanese data for the 28 Our Notes rows

- **Cap 300 clips per row**, stratified across duration bins (< 1, 1–3, 3–8, ≥ 8 s) and expression classes, from training episodes only. v1's 36 rows have a median of 221 rows and a 75th percentile of 259.
  - Total ≈ 20k rows, about **620 steps per epoch, 1.8× E1b**.
  - Uncapped would be 2.4–3.7×. That is the risk to accent (§3.6).
- **Text policies** (§5):
  - drop ellipsis-only and nonverbal-only lines from the baseline (they go to the `[nv]` arm);
  - remove newlines; strip `（）` but keep the audio; strip markup tags; take readings from `<ruby>`;
  - exclude multi-speaker lines, `spot` and `VoiceLocation`;
  - add the new names to the readings table.
- **Gate:** v1's ASR gate for lines of 3+ kana. Lines of 1–2 kana can't be checked by ASR (40% pass, §9.1), so they are gated by speaker probe agreement plus a plausible duration instead.
- **Locked vectors:** v1 recipe. Measured floors used it (§9.4); changing it would change the floors.

### 3.4 Mandarin and English

- **Base pseudo-Mandarin for the new rows**, routed by the floor (§9.4):
  - texts are the official zh-Hans translations of the character's own lines (8–30 hanzi, at most one numeral, none overlapping the eval probes);
  - E1's gate, with the identity bar recalibrated to **0.39** for the larger probe (§9.2);
  - 6 samples per text instead of 2. Six roles (rana, soyo, houka, tomori, mahoro, miyako) pass under 20% of renders, mostly on the accent gate, so they need more tries per text;
  - at most 40 rows per voice, as in E1.
- **VC rows (RVC, E5 recipe) for arale, nonoka, hotaru, yomogi.** Base can't reach these four: their real voices are the furthest from its average output, and their renders land on Chieri (§9.4).
  - Train their four converters locally (~3 h each on the 5060 Ti, $0).
  - E5's VC rows for the 13 v1 rows that stay (187 rows) go in the same arm.
  - E5's `mygo_tomori` and `mujica_sakiko_tsuki` rows (14 + 17) are dropped, because those converters were built on v1-channel audio of rows that now change.
  - VC rows can share the voice's row; no per-language rows are needed (§9.3).
- **English:** unchanged from v1 (replay only). v1 English is already at WER 0.003. Nothing in this data changes that.
- **Replay:** E1's 13.3% share, so the pool must roughly double (p05 with about 4 h zh and 2 h en).

### 3.5 Tags: one arm, `[nv]` first

- The expression labels are real signal (§7.4), but they are a separate claim, so they don't go into the baseline.
- The first tag worth testing is a single atomic `[nv]` token. It is trained on the ~2,770 nonverbal lines and initialised from `息`, `はぁ`, `っ` with `train_sft.py --new_tokens`.
  - v1 has no way to make a gasp.
  - The material is plentiful (about 100 per row).
  - The success test is simple: an `[nv]` render must not come out as a word.
- Thirteen-class expression tags (`[smile]`, `[sad]` …) come after, as their own experiment, if `[nv]` works.

### 3.6 Evaluation: add the metrics that can see the difference

v1's metrics are saturated on mid-length Japanese (§8). E6 adds:
- **Short-line identity** on held-out Our Notes lines of 0.7–1.6 s, for all rows. Real audio through the codec leaks 0.13; v1 leaks 0.37 on overlap rows and 0.44 on controls.
- **F0 range against real audio** for the 28 Our Notes rows.
- **Duration on silence-trimmed audio.** The raw 1.29 "too long" was padding (§9.5).
- **`[nv]` rendering** (arm C only).
- **Leak reported within each set and across sets**, with Mutsumi↔Mortis counted separately (§9.2).

The probe is 63+ classes, trained on codec-resynthesised clips. **E1b_ep2 is re-rendered under the same probe and texts as condition 0**, so the regression guard compares like with like.

**Pre-registered prediction:** about 1.8× E1b's steps will move Mandarin and English accent on the 37 unchanged rows slightly the wrong way (§8: zh lid ρ = +0.27 with Japanese minutes; Kurisu: accent grows with steps). The guard therefore covers zh SER, zh lid and en lid, not only identity. If the baseline fails on accent, a pre-registered conditional arm raises replay to 25%.

## 4. What not to do

| Idea | Why not |
|---|---|
| Merge v1 and Our Notes clips for the 10 overlap characters | A locked vector would average two channels 0.68 apart (§7.1) |
| Train on everything | Minutes buy nothing measurable (§8); steps cost accent |
| Switch models | Nothing newer fits three languages with fixed speakers [W] |
| One row per language | VC Mandarin sits on the voice's own row (§9.3) |
| Fix the 40 ms codec echo first | −39 dB, likely inaudible (§10). The A/B clips are there if you want to listen |
| Train `……` lines as text | `……` is Qwen's pause token; the audio is a reaction (§5, §9.6) |
| Use the old `p_true ≥ 0.5` gate on the larger probe | It passes 74% of real new-role clips instead of 91% (§9.2) |

## 5. Plan and cost

**Local first, $0 (5060 Ti):**
1. Build the 28-row manifest with the policies above and an episode-level split, then run the ASR gate (~1 h).
2. Compute locked vectors.
3. If D3 = yes: train RVC converters for the four roles (~12 h), make about 20 VC rows each, and give you a short listening check as in E5.
4. Encode codes, unless it's faster on the pod.

**One pod, chained end to end (H100, ~$3.99/h):**

| Stage | Time |
|---|---:|
| Bootstrap: environment, replay pool, E1b-recipe pseudo rows for the 37, new pseudo rows for the 28 (28 × 60 × 6 renders) | ~2.5 h |
| Train E6a, E6b, E6c (3-epoch cosine, ~620 steps/epoch) | ~1.2 h |
| Evaluate 7 conditions: E1b_ep2, and epochs 1 and 2 of each arm | ~5.5 h |
| Decision script applies the rule and writes the selection report | — |
| **Total** | **~9–10 h, $40–55** |

The conditional replay arm adds ~1.5 h. Per the E5 lesson, every stage is queued before launch, so nothing waits on a question.

## 6. Decisions for you

| | Question | Recommendation |
|---|---|---|
| **D1** | Add the 16 new characters and replace MyGO / Ave Mujica data with Our Notes? | Yes. Keep the v1 row names for the 12 replaced rows |
| **D2** | Personas: (a) `sakiko_tsuki` ← Our Notes MyGO-arc Sakiko and `mujica_sakiko` ← the other arcs, or (b) keep the anime `sakiko_tsuki` unchanged. Keep `soyo_tsuki` on v1 data? Add Hatsune? | (a), since it drops the last anime-channel row. Keep `soyo_tsuki`. Hatsune optional (one slot, 19 min, distinct delivery). Under (b) the counts become 38 unchanged and 11 replaced |
| **D3** | Run the VC arm (4 new local converters, a short listening check for you)? | Yes. Mandarin is a stated requirement and four new voices can't get it otherwise |
| **D4** | Release form for v2 | Your call; see §7. This is a new decision, not a repeat of v1's |
| **D5** | Approve one pod run, ~$40–55 | Once D1–D3 are settled |

## 7. Release considerations specific to v2

These are facts for D4 [W], not a recommendation:
- **Our Notes terms of service** (operator 株式会社ブシロード):
  - Art. 13(18) bans reverse engineering or analysing the service's structure.
  - Art. 11(4) bans using its content without permission.
- **Our Notes streaming guideline** (dated 2026-09-24) lists as prohibited videos made by altering voices, **"including inputting voices into generative AI for that purpose"**. That clause is explicit and newer than anything that applied to v1's sources.
- **The BanG Dream! fan guideline** (item 10) still prohibits making or distributing derivative works that use the voice audio, as it did for v1.
- **Band 3 (夢限大みゅーたいぷ)** is presented as a virtual band whose cast performs live, credited by character name. The game data has no voice-actor names for them or for Kanata. A model card would credit them by character name.

v1's card wording (unofficial, AI-generated, takedown on request), the no-transcripts and no-audio rules, and keeping converters local would all carry over. Whether v2 is public, gated or private is a separate choice from v1.

## Sources

- Qwen3-TTS Base model API (last modified 2026-01-23): https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- qwen-tts on PyPI (0.1.1): https://pypi.org/project/qwen-tts/
- Qwen3-TTS issue #386 (40 ms echo): https://github.com/QwenLM/Qwen3-TTS/issues/386 · PR #387 (fine-tuning alignment): https://github.com/QwenLM/Qwen3-TTS/pull/387
- Audio8-TTS: https://github.com/Edge0-AI/Audio8_TTS · https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air
- Our Notes terms: https://bang-dream-on.bushimo.jp/rule/
- BanG Dream! fan guideline: https://bang-dream.com/bdp-guideline/
- 夢限大みゅーたいぷ: https://bang-dream.com/yumemita · https://gamebiz.jp/news/377215 · https://animeanime.jp/article/2026/07/02/100615.html
- Our Notes EN official site: https://bdon.biligames.com/details/?id=181789561021989038&lang=en-us
- Existing community voice models (RVC only, no Our Notes TTS found): https://huggingface.co/Wanlau/RVC_BanGDream · https://huggingface.co/TogetsuDo
