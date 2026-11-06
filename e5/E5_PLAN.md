# E5 plan: RVC-converted own-voice Mandarin rows for the voices E1b renders generic, then a retrain (DRAFT, 2026-10-31)

Tags: [L] measured, [I] inference. Builds on `../e4/E4_PLAN.md` and `../e4/E4_RESULTS.md`.

**Status: pre-registered 2026-10-31, after the user's decisions (§8).** All 13 converters are queued.

## 0. Premise check, before any new converter trains

E4 established three things on raw RVC audio for Kasumi and Eve:
- the accent is native by ear;
- the voice sounds like the character by ear;
- the probe confirms identity, both raw and after the codec.

**Gap.** The TTS learns from codec tokens, and nobody has listened to the codec-resynthesised audio yet.

**Check.** The user hears 3 new sentences per voice (sources 6–8), each in two forms: raw VC + shift, and its Qwen3-TTS-Tokenizer resynthesis (`../e4/vc/codec/200/`). The folder is `../e4/listen/ep200_codec/`.

**Rule.** If the codec versions sound accented or lose the character, E5 stops before any training.

**Result (user, 2026-10-31):** "y codec clip still sound native". The premise holds, so E5 proceeds. The user did not comment on character after the codec; the probe gives codec pred == row 1.00 for both voices [L].

## 1. Target voices (rule, then list)

**Rule.** A voice is a target if its zh leak on the E1b_ep2 eval is > 0.15 (`e1/results/eval/E1b_ep2/scored.jsonl`, pred ≠ row over 32 zh renders). The 0.15 threshold is the one the adoption count uses.

**Why not the 10 voices in the E4 outline.** That outline reused E3's list, which came from E2's pseudo-row yields. E5 retrains from E1b's mix, so E1b's own per-voice leak is the right criterion.

**The 15 targets** (measured [L]):

| Voice | zh leak | Rows in E1b mix | Train clips | Voice | zh leak | Rows in E1b mix | Train clips |
|---|---|---|---|---|---|---|---|
| ppp_rimi | 0.75 | 0 | 155 | hhw_kanon | 0.34 | 0 | 231 |
| ppp_kasumi | 0.75 | 0 | 248 | ras_chuchu | 0.25 | 16 | 272 |
| ro_ako_maou | 0.69 | 2 | **40** | mujica_sakiko_tsuki | 0.22 | 4 | 193 |
| mygo_tomori | 0.56 | 0 | 206 | ras_masking | 0.19 | 5 | 143 |
| mygo_soyo_tsuki | 0.56 | 0 | 120 | hhw_hagumi | 0.19 | 0 | 142 |
| ras_lock | 0.50 | 0 | 216 | pp_aya | 0.16 | 0 | 362 |
| ro_rinko | 0.47 | 0 | 260 | hhw_kaoru | 0.16 | 22 | 187 |
| pp_eve | 0.38 | 0 | 289 | | | | |

This list differs from the narrative in E1_RESULTS, which named kokoro and himari. Measured, those two are at ≤ 0.12, while chuchu and kaoru are above 0.15.

With n = 32 the standard error is about 0.07, so aya, kaoru, hagumi and masking sit near the line. They are included, because extra in-voice rows did not hurt identity elsewhere (E1).

**The pooled E1b zh leak of these 15 is 0.41.**

## 2. Converters

- **Recipe:** E4's, unchanged (Applio 59c1131).
  - Data: the voice's train split only.
  - Preprocess: 32 kHz, contentvec, rmvpe.
  - Training: 200 epochs, batch 8, from the HiFi-GAN pretrained weights.
  - Weights used: epoch 200 only.
  - Inference: index_rate 0; proposed_pitch, threshold = the voice's pooled training median F0.
- **New converters:** 13. The Kasumi and Eve converters from E4 are reused.
- **ro_ako_maou:** only 40 train clips, about 5 min, so it is the low-data converter. If its rows fail the gates or the listening gate (§4), it contributes no rows and stays generic. That is reported, not patched.

## 3. Sources

- **Texts:** AISHELL-3 test texts from p11's pool (8–30 hanzi, ≤ 1 numeral). They exclude:
  - the eval probes;
  - every text in the E1b mix;
  - the E2 candidates;
  - the E4 sources.

  Each voice gets **40 texts**, disjoint across voices, fixed seed. If the unused pool is short, this is reported, and E2-candidate texts not in any training mix fill the gap.
- **Rendering:**
  - Model E1b_ep2, `generate_custom_voice` in the target voice's own row, Chinese, **6 samples per text** (as in E4).
- **Selection:** the most native sample (lowest lid) that passes E2's accent, text and length gates. This is the native gate.

  The lid gate is biased toward low-pitched, generic-timbre renders. That is acceptable here: the source only has to carry native articulation, and the converter supplies pitch and timbre [I].

## 4. Rows: conversion, gates and listening

**Conversion.** Each source is converted with its voice's epoch-200 converter and the shift, exactly as E4's gated cell. The VC no-shift and shift-only cells are also rendered for the diagnostic below.

**Row gates.** All are scored with bdscore.

| Gate | Applied to | Condition |
|---|---|---|
| accent | the **source**, pre-conversion | native gate (§3) |
| identity (raw) | converted | pred == row and p_true ≥ 0.5 × real_resynth_p_true |
| identity (codec) | converted, after Qwen3-TTS-Tokenizer resynthesis | pred == row and ratio ≥ 0.5 |
| text | converted | Qwen3-ASR CER ≤ 0.05 |
| length | converted, after the training audio policy | 2–12 s, not runaway |

**No post-conversion accent metric is a gate.** E4 Amendment C's C4 (the no-shift twin) is withdrawn: the user heard Eve's no-shift clips as native at lid −14.9 and SER 0.20, so neither lid nor SER on converted audio is valid for these voices [L].

**Diagnostics, reported per voice and not gating:**
- paired SER(VC + shift) − SER(shift only);
- lid and SER for every cell;
- text-gate failure rate, converted against source.

A voice is **flagged** if more than 25% of its conversions fail the text gate. E4 had 15% for Kasumi and 11% for Eve. One of those failures was an -n/-ng confusion (变阵 → 辩证) on a source with CER 0. A flagged voice gets the larger listening set (below).

**Dose.**
- **Cap:** 20 VC rows per voice. The 34 working voices had a median of 13 and a mean of 15 gated rows in the E1b mix, and voices with 10–14 rows reached zh leak 0.019.
- **Order:** highest raw p_true ratio first.
- **Too few rows:** a voice with fewer than 10 passing rows is reported as weak. It still contributes whatever passed.
- **Existing rows:** E1b pseudo rows are kept for chuchu (16), kaoru (22), masking (5), sakiko_tsuki (4) and ako_maou (2). The E1b mix is not modified, so the comparison stays clean; VC rows are added on top.

**Audio path into training.** RVC output (32 kHz) is resampled to 24 kHz, then the Kurisu v3 audio policy (`p03_build.normalise_audio`) is applied, then the 2–12 s check. This is E3's collect step. Row ids are `pseudo_zh_<row>_rvc_<utt>`, source `pseudo_vc`, 3% val.

**Listening gate (per voice, blind, before the mix is built).**
- **Set:** 3 rows per voice, picked at random (seed 5) from the voice's passing rows, played as their codec resynthesis.
- **Flagged voices:** 6 rows, 3 of which are text-gate failures.
- **Questions, as in E4:**
  1. native or accented?
  2. does it sound like the character?
- **Rule:** a voice judged accented, or not the character, contributes **zero** rows.

## 5. Retrain (pod)

- **Training:** full SFT from Base with `kurisu_v2/pod/train_sft.py`, on **E1b's mix plus the VC rows**. E2's mix is not reused.
- **Schedule:** as in E2/E3: a 3-epoch cosine, stopped at 2.0, with epochs 1 and 2 evaluated.
- **Size:** about 300 added rows (15 × 20 at most), which takes the pseudo share from 5.0% to about 7.5%. E2's dilution happened at 10.7% with rows that were not truly in-voice.
- **E1b's own pseudo rows (564) are not kept locally.** They were deleted with the E1 pod; only `e1/results/exp/E1b/mix.json` survives. They are regenerated on the pod with `e1_pseudo.py`, using the same texts (p11 seed 31), seeds, gates and cap.
  - GPU nondeterminism can change individual renders, so the regenerated per-voice counts are reported against E1b's `mix.json` before training.
  - If any voice differs by more than 3 rows, this is disclosed. It is not corrected.
- **Cost:** about 30–45 min more H100 time than first estimated, so the pod total is about $8–12, not $6–10.
- **Evaluation:** the E1/E2 eval harness, unchanged: 32 zh, ja and en renders per voice, and bdscore.

## 6. Selection and adoption (numbers fixed now)

**Eligible.** All E1 guards (`e1_select.GUARDS`) as in E2, with one change: **zh_ser ≤ 0.087, zh_lid ≤ −18.0 and zh_ser_voice_max ≤ 0.15 are computed over the 34 non-target voices only.**

The reason: the 15 targets will now speak at their own pitch. E4's shift-only control showed the metrics fail at that pitch with native articulation (Kasumi SER 0.14, Eve 0.21), so applying these guards to the targets would reject a correct model.

The other guards are applied as written:
- ja_leak ≤ 0.03;
- no voice's ja leak worse than the floor by more than 0.2;
- ja_cer ≤ 0.035;
- en_wer ≤ 0.02;
- runaway ≤ 0.01;
- pooled en_lid ≤ −9.3.

**Pick.** Among eligible checkpoints, the one with the lowest pooled zh leak.

**Adopt over E1b_ep2 only if all of the following hold:**
1. pooled zh leak ≤ 0.08 (E1b: 0.155);
2. at least 43 of 49 voices at zh leak ≤ 0.15 (E1b: 34);
3. the 15 targets' pooled zh leak ≤ 0.15 (E1b: 0.41);
4. **target accent by ear.** The user hears 2 blind zh renders per target voice from the pick (30 clips). At least 12 of 15 voices must be judged native. Whether each sounds like the character is reported per voice.

Otherwise E1b_ep2 stays the model.

**Reported regardless:**
- zh leak, lid and SER per target voice, each against E1b;
- en lid for the target group;
- per-voice row counts;
- the listening outcomes.

## 7. Order of work

| Step | Where | Cost |
|---|---|---|
| 0. Codec listening check, Kasumi and Eve | local | none |
| 1. Shakedown: Kasumi and Eve rows end to end (§3–4, sources through gated rows), with the existing converters | local | $0 |
| 2. The 13 new converters | per §8 | |
| 3. Rows for the 13, then the listening gate | local | $0 |
| 4. Retrain and eval | pod | |
| 5. Selection, then target listening | | |

Pods are terminated after each use and checkpoints are deleted there. Nothing is published (BanG Dream guideline item 10).

## 8. Decisions (user, 2026-10-31)

1. **Converters:** trained locally on the 5060 Ti ($0, about 29 h for 13).
   - The queue is `e5_rvc_queue.sh`, which runs E4's `e4_rvc_train.sh` voice by voice.
   - It re-reads `queue.txt` before each voice and waits while `PAUSE` exists.
2. **Retrain on a pod:** approved, at an estimated $6–10.
3. **The E4 verdict covers both Kasumi and Eve.**
4. **Target set: the 15 (user, 2026-10-31).** The user asked what each list means, was answered, then chose 15. The §1 rule stands.
   - The 10 (E3's list) are a subset of the 15, so the 8 new converters they share are queued first, worst leak first: rimi, ako_maou, tomori, soyo_tsuki, lock, rinko, kanon, hagumi.
   - The 5 extras (chuchu, sakiko_tsuki, masking, aya, kaoru) are appended only if the user picks 15.
   - If the user picks 10, the §1 rule becomes "E3's list". The adoption count in §6(2) stays at 43/49, and §6(3) is computed over the 10.

## Amendment E5-A (2026-10-31): source donor fill

Written after the Kasumi/Eve shakedown (§7 step 1). It comes before any user listening of E5 rows, and before rows exist for any other voice.

**Observation [L].** The native gate (§3) kept **24 of 40 texts for Kasumi but only 5 of 40 for Eve** (6% of Eve's renders). All 5 Eve sources that passed went on to pass every row gate. So the bottleneck is the source gate, not the conversion.

The source gate has the E4 pitch confound:
- Median F0 of the own-row renders: Eve 385 Hz, Kasumi 329 Hz.
- lid correlates with log F0 within each voice's 240 renders: r = 0.48 for Eve, 0.38 for Kasumi.
- Eve's lid-pass rate falls across F0 terciles: 0.18, then 0.06, then 0.03.
- The share of renders already in-voice (pred == row): Eve 0.68, Kasumi 0.12.

So the gate rejects Eve's renders mostly for being at her pitch, which is not demonstrated accent. Voices that E1b already renders partly in-voice are expected to hit the same wall.

**E5-A1. Donor fill.**
- **Per text, own row first (unchanged).** If any own-row sample passes `native()`, the lowest-lid one is kept.
- **Otherwise, the donor row.** The text is rendered 6 times in the **ppp_kasumi** row:
  - E1b, Chinese;
  - seed 2510000 + 100000·voice index + 1000·sample + batch;
  - scored with row = donor.
  
  The lowest-lid sample that passes `native()` is kept, and `src_row` records the donor.
- **Why Kasumi.** E3/E4 already used this row for 9 of Eve's 18 sources. Its E1b Mandarin is generic (pred == row 0.12) and lower-pitched (329 Hz), and 45% of its renders pass the native gate. No other donor is tried.
- **Kasumi itself** gets no donor fill.
- **Unchanged:** the native gate, and the 40 texts per voice. No top-up texts are drawn.

**E5-A2. Pitch-clip check.** Applio clips the proposed shift at ±12 semitones. Reaching Eve's 500 Hz from a donor take at about 313 Hz needs about +8, so no clipping is expected. Per voice, the share of clips whose logged offset sits at ±12 is reported.

**E5-A3. Listening set.**
- **Drawn from the trained rows.** The 3 clips now come from the rows that would actually be trained: the top 20 by raw ratio that also pass the post-policy 2–12 s check. Before, they came from all passing rows.
- **Both source kinds heard.** If a voice's trained rows include both own-row and donor sources, at least one of each is in the 3.
- **Flagged voices:** as before.

**E5-A4. Collect.** If a (cell, row, utt) is scored in more than one batch file, the newest file wins.

**Effect on the shakedown voices.**
- **Kasumi:** not affected. There is no donor fill for her and her passing pool is the same. Her listening pick is checked to be the same after the rerun.
- **Eve:** the earlier 3-clip set was built from 5 rows. It is withdrawn before the user hears it.
- **Shakedown batch files:** moved to `vc/shakedown/`.

## Amendment E5-B (2026-10-31): pick the most in-voice native source

Written after batch 1 (Kasumi, Eve, Rimi). It comes before any listening verdict and before rows exist for any other voice. The diagnostics are in `vc/diag/` and `score_diag_rimi.log`; all of them use batch-1 sources or real held-out clips.

**Batch-1 outcome [L].** Rimi is weak: 1 of 24 conversions passes. Her clips are classified as Rimi 92% of the time, but the median raw ratio is 0.34.

**Rimi diagnostics [L].**
- **Training length.** Identity falls with training. By epoch:

  | Epoch | Raw ratio (median) | Raw pass |
  |---|---|---|
  | 50 | 0.44 | 0.38 |
  | 100 | 0.37 | 0.17 |
  | 150 | 0.35 | 0.08 |
  | 200 | 0.34 | 0.08 |

  So the epoch count is not changed.
- **Self-reconstruction.** Her held-out real clips through her own converter: ratio 1.02, 8 of 8 pass. The converter reproduces Rimi.
- **Cross-speaker.** Her converter imposes her voice less firmly on other speakers' real speech than Kasumi's converter does:

  | Real input | Converter | Ratio (median) |
  |---|---|---|
  | Eve | Rimi | 0.66 |
  | Kasumi | Rimi | 0.48 |
  | Eve | Kasumi | 0.93 |
  | Rimi | Kasumi | 0.93 |

  So the source's delivery matters.

**Paired test [L].** On own-row texts where the two rules choose different samples, each text was converted once from the lowest-lid sample and once from the highest-p_true sample. Same text, same converter:

| Voice | Texts compared | Highest p_true better | Median ratio change | Raw identity passes |
|---|---|---|---|---|
| Kasumi | 15 | 13 | +0.22 | 12 → 14 |
| Rimi | 7 | 6 | +0.09 | 1 → 4 |
| Eve | 3 | 1 | −0.01 | 3 → 3 |

Pooled, the highest-p_true sample was better on 20 of 25 texts.

**E5-B1. Selection rule.**
- **Own-row texts:** among the samples that pass `native()`, keep the one with the **highest source p_true** (the probe's p(row) on the raw render). This replaces the lowest-lid pick.
- **The native gate is unchanged.**
- **Donor-filled texts** keep the lowest-lid pick. Their renders were scored against the donor row, and there is no paired evidence for them.

**Scope.**
- E5-B applies to every voice whose listening set has not been issued: Rimi (re-selected; her 1-clip set was never shown) and the other 12.
- **Kasumi and Eve keep the lowest-lid selection**, because their sets went to the user before this amendment. This is disclosed in the results.

**E5-B2. Stale outputs.** When a (row, utt)'s chosen source changes, `e5_sources` deletes that utt's converted outputs (every `out/ep*` cell and the shift control) and its logged offset before conversion. Codec resyntheses are always regenerated.

## Rows outcome, all 15 voices (2026-11-01, before any listening verdict)

Written after batch 3 (`e5_rows_b3.log`, `vc/collect_report.json`). No verdict has been received for any voice. Columns: own + donor source texts (donor share in brackets), rows passing every gate, rows trained (top 20, donor share in brackets), codec identity pass rate, the share of conversions at the ±12-semitone clip (E5-A2), and paired mean SER / median lid for source, shift-only control and the epoch-200 conversion.

| Voice | Sources (donor) | Passing | Trained (donor) | Codec identity | Pitch-clipped | SER source / shift / VC | lid source / shift / VC |
|---|---|---|---|---|---|---|---|
| ppp_kasumi | 24 (0) | 17 | 15 (0) | 0.92 | 8% | 0.036 / 0.194 / 0.276 | -23.9 / -15.5 / -12.9 |
| pp_eve | 23 (18) | 12 | 11 (6) | 0.57 | 9% | 0.048 / 0.199 / 0.294 | -22.5 / -13.4 / -7.3 |
| ppp_rimi (weak) | 24 (11) | 2 | 2 (0) | 0.17 | 8% | 0.048 / 0.177 / 0.250 | -21.0 / -13.1 / -10.3 |
| ro_ako_maou | 28 (2) | 12 | 12 (0) | 0.46 | 0% | 0.010 / 0.028 / 0.099 | -22.7 / -22.0 / -16.6 |
| mygo_tomori | 25 (4) | 14 | 14 (0) | 0.60 | 0% | 0.028 / 0.033 / 0.106 | -22.5 / -22.7 / -15.1 |
| mygo_soyo_tsuki | 32 (7) | 22 | 20 (2) | 0.69 | 0% | 0.061 / 0.101 / 0.229 | -21.7 / -17.6 / -12.0 |
| ras_lock | 27 (7) | 21 | 19 (4) | 0.81 | 7% | 0.079 / 0.242 / 0.309 | -20.1 / -10.2 / -6.6 |
| ro_rinko | 22 (6) | 13 | 13 (2) | 0.64 | 0% | 0.041 / 0.116 / 0.191 | -21.6 / -16.0 / -10.2 |
| hhw_kanon (weak) | 26 (16) | 6 | 6 (3) | 0.27 | 0% | 0.077 / 0.146 / 0.184 | -21.9 / -17.7 / -11.5 |
| ras_chuchu | 20 (1) | 18 | 18 (1) | 1.00 | 0% | 0.092 / 0.103 / 0.220 | -21.7 / -20.9 / -16.1 |
| mujica_sakiko_tsuki | 25 (2) | 20 | 17 (0) | 0.80 | 0% | 0.058 / 0.070 / 0.152 | -21.7 / -20.9 / -15.3 |
| ras_masking | 28 (3) | 11 | 11 (0) | 0.54 | 0% | 0.022 / 0.031 / 0.072 | -21.1 / -20.0 / -17.2 |
| hhw_hagumi | 26 (10) | 23 | 20 (8) | 0.88 | 8% | 0.039 / 0.233 / 0.269 | -21.1 / -12.0 / -8.1 |
| pp_aya | 27 (8) | 21 | 20 (3) | 0.81 | 0% | 0.056 / 0.093 / 0.214 | -21.2 / -19.0 / -13.4 |
| hhw_kaoru | 30 (3) | 26 | 20 (0) | 0.97 | 0% | 0.040 / 0.059 / 0.121 | -22.9 / -21.5 / -15.5 |

- **Size.** 218 rows if every voice passes listening, not the ~300 budgeted. The pseudo share goes from 4.97% to about 6.7%.
- **Weak (< 10 passing).** Rimi (2) and Kanon (6). Both still contribute what passed (§4).
- **Flagged.** None; the highest text-gate failure rate is kaoru's 4 of 30 (13%), under the 25% bar.
- **Accent metrics where they are valid.**
  - For six voices, the shift-only control stays within 0.03 SER of the source, so pitch does not explain a rise: ako_maou, tomori, sakiko_tsuki, chuchu, masking and kaoru.
  - For these six, conversion still raises SER by +0.04 (masking) to +0.12 (chuchu) over the shift-only control, and moves lid from about −21 to about −16.
  - Soyo_tsuki and aya (shift costs +0.04) rise by a similar +0.12.
  - For reference, E1b's eval zh renders for the 34 non-target voices average SER 0.065 (median lid −19.9); the highest voice is kokoro at 0.135.
  - Chuchu, soyo_tsuki, aya and sakiko_tsuki convert to above that, while their shift-only controls sit at 0.07–0.10.
  - Under §4, the user's listening still decides. If a voice with these numbers is judged native, that is recorded as ear over metric, with the numbers alongside, as in E4.
- **Blindness.** Ako's set went out with a note to listen for accent (from batch 2's numbers), so her verdict is not fully blind. The other 14 sets go out without per-voice notes.
