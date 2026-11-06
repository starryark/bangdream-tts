# E4 plan: per-voice RVC for native Mandarin in the voices E1b renders generic (pre-registered 2026-10-31)

Tags: [L] measured, [I] inference. Follows `e2/E2_RESULTS.md` and `e3/E3_PLAN.md` (user chose option 2, RVC, on 2026-10-31).

## Why RVC after Seed-VC failed

E3 showed that zero-shot Seed-VC fixes identity (40/40 clips assigned to their own voice) but re-imposes the Japanese accent: SER rose from 0.06 to 0.18–0.21, and lid from −28 to −12 [L]. The likely cause is the in-context Japanese reference mel, which the DiT copies along with timbre [I].

RVC differs on three points:
- **No reference prompt.** Content comes from ContentVec features of the source, and timbre lives in the trained per-voice generator.
- **Index off.** With `index_rate 0` the FAISS index of Japanese training features is never loaded (Applio `rvc/infer/pipeline.py`: the index is read only when `index_rate > 0`), so no Japanese frames are retrieved.
- **F0 conditioning.** The source F0 contour (rmvpe) drives the decoder with a constant per-clip shift, so Mandarin tone shapes survive by construction. The E3 Seed-VC configuration had no F0 conditioning.

**Hypothesis [I].** Accent should survive better than with Seed-VC.

**Risk.** Each generator is fine-tuned only on Japanese output. Phones that Japanese lacks (retroflexes, ü, -ng codas, er) may be rendered toward Japanese realisations. The shared-character SER is the metric that would catch this.

## Feasibility test (local RTX 5060 Ti, no pod)

Mirrors E3: two voices, Mandarin only.

### Sources

The 40 E3 feasibility texts and source rows, taken from `e2/results/e3/vc/jobs.json`:
- Kasumi: 20 own texts.
- Eve: 11 own texts plus 9 of Kasumi's.

E3's source wavs were deleted with the pod, so they are re-rendered locally:
- **Model:** E1b_ep2, `generate_custom_voice` in the job's `src_row`, Chinese.
- **Samples:** 6 per text, with fixed seeds.
- **Selection:** for each job, the most native sample (lowest lid) that passes E2's accent, text and length gates (P(zh) ≥ 0.9, lid ≤ −19, CER ≤ 0.05, 2–12 s, not runaway). Identity is ignored.
- Jobs with no passing sample are dropped and counted.
- The sources' median lid and mean SER are reported next to E3's (−28 / −24).

### Converter

- **Software:** Applio (MIT) at commit `59c113106c0dfff647324ab1a80c286f17b51929`, run in its own venv (`bangdream/.tools/rvcenv`, torch 2.11 cu128).
- **Training data:** the voice's own TTS train split (`data/split.jsonl`, `data/wav24`). That is 248 clips / 34 min for Kasumi and 289 / 44 min for Eve. Test, val and reference clips are excluded.
- **Preprocess:** 32 kHz; cut Automatic, chunk 3.0 s, overlap 0.3; no effects; no normalisation.
- **Extract:** rmvpe F0 and the `contentvec` embedder, with include_mutes 2.
- **Train:** HiFi-GAN vocoder from Applio's default 32 kHz pretrained G/D; batch 8; **200 epochs**, saving weights every 50.
- **Gated checkpoint:** only the **epoch-200** weights are gated. Epoch 100 is reported as a secondary result only.
- **Inference:**
  - Settings: `index_rate 0`; f0 rmvpe; pitch 0 with `--proposed-pitch`; `--proposed-pitch-threshold` = the voice's median voiced F0 over its training clips (pooled frames from the training f0 files).
  - What the proposed-pitch option does: it shifts each source by round(12·log2(threshold / source median F0)) semitones, clipped to ±12.
  - Defaults kept: protect 0.33 (a no-op when the index is off), volume envelope 1.0, contentvec.
  - Not used: autotune, formant shift, cleanup.
  - Output: WAV.

### Scoring and gates

Scoring is identical to E3 (`e3_collect.py`): `bdscore` on the raw converted audio, using `bundle/probe_codec.npz` and `e2_baselines.json` real_resynth_p_true. The gates are:
- **identity:** pred == row, and p_true ≥ 0.5 × real p_true
- **accent:** P(zh) ≥ 0.9 and lid ≤ −19
- **text:** CER ≤ 0.05
- **length:** 1.5–13 s, and not a runaway

**Feasibility gate.** E3's gate, plus an SER condition. It passes if, for at least one of Kasumi and Eve, all three hold for the same voice:
1. at least 20% of converted clips pass all gates;
2. the median converted lid is ≤ −19;
3. the mean shared-character SER of converted clips (m23-valid rows) is ≤ 0.10.

For reference, the SER values seen so far:

| Condition | SER |
|---|---|
| Native sources | ~0.06–0.07 |
| E0 Base floor | 0.077 |
| Seed-VC | 0.18–0.21 |

**If the gate fails, E4 stops.** It is reported and nothing is trained.

### Reported regardless (diagnostic, not gating)

- The pass rate for each gate, per voice.
- Identity split into two parts: pred == row alone, and the p_true ratio. `probe_codec` was fit on codec-resynthesised audio and RVC output is not codec audio, so a ratio-only failure may be a domain effect rather than a wrong voice.
- Identity re-scored after a Qwen3-TTS-Tokenizer encode→decode of each converted clip. This is the form the TTS would actually learn from.
- Paired source→VC deltas in lid and SER.
- The epoch-100 checkpoint on the same gates.
- A few listening samples, local only.

## Amendment B (2026-10-31, after the epoch-50 smoke test, before any control or epoch-200 result)

**What the smoke run showed (disclosed).** The epoch-50 Kasumi weights were run once as a pipeline smoke test. Kasumi's target median F0 is 452.7 Hz (rmvpe), and librosa pyin agrees (median 463 Hz over 25 train clips). The E1b sources sit near 300 Hz, so each clip was shifted +2 to +12 semitones.

| | Raw pred == row | Codec pred == row | lid, source → VC | SER, source → VC | CER |
|---|---|---|---|---|---|
| Kasumi epoch 50 | 19/20 | 15/20 | −28.8 → −14.8 | 0.036 → 0.23 | 0.011 |

**Why it is not yet interpretable.**
- lid (VoxLingua107) and SER (charsiu) have never been validated near 460 Hz. SER's human validation (ρ = −0.84) was on Kurisu, at about 220 Hz.
- A large F0 shift alone could degrade both metrics with no change in articulation.
- E3's Seed-VC output was also at Kasumi's and Eve's pitch, so E3's "accent re-imposed" conclusion carries the same caveat. It is qualified, not withdrawn, until the controls below are in.

**Controls, added now.** Together with the existing source and VC + shift cells, they complete a 2×2:

| Cell | Audio | Tests |
|---|---|---|
| **source** (exists) | the native E1b sources | — |
| **shift** (no VC) | each source pitch-shifted by its own per-clip offset (the offset Applio's proposed-pitch computed for the gated conversion) with Praat PSOLA (parselmouth "Change gender", formant ratio 1.0, so formants and articulation are kept) | the metrics' F0 sensitivity |
| **VC + shift** (the gated cell) | RVC as planned | — |
| **VC, no shift** | RVC with proposed_pitch off, pitch 0 | the generator's articulation at the source's own pitch |

The no-shift cell runs at epoch 50 now and at epoch 200 later. Each cell is scored with bdscore exactly as the gated cell, for both voices.

**Predictions and how they are read.**
- **Metric confound.** If the shift cell degrades to near the VC + shift levels (lid > −19 or SER > 0.10), then lid and SER are F0-sensitive at these pitches. The accent condition of the gate is then **not evaluable** for that voice. It is reported as "not evaluable on accent", not as pass or fail, and the user's listening judgement decides.
- **Real accent.** If the shift cell stays native (lid ≤ −19 and SER ≤ 0.10) while VC + shift degrades, the RVC generator imposes articulation, and the gate result stands.
- **VC, no shift** shows whether the generator alone moves the accent. Identity is reported there too, since pitch is part of identity.

**Unchanged.** The epoch-200 gate itself, its thresholds and its decision rule. The gate outcome is reported as written, alongside whether the accent condition was evaluable.

**Also watched.** Identity after codec resynthesis: 15/20 pred == row, and only 15% with ratio ≥ 0.5, at epoch 50. If it stays low at epoch 200, RVC rows would mostly fail the identity check in the domain the TTS learns from.

**Listening.** A local folder for the user holds 5 sources, the same 5 after shift only, and the same 5 after VC + shift, plus VC, no shift.

### Control results, Kasumi at epoch 50 [L] (diagnostic; the gate is at epoch 200)

Report: `vc/report_ampB_ep50.json`. 20 clips per cell.

| Cell | Median F0 | pred == row | codec pred == row | lid (median) | SER (mean) | CER |
|---|---|---|---|---|---|---|
| source | ~300 Hz | 0.00 | 0.00 | −28.8 | 0.036 | 0.000 |
| shift (PSOLA only) | ~465 Hz | 0.40 | 0.20 | −19.4 | **0.143** | 0.000 |
| VC, no shift | ~300 Hz | 0.40 | 0.30 | −21.6 | 0.112 | 0.026 |
| VC + shift | ~460 Hz | 0.95 | 0.75 | −14.8 | 0.229 | 0.011 |

**Reading.** The effects are roughly additive:
- The pitch shift alone costs about 9 lid points and +0.11 SER.
- RVC alone costs about 7 lid points and +0.08 SER.

By the Amendment B rule (the shift cell's SER of 0.143 is above 0.10), **the accent condition is not evaluable for Kasumi**. The metrics cannot certify native Mandarin at her pitch even for a natively articulated clip.

RVC's own contribution is visible at both pitches. Whether it is audible accent or vocoder artifacts the phone recogniser dislikes is undecided [I]; the user's listening judgement decides.

**Side findings:**
- Pitch alone moves the identity probe: the shift-only generic voice is assigned to Kasumi on 8 of 20 clips.
- E2's accent gate (lid ≤ −19, SER-blind) may have been biased against own-voice renders of high-pitched voices. This is a hypothesis [I], not yet tested.

Listening set: `listen/kasumi_ep50/` (README lists each clip's scores).

### Listening result, Kasumi epoch 50 (user, 2026-10-31)

**Verdict:** "both c and d sounds native chinese". Here c is VC with no shift and d is VC + shift. The user made no comment on b (shift only), so nothing is inferred about b.

**Scope:**
- one rater;
- 5 clips (the first 5 Kasumi sources);
- epoch-50 weights;
- not blind, because the README listed each clip's scores.

**Reading [I]:** On these clips, the user does not hear RVC's extra cost (about 7 lid points and +0.08 SER) as accent. The likeliest cause is timbre and vocoder artifacts that the LID and phone models react to. Taken together with the shift control, lid and SER are not valid accent measures for high-pitched converted clips.

E3's "accent re-imposed" verdict for Seed-VC rested on the same metrics, so it is now unsupported. Its audio was deleted with the pod and cannot be heard again.

## Amendment C (2026-10-31, after the epoch-50 listening, before any epoch-200 conversion or score)

This was written while Eve was still training (epoch 129 of 200). No epoch-200 audio exists yet for either voice.

**C1. The gate for a voice whose accent is not evaluable.** Amendment B said that "the user's listening judgement decides", but not how the rest of the gate is computed. That is now fixed.

The rule applies to any voice whose shift cell has lid > −19 or SER > 0.10. That is Kasumi already, and Eve if her shift cell turns out the same way. Such a voice passes feasibility if both of the following hold:
- **(a)** At least 20% of its epoch-200 VC + shift clips pass the identity, text and length gates. These are scored on raw audio, as written; only the accent gate is dropped from the per-clip pass.
- **(b)** The user's listening verdict on an **epoch-200** listening set is "native". The epoch-50 verdict above is provisional and does not count for (b).

The original gate is still computed and reported as written (all four per-clip gates, median lid and SER), labelled "not evaluable on accent".

**C2. Eve is classified only after her own shift cell is scored.** The order is:
1. convert at epoch 200, which records her offsets;
2. run the shift control;
3. score.

If her shift cell stays native (lid ≤ −19 and SER ≤ 0.10), her gate stands as originally written, and listening is diagnostic only.

**C3. A branch condition for the 10-voice outline.** This is new and is not part of the feasibility gate.

The probe was fit on codec-resynthesised audio, so identity after the codec is the honest measure of what the TTS would learn from. At epoch 50, only 15% of Kasumi's clips reached a codec ratio ≥ 0.5.

The 10-voice branch proceeds only if a voice that passes feasibility also has **codec identity (codec pred == row and codec ratio ≥ 0.5) on at least 40% of its epoch-200 VC + shift clips**. Below 40%, E4 reports and stops. Scaling the sources up by 1/rate is then offered to the user as an option, not taken by default. The reason is E2: synthetic rows chosen from a low-yield tail dilute identity.

**C4. The row-level accent gate in the 10-voice branch.** VC + shift rows would fail any lid or SER gate for the reason above. Each row's accent gate is therefore evaluated on two things:
- its source, which is already gated as native;
- its VC no-shift twin, converted from the same source with the same weights, which must have P(zh) ≥ 0.9 and lid ≤ −19.

Caveat: the generator is F0-conditioned, so articulation at ~460 Hz need not match articulation at ~300 Hz. Listening to a sample of each voice's rows remains the arbiter.

**C5. An identity diagnostic.** At epoch 50, shift only and VC with no shift each reached 0.40 pred == row; only the two together reached 0.95. At epoch 200, if VC no-shift pred == row does not exceed shift only by more than 0.10, the identity gain is reported as mostly a pitch effect, whatever the gate says.

**Listening set at epoch 200** (`listen/ep200/`):
- **Kasumi:** the same 5 sentences as at epoch 50.
- **Eve:** her first 5 sources.
- **Cells:** a, b, c and d for both voices, plus real Japanese reference clips.
- **Scores:** the README carries none until the user has answered.
- **Questions:**
  1. Do c and d sound native or Japanese-accented? This answer decides C1(b).
  2. Does d sound like the character? This one is diagnostic.

## If feasibility passes (outline; pre-registered in full before any training)

- **Converters:** train RVC converters for the 10 E3 target voices with the same recipe. `ro_ako_maou` has only 5 min, so it is flagged and handled as a low-data case.
- **Rows:** convert native Mandarin sources and gate them as above, capped at 25 rows per voice. Identity is re-checked after codec resynthesis (C3), and the accent gate is applied to the source and to the no-shift twin (C4).
- **Retrain:** on a pod, from Base, with E1b's mix plus the VC rows.
- **Reference:** E1b epoch 2. E2 showed that extra synthetic rows outside these voices dilute identity, so E2's mix is **not** reused.
- **English:** handled the same way only after Mandarin works.

## Constraints

Everything stays local/private (BanG Dream guideline item 10). No converter, clip or dataset is uploaded anywhere.
