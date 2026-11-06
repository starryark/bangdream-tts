# E2 plan: own-voice Mandarin for every voice, and English pseudo rows (pre-registered 2026-10-30)

Tags: [L] measured, [I] inference. Builds on `e1/E1_PLAN.md` and `e1/E1_RESULTS.md`.

## Starting point

E1b epoch 2 (`models/E1b_ep2`) [L]:
- **Japanese identity** is at the real-clip level for all 49 voices (leak 0.014).
- **Mandarin accent** is at about the floor for all voices (SER 0.069).
- **Mandarin identity is wrong for 15 voices**: zh leak ≥ 0.15, about 0.40 for the 12 voices that had no in-voice pseudo rows. These are the problem voices in `bundle/e2_baselines.json`.
- **English is accented:** en lid −9.3, against floor −12.2.

## Why not just "more pseudo rows from E1b"

The local check on E1b's own eval renders, for the 15 problem voices [L]:
- Only 2% of their Mandarin renders pass the E1 gates (identity p_true ≥ 0.5, plus accent).
- Own-voice renders are **more** accented: lid −15.0 when assigned to their own voice, −18.4 otherwise.

So identity and native accent are anti-correlated for these voices. Two changes address this:
- **(a) A per-voice identity threshold.** Require pred = own row **and** p_true ≥ 0.5 × that voice's real-clip p_true. The real-clip baselines (through the codec) are 0.49–0.90; kasumi's own real clips reach only 0.57, so a flat 0.5 demanded near-real confidence from her renders.
- **(b) Many more candidates for the problem voices**, so that enough land in the overlap. The estimated overlap rate with the top-1 identity check is 16%.

If E2 does not fix them, E3 is the VC route (`research/vc_and_data_2026-10-29.md`).

## Data

| Part | Rows | Notes |
|---|---|---|
| Japanese | 9,268 train + 111 val | as E1 |
| Replay | 1,509 (14% of Japanese) | as E1, the same p05 settings + distance filter |
| Pseudo zh | ≤ 25 per voice | **Sources:** Base (x-vector, as in E1) for the 34 other voices, on their E1 texts × 2 samples; E1b (`generate_custom_voice`) for all 49 voices. **E1b texts:** the 34 others get their E1 texts × 1 sample; the 15 problem voices get their E1 texts + 100 new texts × 2 samples. Texts: AISHELL-3 test (p11 pool), disjoint per voice |
| Pseudo en | ≤ 12 per voice | Sources Base and E1b, 30 texts per voice, 1 sample each. Texts: LibriTTS-R `test.clean` `text_normalized`, 6–20 words, no digits, `p05.en_text_problem` = None, deduplicated, not in `probes_en_v3`, disjoint per voice |

**Gates** (all required; scored with `bdscore` as in the E1 eval):

| Gate | Mandarin | English |
|---|---|---|
| Identity | pred = row, and p_true ≥ 0.5 × real_p_true[row] | same |
| Accent | P(zh) ≥ 0.9 and lid ≤ −19 | P(en) ≥ 0.9 and lid ≤ −12 (the floor mean) |
| Text | CER ≤ 0.05 | WER ≤ 0.05 |
| Length | not a runaway; 2–12 s after the training audio policy | same |

**Selection of rows:**
- Per (voice, language, text), keep the passing candidate with the highest p_true / real_p_true across sources and samples.
- Then apply the cap, keeping the highest ratios first.
- Report the pass rate per voice, language and source before training.

**Expected size** ≈ 9,268 + 1,509 + ≤ 1,225 zh + ≤ 588 en ≈ 12.6k rows, about 395 steps per epoch [I].

## Training

- The E1b command unchanged: full SFT from Base, fp32 master weights, lr 5e-6, effective batch 32, `--p_auto_lang 0.25`, the same 49 locked speaker rows and 55 aux speakers, seed 20261028.
- `--epochs 3 --save_at 1.0,2.0 --stop_after 2.0`: the same LR schedule as E1b up to epoch 2, which was E1b's pick.

## Evaluation

The E1 eval plan, code and seeds unchanged (4,356 renders per checkpoint). Floor, real and E1b epoch 2 numbers come from the E1 run (`e1/results`).

## Selection rule (fixed now)

**Candidates:** E2 epoch 1 and E2 epoch 2.

**Eligible only if all hold:**
- The E1 guards: ja CER (held-out) ≤ 0.035; zh SER ≤ 0.087; zh lid ≤ −18; any voice's zh SER ≤ 0.15; en WER ≤ 0.02; runaway ≤ 1%; no voice's ja leak > its floor + 0.2.
- No Japanese regression: pooled ja leak ≤ 0.03.
- No English regression: pooled en lid ≤ −9.3 (E1b epoch 2).

**Primary metric:** pooled zh leak. Pick the eligible checkpoint with the lowest.

**Adoption over E1b epoch 2** requires both:
- pooled zh leak ≤ 0.10 (E1b: 0.155);
- at least 40 of 49 voices with zh leak ≤ 0.15 (E1b: 34).

Otherwise E1b epoch 2 stays the model, and E3 (VC pseudo-data) is next.

**Reported regardless:**
- zh leak and lid per problem voice;
- en lid pooled and by group (problem voices / others);
- persona pairs;
- pseudo-row pass rates.

## Compute

- **Hardware and storage:** one Secure H100, container disk only. Audio, refs and the E1b checkpoint go up over SSH; nothing is published.
- **Time estimate [I]:**
  - upload ≈ 20 min, in parallel with the environment setup;
  - pseudo ≈ 13.9k renders (≈ 35 min) + scoring (≈ 40 min);
  - training ≈ 10 min;
  - eval 2 × 15 min.
  - **Total ≈ 2.5–3 h, ≈ $10–12.**
- **Deletion:** the pod is terminated after the results (and, if adopted, the checkpoint) are copied home and hash-checked.
