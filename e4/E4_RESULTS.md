# E4 results: per-voice RVC feasibility (Kasumi, Eve), epoch 200 (2026-10-31)

Tags: [L] measured, [I] inference. The plan and amendments B and C are in `E4_PLAN.md`; all of them were written before the results they govern.

**Status: feasibility PASSED for both voices under Amendment C.** The original gate fails as written, and the accent metrics are not evaluable at these pitches.

**Where the numbers live.** The report is `vc/report_ep200.json`; the rows are `vc/scored_ep200.jsonl`; the log is `e4_score_ep200.log`.

**Run.**
- Everything ran locally on the RTX 5060 Ti, at $0.
- Training took about 2 h 40 min for Kasumi and 3 h 25 min for Eve.

**F0 targets.** These are pooled rmvpe medians over each voice's training slices:

| Voice | Target (rmvpe) | pyin check |
|---|---|---|
| Kasumi | 452.7 Hz | 463 Hz |
| Eve | 499.9 Hz | 513 Hz, 10–90th percentile 363–709 |

The sources sit near 300 Hz, so the per-clip shifts were:
- Kasumi: +2 to +12 semitones;
- Eve: +3 to +12 semitones.

## Results [L]

| Cell | Voice | n | Pass (as written) | pred == row | Codec pred == row | Codec ratio ≥ 0.5 | lid (median) | SER (mean) | CER |
|---|---|---|---|---|---|---|---|---|---|
| source | Kasumi | 20 | 0.00 | 0.00 | 0.00 | 0.00 | −28.8 | 0.036 | 0.000 |
| shift only | Kasumi | 20 | 0.05 | 0.40 | 0.20 | 0.05 | −19.4 | 0.139 | 0.000 |
| VC, no shift (ep200) | Kasumi | 20 | 0.15 | 0.75 | 0.80 | 0.40 | −22.3 | 0.136 | 0.017 |
| **VC + shift (ep200, gated)** | Kasumi | 20 | **0.20** | 1.00 | 1.00 | **0.95** | −14.3 | 0.167 | 0.014 |
| VC + shift (ep100) | Kasumi | 20 | 0.10 | 0.95 | 0.95 | 0.40 | −15.5 | 0.172 | 0.010 |
| source | Eve | 18 | 0.00 | 0.17 | 0.00 | 0.00 | −26.4 | 0.047 | 0.000 |
| shift only | Eve | 18 | 0.00 | 0.39 | 0.44 | 0.17 | −15.1 | 0.210 | 0.008 |
| VC, no shift (ep200) | Eve | 18 | 0.00 | 0.72 | 0.50 | 0.00 | −14.9 | 0.200 | 0.032 |
| **VC + shift (ep200, gated)** | Eve | 18 | **0.00** | 1.00 | 1.00 | **0.61** | −9.4 | 0.264 | 0.012 |
| VC + shift (ep100) | Eve | 18 | 0.00 | 1.00 | 1.00 | 0.61 | −9.8 | 0.318 | 0.022 |

The scoring is not perfectly reproducible. The same shift-only files gave a Kasumi SER of 0.143 in the Amendment B run and 0.139 here. A difference of about ±0.005 is batching noise, not a change in the audio.

## Gate outcomes

| Test | Kasumi | Eve |
|---|---|---|
| Pre-registered gate, as written | **fails**: pass rate 0.20 is met, but lid −14.3 > −19 and SER 0.167 > 0.10 | **fails**: pass rate 0.00 |
| Is accent evaluable? (Amendment B: shift cell must stay at lid ≤ −19 and SER ≤ 0.10) | **no**: shift SER 0.139 | **no**: shift lid −15.1, SER 0.210 |
| C1(a): clips passing identity + text + length ≥ 20% | **0.80, pass** | **0.78, pass** |
| C1(b): the user hears c and d at epoch 200 as native | **native, pass** | **native, pass** |
| C3: codec identity ≥ 40% (branch condition) | **0.95, pass** | **0.61, pass** |
| C5: VC no shift minus shift only, pred == row (must be > 0.10) | 0.75 − 0.40 = **+0.35** | 0.72 − 0.39 = **+0.33** |

C5 means RVC adds identity beyond pitch. The codec version is weaker for Eve: 0.50 against 0.44.

**Overall.** The gate as originally written fails for both voices. Under Amendment C, both voices pass every condition, including C1(b).

## Observations

- **Identity matured between epochs 100 and 200 for Kasumi.** Codec ratio ≥ 0.5 went 0.15 (ep50) → 0.40 (ep100) → 0.95 (ep200). Eve was already at 0.61 by epoch 100 [L]. The epoch-50 worry in Amendment B is resolved for Kasumi.
- **Kasumi's accent costs are no longer additive at epoch 200.** An additive model predicts SER ≈ 0.24 for VC + shift; the measured value is 0.167. Compared with shift only, VC + shift adds just +0.03 SER [L]. Eve is sub-additive too: 0.264 measured against a predicted 0.36.
- **Text failures:**
  - Five clips fail CER: 3 for Kasumi, 2 for Eve.
  - Four of the five ASR hypotheses are near-homophones (途/图, 刚/钢, 建/俊).
  - One is **变阵 biànzhèn → 辩证 biànzhèng in both voices**, on the same sentence, whose source had CER 0. That is an -n → -ng confusion, a coda contrast Japanese lacks. It is sentence 1 in both listening folders [L/I].
- **Eve's no-shift cell is poor on the metrics:** lid −14.9, SER 0.20, and only 17% pass the accent gate. Her generator was trained at around 500 Hz, so at around 300 Hz it runs out of range [I].
  - This undercuts C4's proxy, which is the no-shift twin. For voices as high as Eve, C4 would discard most rows.
  - The user's verdict on Eve's c clips tells whether this is audible or only a metric effect.

## Listening verdict (user, 2026-10-31, epoch 200; blind to scores)

> "c and d sound like native mandarin, d sound alike the character"

The user gave one answer covering both folders (`listen/ep200/ppp_kasumi/` and `listen/ep200/pp_eve/`), so it is read as applying to both voices. It is one rater and 5 sentences per voice; this time the README showed no scores.

- **C1(b) passes for Kasumi and for Eve.** With C1(a) and C3 also passing, E4 feasibility passes, and the 10-voice branch is open.
- **Eve's c clips sound native**, even though they score lid −14.9 and SER 0.20. So even at the source's pitch, the converted audio throws the metrics off. C4's no-shift twin is therefore not a valid accent proxy either [L/I], and the branch's row-level accent gate must be re-planned before it is used.
- **d sounds like the character** for both voices. That agrees with the probe: pred == row 1.00, both raw and after the codec.

Next step: a pre-registered plan (E5) for converters and rows for the remaining voices, then a retrain. This is a new decision because of time and spend.
