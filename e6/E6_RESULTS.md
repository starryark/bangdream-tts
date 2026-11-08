# E6 results: Our Notes data, 16 new voices, VC and `[nv]` arms (2026-11-08)

Tags: [L] measured, [I] inference. Plan, deviations and amendments 1–10: `E6_PLAN.md` (§5 guards and §6 rule fixed before training).

**Verdict under the pre-registered rule: v2 is not adopted. v1 (E1b epoch 2, 49 voices) stays the model, and nothing is published.** [L]
- No E6a candidate passes G1–G3, so there is no E6a\*.
- E6a fails on Mandarin identity, Japanese CER and coverage as well as accent, so E6d is not triggered.
- E6b and E6c can only replace an E6a\* (§6 steps 2–3), so their results can't release a model either. E6b's epoch-1 candidate failed only one guard.

**Compute and cost.** [L]

| Pod | GPU | Window | What ran | Cost |
|---|---|---|---|---:|
| RVC converters | RTX 4090 ($0.69/h) | 2026-11-07 22:50 → 11-08 01:24 | 5 launches (4 failed on script faults), ~2.5 h in total | ~$1.8 |
| Main | H100 SXM ($3.99/h), `dsrojh8on3fmaz` | 2026-11-08 01:43 → 10:07 | ~8.4 h | ~$33.5 |

- On the main pod, the upload took 75 min.
- Bootstrap then stopped twice, at the replay shortfall (amendment 9) and at duplicate clip ids (amendment 10). Both were fixed on the pod without a new upload; about 30 min was idle.
- Training plus evaluation of 7 conditions and 2 reference sets took 5.4 h (04:41 → 10:07).
- Total ≈ **$35**, against the $40–55 estimate.

**Where the numbers live.**
- `results/eval/selection.{json,md}` (from `e6_select.py`).
- `results/eval/<cond>/{scored.jsonl,summary.json}` for C0, E6{a,b,c}_ep{1,2}, `real` and `floor_on`.
- `results/exp/<arm>/mix.json`, `results/logs/`, `results/pseudo_*`, `results/vc/`.
- No checkpoints were downloaded. `download` pulls only a released checkpoint, so every E6 checkpoint, E6b ep2 included, was deleted with the pod. A follow-up would retrain.

## Data that went in [L]

| Data | Count |
|---|---|
| Japanese | 15,514 unique train rows over 65 voices (54 repeats collapsed, amendment 10) |
| Replay | 2,526 rows per arm, from a 6,143-row pool (zh 5.5 h, 70 speakers; en 2.5 h) |
| E1b-recipe pseudo zh for the 37 kept v1 voices | 374 rows, identical to E1b's |
| New Our Notes pseudo zh | 460 rows (bar 0.3896) |
| E6b VC rows | 223: E5's 187 kept, plus 36 new (arale 20, nonoka 7, hotaru 8, yomogi 1) |
| E6c `[nv]` rows | 765 over 25 voices |

- The four new converters were trained to epoch 200 (arale and nonoka locally; hotaru and yomogi on a pod after three local resets).
- Their gate pass rates were 0.92 / 0.31 / 0.38 / 0.04, and the user judged all four listening sets native and in character.

## Selection (§6) [L]

`selection.md`, with columns as defined in §5:
- v1 = the 37 unchanged voices.
- ON = the 28 Our Notes voices.
- VC4 = arale, nonoka, hotaru and yomogi.
- replaced = the 12 MyGO / Ave Mujica voices.

| cond | v1 ja-mid leak | v1 ja-short leak | v1 zh leak | v1 zh SER | v1 zh lid | v1 en lid | all ja CER | ON ja-mid leak | ON zh leak (Base-routed) | VC4 zh leak | replaced ja-short leak | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 (v1) | 0.010 | 0.355 | 0.160 | 0.074 | −18.17 | −9.11 | 0.032 | 0.206\* | 0.406\* | — | 0.333 | |
| E6a ep1 | 0.013 | 0.351 | 0.219 | 0.070 | −19.54 | −9.12 | 0.035 | 0.021 | 0.038 | 0.672 | 0.071 | G1 zh leak, G1 per-row, G2 ja CER, G3 per-row |
| E6a ep2 | 0.014 | 0.334 | 0.211 | 0.066 | −19.27 | −8.71 | 0.039 | 0.021 | 0.040 | 0.695 | 0.074 | G1 zh leak, G1 per-row, G2 ja CER, G3 per-row |
| E6b ep1 | 0.010 | 0.372 | 0.142 | 0.065 | −19.62 | −9.17 | 0.035 | 0.032 | 0.040 | 0.445 | 0.111 | G3 per-row |
| E6b ep2 | 0.009 | 0.327 | **0.096** | 0.077 | −17.97 | −8.60 | 0.033 | 0.018 | 0.013 | **0.352** | 0.103 | G2 per-row zh SER, G3 per-row |
| E6c ep1 | 0.013 | 0.329 | 0.209 | 0.068 | −19.55 | −8.54 | 0.037 | 0.026 | 0.061 | 0.695 | 0.090 | G1 zh leak, G1 per-row, G2 ja CER, G3 per-row |
| E6c ep2 | 0.019 | 0.355 | 0.242 | 0.061 | −19.69 | −8.68 | 0.039 | 0.020 | 0.060 | 0.688 | 0.077 | G1 zh leak, G1 per-row, G2 ja CER, G3 per-row |

\* C0 has only the 12 replaced voices under their v1 names; v1 has no voice for the 16 new ones.

**What each failure is** [L]:
- **G1, zh leak, 37 unchanged voices.**
  - E6a and E6c lose Mandarin identity on the voices v1 already struggled to keep in Mandarin (E1b's Base-fallback set).
  - E6a ep1 per-voice zh leak, C0 → E6a: pp_eve 0.22 → 0.69, hhw_hagumi 0.16 → 0.56, ppp_rimi 0.50 → 0.78, ro_rinko 0.63 → 0.84.
  - Pooled over the 37: 0.160 → 0.21–0.24, against a bound of +0.03.
  - E6b, which carries E5's own-voice VC rows for these voices, has no G1 failure at either epoch. Its pooled leak is 0.142 (ep1) and 0.096 (ep2).
- **G2 ja CER ≤ 0.035 over all 65 voices.** E6a ep1 0.0354, ep2 0.0386; E6c 0.037–0.039; E6b 0.035 / 0.033 (pass).
  - The new voices carry it. VC4's ja CER is 0.07–0.10, and the 12 replaced voices were already at 0.057 when rendered by v1 (C0).
  - On the 37 unchanged voices CER is 0.025–0.028, the same as C0's 0.025.
- **G2, per-voice zh SER ≤ 0.15 (E6b ep2 only):** hhw_kokoro 0.166, pp_eve 0.174, ras_lock 0.163.
- **G3, per-voice ja-mid leak ≤ 0.20 on the Our Notes voices.**
  - ikka_yomogi fails in every arm except E6b ep2: 0.25 / 0.275 (E6a), 0.475 / 0.125 (E6b), 0.30 / 0.325 (E6c).
  - mewtype_arale fails in E6a ep1 (0.225) and E6b ep2 (0.25).
  - The leaks spread across the new voices' neighbourhood rather than one look-alike. Yomogi → arale, hotaru, houka, raika, chieri; arale → houka, hina, chieri.
  - The bound means ≥ 9 of 40 renders off-voice. The exemptions (Rāna, and the Mutsumi↔Mortis and Sakiko↔Sakiko-tsuki pairs) are applied as §5 says.

## What E6 did achieve (not shippable under the rule) [L]

- **Coverage of the Our Notes voices.**
  - On the 12 replaced voices, ja-mid leak fell from 0.206 to 0.00–0.01, and Mandarin leak from 0.41 to 0.02–0.07.
  - All 28 Our Notes voices pooled reach ja-mid leak 0.02–0.03.
  - C1 (the short-line claim, not a gate) would hold in every candidate: replaced-voice ja-short leak fell from 0.333 to 0.07–0.11, a drop of at least 0.22 against the 0.10 needed.
- **E6b ep2 is the strongest condition.**
  - Mandarin leak on the 37 unchanged voices is 0.096, against v1's 0.160. The ON Base-routed voices are at 0.013 and VC4 at 0.35, against 0.67–0.70 without VC rows.
  - Japanese is unchanged on the 37.
  - It misses on 3 voices' zh SER (by 0.013–0.024) and on arale's Japanese coverage (0.25).

## Pre-registered prediction (§2) [L]

The prediction was that E6a moves accent on the 37 unchanged voices the wrong way (zh lid, en lid and zh SER up), more at ep2. **It failed for Mandarin.**
- zh lid fell (more native): −18.17 → −19.54 / −19.27.
- zh SER fell: 0.074 → 0.070 / 0.066.
- en lid was flat at ep1 (−9.12) and rose 0.40 at ep2 (−8.71), the one part that moved as predicted.
- E6a's failure was identity, not accent.

## What this means [I]

- Adding 16 voices costs the weakest v1 Mandarin voices their identity unless they have own-voice Mandarin rows.
  - E6b's VC rows are what kept G1.
  - Any future run should treat E5's VC rows as part of the base recipe, not as an arm.
- The rule needed an E6a\* before E6b could ship. With VC rows as the baseline, E6b ep1 would have failed only on yomogi's Japanese coverage.
  - That is the design lesson for a next plan, not a reason to reinterpret this one.
- Yomogi is the hard new voice.
  - Her converter passed only 1 of 28 Mandarin conversions.
  - Her Japanese leaks into the other new voices at 0.25–0.48 in five of six candidates.
  - Supplementing her data, or reporting her as weak the way §1 treats Rāna, would have to be fixed in a new plan before any run.
- The G2 CER bound was set on E1's 49 voices. The Our Notes text, with its 〜, ——, 『』 and band names, raises CER: the 12 replaced voices already sit at 0.057 with v1. Whether that is pronunciation or normalisation hasn't been checked.

## Also reported (§6), from the scored files [L]

- **Our Notes Mandarin vs the floor** (`floor_on`: Base, x-vector only, E6a ep1's vectors, Mandarin, 28 voices).
  - The floor's pooled zh leak is 0.094. It is worst on yomogi 0.72, nonoka 0.78, arale 0.41 and hotaru 0.41, which confirms §3.6's premise for the VC arm.

    | cond | pooled Our Notes zh leak |
    |---|---|
    | E6a | 0.138 / 0.140 |
    | E6b | 0.113 / **0.068** |
    | E6c | 0.160 / 0.156 |

  - Only E6b ep2 beats the floor overall. In the other arms, 6–8 voices end up more than 0.10 worse than the floor, typically yomogi, arale, hotaru, Sakiko, Tomori, Mutsumi and Mahoro. Nonoka is always better than the floor.
  - In E6b ep2 only mujica_sakiko is worse than the floor, and arale and nonoka are better. [I] Japanese-only training pulls Our Notes voices' Mandarin away from what Base did with the same vector, and VC rows offset it.
- **`[nv]` in E6c** (§6 step 3 thresholds; 520 renders per candidate, 8 per voice).
  - ≤ 2 kana by ASR: 0.97 / 0.96.
  - 0.3–2.0 s: 0.96 / 0.93 (median 1.12 / 1.28 s).
  - No runaways.
  - The duration share is on untrimmed renders, because the audio was deleted with the pod. Trimmed durations are shorter, so the share at or under 2.0 s can only rise, but some renders could drop below 0.3 s.
  - E6a, which never saw `[nv]` and was given the literal text `……っ`, scores 0.96 / 0.97 and 0.98 / 0.97 (median 0.88 / 0.96 s), with no runaways. [I] The token works, but on these metrics it doesn't beat the literal text.
- **Not analysed:** within-set vs cross-set leak, gate removal per expression class, and F0 range and duration ratios. The trimmed-audio measures need the renders, which were deleted with the pod; the leak splits can still be computed from `results/eval/*/scored.jsonl`.

## Published model

Unchanged: `starrydark/BanG_Dream_Qwen3_TTS` stays at v1 (E1b epoch 2). No v1 tag was created and no files were uploaded, since `publish_hf_v2.py` refuses without a released checkpoint.
