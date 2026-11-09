# E9 results: v1 voices kept, Our Notes added per era, 16 new voices (2026-11-09)

Tags: [L] measured or the listener's judgment, [I] inference. Plan: `E9_PLAN.md`, fixed before launch.

## Verdict: not adopted, v1 stays [L]

E9 passed every screen, but two of the pre-registered listening rules (§6) failed:
- **The v1-era voices were worse than v1.** Over 33 items the mean CCR was −0.58 (two-sided Wilcoxon p = 0.046).
- **One character summed to ≤ −5:** Uika at −7.

| Rule (§6) | Result | Pass |
|---|---|---|
| 1. Screens | all pass | yes |
| 2a. v1-era voices not worse (33) | mean −0.58, p = 0.046 | **no** |
| 2b. No character's three items sum to ≤ −5 | Uika −7 | **no** |
| 3. Persona + zh + unchanged not worse (18) | mean +0.11, p = 0.76 | yes |
| 4. No voice flagged "wrong voice" twice | no flags at all | yes |
| 5. Weak new voices ≤ 3; weak Our-Notes-era ≤ 2 | 0 and 0 | yes |
| 6. Consistency | repeats 4/6 agree; nulls 0, 0, 0 | yes (at the threshold) |

Nothing is published to Hugging Face, and the Space keeps serving v1.

**Cost [L]:** one H100, pod `wwtyz192612cp5`, 01:18 → 03:41, about $9.5. Running total for E6–E9: about $75 (RVC $1.8, E6 $33.5, E7 $12.4, E8 $18, E9 $9.5).

## What ran [L]

- **Recipe:** E8's winner (B: `--spk_lr 1e-4`, average of the epoch 1.0/1.5/2.0 checkpoints) on E9's speaker set:
  - 72 rows (74 ids with the two aliases);
  - 16,790 training clips after pruning 244 speaker outliers.
- **Final validation loss:** 2.3143.
- **Learned speaker offsets:** |delta| mean 0.178 against a row norm of 17.3. The median nearest-neighbour centred cosine went 0.647 → 0.635.
- **Screens (§5), `E9_avg`: all pass.**

  | Screen | E9_avg | Bound |
  |---|---|---|
  | S1 gated ja-mid leak (new + Our-Notes-era + single ids, all seeds) | 0.0155 | ≤ 0.06 |
  | S2 gated voice > 0.40 | none (yomogi 0.175, the only one > 0.10) | none |
  | S3 v1 ja-mid leak | 0.0112 | ≤ C0 0.0099 + 0.03 |
  | S3 v1 zh leak | 0.138 | ≤ C0 0.155 + 0.10 |
  | S4 v1 ja CER | 0.0258 | ≤ C0 0.0251 + 0.01 |
  | S4 Our Notes ja CER | 0.0541 | ≤ real 0.042 + 0.02 |
  | S5 v1 zh SER | 0.0713 | ≤ C0 0.0733 + 0.02 |
  | S6 runaway | 0.0001 | ≤ 0.01 |

- **Reported, not gated:** the share of each v1-era id's renders that the probe assigned to its own character was 0.91–1.00, except Anon 0.56 and Taki 0.46. The probe's classes for those two are fitted on Our Notes data.

## Listening [L]

**Pace:**

| Page | Trials | Time | s / trial |
|---|---:|---:|---:|
| p1 | 30 | 12.8 min | 26 |
| p2 | 30 | 7.5 min | 15 |
| abs | 50 | 13.0 min | 16 |

About 30 s per trial was requested. The consistency checks passed, so the data stands under the plan.

### Pair trials: E9 vs v1, same voice, text and seed (CCR −3…+3, + favours E9)

| Group | n | Mean | p (two-sided) |
|---|---:|---:|---:|
| v1-era voices, Japanese | 33 | **−0.58** | **0.046** |
| persona (`mygo_soyo_tsuki`, `mujica_sakiko_tsuki` through the alias) | 4 | +1.00 | 0.50 |
| Mandarin (6 v1-era voices + Eve, Hagumi) | 8 | −1.00 | 0.27 |
| unchanged voices, Japanese | 6 | +1.00 | 0.13 |

**Per v1-era character** (sum of three items: held-out v1 line, novel sentence, short line):

| Character | Sum | Character | Sum |
|---|---:|---|---:|
| Uika | **−7** | Mortis | −2 |
| Anon | −4 | Soyo | −1 |
| Mutsumi | −3 | Nyamu | −1 |
| Sakiko | −3 | Taki | +1 |
| Rāna | −3 | Tomori | +2 |
| | | Umiri | +2 |

In E8, which replaced the v1 data with Our Notes data, Anon and Umiri were −7 each. With both eras merged, Umiri recovered to +2, while Anon improved to −4 but stays negative.

### Absolute: does it sound like the character next to a real clip? (yes / partly / no)

- **The 9 Our-Notes-era ids:** all passed. 16 of 18 answers were "yes" and 2 were "partly" (Uika, Taki).
- **The 16 new voices:** all passed, with none weak.
  - "yes, yes": 9 voices.
  - "yes, partly": 3 voices (Chieri, Miku, Kanata), plus Yomogi "partly, yes".
  - Miyako: "partly, partly".
  - Nonoka and Mahoro: "yes, no". The rule counts a voice as weak only with no "yes" and at least one "no".

## Exploratory (not part of the rule)

- **[L] The regression sits in the held-out v1 game lines:**

  | Line type | Mean | p | Items worse |
  |---|---:|---:|---:|
  | held-out | −1.45 | 0.008 | 9 of 11 |
  | novel sentence | −0.27 | 0.73 | — |
  | short line | 0.00 | 0.86 | — |

  - The held-out effect shows on both pages: p1 −2.17 (6 items), p2 −0.60 (5 items).
  - p1, the page listened to more slowly, was more negative for every line type.
- **[L] Mandarin, 5 of 8 worse, not significant:**
  - worse: Soyo −3, Hagumi −3, Uika −2, Sakiko −2, Eve −1;
  - better: Anon +3, Tomori +1.

  The pooled rule 3 passed. The automatic Mandarin screens were at or better than v1 (leak 0.138 vs 0.155, SER 0.071 vs 0.073).
- **[L] Consistency passed at exactly the threshold.** 4 of 6 swapped repeats agreed in sign. The repeat of Uika's novel sentence agreed (−3, then −1).
- **[I] Why the v1-era ids lost:**
  - They train on the same v1 data as v1, and the unchanged voices did not lose (+1.00). So the recipe alone is an unlikely cause.
  - The likelier cause is sharing: each of these characters now also has an Our-Notes-era id trained in the same model. Some of the Our Notes delivery appears to carry into the v1-era id.
  - It shows most on held-out game lines, where an in-character delivery is easiest to judge. This is a hypothesis; E9 does not test it.

## What E9 does establish

- **All 16 new voices and all 9 Our-Notes-era ids sound like their characters** (absolute check, no weak voice).
- **The persona aliases and the unchanged voices were not worse than v1.**

They cannot ship as v2 under this plan, because the v1-era ids of the 11 shared characters regressed.

## Decision after the verdict (user, 2026-11-09)

**Hybrid release.** This is a new decision, separate from the rule; v1 stays the default everywhere.
- **Hugging Face:** `main` stays v1 (49 voices). `E9_avg` is published as the separate, non-default branch `ournotes`, recommended only for its 25 passing voices: the 16 new voices and the 9 `_ournotes` ids.
- **Demo Space:** each voice is served from v1 if v1 has it, otherwise from `ournotes`. So the 49 v1 voices come from v1 and the 25 from E9.
