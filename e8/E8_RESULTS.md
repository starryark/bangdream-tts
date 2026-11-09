# E8 results: learnable speaker rows and averaged checkpoints, judged by listening (2026-11-08/09)

Tags: [L] measured or the listener's judgment, [I] inference. Plan: `E8_PLAN.md` (fixed before launch).

**Round 2 did not produce a verdict under the plan, and v1 stays meanwhile.** [L]
- Round 2 failed its consistency checks: 4 of 8 swapped repeats agreed in sign (≥ 6 needed), and one null pair was rated 2 (bound ±1).
- Under §6.7 it therefore does not decide: the plan calls for one repeat with fresh randomisation, and the repeat decides. Until then v1 stays by default.
- Separately, the primary criterion was not met in this round: the replaced (MyGO!!!!! / Ave Mujica) voices were not better than v1 in Japanese (mean CCR −0.46, one-sided p = 0.80).
- **Recommendation [I]: the repeat is unlikely to change the outcome.** It could only lead to adoption if the replaced voices turned significantly better. Anon and Umiri were preferred in v1 on every item, and Anon's repeat agreed.
- **Whether to run the repeat or skip it is the user's call.** (The analysis code's note said a repeat applies only when consistency is the *only* failure; the plan's wording governs.)

**Cost:** one H100, pod `jc6hj3x69268j5`, 14:42 → 19:13, ≈ 4.5 h, **≈ $18**. The listener spent about 45 min: Round 1 took 5 min and Round 2 took 31 min. Running total for E6–E8: about $66 (RVC $1.8, E6 $33.5, E7 $12.4, E8 $18).

## What ran [L]

- **Arms:**
  - A: E7a's data and seed.
  - B: the same plus learnable speaker rows (`--spk_lr 1e-4`).
  - Each arm's candidate was the weight-average of its epoch 1.0, 1.5 and 2.0 checkpoints.
- **B's offsets at epoch 2:** |delta| mean 0.176 (max 0.235) against a row norm of 17.3.
  - The five most crowded pairs each moved apart slightly, e.g. Eve–LOCK 0.824 → 0.806 and Kasumi–Himari 0.787 → 0.762.
  - The median nearest-neighbour centred cosine went 0.658 → 0.641.
- **Final validation loss:** A 2.3208 (identical to E7a, same seed), B 2.3192.
- **Screens (§5):** both `avg` candidates passed every one.

  | | A | B |
  |---|---|---|
  | Our Notes ja-mid leak, 80 renders (real audio 0.027) | 0.016 | 0.017 |
  | v1 zh leak (C0 0.160) | 0.128 | 0.144 |
  | v1 zh SER (C0 0.074) | 0.078 | 0.074 |
  | v1 ja CER (C0 0.025) | 0.025 | 0.026 |
  | Our Notes ja CER (real 0.042) | 0.051 | 0.051 |
  | yomogi (the only voice > 0.10) | 0.225 | 0.225 |

  Averaging helped on the automatic measures. Our Notes leak fell to below the real-audio level (E7a's single epochs were at 0.023–0.025), and Mandarin leak on the v1 voices is below v1's.
- **Downloaded:** `models/E8A_avg`, `models/E8B_avg`, all 352 listening renders (`e8/listen_renders/`), the eval scores and the pseudo data.

## Round 1: A vs B on crowded voices [L]

- **Result:** 20 trials; mean preference for B +0.25; no "wrong voice" flags; consistency passed (2/2 repeats agree, null rated 0). **B won under the rule.**
- **It was narrow:**
  - B preferred on 9 items, A on 8, 3 ties.
  - The scores were −3 ×2, −1 ×6, 0 ×3, +1 ×5, +3 ×4.
- **Per voice (summed preference for B):**

  | voice | sum |
  |---|---|
  | mewtype_arale | +7 |
  | ournotes_kanata | +4 |
  | ikka_yomogi | +3 |
  | ikka_chieri | 0 |
  | millsage_nagi | 0 |
  | mewtype_yuno | −1 |
  | ikka_raika | −4 |
  | mewtype_nonoka | −4 |

  The learnable rows helped most on the most confused pair, arale and yomogi [I].

## Round 2: B vs v1 (C0) [L]

**Paired comparisons** (+ means the new model is preferred, − means v1 is):

| group | n | mean | p (one-sided, >0) | p (two-sided) | rule |
|---|---|---|---|---|---|
| Japanese, replaced voices | 24 | −0.46 | 0.80 | 0.39 | needed > 0, p < 0.05: **fail** |
| Japanese, unchanged voices | 24 | +0.04 | 0.36 | 0.71 | not worse: pass |
| Mandarin | 16 | −0.31 | 0.76 | 0.48 | not worse: pass |
| English (reported only) | 8 | −0.13 | | 1.00 | |

**Per voice** (held-out, novel and short line, then the sum):

| voice | held-out | novel | short | sum |
|---|---|---|---|---|
| mujica_umiri | −3 | −2 | −2 | **−7** |
| mygo_anon | −3 | −2 | −2 | **−7** (repeat −2) |
| mujica_mutsumi | −2 | −1 | +2 | −1 |
| mujica_nyamu | +2 | −2 | −1 | −1 |
| mujica_uika | −2 | +2 | −1 | −1 |
| mygo_tomori | −1 | −1 | +2 | 0 |
| mujica_sakiko | −2 | +2 | +3 | +3 |
| mygo_taki | +2 | +2 | −1 | +3 |

- Anon and Umiri are consistent regressions.
- The unchanged voices sum between −3 (Aya, Kasumi) and +4 (Yukina), with no pattern.
- By line type (Japanese): held-out −0.25, novel +0.06, short −0.44.

**No "wrong voice" flags anywhere.**

**Consistency failed:**
- The repeats that disagreed were Kasumi short, Uika zh, Taki held-out and Sakiko zh. They concentrate in Mandarin and short lines, where the two models are hard to tell apart.
- The null pair rated 2 was Chuchu.
- Where the difference is large (Anon), the repeat agreed.

**New voices (absolute check against a real clip): all 16 pass, none weak.**
- 14 voices were "yes" on at least 2 of 3 renders, and 9 were "yes" on all 3.
- ikka_yomogi: yes / partly / partly.
- ournotes_kanata: yes / yes / no.

## What this means [I]

1. **The 16 new voices work.** All 16 were judged in character against real clips, including yomogi, whom every automatic gate in E6–E8 flagged. That is the clearest gain of the whole v2 effort, and the automatic metrics never showed it.
2. **Replacing the MyGO!!!!! / Ave Mujica training data was not an improvement as heard.**
   - Anon and Umiri are clearly worse to this listener than in v1; the others are mixed.
   - The cause is not one thing:
     - **Umiri**'s v1 voice is anime audio (3 min). Anime vs game direction, with the listener's reference being the anime, could explain her.
     - **Anon**'s v1 voice is Girls Band Party game audio, so she went game → game. Her regression is unexplained by source: a different game, years apart, possibly different direction.
   - The automatic probe can't see this. It measures "which of 65 voices", not "the version of the character the listener expects".
3. **On the 37 unchanged voices v2 is a tie with v1:** no regression, no gain.
4. **The paired test was harder than planned, and two explanations fit.**
   - Many v2-vs-v1 pairs are near-identical, so choices on them are close to chance.
   - Round 2 also went fast: 84 pairs and 48 checks in 31 min, about 14 s per trial including replays, against the ~30 s budgeted.
   - This data can't separate the two. A repeat would need slower listening, headphones and one session per file.
5. **B's learnable rows are a small, real step for crowded voices** (Round 1: arale +7, yomogi +3). The learning rate let the rows move less than 1% of their norm; a stronger setting could do more.

## The design lesson for a next plan

What the listening supports is "v1's voices as they are, plus the 16 new voices", not "replace MyGO!!!!! / Ave Mujica with Our Notes".
- A next plan could keep v1's data for the 49 existing voices, or let each replaced voice keep whichever version the listener prefers.
- It could add the 16 Our Notes voices, with learnable rows and checkpoint averaging.
- It could be judged by listening: non-inferiority to v1 on shared voices, plus the absolute check for the new voices.

That is a decision for the user. It reverses D1's "replace" for at least Anon and Umiri.
