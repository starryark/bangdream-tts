# E5 results: RVC-converted own-voice Mandarin rows for 15 voices, then a retrain (2026-11-06)

Tags: [L] measured, [I] inference. Plan and amendments: `E5_PLAN.md` (§6's rule fixed before training).

**Verdict under the pre-registered rule: E5 is not adopted. E1b epoch 2 stays the model.** Both E5 checkpoints are eligible (every guard passes), but the pick misses all three metric adoption conditions (§6 1–3), so condition 4 (target listening) was not triggered.

E5 nonetheless moves Mandarin identity the way it was designed to: pooled zh leak 0.155 → 0.088, voices with own-voice Mandarin 34 → 40–41 of 49, targets 0.41 → 0.20–0.22, with Japanese and English unchanged at epoch 2.

**Compute.** One Secure H100 (pod `7blemrvqfcdnaj`, $3.99/h), 2026-11-06 09:29 → 20:24 UTC, about 10.9 h, **about $44 against the $8–12 estimate**. About 9 h of that was idle: the data stage finished at 10:12 and training waited on a release question to the user, because the queue was not chained to start automatically. The work itself (upload, data rebuild, training, two evals) took about 2 h. The pod was terminated after both checkpoints were copied home and their sha256 verified. The converters cost $0 (local RTX 5060 Ti).

**Where the numbers live.** `results/selection_e5.{json,md}` (from `e5_select.py`), `results/eval/E5_ep{1,2}/{scored.jsonl,summary.json}`, `results/logs/`, `results/mix.json`, `results/pod_rows.txt`. Checkpoints: `models/E5_ep1/`, `models/E5_ep2/` (each with `SHA256SUMS`).

## Listening gate for the rows (§4) [L]

User, 2026-11-06: "all voices in e5/listen sound native and like the character". All 15 voices passed, so all 218 gated VC rows went in. One verdict covered all 15 sets in one batch. Ako's set was not fully blind (disclosed in the plan).

## Data rebuild on the pod (§5) [L]

- **Aux filter:** dropped 0 of 55 speakers, the same as E1. `aux_filter.json` differs from E1's only in third-decimal cosines (e.g. 0.402 vs 0.401), which is GPU nondeterminism.
- **E1b pseudo rows regenerated:** 573 against E1's 583. Seven voices are off by more than 3 (disclosed, not corrected, as pre-registered): ag_moca −4, ag_tsugumi −5, pp_chisato −4, pp_maya +4, hhw_kaoru −4, mujica_mortis +6, mujica_sakiko −5.
- **Mix:** 11,538 train / 219 val (E1b: 11,341 / 208). Japanese 9,268 and replay 1,509 are identical to E1b; pseudo 761 (E1b 564), i.e. 6.6% of training (E1b 5.0%). VC rows: 218 (207 train, 11 val).
- **Training:** E1b's command unchanged; 361 steps/epoch, checkpoints at steps 361 and 722 (epochs 1 and 2). Epoch-1 val loss 2.390 (E1b 2.397).

## Selection (§6) [L]

| cond | ja leak | ja CER | ja p_true | zh SER (34) | zh lid (34) | zh leak | voices zh leak ≤ 0.15 | target zh leak | en lid (targets) | en leak | en WER | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1b_ep2 (model) | 0.014 | 0.028 | 0.802 | 0.065 | −19.55 | 0.155 | 34/49 | 0.410 | −9.34 (−7.93) | 0.040 | 0.003 | |
| E5_ep1 | 0.025 | 0.030 | 0.761 | 0.058 | −19.88 | **0.088** | 41/49 | 0.221 | −9.66 (−7.82) | 0.064 | 0.005 | yes |
| E5_ep2 | 0.014 | 0.025 | 0.815 | 0.069 | −19.55 | **0.088** | 40/49 | 0.202 | −9.54 (−7.85) | 0.040 | 0.003 | yes |

**Adoption conditions** (pick = E5_ep1): pooled zh leak ≤ 0.08: **no** (0.088); ≥ 43/49 voices ≤ 0.15: **no** (41); targets ≤ 0.15: **no** (0.221). Condition 4 was not reached.

**Tie.** E5_ep1 and E5_ep2 have exactly the same pooled zh leak, 138 of 1,568 renders. The rule has no tie-break; `min()` returns ep1 because it is listed first. With E1's tie-break (Japanese p_true) the pick would be ep2 (0.815 vs 0.761). Either way the adoption conditions fail: ep2 has 40/49 and targets 0.202.

## Targets, per voice [L]

zh leak, then shared-character SER, then mean lid, for E1b_ep2 / E5_ep1 / E5_ep2 (32 zh renders each):

| Voice | zh leak | SER | lid | VC rows |
|---|---|---|---|---|
| ppp_rimi | 0.75 / 0.44 / 0.34 | 0.087 / 0.127 / 0.116 | −13.0 / −10.5 / −10.2 | 2 |
| ppp_kasumi | 0.75 / 0.72 / 0.69 | 0.066 / 0.101 / 0.099 | −18.4 / −16.3 / −15.3 | 15 |
| ro_ako_maou | 0.69 / 0.59 / 0.47 | 0.044 / 0.042 / 0.041 | −23.3 / −23.7 / −21.8 | 12 |
| mygo_tomori | 0.56 / 0.50 / 0.44 | 0.067 / 0.060 / 0.065 | −19.1 / −19.4 / −19.5 | 14 |
| mygo_soyo_tsuki | 0.56 / 0.16 / 0.16 | 0.083 / 0.120 / 0.144 | −16.5 / −14.9 / −12.1 | 20 |
| ras_lock | 0.50 / 0.19 / 0.16 | 0.124 / 0.154 / 0.171 | −12.6 / −9.5 / −9.6 | 19 |
| ro_rinko | 0.47 / 0.31 / 0.19 | 0.091 / 0.067 / 0.085 | −15.4 / −15.2 / −14.5 | 13 |
| pp_eve | 0.38 / 0.03 / 0.06 | 0.121 / 0.175 / 0.135 | −9.6 / −6.7 / −8.0 | 11 |
| hhw_kanon | 0.34 / 0.03 / 0.06 | 0.104 / 0.102 / 0.109 | −12.0 / −10.7 / −11.1 | 6 |
| ras_chuchu | 0.25 / 0.06 / 0.16 | 0.064 / 0.060 / 0.062 | −18.2 / −19.1 / −17.7 | 18 |
| mujica_sakiko_tsuki | 0.22 / 0.09 / 0.09 | 0.068 / 0.076 / 0.090 | −19.4 / −20.9 / −18.6 | 17 |
| ras_masking | 0.19 / 0.06 / 0.12 | 0.033 / 0.032 / 0.059 | −21.8 / −22.0 / −20.9 | 11 |
| hhw_hagumi | 0.19 / 0.03 / 0.03 | 0.084 / 0.133 / 0.154 | −13.7 / −10.7 / −9.7 | 20 |
| pp_aya | 0.16 / 0.09 / 0.00 | 0.080 / 0.095 / 0.087 | −16.1 / −16.0 / −15.3 | 20 |
| hhw_kaoru | 0.16 / 0.00 / 0.06 | 0.079 / 0.069 / 0.073 | −19.0 / −17.7 / −19.3 | 20 |

**Reading.**
- **11 of 15 targets now have own-voice Mandarin** (≤ 0.15 at either checkpoint, ≤ 0.16 for lock and soyo_tsuki at ep2). Eve, Kanon, Hagumi, Aya and Kaoru went to ≤ 0.06.
- **Still generic:** Kasumi (0.69–0.72, 15 rows), Ako/demon princess (0.47–0.59), Tomori (0.44–0.50), Rimi (0.34–0.44, only 2 rows). Kasumi is the exception to "rows fix identity": 15 rows that passed both identity gates moved her only 0.03–0.06 [L]. Her E1b Japanese leak was also the highest of the game voices (0.159), so her row may simply sit close to the Base default voice [I].
- **SER and lid rose for six targets** (rimi, kasumi, soyo_tsuki, lock, eve, hagumi). These are the voices now speaking at their own, higher pitch, which is where E4's shift-only control showed both metrics fail on native speech (§6 "why"). Whether their Mandarin is native is an ear question that was not asked, because the rule stopped at conditions 1–3.

## Non-target voices [L]

- Pooled zh leak over the 34 non-targets: 0.042 (E1b) → 0.029 (ep1) / 0.038 (ep2).
- **Regressions:**
  - mujica_sakiko: zh 0.062 → 0.188 / 0.156, and ja 0.167 → 0.250 / 0.333. Her Mandarin and Japanese renders now drift toward her persona twin `mujica_sakiko_tsuki`, which got 17 VC rows [I: probe confusion inside the pair, which is 89–91% on real clips].
  - ras_layer: zh 0.000 → 0.000 / 0.125 at ep2.
- No voice's Japanese leak is worse than the floor by more than 0.2 (guard).
- Japanese leak > 0.05 at ep2: ro_ako 0.059, ro_ako_maou 0.094, mygo_soyo 0.125, mujica_sakiko 0.333. These are all persona pairs. Kasumi's Japanese leak fell from 0.159 to 0.136 (ep1) and ≤ 0.05 (ep2).

## What this means [I]

- **The VC route works for most voices,** and E1b's mix plus 218 VC rows costs nothing in Japanese or English at epoch 2. The pre-registered bar (0.08 pooled, 43/49, targets 0.15) was set for a full fix. E5 got about half-way on the targets (0.41 → 0.20).
- **The remaining gap is four voices** (Kasumi, Ako/demon princess, Tomori, Rimi), and Rimi's is a row-count problem (2 rows).
- **Persona pairs trade off.** Giving one persona in-voice Mandarin pulls its twin's renders toward it (Sakiko).

## Published model

**User, 2026-11-06: publish E1b_ep2, the pre-registered rule's model.** The override (E5_ep2 after a 30-clip target listening check, set built in `listen_final/` from `results/listen_key.json`) was offered and declined, so that set was not judged.

- Hugging Face: `starrydark/BanG_Dream_Qwen3_TTS` (public; the TTS checkpoint only, no RVC converters, no audio, no transcripts), from `models/E1b_ep2` (sha256 `27652e20…`).
- Code: `github.com/starryark/bangdream-tts`.
- E5_ep1 and E5_ep2 stay local (`models/E5_ep{1,2}`).
