# E8: learnable speaker rows and averaged checkpoints, decided by listening (pre-registered plan)

**Listening done 2026-11-09 (`E8_RESULTS.md`).**
- **Round 1:** B won, narrowly (mean +0.25).
- **Round 2:** failed its consistency checks (4 of 8 repeats agreed, one null rated 2), so under §6.7 it does not decide; a repeat would. v1 stays meanwhile.
  - Replaced voices in Japanese: −0.46 (Anon and Umiri consistently worse than v1).
  - Unchanged voices: +0.04. Mandarin: −0.31.
  - **All 16 new voices passed the absolute check; none weak.**

**Run 2026-11-08, 14:42–19:13, pod `jc6hj3x69268j5`, ≈ $18.**
- **Smoke passed:** A saved the locked rows exactly; B saved the locked rows plus its offset; the extra seeds and listening renders worked.
- **B's offsets at epoch 2.0:** |delta| mean 0.176 (max 0.235, ppp_rimi) against a row norm of 17.3. The crowded pairs moved apart slightly; the median nearest-neighbour centred cosine went 0.658 → 0.641. Final val loss: A 2.3208 (= E7a), B 2.3192.
- **Both `avg` candidates passed every screen (§5):**

  | | A | B |
  |---|---|---|
  | Our Notes ja-mid leak, 80 renders | 0.016 | 0.017 |
  | v1 zh leak (C0 0.160) | 0.128 | 0.144 |
  | v1 zh SER (C0 0.074) | 0.078 | 0.074 |
  | v1 ja CER (C0 0.025) | 0.025 | 0.026 |
  | Our Notes ja CER (real 0.042) | 0.051 | 0.051 |
  | yomogi (the only voice > 0.10) | 0.225 | 0.225 |

- **Downloaded:** both checkpoints (`models/E8A_avg`, `models/E8B_avg`) and all 352 listening renders.
- **Round 1 page:** `e8/listen/round1/index.html`.

History: **Status: approved by the user 2026-11-08 ("proceed", on `research/improvement_2026-11-08.md` §5).** This plan and the code (`e8/`, plus the `--spk_lr` option in `kurisu_v2/pod/train_sft.py`) are fixed before the pod is launched. Results go to `E8_RESULTS.md`.

## 1. What changes from E6/E7, and why

1. **Listening decides; automatic metrics only screen.** E6 and E7 were judged by automatic gates alone. On our own data those gates misjudge real recordings:
   - on short lines the probe gives 14–17% leak, and 9 voices over 0.20;
   - real recordings of the Our Notes lines score ja CER 0.042 through the codec, against the 0.035 bound.

   So in E8 the gates become catastrophe screens calibrated on real audio and v1 (§5), and a blind listening comparison against v1 decides (§6).
2. **Arm B, learnable speaker rows.**
   - Each voice's row is today a fixed Base speaker-encoder centroid. Neighbouring voices' centroids have a median raw cosine of 0.993.
   - After centring, yomogi's nearest voices are arale (0.68), raika (0.56) and chieri (0.53), exactly where her renders leak.
   - B adds a trainable fp32 offset per shipped voice with its own learning rate, so the model can move crowded voices apart.
3. **Averaged checkpoints.** Each arm's candidate is the uniform weight average of its checkpoints at epochs 1.0, 1.5 and 2.0. That should cut the epoch-to-epoch swings seen in E6b and E7a (yomogi 0.475 → 0.125 and 0.50 → 0.15) and sit between epoch 1's accent and epoch 2's identity.
4. **Nothing is deleted before listening.** The candidate checkpoints and every listening render are downloaded, with resumable, size-checked transfers.

## 2. Arms

| Arm | Data | Change | Training seed |
|---|---|---|---|
| **A** | E7a's data exactly (E6b's mix: Japanese 65 rows, replay 14%, pseudo_v1, pseudo_on, 223 VC rows) | none (a replication of E7a) | 20261108 (E7a's) |
| **B** | same | `--spk_lr 1e-4`: a learnable fp32 offset on each of the 65 shipped speaker rows; no weight decay; folded into codec rows 3000+ at save; aux replay speakers stay locked | 20261108 |

- **Shared settings:** E1b/E6/E7 hyperparameters (lr 5e-6, effective batch 32, fp32 master weights, `p_auto_lang` 0.25, 3-epoch cosine laid out, stopped at 2.0). Saves at epochs 1.0, 1.5 and 2.0.
- **Why `--spk_lr` = 1e-4:**
  - It is the learning rate E6c's `[nv]` token offset used.
  - Over about 1,190 steps it lets a row move by roughly the size of the gap between neighbouring voices: about 2 on a norm of about 17, from √(2·(1−0.993)) × 17.
  - It is not tuned.
- **Arm C dropped** (lr 2e-6): listening time, not GPU time, is the constraint.

## 3. Candidate per arm (fixed now)

- **The candidate is `avg`:** the elementwise mean, in fp32 and then cast to bf16, of `model.safetensors` from the epoch 1.0, 1.5 and 2.0 checkpoints. The config, tokenizer and `speech_tokenizer/` are copied from epoch 2.0. Script: `e8/e8_average.py`, tested locally (§8).
- **Fallback:** if `avg` fails any screen in §5, epoch 2.0 is evaluated and becomes the candidate if it passes. If both fail, the arm has no candidate.

## 4. Data and pod stages

- **Data:** E7's data path, unchanged: same bundle, the replay flags from E6 amendment 9, and duplicate collapse from E6 amendment 10. Mix: `e6_mix.py --arm E8 --sources pseudo_v1,pseudo_on,vc`. Both arms use this one mix.
- **Queue:** smoke (A and B, 8 steps each; checks below) → train A → train B → average → full screen eval of A's and B's `avg` (`e6_eval.py --extra_on_seeds`) → [fallback evals] → listening renders (§6) → done.
- **Smoke checks for B** (plumbing only; at 1e-4 × 8 steps the drift is tiny):
  - the saved rows 3000+ differ from the locked table;
  - the per-voice |offset| is printed;
  - E7's existing smoke checks still pass (Umiri's own-voice rate ≥ 0.7 on 80 renders).
- **Reported at every B save:**
  - per-voice |offset| (mean and maximum);
  - the centred cosine of the crowded cluster (yomogi, arale, nonoka, chieri, raika, yuno, nagi) before and after.
- **Downloads** (`e8/pod_up.sh download`, resumable per file, sizes verified):
  - both arms' candidate checkpoints;
  - all listening renders;
  - the eval scored files;
  - the pseudo rows and wavs (`pseudo_v1/`, `pseudo_on/`), so a later run can skip about 1.3 h of bootstrap.

## 5. Screens (catastrophe level, calibrated now)

The anchors come from E6's `real` reference and C0's scores (`e6/results/eval/{real,C0}`). E6 and E7 informed the margins: each one is meant to let a sound model through and catch a broken one. A candidate passes only if all of these hold.

| | Screen | Anchor |
|---|---|---|
| S1 | Our Notes pooled ja-mid leak, all seeds, ≤ 0.06 | real 0.027; E7a 0.023–0.025 |
| S2 | No Our Notes voice ja-mid leak > 0.40 at 80 renders, `mygo_raana` excepted (reported) | catches a voice that has collapsed |
| S3 | 37 unchanged voices: pooled ja-mid leak ≤ C0 + 0.03, and pooled zh leak ≤ C0 + 0.10 | gross regressions only |
| S4 | ja CER: unchanged voices ≤ C0's (0.025) + 0.01; Our Notes voices ≤ real-through-codec (0.042) + 0.02 | the ASR misreads names, so real audio is the yardstick |
| S5 | Unchanged voices: pooled zh SER ≤ C0 + 0.02, zh lid ≤ C0 + 2.0, en lid ≤ C0 + 2.0 | gross accent collapse only |
| S6 | Runaway ≤ 1% | |

`e8/e8_screens.py` computes these from `eval/<cond>/scored.jsonl` and writes `eval/<cond>/screens.json`.

## 6. Listening (decides)

- **Setup:**
  - Local HTML page (`e8/listen/round*/index.html`) built by `e8/e8_listen_page.py`. It stays on this machine: the clips are game-voice renders and are never published.
  - Blind A/B labels, side randomised per trial, trial order randomised.
  - Both clips loudness-normalised to −23 LUFS.
  - Unlimited replay.
  - Rating: CCR from −3 ("A much better") to +3 ("B much better"), plus "no difference" (0), plus optional flags: wrong voice, misreading, accent, artefact.
  - Hidden repeats with the sides swapped, and null pairs (the same clip twice).
  - Results are kept in the page and downloaded as JSON. The answer key is a separate file, never loaded by the page.
- **Renders:** made on the pod with `e8/e8_listen_render.py`.
  - One take per (system, item), batch size 1, seed = 900000 + item index. Every system gets the same seed for the same item.
  - First take only, no cherry-picking.
- **Items** (fixed now; `e8/listen_items.json`, built by `e8/e8_listen_items.py` with random seed 8):
  - **Round 1 (arm choice; about 12 min):** A's candidate vs B's candidate.
    - 8 crowded voices (`ikka_yomogi`, `mewtype_arale`, `mewtype_nonoka`, `ikka_chieri`, `ikka_raika`, `mewtype_yuno`, `millsage_nagi`, `ournotes_kanata`), each with 1 held-out mid-length line and 1 short line (16 trials).
    - The first 4 of those voices with 1 Mandarin probe each (4 trials).
    - 2 repeats with the sides swapped, and 1 null pair: 23 trials in total.
  - **Round 2 (release decision; about 50 min, in two or three sessions):** the Round 1 winner vs v1 (C0, `models/E1b_ep2`) on the 49 shared voice ids.
    - **Japanese, replaced voices** (8: `mygo_tomori`, `mygo_anon`, `mygo_taki`, `mujica_uika`, `mujica_mutsumi`, `mujica_umiri`, `mujica_nyamu`, `mujica_sakiko`) × 3 texts (1 held-out mid-length line, 1 novel sentence, 1 short line) = 24.
    - **Japanese, unchanged voices** (8: `ppp_kasumi`, `ag_ran`, `pp_aya`, `ro_yukina`, `hhw_kokoro`, `mo_mashiro`, `ras_chuchu`, `mygo_soyo_tsuki`) × the same 3 kinds = 24.
    - **Mandarin** (8: `mygo_tomori`, `mygo_anon`, `mujica_uika`, `mujica_sakiko`, `pp_eve`, `hhw_hagumi`, `ppp_kasumi`, `ro_rinko`) × 2 probes = 16.
    - **English** (the 8 Japanese-unchanged voices × 1 probe) = 8, reported only.
    - 8 repeats with the sides swapped, and 4 null pairs: 84 paired trials in total.
    - **New voices (absolute check, no v1 counterpart):** each of the 16 new voices × 3 renders (1 mid-length and 1 short Japanese, 1 Mandarin), each played next to a real reference clip of that character. Question: "Does this sound like the character?" (yes / partly / no). 48 quick judgments.
- **Total listening time:** about 60 min over 3 sessions of 20 min or less. This is up from the 45 min estimated in the research note, because of Round 1 and the new-voice check.
- **Round 1 rule:** B is chosen if its mean CCR over the 20 non-control trials is > 0 and it has no more "wrong voice" flags than A. Otherwise A is chosen: on a tie, the simpler model.
- **Round 2 rule:** v2 is adopted only if all of these hold.
  1. The candidate passes the screens (§5).
  2. **Replaced voices, Japanese:** mean CCR > 0, one-sided Wilcoxon signed-rank p < 0.05 over the 24 trials (zeros dropped, the Wilcoxon default).
  3. **Unchanged voices, Japanese:** not significantly worse (two-sided Wilcoxon p ≥ 0.05, or mean CCR ≥ 0).
  4. **Mandarin:** not significantly worse (same test).
  5. **No shared voice** is flagged "wrong voice" in 2 or more of its items.
  6. **New voices:** at most 3 of the 16 are judged weak, where weak means "no" on 2 or more of their 3 renders. Weak voices are listed as weak in the card.
  7. **Consistency:**
     - the swapped repeats agree in sign (or both are 0) in at least 6 of 8;
     - every null pair is rated within ±1.

     If this fails, the round is repeated once with a fresh randomisation, and that second round decides.

  Otherwise v1 stays. English is reported, not decisive.

## 7. Publish path

- **If Round 2 adopts:**
  1. Write `release/MODEL_CARD_v2.md`. Weak new voices are listed, and the card says how the model was chosen (listening).
  2. Point `release/publish_hf_v2.py` at E8's decision (`e8/results/decision.json`) and checkpoint. It tags the current main revision `v1` first.
  3. Upload the model only: no audio, transcripts or reference lists.
  4. Set the Space's `MODEL_VERSIONS` to `v2=main,v1=v1`.
  5. GitHub: E8's code and results.
- **If not:** v1 stays; GitHub gets E8's code and results.

## 8. Tests before launch ($0, local)

- **Trainer:** a standalone test of the `--spk_lr` path: int and tensor indexing, the gradient reaches the offset, the dtype is kept, and the saved rows equal the locked rows plus the offset. With the flag off, the code path is unchanged.
- **`e8_average.py`:** averaging `models/E1b_ep2` with itself three times reproduces its tensors exactly, and non-weight files are copied.
- **`e8_screens.py`:** run on E6's and E7's scored files; values printed for E6b and E7a.
- **The listening page:** generated from `samples/E1b_ep2` as a dry run.

## 9. Cost

About 5–6 h on one H100 ($3.99/h), roughly **$20–25**:

| Stage | Time |
|---|---|
| Upload | 15–75 min |
| Bootstrap | about 1.6 h |
| Training, 2 arms | about 35 min |
| Averaging | about 10 min |
| 2 full evals | about 1.8 h |
| Listening renders | about 20 min |
| Download (about 9 GB, resumable) | |

Plus about 60 min of the user's listening.

## 9.1 Amendments

(none yet)
