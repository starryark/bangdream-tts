# E6: BanG Dream v2 on Our Notes data (pre-registered plan; approved 2026-11-07, run 2026-11-08)

**Outcome (2026-11-08 10:07): v2 not adopted; v1 (E1b_ep2) stays and nothing is published.** No E6a candidate passed G1–G3. E6a failed on Mandarin identity of 4 unchanged voices, ja CER and yomogi/arale coverage, not on accent alone, so E6d was not triggered, and E6b/E6c can only replace an E6a\*. Total pod cost was about $35. See `E6_RESULTS.md`. History: **Status: approved 2026-11-07; local stage running.** As of 2026-11-07 12:00, done: data build, ASR, rows, probeE6 (bar recorded in §8 item 13), VC source renders, and a local smoke test of every pod script (passed). Converters: arale and nonoka finished 200 epochs (14:05, 16:03). The GPU stopped responding at about 16:45 and the machine reset uncleanly at 16:55 (Kernel-Power 41, no driver or bugcheck event; free RAM had been ~2 GB with another app open). Hotaru had reached epoch 50, and yomogi failed at CUDA init. Both partial model folders were moved aside (`*_crashed_20261107`), and both converters were retrained from scratch from 20:30 with the same recipe (`FROM=rvc bash e6/local_chain.sh`). The queue now also requires the epoch-200 weights before it marks a row done. A second freeze came at 22:01: during hotaru epoch 140 the trainer stopped with "CUDA error: unknown error", the OS froze, and the machine was reset with the power button at 22:02 (Kernel-Power 41, bugcheck 0). Blender was closed and about 36 GB RAM was free, so memory pressure doesn't explain it; the shape points to the GPU, power or temperature, and the cause is unknown. The same driver trained 17 converters without a fault on Oct 30–Nov 1. Retry from 22:12: hotaru resumes from its epoch-100 state (Applio loads the latest G_/D_), and yomogi starts from scratch. The only change is checkpoint cadence (`--save-every-epoch 25`, not part of the recipe). GPU and RAM telemetry is logged every 15 s to `e6/logs/gpu_telemetry_run3.log`. If there's a third reset, local RVC training stops, and the user chooses between training hotaru and yomogi on the pod and running E6b with arale and nonoka only (a recorded deviation). **The third reset came at 22:35.** Hotaru had resumed at epoch 101 and reached epoch 121, then failed with "CUDA error: unknown error" at 22:34:57. Telemetry then read "GPU is lost", and the machine reset at 22:35. The telemetry shows what led up to it:
- RVC alone used 5–7 GB VRAM and 40–90 W (22:14–22:19).
- From 22:20, VRAM in use rose to 10.7 GB and then 14.6–15.3 GB of 16. Power reached 160 W and temperature 82 °C, with no throttle flags.
- Free system RAM fell from 26 GB to 7.4 GB in the last 90 s.

That pattern points to a second GPU workload filling the card alongside RVC. Other local GPU jobs, ComfyUI and a headless Blender render, restarted within 3 minutes of the reboot. Local RVC training is stopped (hotaru's last state is epoch 100; yomogi has none). **The user chose a rented pod (2026-11-07 22:50).** Hotaru and yomogi train there from scratch with the same recipe and torch 2.11.0+cu128, as in the local venv (`e6/rvc_pod.sh all` → `e6/rvc_pod_run.sh`). Only the training location changes, which is a recorded deviation from §3.6's "trained locally". The pod gets Applio's code, the four model files the recipe uses and the two rows' training clips. Hotaru has 295 clips: two of its 297 bundle rows share an audio file, and the local queue's copy deduplicated them in the same way. Back come weights, config and f0 folders, with no optimizer states or audio, and the pod is deleted. **Done 2026-11-08 01:24:** both converters reached epoch 200 on an RTX 4090 Secure pod, training side by side from 23:37 to 01:21 (hotaru about 21 s/epoch, yomogi 24–28 s/epoch). They were downloaded with their f0 folders (591 and 752 slices), and the pod was deleted (the API lists no pods). Pod cost was about $1.3 for this run plus about $0.5 for four earlier launches that stopped on script faults. The fixes were a stray CR in the ssh port, ssh taking the tar stream's stdin, a requirements filter that dropped onnxruntime, yomogi clips corrupted by the 22:35 reset (re-staged from `data_on/wav24` with sha256 and decode checks, then checked again on the pod), and a remote start that blocked. Logs: `e6/logs/rvc_pod*.log`. **Conversion and scoring ran locally 01:34–01:38, with no other GPU job, and telemetry in `e6/logs/gpu_telemetry_run4.log`.** Gate pass rates (raw identity, codec identity, CER; `e6/vc/collect_report.json`): arale 0.92 (20 rows), nonoka 0.31 (7), hotaru 0.38 (8) and yomogi 0.04 (1). The last three are below `MIN_ROWS` = 10 and flagged weak, as E5's Rimi was with 2 rows. Nonoka, hotaru and yomogi mostly fail on identity strength, not on the top prediction. The raw probe still names the right role on 25/26, 24/24 and 23/28 clips, but the median p_true is 0.52, 0.71 and 0.34 of the role's real-clip base, and it falls further after the codec (0.39, 0.44, 0.24). Listening sets are in `e6/listen/` (yomogi has 1 clip). **Listening (user, 2026-11-08 ~01:45): all four native and in character** (`e6/vc/listening.json`), so `e6_vc.py rows` wrote 36 VC rows: arale 20, nonoka 7, hotaru 8, yomogi 1. Yomogi's single row is in the train split; one arale row is val. With E5's 187 kept rows, E6b has 223 VC rows. Pod run: `bash e6/pod_up.sh all > e6/logs/pod_up.log 2>&1`. It runs launch (H100-class only, 6 rounds), upload with bootstrap then the queue starting detached, wait for QUEUE_DONE, download and stop, unattended. A dead queue pulls logs to `e6/results_partial_*` and stops the pod. **Pod `dsrojh8on3fmaz` (H100 SXM, $3.99/h) launched 01:43. Upload took 75 min, ending 02:59.** Bootstrap had already died during the upload, at p05's replay shortfall check: the zh pool reached 2,100 clips / 2.53 h of 5.0 h, capped by p05's default `--zh_speakers 30` × 70 clips. That pool covers E6a–c's draw (2,534 rows) but not E6d's (~5,189). The wrapper was stopped before it deleted the pod. Replay was rerun on the pod with `--zh_speakers 70 --zh_hours 5.5` (amendment 9), and bootstrap and the queue restarted at 03:08. `e6/pod_wait.sh` now watches the queue: QUEUE_DONE leads to download and stop. A dead queue keeps the pod 60 min for a fix, and watching resumes if the queue runs again. **At 04:36 bootstrap stopped again, at the duplicate-id guard (amendment 10). The fix went in and the queue started at 04:41.** The local partial folders are moved aside as `*_local_partial_<time>`. Conversion and scoring of all four roles then run locally (short GPU jobs, started only when no other GPU job is running). Next: conversion and scoring, then the user's listening check (`e6/listen/`), then one pod run (`e6/pod_up.sh launch|upload|status|download|stop`). If adopted: `release/publish_hf_v2.py` (needs `release/MODEL_CARD_v2.md`). The user answered D1–D5 in `../research/recommendation_2026-11-07.md` §6:
- D1: yes, add the 16 and replace the MyGO / Ave Mujica data.
- D2: option (a), Sakiko split by arc; keep `soyo_tsuki`; Hatsune not added. The recommendation left Hatsune optional, and the user's instruction to follow the recommendations is not a request to add it.
- D3: yes, run the VC arm.
- D4: release v2 if E6 adopts a candidate (same public form as v1).
- D5: pod budget approved.

The rules below are now fixed. Any later change is an amendment, logged in §8 with date and reason, made before results are seen.

Evidence: `../research/data_facts_2026-11-06.md` (§ numbers below). Engine and hyperparameters are as in E1b.

## 1. Questions

1. **Non-inferiority.** With 28 Our Notes rows added, do the 37 unchanged v1 rows keep v1 quality, accent included?
2. **Coverage.** Do the 28 Our Notes rows (from 27 roles: 12 replacing v1 MyGO / Ave Mujica rows, 16 new) reach v1-level identity and text accuracy?
3. **Short lines (the claim v2 exists to make).** Do the replaced rows leak less on 1 s lines than v1 does?
4. **Secondary.**
   - Do VC Mandarin rows fix the four roles Base can't reach (E6b)?
   - Does an `[nv]` token produce nonverbal audio without harming anything else (E6c)?

## 2. Arms

| Arm | Data | Runs |
|---|---|---|
| **E6a** | Baseline (§3) | always |
| **E6b** | E6a + VC Mandarin rows (§3.6) | if D3 = yes |
| **E6c** | E6a + `[nv]` rows and token (§3.7) | always |
| **E6d** | E6a with replay share 25% instead of 13.3% | **only if** no E6a candidate passes because of an accent guard (§6, step 1) |

- All arms use E1b's hyperparameters: lr 5e-6, effective batch 32, fp32 master weights, `p_auto_lang` 0.25, 3-epoch cosine.
- Checkpoints are saved at the end of epochs 1, 2 and 3. **Candidates are epoch 1 and epoch 2.**
- At ~620 steps per epoch, epoch 1 (~620 steps) is close to E1b_ep2's 686 in total steps.

**Pre-registered prediction:** E6a moves Mandarin and English accent on the 37 unchanged rows slightly the wrong way (zh lid and en lid up, zh SER up), more at epoch 2 than epoch 1. The basis is §8 (ρ = +0.27 between Japanese minutes and zh lid) and Kurisu's finding that accent grows with steps.

## 3. Data

### 3.1 Rows (65, or 66 with Hatsune)

| Set | n | Names |
|---|---:|---|
| v1 rows, unchanged | 37 | the 36 non-MyGO / non-Ave-Mujica rows + `mygo_soyo_tsuki` (GBP game audio) |
| Replaced, v1 names kept | 12 | `mygo_tomori`, `mygo_anon`, `mygo_raana`, `mygo_soyo`, `mygo_taki`, `mujica_uika`, `mujica_mutsumi`, `mujica_mortis`, `mujica_umiri`, `mujica_nyamu`, `mujica_sakiko` (Sakiko lines outside `adv_script_mygo_*`), `mujica_sakiko_tsuki` (Sakiko lines in `adv_script_mygo_*`) |
| New | 16 | arale, nonoka, ritsu, miyako, yuno, hotaru, natsume, nagi, mahoro, houka, raika, miku, yomogi, chieri, shizuku, kanata. Row names are fixed at build from the game's band ids and written into this file before training |
| Optional (D2) | 1 | hatsune |

- Speaker slots run from 3000 contiguously. The build asserts that every row has a trained slot and that no row uses a never-trained one.
- Persona handling per §7.2.
- If D2 = (b), `mujica_sakiko_tsuki` stays on its v1 (anime) data, `mujica_sakiko` takes all Sakiko lines, and every count below becomes 38 unchanged / 11 replaced / 27 Our Notes rows.
- The Sakiko arc split was not measured at the floor; the pre-check's `sakiko` used all arcs.

### 3.2 The 37 unchanged rows: E1b recipe

- Japanese rows: E1b's E0 train split (verdict ok), with val as in E1.
- Pseudo-zh: regenerated with `e1_pseudo.py`, using the same texts (p11, seed 31), 2 samples, cap 40 and the 49-class `probe_codec` gate (`p_true ≥ 0.5`), exactly as E5 did. This is not byte-identical, because E1b's pseudo audio was deleted with its pod. The regenerated yield per row is recorded.

### 3.3 Japanese rows for the 28 Our Notes rows

- **Source.** `adv_voice` only, single speaker, role from `TargetName`, 0.6–15 s.
- **Split.** By episode: a deterministic 1 in 20 episodes per arc (hash of episode id) is test. Test lines never appear in training, locked-vector references or pseudo texts.
- **Text.**
  - Drop ellipsis-only and nonverbal-only lines (§5); they are E6c material.
  - Remove newlines, strip `（）` and markup tags, take readings from `<ruby>`.
  - Exclude multi-speaker lines.
  - Add the new-cast readings to `BD_READINGS`.
- **Gates.**
  - Lines with ≥ 3 kana: v1's `cer_bar`, better of two ASR systems as in `b02_asr.py`.
  - Lines with 1–2 kana: the out-of-fold 27-role clip probe predicts the row's role (or its pair, for Mutsumi/Mortis; both Sakiko rows map to role sakiko), and trimmed duration is 0.25–2.0 s.
  - All lines: drop the clip if the out-of-fold clip probe predicts another role with p > 0.5 (expected 2–4%, §7.3).
  - Report what the gates remove per expression class.
- **Cap.** 300 per row. Sampling is stratified, proportional to (duration bin < 1 / 1–3 / 3–8 / ≥ 8 s) × (expression class, §6) cells of the gated pool, seed 61. Rows with fewer clips take all of them.
- **Locked vectors.** v1 recipe (≥ 2 s, no monologue, no ellipsis, drop the 20% least typical, cap 64), from training episodes.
- **Rāna** uses Our Notes data only. If `mygo_raana` alone fails G3, E6 still adopts, and the card lists it as weak. Supplementing it with v1 clips is a follow-up, not part of E6.

### 3.4 Pseudo-zh for the 28 Our Notes rows

- **Texts.** Official zh-Hans translations of the row's own training lines. 8–30 hanzi after normalisation, at most one numeral, no Latin, and no overlap with `probes_zh_v3` (asserted). 60 texts per row, seed 31.
- **Generation.** Base model from the locked vector, **6 samples per text**.
- **Identity gate.**
  - The prediction must be the row itself (for the pairs Mutsumi/Mortis and the two Sakiko rows, the exact row).
  - `p_true` must clear the matched bar on probeE6. The bar is recomputed by the method of §9.2 (the value that passes 91.0% of real resynthesised clips, 5-fold) and written here before generation. It was 0.39 on probe63.
- **Accent gate.** `p_target ≥ 0.9`, `lid ≤ −19`.
- **Text gate.** CER ≤ 0.05.
- **Selection.** Keep the best sample per text, at most 40 per row.
- Arale, nonoka, hotaru and yomogi go through the same process. Their expected yield is near zero (§9.4).

### 3.5 Replay

E1 recipe at the E1b share (13.3%), with a larger pool: `p05_replay.py --zh_hours 4.0 --en_hours 2.0 --max_clips_per_speaker 70`. If `e1_mix` reports the pool short, raise the hours and record it.

### 3.6 E6b only: VC Mandarin rows

- E5's VC rows for the 13 retained v1 rows with converters (187 rows, E5's listening outcome kept). E5's `mygo_tomori` (14) and `mujica_sakiko_tsuki` (17) rows are not used: those converters were built on v1-channel audio of rows that now change.
- New converters for arale, nonoka, hotaru and yomogi:
  - E5 recipe: Applio 59c1131, 32 kHz, contentvec, rmvpe, 200 epochs, batch 8, HiFi-GAN, index_rate 0, proposed-pitch shift;
  - trained locally on each role's training clips;
  - E5 sources (`e5_sources.py`) and gates;
  - at most 20 rows per role after the user's listening check.
- Audio path into training as in E5: resample to 24 kHz, `p03_build.normalise_audio`, 2–12 s.

### 3.7 E6c only: `[nv]`

- Rows: ellipsis-only or nonverbal-only lines from training episodes of the 28 Our Notes rows.
  - 0.3–3.0 s, with the clip probe agreeing with the role.
  - Text `[nv]`, language japanese, at most 120 per row.
- Trainer: `--new_tokens '{"[nv]": ["息", "はぁ", "っ"]}' --new_token_lr 1e-4`.

## 4. Evaluation

**Conditions.**
- C0 = **E1b_ep2** (v1 as shipped). It renders the 37 unchanged rows and the 12 replaced rows under their v1 names. The 16 new rows don't exist in C0.
- E6a, E6b and E6c at epochs 1 and 2.
- E6d at epochs 1 and 2, if triggered.

**ProbeE6.**
- Logistic regression on codec-resynthesised training clips of all rows, global-mean centring as in v1. Report 5-fold accuracy.
- Leak is reported **within set** (28 Our Notes / 37 v1) and **across sets**.
- Mutsumi↔Mortis confusions are counted as pair confusions: reported, but not counted as leak.

**Eval sets per row (2 seeds each).**

| Set | Texts |
|---|---|
| ja-mid | E1's Japanese probes, same texts as E1 |
| ja-short | 16 test-episode lines, real duration 0.7–1.6 s, ≥ 3 kana. The 28 Our Notes rows use their own lines. The 37 rows share one fixed set of 16 drawn from several Our Notes characters, with no Latin, no digits and no Our Notes character names (a Roselia row saying 寧月 only adds CER noise) |
| zh | `probes_zh_v3`, as E1 |
| en | as E1 |
| nv | E6c: `[nv]` × 8 per row. E6a comparator: `……っ` × 8 |

**Metrics.**
- Identity leak and `p_true` (probeE6).
- ja CER on ≥ 3-kana lines.
- zh shared-character SER and lid; en lid and WER; runaway.
- For the 28 Our Notes rows: render/real duration ratio on silence-trimmed audio (−40 dB), and F0 range ratio render/real.
- nv: ASR kana count, trimmed duration.
- **Real reference:** test-episode real clips through the codec, scored with probeE6. This is the ceiling for short-line leak.

## 5. Guards

**G1: non-inferiority on the 37 unchanged rows, against C0 (same probe, same texts).**
- Pooled ja-mid leak ≤ C0 + 0.02.
- Pooled ja-short leak ≤ C0 + 0.05.
- Pooled zh leak ≤ C0 + 0.03.
- Pooled zh SER ≤ C0 + 0.010.
- Pooled zh lid ≤ C0 + 1.0, and pooled en lid ≤ C0 + 1.0 (higher lid is more accented).
- Per row: ja-mid leak and zh leak each ≤ that row's C0 value + 0.20.
- In E6b, G1 is reported separately for the 24 rows whose data is unchanged and the 13 with VC rows. The pooled bounds apply to all 37.

**G2: E1's absolute guards, all rows.**
- Pooled zh SER ≤ 0.087 and zh lid ≤ −18.
- Every row's zh SER ≤ 0.15.
- ja-mid CER ≤ 0.035; en WER ≤ 0.02; runaway ≤ 1%.

**G3: coverage of the 28 Our Notes rows.**
- Pooled ja-mid leak ≤ 0.05.
- Every row ja-mid leak ≤ 0.20, except `mygo_raana`, which is reported (§3.3). Pair confusions are excluded: Mutsumi↔Mortis and Sakiko↔Sakiko-tsuki.
- Pooled zh leak ≤ 0.15 over the 24 Base-routed rows. Arale, nonoka, hotaru and yomogi are reported; they are E6b's target.

**C1: the short-line claim (not a shipping gate).** Pooled ja-short leak over the 12 replaced rows is lower than C0's on the same rows by ≥ 0.10. The pre-check had v1 at 0.37 and real audio at 0.13 (§9.5). The card claims better short lines only if C1 holds.

## 6. Selection rule

1. **E6a.** Check the candidates in order (ep1, then ep2); the first to pass G1–G3 is E6a\*.
   - If no candidate passes, but at least one passes G3 and fails **only** on accent bounds, run E6d. The accent bounds are G1 zh SER, zh lid and en lid, and G2 zh SER and zh lid. This is the predicted failure pattern: ep1 short of coverage, ep2 over on accent. E6d's candidates are judged by the same rule and take E6a's place.
   - Otherwise, or if E6d also fails, v2 is not adopted. v1 stays, and the results are reported.
2. **E6b** replaces E6a\* if its first passing candidate (same order) meets all of:
   - G1–G3;
   - pooled zh leak over arale, nonoka, hotaru and yomogi is ≤ 0.30 and at least 0.20 below E6a\*'s;
   - pooled zh leak over the other rows is no more than 0.02 above E6a\*'s.
3. **E6c** is adopted as the release only if E6b is not adopted, and its first passing candidate meets all of:
   - G1–G3;
   - against E6a\* at the same epoch: pooled ja-mid leak ≤ +0.01 and ja CER ≤ +0.003;
   - `[nv]` works: ≥ 80% of `[nv]` renders contain ≤ 2 kana by ASR, ≥ 80% are 0.3–2.0 s trimmed, and none runs away.

   If E6b and E6c both pass, E6b ships and `[nv]` goes into a later E6b+`[nv]` run. **No untested combination ships.**
4. **Listening before release** (a no-go check, not selection): the user hears 3 ja-short and 3 zh renders for each of the 16 new rows of the selected candidate. Any row marked wrong-voice is listed as weak in the card.

**Also reported:**
- the prediction's outcome (§2);
- trained Mandarin vs floors for the 28 Our Notes rows;
- within-set and cross-set leak;
- gate removal per expression class;
- F0 range and duration ratios.

## 7. Schedule and cost

| Where | Stage | Time |
|---|---|---:|
| Local (5060 Ti, $0) | Manifest, split, ASR gate, clip-probe gate, locked vectors | ~2 h |
| Local | RVC converters ×4 and VC rows (D3), user listening | ~12 h GPU |
| Pod (H100, chained) | Bootstrap: env, replay, E1b-recipe pseudo (37 rows), new pseudo (28 × 60 × 6), codes | ~2.5 h |
| Pod | Train E6a, E6b, E6c | ~1.2 h |
| Pod | Evaluate 7 conditions (~6.5k renders each) | ~5.5 h |
| Pod | Selection script and report | — |
| | **Pod total** | **~9–10 h, $40–55**; E6d adds ~1.5 h |

Per the E5 lesson, everything is queued before launch. Results go to `E6_RESULTS.md`.

## 8. Amendments

### Pre-run clarifications (2026-11-07, before any E6 result exists)

These fill details the plan left open. Code: `e6/*.py`, `e6/*.sh`.

1. **Row names for the 16 new rows:** `mewtype_{arale,nonoka,ritsu,miyako,yuno}` (夢限大みゅーたいぷ), `millsage_{hotaru,natsume,nagi,mahoro,houka}`, `ikka_{raika,miku,yomogi,chieri,shizuku}` (一家Dumb Rock!), `ournotes_kanata`.
2. **Sakiko split:** lines from story assets `adv_script_mygo_*` → `mujica_sakiko_tsuki`; all other Sakiko lines → `mujica_sakiko`.
3. **Test split (replaces §3.3's episode split) and text details.**
   - §3.3's episode split didn't work: the story has only 124 `<arc>_<chapter>_<part>` episodes, so 1 in 20 left 12 rows with no test lines. A scene-level split needed about 25% of all clips and still found only 2–6 short lines for some rows.
   - Replacement, as v1's E0 test split: **line-level holdout.** Per row, 24 short-eligible lines (0.7–1.6 s, ≥ 3 kana) and 12 held-eligible lines (3–8 s, ≥ 5 kana) are taken in sha256(id) order, with distinct texts and no Latin or digits.
   - Any other clip of the same row with a test line's text never trains.
   - Episodes (`<arc>_<chapter>_<part>`) are kept only as the clip probe's group folds.
   - Chieri and Kanata have only 8 eligible short lines each; Sakiko-tsuki has 14 and Mortis 20.
   - A ruby reading replaces its base text, except emphasis dots (傍点), which keep the base.
   - The new-cast readings OpenJTalk misreads went into `pipeline/bdcommon.py` `BD_READINGS`. They apply to every scored condition, C0 included.
4. **Clip-probe gate:** a 27-role logistic regression on the Qwen speaker-encoder vectors, with 5 group folds by episode.
5. **Eval sets for the 28 Our Notes rows** (from the held-out lines, both gates passed, ASR verdict ok, no Latin or digits):
   - ja_heldout: 8 lines of 3–8 s with ≥ 5 kana;
   - ja_short: 16 lines of 0.7–1.6 s with ≥ 3 kana. A row with fewer gets all it has, and the count is reported.
   - The shared ja_short set for the 37 v1 rows is drawn round-robin, without Our Notes character names.
6. **nv rows** are 0.6–3.0 s of source audio (the build drops clips under 0.6 s).
   - Eval nv is 4 texts × 2 seeds = 8 renders per row. E6a renders `……っ` (the comparator), E6c renders `[nv]`; C0 and E6b render none.
7. **Arms are trained as E5:** 3-epoch cosine layout, checkpoints at 1.0 and 2.0, stopped at 2.0.
8. **Replay pool:** `p05_replay.py --zh_hours 5.0 --en_hours 2.5 --max_clips_per_speaker 70`, so E6d's 25% share (~5.5k rows) can't run short. Every arm draws `0.14 / 0.86 × Japanese train rows` (E1's formula, `e6_mix.py`) with the same seed; E6d uses `--replay_share 0.25`.
9. **Replay pool, zh speakers (2026-11-08, on the pod, before any training):** amendment 8's flags gave a zh pool of 2,100 clips / 2.53 h, because p05's default `--zh_speakers 30` with `--max_clips_per_speaker 70` caps it at 2,100 clips. §3.5 allows raising the hours when the pool is short, and that only works with more speakers. Rerun: `p05_replay.py --zh_hours 5.5 --en_hours 2.5 --zh_speakers 70 --max_clips_per_speaker 70` gave zh 4,556 clips / 5.50 h from 70 speakers (49 F, 21 M) and en 1,587 clips / 2.50 h (unchanged), 6,143 rows. Every arm still draws from one pool with one seed. The short pool is kept on the pod as `replay_short_30spk/`.
10. **Duplicate clip rows (2026-11-08, on the pod, before any training):** bootstrap stopped at `e6_mix.py`'s duplicate-id guard. The Our Notes story reuses some voice files across scenes, and `e6_onbuild.py` listed each once per scene: 52 Japanese ids (55 extra rows) and 5 `[nv]` ids. Every pair has the same speaker, audio file and split. 4 differ only in punctuation, and none overlaps the eval clips. `e6_mix.py` now collapses exact repeats (same id, speaker and audio, first kept) and records the counts in `mix.json`, and an id naming two different clips still stops the mix. Every arm draws 0.14/0.86 × the unique Japanese train rows (15,514, so 2,526 replay rows; E6d's 25% → ~5,171). The three mixes were rebuilt on the pod (E6a 18,844 / E6b 19,058 / E6c 19,609 train rows), and the queue started at 04:41. The local probe and gate vectors were fitted with the repeats in (55 of ~15.7k rows); they are left as they are.
9. **VC sources for the four new roles:**
   - E5-A's donor path: E1b_ep2 renders in `ppp_kasumi`, 6 per text, lowest-lid native sample.
   - 40 AISHELL-3 test texts per role, excluding all texts used by E1–E5.
   - RVC training clips are the row's E6a Japanese training rows (≤ 300).
10. **VC identity gates** use probeE6 and its out-of-fold real-resynth `p_true` per row as `base_p`.
11. **E6d trigger, accent bounds:** G1 zh SER, zh lid, en lid, and G2 zh SER, zh lid and the per-row zh SER maximum. The per-row maximum is the per-row form of G2's zh SER bound.
12. **floor_on** (report only): Base x-vector-only from E6a epoch 1's rows, Mandarin only, the 28 Our Notes rows.
13. **Gate and probe:** the matched identity bar and probeE6's accuracy are written to `bundle/probe_e6_calib.json` by `e6_probe.py`, and the value is copied here before pseudo generation: **bar = 0.3896** (2026-11-07 11:01; 65 classes, 11,931 clips, 5-fold accuracy 0.972; p_true ≥ 0.5 is passed by 0.855 of real clips, against f49 = 0.910 for E1's 0.5 under the 49-class probe). Rows under 0.9 accuracy: mujica_mortis 0.887, mujica_sakiko 0.893, mujica_sakiko_tsuki 0.883, mewtype_arale 0.893. The two Sakiko rows and Mortis↔Mutsumi are the excluded pairs (G3). Also logged: the probe's resynthesis now runs in batches of ≤ 8 clips / 40 s with the cache freed per batch (a memory fix for the local 16 GB GPU; the method is unchanged).
14. **Smoke-test observation (2026-11-07 11:53, v1 checkpoint only, no rule change):** the local smoke test ran every pod script on 2–3 rows with E1b_ep2 standing in for a candidate. Under probeE6, C0's ppp_kasumi renders are their own row in 22/24 novel Japanese lines but only 2/32 shared short lines (11/32 go to mewtype_arale, 5 to pp_eve) and 4/32 Mandarin probes (Kasumi is one of E1's Base-fallback voices in Mandarin). G1 compares each arm with C0 under the same probe, so Kasumi's baseline leak is in both sides; Kasumi↔Arale is not added to the excluded pairs. The VC source renders (E1b_ep2 in ppp_kasumi, 6 per text) gave a native-accent sample for 26, 26, 24 and 28 of the 40 texts for arale, nonoka, hotaru and yomogi.
