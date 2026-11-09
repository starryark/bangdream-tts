# E9: v1 voices kept, Our Notes added per era, 16 new voices; decided by listening (pre-registered plan)

**Run 2026-11-09 01:18–03:41 (pod `wwtyz192612cp5`, ≈ $9.5). Listening 12:25–12:59: NOT ADOPTED, v1 stays.** The v1-era voices were worse than v1 (mean −0.58, p = 0.046), and Uika summed to −7. All other rules passed. Details: `E9_RESULTS.md`, `results/listening.json`.
- **`E9_avg` passed every screen:**

  | screen | value |
  |---|---|
  | gated ja-mid leak | 0.0155 |
  | yomogi | 0.175 (the only voice > 0.10) |
  | v1 zh leak (C0 0.155) | 0.138 |
  | v1 zh SER (C0 0.073) | 0.071 |
  | v1 ja CER (C0 0.025) | 0.026 |
  | Our Notes ja CER | 0.054 |

- **v1-era ids assigned to their own character (reported, not gated):** 0.91–1.00, except Anon 0.56 and Taki 0.46. The probe's classes for those two are Our Notes-fitted.
- **Downloaded:** `models/E9_avg` (74 ids on 72 rows) and 152 listening renders. The pages are `e9/listen/{p1,p2,abs}.html`.

History: approved by the user 2026-11-09; pod launched 01:18. This plan and the code (`e9/`) are fixed before the pod is launched. Results go to `E9_RESULTS.md`. Sources:
- E8's listening (`../e8/E8_RESULTS.md`);
- the persona-distinctness check;
- the curation audit (`../research/curation_2026-11-09.md`).

## 1. Why, and what the user decided

In E8, the 16 new voices were all judged in character, and the 37 unchanged voices tied with v1. But replacing the MyGO!!!!! / Ave Mujica data with Our Notes made **Anon and Umiri clearly worse** than v1. The user then decided:

1. **Merge, don't replace:** these characters train on v1 data **and** Our Notes data.
2. **Persona rows:**
   - **Soyo + Soyo-tsuki** become one voice, and so do **Sakiko + Sakiko-tsuki**. The old ids stay as aliases.
   - **Mortis** stays separate from Mutsumi: their clips are 94–98% separable, against 85–89% for the Sakiko pair.
3. **Two ids per character, one shared model.**
   - The curation audit found the v1 and Our Notes recordings of one character 94–100% separable. Part of that is the channel (Our Notes is brighter and slightly noisier); part is performance years apart.
   - So each character gets a v1-era id, under the usual name and the default, and an Our-Notes-era id `<name>_ournotes`.
4. **Pruning:** drop 244 training clips (about 1.5%) that are speaker outliers. The rule: ≥ 2 s, robust z < −3.5 within the voice, *and* nearer another voice's centroid, computed per era.

## 2. Voices (72) and aliases

| Group | n | Data | Locked-vector references |
|---|---:|---|---|
| Unchanged v1 voices | 36 | v1 | E6's |
| v1-era ids of 9 characters (`mygo_tomori`, `mygo_anon`, `mygo_soyo`, `mygo_taki`, `mujica_uika`, `mujica_mutsumi`, `mujica_mortis`, `mujica_umiri`, `mujica_sakiko`) | 9 | v1 (Soyo with Soyo-tsuki, Sakiko with Sakiko-tsuki) | E1's v1 references (persona lists united) |
| Our-Notes-era ids `<name>_ournotes` of the same 9 | 9 | Our Notes (Sakiko with the MyGO-arc Sakiko lines) | E6's Our Notes references |
| `mygo_raana`, `mujica_nyamu` | 2 | both eras under one id | both lists |
| New Our Notes voices | 16 | Our Notes | E6's |

- **Why Rāna and Nyamu have one id:** the codec embedding has 3,072 rows and speakers start at 3000, so at most 72 voices fit.
  - Rāna's Our Notes data is only 4.8 min.
  - Nyamu's two eras are the closest (centroid cosine 0.74).
- **Aliases:** `mygo_soyo_tsuki` → `mygo_soyo` and `mujica_sakiko_tsuki` → `mujica_sakiko`, both v1-era (`e9/e9_aliases.py`, added to every checkpoint).
- **Data** (`e9/e9_bundle.py`, linted):
  - 16,790 training clips after pruning; per-voice minimum 35.
  - Pseudo-zh: E8's E1-recipe rows for all 49 v1 voices (583) plus E8's Our Notes rows (460), relabelled to the era ids. Not regenerated.
  - VC rows: E5's 218 (with Tomori and Sakiko-tsuki now valid again, mapped to v1-era ids) plus E6's 36.
  - Replay: E6/E8's flags.

## 3. Recipe

E8's winner, B, unchanged except for the speaker set:
- E1b hyperparameters, `--spk_lr 1e-4` (learnable speaker rows), training seed 20261108;
- saves at epochs 1.0, 1.5 and 2.0;
- **candidate = the weight-average of the three (`avg`).** If `avg` fails a screen, epoch 2.0 is evaluated, and used if it passes.

## 4. Pod

- **Run:** `e9/pod_up.sh launch` → `upload` → `e9/pod_wait.sh`.
  - The bootstrap (`e9/bootstrap_e9.sh`) skips pseudo generation: env, replay, input check, aux filter, codes, mix.
  - The queue (`e9/queue_e9.sh`): smoke → train → average → aliases → screen eval → listening renders.
- **Smoke checks:**
  - 72 voices at rows 3000–3071, aliases in place;
  - the learned rows differ from the locked rows;
  - aliases and era ids render;
  - extra-seed eval; Umiri-ournotes assigned to Umiri ≥ 0.7;
  - two listening renders.
- **Download:** resumable (`e9/sync_down.sh`). The candidate checkpoint and every listening render are kept.
- **Estimate:** about 3.5–4.5 h, **about $15–18**.

## 5. Screens (`e9/e9_screens.py`; catastrophe level)

- **Probe-based identity (S1, S2)** applies to the 16 new ids, the 9 `_ournotes` ids and Rāna and Nyamu: pooled ja-mid leak ≤ 0.06 on all seeds, and no voice > 0.40.
  - probeE6's MyGO/Mujica classes were fitted on Our Notes clips; `X_ournotes` is scored against class X.
  - The **9 v1-era ids are not probe-gated.** For them the probe measures "how Our-Notes-like" (it calls v1's own Anon 0.30 and Taki 0.53 "leaky"). Their share assigned to their own character is reported.
- **S3–S6 as in E8, on the 36 unchanged voices against C0.** C0 is E6's scored v1. Comparisons are pooled, not seed-paired, because the eval batch layout changes with 72 voices.

## 6. Listening (decides)

- **Pages:** local pages `e9/listen/{p1,p2,abs}.html` (`e9/e9_listen_page.py`); the clips never leave this machine. Items are fixed in `e9/listen_items.json` (101 items, render seed 910000 + index, identical for both systems).
- **Pairs: E9 vs v1, same voice id and text, CCR −3…+3** (51 trials):
  - **v1era_ja:** the 11 characters' v1-era voices (9 two-era characters + Rāna + Nyamu) × 3 lines each: one v1 held-out line, one novel sentence, one short line (33).
  - **persona:** `mygo_soyo_tsuki` and `mujica_sakiko_tsuki` × 2 v1 lines (E9 through the alias) (4).
  - **zh:** 8 voices, 6 v1-era plus Eve and Hagumi (8).
  - **unch_ja:** 6 unchanged voices (6).
  - Plus 6 hidden repeats with the sides swapped and 3 null pairs: 60 trials over two pages.
- **Absolute (yes / partly / no next to a real clip):** the 9 `_ournotes` ids × 2 and the 16 new voices × 2: 50 trials.
- **The listener is asked to take about 30 s per trial, with headphones** (E8's round went at about 14 s per trial).
- **Rule.** E9 is adopted only if all of these hold:
  1. The screens pass.
  2. **v1-era voices (33):** not worse than v1, i.e. mean ≥ 0 or two-sided Wilcoxon p ≥ 0.05; **and** no character's three items sum to ≤ −5 (E8's Anon and Umiri were −7).
  3. **persona + zh + unch_ja (18):** not worse (same test).
  4. No voice is flagged "wrong voice" twice.
  5. Weak new voices ≤ 3, and weak Our-Notes-era ids ≤ 2. Weak means no "yes" among its 2 answers and at least one "no".
  6. **Consistency:** ≥ 4 of 6 repeats agree, and every null is within ±1. If only this fails, the pair pages are repeated once with fresh randomisation, and that repeat decides.

  Otherwise v1 stays.

## 7. If adopted: release

1. `release/MODEL_CARD_v2.md`:
   - 72 voices plus 2 aliases;
   - the era ids explained;
   - weak voices listed;
   - selection by listening;
   - the v1 card's unofficial / AI-generated / takedown wording kept.
2. `release/publish_hf_v2.py`, adapted to E9: 72 voices and aliases (it currently asserts 65 voices and `set(spk) == set(names)`); E9's decision file.
   - Tag `v1` first, then upload the model only (no audio, transcripts or reference lists).
3. Space: `MODEL_VERSIONS=v2=main,v1=v1`.
4. GitHub: E8 and E9 code and results.

## 8. Amendments

1. **Scoring of `_ournotes` ids (2026-11-09 01:58, smoke stage, before any training).**
   - `bdscore.score` looks up each row's own probe class (`classes.index(row)`), and probeE6 has no `_ournotes` classes, so the smoke eval stopped.
   - `e6/e6_eval.py` now scores an `X_ournotes` render against probe class X and restores the id afterwards. That is exactly what §5 already prescribes (`e9_screens.probe_name`).
   - The queue restarted at 02:00, reusing the finished smoke training. No rule changed.
