# E0 pipeline review: correctness, maths, methodology

Scope: `bangdream/pipeline/` b01–b07 + `bdcommon.py`, and the `kurisu_v2/pod` code they call (`p03_build`, `metrics`, `accent_metrics`, `train_sft.speaker_embedding`). Outputs in `data/`, `vectors/`, `refs/`, `e0/` were tested read-only. No pipeline file was modified.

Tags: **[run]** = reproduced by a check script in this folder (`check1_data.py` … `check10_delta_splithalf.py`, run with the `.venv-tts` python, CPU only). **[read]** = from reading the code. The speaker-encoder checks (6–8, 10) load only the 76 `speaker_encoder.*` tensors from `model.safetensors` into `Qwen3TTSSpeakerEncoder` on CPU, in fp32. They reproduce b03's bf16 clip vectors at cos 1.0000 and b05's stored p_true to within 0.007 (1587/1590 identical argmax).

---

## Critical

### C1. The probe judges renders in a different channel from the one it was trained on, and b07's control was read the wrong way [run]
**Where:** `b03_vectors.py:140-149` (probe fit on real clips only), `b05_score.py:98-105` (renders scored with that probe and the real-clip `mu`), `b07_resynth_check.py:72-78` (the control is summarised only as argmax leak and mean p_true). Downstream: `e0/e0_results.md:37-40` and the E1 gate in `:153-155`.

This is a methodology flaw, not a code bug: the code computes what it says. But it silently distorts the leak numbers that the E1 plan acts on.

**What b07 actually shows.** For each held-out clip, the codec encode→decode shift in the probe's centred unit space is mostly one common direction δ: |mean shift| / mean |per-clip shift| = 0.73 (`check6`). That direction raises exactly the anime/Mujica logits (`coef @ δ`):

| class | sakiko_tsuki | nyamu | uika | sakiko | mortis | umiri |
|---|---|---|---|---|---|---|
| Δlogit | +1.27 | +1.20 | +0.77 | +0.70 | +0.50 | +0.40 |

cos(δ, sakiko_tsuki centroid) = 0.36. Real clips survive this shift because their median own-vs-best-other logit margin is 3.45 and the codec costs only 0.65 of it. That is why b07 shows leak going only from 1.0% to 1.7%. Renders have small margins, so the same shift tips them over, and they tip toward the same place the floor leaks go (`mujica_sakiko_tsuki`, 145 of 279).

**Sensitivity test.** I removed the mean codec shift from every floor render vector and re-scored (`check7`). I then repeated it with δ estimated on the other half of the rows, 200 random splits (`check10`), and with δ from the 31 rows that have zero leak only:

| | uncorrected | δ removed | cross-half δ (2.5–97.5%) |
|---|---|---|---|
| ja leak | 75/414 = 18.1% (b05's stored bf16 run: 74) | 28/414 = **6.8%** | 6.5–7.2% |
| zh leak | 19.1% | 14.9% | 14.8–15.7% |
| en leak | 14.3% | 7.7% | 7.4–8.4% |

δ from the zero-leak rows only gives 7.0% ja.

Per voice, Japanese leak. The uncorrected column is b05's stored result; the CPU recompute differs only for eve (6/10). The corrected column gives 95% Wilson CIs:

| voice | uncorrected (b05) | δ removed (CI) |
|---|---|---|
| ras_pareo | 6/8 | **0/8** (0.00–0.32) |
| hhw_kokoro | 6/10 | **0/10** (0.00–0.28) |
| hhw_hagumi | 7/8 | 2/8 (0.07–0.59) |
| ppp_rimi | 6/8 | 2/8 (0.07–0.59) |
| ras_lock | 10/10 | 4/10 (0.17–0.69) |
| ppp_kasumi | 10/10 | 8/10 (0.49–0.94) |
| pp_eve | 5/10 | 5/10 (0.24–0.76) |

After correction the destinations spread out (pareo 31, soyo 18, sakiko_tsuki 17). Across rows, the codec p_true drop correlates with floor p_true: Spearman ρ = −0.52, p = 1e-4.

**Premise check (`check11`): do renders carry the same channel?** For every row, I computed the render shift (mean floor-ja vector minus mean real held-out vector). It aligns with δ at a median cos of **0.60** (range 0.16–0.68, n = 49; a random 2048-D direction gives sd 0.022). This holds equally in the 31 zero-leak rows, where identity is intact (median 0.60), and in the 18 leaking rows (0.59). The render shift's projection on δ is **1.36 × |δ|** (median), so renders carry at least the full codec shift. Removing one δ is, if anything, an under-correction along that direction.

**Failure scenario.**
- "The floor's 18% leak is therefore real voice collapse" (e0_results:39) is not supported: more than half of the Japanese leak follows the codec channel direction.
- "Converge on sakiko_tsuki = the Base default voice" (:40) is mostly that same channel direction.
- The planned E1 gate ("p_true above that row's real-clip 25th percentile", :154) sets its threshold in the no-codec channel. Every Base render carries the codec, so the gate would reject renders systematically, and unevenly by voice (pareo and kokoro especially). That changes which voices get pseudo-zh rows and which are pushed to round 2 (:155).
- Kasumi, eve and lock are still weak after correction (p_true 0.12 / 0.19 / 0.16 against real 0.61 / 0.82 / 0.79), so some collapse is real. But the list and the counts are wrong.

**Fix.**
- Pass the **train** clips through the 12 Hz codec, as b07 already does for held-out clips.
- Fit the probe on resynthesised vectors (or on real ∪ resynthesised), and centre with the resynthesised train mean.
- Take every baseline and gate threshold from **resynthesised held-out** clips, never from raw real clips.
- Until then, report the δ-corrected numbers alongside the raw ones.
- Save render vectors and full probability vectors in b05 so that checks like this do not need the encoder re-run.

---

## Major

### M1. The "floor identity gets worse as data grows" conclusion is not supported [run]
**Where:** `b06_report.py:44` (pooled leak), `:56-62` (bucket means); e0_results:35-36 and table :129-135.

`leak` pools ja with zh/en. About 24 of the ~34 clips per row are zh/en, which have no real-clip baseline and are out of the probe's language. Even so, the trend is not significant (`check3`, `check5`):
- Spearman(minutes, all-language leak) = 0.20 (p = 0.17). Spearman(minutes, ja leak) = 0.10 (p = 0.51).
- 30–60 min bucket minus <10 min, row bootstrap: all-language +0.117 (95% CI −0.06 … +0.28); ja +0.026 (−0.16 … +0.22).
- Japanese-only bucket means are not monotone: 0.156 / 0.067 / 0.229 / 0.182.

**Failure scenario:** E1 planning acts on "the fine-tune has the most work where it has the most data" and "the <10 min risk bucket starts from the best floor". Neither is supported. The pattern is a handful of bright game voices, and C1 changes several of those too.

**Fix:** bucket on Japanese leak only, with row-bootstrap CIs, and show the languages as separate columns. Drop the dose interpretation until E1 has more texts and seeds.

---

## Minor

### m1. `metrics._norm_text` lowercases before `reading()`, so the Latin keys in BD_READINGS never match in b05 [run]
**Where:** `kurisu_v2/pod/metrics.py:141-144`, called from `b05_score.py:68-69`. Examples:

| input | `C.reading` (b02) | `_norm_text` (b05) |
|---|---|---|
| Roselia | ロゼリア | アルオエスイエルアイエ |
| MyGO | マイゴ | エムワイジオ |
| CiRCLE | サクル | シアイアルシエルイ |

The same holds for every other Latin key (Poppin'Party, RAISE A SUILEN, Chu², …).

**E0 impact: none.** The held-out references exclude Latin by construction (`b03:59-60`), and only 1 of 828 ja ASR hypotheses contains Latin ("Tプレス", where the CER is unchanged).

**Failure scenario:** E1 novel texts, or ASR output, containing band names get a hugely inflated ja CER, on both reference and hypothesis.

**Fix:** NFKC + lowercase the table keys inside `reading()`, or apply `reading` before `.lower()` for ja.

(The in-place READINGS patch itself does reach `metrics`: `_norm_text('有咲')` returns アリサ [run].)

### m2. b07 writes stale `sim`, `cer`, `asr` and `dur` into `resynth/scored.jsonl` [run]
**Where:** `b07_resynth_check.py:44` (`{**r, ...}` copies the real row) and `:63-68` (only `p_true` and `pred` are recomputed). In 414/414 rows, `sim` equals the real clip's value. Anyone reading resynth sim sees real numbers, and the channel effect on sim was never measured (see m4).

**Fix:** drop those keys, or recompute them.

### m3. The centred-cosine `sim` is biased in two directions; b06 calls real sim "the target" [run]
**Where:** `b05_score.py:98-100`, `b06_report.py:8`.

- **Target bias.** Renders are conditioned on the centroid itself. Real clips are samples scattered around it.
- **Channel bias.** The codec alone lowers the row-mean sim from 0.653 to 0.597. The effect is row-dependent, from −0.107 (pp_aya) to +0.06 (sakiko_tsuki) (`check8`).
- Floor ja sim exceeds the codec-resynth target in 13 rows (chisato +0.16, sakiko_tsuki +0.10, sakiko +0.10, …).
- Within the floor, sim and p_true rank clips alike (ρ = 0.86), so sim is usable within one row and one condition only, as e0_results already says.

**Fix:** use resynthesised held-out sim as the target. Never compare floor sim with real sim across rows.

### m4. The pooled Japanese leak CI ignores clustering, and one seed confounds seed with text [run]
- e0_results:30 gives 74/414 with CI 14–22% (binomial). Clips cluster within a voice (one vector each); a row-cluster bootstrap gives **10.0–27.0%**.
- The per-voice Wilson CIs are the honest ones, and they are wide: kasumi ja 10/10 → 0.72–1.00; all languages 30/34 → 0.73–0.95.
- With one seed per text, the CI covers text and seed variance together. No seed-reliability estimate exists; Kurisu p10_e0 warns about exactly this.
- The E1 plan of ~20 texts × 2 seeds is the right fix.

### m5. b01 dedup keeps the first filename even when duplicate filenames give different transcripts [run]
**Where:** `b01_build.py:87-92`. All 46 duplicates are within a set; none cross sets. Dedup is otherwise correct: the manifest has 0 duplicate sha256 values, 0 missing files and 0 orphan wavs.

Of the duplicate groups, 23 differ only in punctuation or spacing, and 5 differ in the words themselves, for example:
- kept `うん、ひなたぼっこ`, other `ん……ひなたぼっこ`
- kept `ん？時間通り`, other `時間通り`

**Fix:** when byte-identical clips disagree on text, keep the variant with the lower ASR CER, or flag the group.

### m6. Minutes and buckets in b06 [run]
**Where:** `b06_report.py:25`, `:36`; `b01_build.py:100`.

- Minutes include the 0.37 s pad per clip (35 min in total) and count test and review clips.
- Five rows (ag_tomoe, ag_tsugumi, hhw_kaoru, ppp_tae, ro_sayo) are 30.25–31.46 min padded but 28.7–29.95 min of train/ok audio, so they sit in the wrong bucket for a training-dose question.
- `BUCKETS` uses hi = 99, so a row of ≥ 99 min would silently vanish from the table (none exists today).
- b01 rounds per clip while summing; the worst error is 0.15 min.

**Fix:** bucket on unpadded train/ok minutes, use an open upper bound, and round only once at the end.

### m7. b05 does not gate SER on `m23_valid`; Kurisu p10_e0 does [run]
**Where:** `b05_score.py:116-118` vs `p10_e0.py:340-344`. No effect today: all 16 zh texts are m23-valid. The gate would matter as soon as the text set changes.

**Fix:** copy `m23_valid` back and set SER/TER to None when it is false.

Related note: the zh SER of 0.077 includes the 6 cap-hit runaways, with SER 0.67–0.93. Without them it is 0.071. The Kurisu anchor does not filter runaways either, so the comparison is consistent, but both numbers are worth reporting.

### m8. Write-up errors [run]
- e0_results:40 says "145 of 287 misassigned". The actual count is 279 (74 ja + 150 zh + 55 en).
- The held-out set is the easier population (2–12 s, ≥ 8 kana, ok transcript). Train clips that meet the same filter score p_true 0.825, against 0.780 for the rest. The 99.0% real accuracy is therefore slightly optimistic. Renders use the same texts, so the floor-vs-real comparison is still matched.
- In `floor_table.json`, `floor_en.lid_ratio` means log P(ja) − log P(en): the same key with a different definition. b06 does not use it.

---

## Verified clean
- **Split leakage [run]**
  - 414 test clips (4–10 per row); 0 test clips in any ref list; 0 ref clips from another row or failing the ok/≥ 2 s rule.
  - `mu` equals the train-only mean (max |Δ| 4e-7). The same `mu` (from `probe.npz`) is used in b03, b05 and b07.
  - Locked vectors only see refs, which are train-only.
  - No near-duplicates across the split: the test→train centred nearest-neighbour cosine peaks at 0.928, against a train→train p99 of 0.905. No test text exists elsewhere.
- **Labels [run].** `rsplit("_", 1)[0]` equals the manifest row for all 9,804 ids. There are no duplicate ids, no sha10 collisions, and no underscore in any hex part.
- **Probe maths [run]**
  - sklearn 1.9.0 has no `multi_class` argument; lbfgs is multinomial.
  - The hand softmax equals `predict_proba` exactly on a refit (max |Δ| 0.0).
  - b03's reported per-row p_true (from `predict_proba`) matches b05's hand softmax to within 0.0005.
  - `classes == rows` (sorted), consistent with `clf.classes_ == arange(49)`.
  - Train accuracy 99.5% vs held-out 99.0%; the 5-fold CV figure (98.7%) is reported separately, so both are honest.
  - `class_weight="balanced"` is sensible: class sizes run 40–424, and it gives a uniform prior. An isotropic random input goes to pp_aya (the largest intercept), not to the sink, so the sink is not an intercept artefact (which is what led to C1).
- **Locked vector [run].** `speaker_embedding` rescales the mean to the mean clip norm. The factor is only 1.005–1.020, so cos(nrm(T − mu), nrm(mean_refs − mu)) ≥ 0.994 and separability is unchanged (median −0.029 vs −0.039, max 0.822 vs 0.821).
- **Mel and encoder input [read + run].** b03, b05 and b07 use exactly train_sft's recipe (24 kHz, n_fft 1024, 128 mels, hop 256, win 1024, fmin 0, fmax 12000). All renders are 24 kHz, so b05's `resample_poly` path is never used; it is correct anyway. Loudness is not a confound (median LUFS: real −19.9, resynth −20.4, floor ja −21.4). b07 does not downmix stereo, but the codec output is mono.
- **CER/WER [run].**
  - Recomputing ja CER from the stored ASR gives 0/200 mismatches.
  - Readings are applied to both reference and hypothesis.
  - `is_runaway` gets `dur`/`dur_ratio`; it flags 6 clips, all at the 400-frame cap (31.92 s).
- **Accent [read + run].** `score_rows` handles absolute paths; `shared_positions` reaches 784/784 zh rows. All five keys come back on every zh row, and en rows get lid. lid_ratio = log P(ja) − log P(zh), where higher means more Japanese, as b06 says.
- **Aggregation [run].** b06's n-weighted leak equals the exact all-language share in every row. pids are unique, and the floor-ja pids equal the real pids with no collision (accent joins only on zh/en).
