# train_sft.py review: going from 3 to 49 shipped speakers plus replay (E1)

Reviewed 2026-10-30. The files were read but not modified.

**Scope.** `kurisu_v2/pod/train_sft.py` (1,116 lines), plus `ft_config.py`, `mix_v3.py`, `p04_codes_v3.py`, `p05_replay.py`, `p11_pseudo_zh.py`, `queue_v3.sh` and `bootstrap_v3.sh`. Upstream code: `qwen_tts` 0.1.1 and transformers 4.57.3 from `kurisu_qwen3tts/finetune/.venv-tts`.

**Method.** Each finding is marked by how it was checked:

- **[run]**: verified by running a CPU-only script. The script is `E:\STEINS;GATEVoice\bangdream\review\check5_trainer_verify.py`; run it with `.venv-tts\Scripts\python.exe -I`. It imports `train_sft` and drives `CodedDataset.collate`, `length_batches`, `speaker_table` and `save_checkpoint` with mock data: 49 shipped and 55 aux speakers, using the real Base `config.json`. It also reads the codec embedding row norms straight from the Base and CustomVoice safetensors. The 1.7B model is never loaded.
- **[read]**: checked by reading the code.
- **[log]**: checked against the Kurisu v3 pod logs in `kurisu_v2/artifacts/v3/logs/`.

## Bottom line

**Nothing in the core training path breaks or trains wrongly at 49 shipped + 55 aux speakers.** The following all check out:

- the speaker table and its capacity;
- the per-row speaker lookup;
- both prefill layouts and their labels, compared against `generate()`;
- the sub-talker shift;
- the schedule;
- the 49-row export.

The remaining risks are of three kinds:

- **The trainer accepts bad speaker setups without complaint.** A shipped speaker with zero training rows is exported anyway, and a cached speaker table is reused even if its reference lists changed.
- **The diagnostics do not scale to 49 similar voices.** Only the first 8 cosines are printed, and there is no per-speaker count or validation loss.
- **The data scripts around the trainer assume one speaker.** This applies to `p11_pseudo_zh.py` in particular, and partly to `mix_v3.py` and `queue_v3.sh`.

## Findings

| # | Severity | Where | Failure scenario | Fix | How checked |
|---|---|---|---|---|---|
| 1 | **major** | `train_sft.py:729-743`, `:864-866`, `:871-878`; `mix_v3.py:40,60,68` | Three cases: a speaker listed in the trainer's `--speakers` is missing from (or misspelt in) `mix_v3.py --speakers`; all of its clips exceed `--max_frames`; or the builder spells the name differently. In each case that speaker has **0 training rows**. It still gets codec row `3000+i` and a `spk_id` entry in the export. It then sounds like a Base x-vector clone, and nothing warns. `mix_v3` silently drops rows for any speaker not in its `--speakers` set (default `kurisu`). The trainer only checks that each row's speaker is known, not that each speaker has rows. `[data] speakers` prints only the top 12. | After building `spks` (line 875), fail if any `saved_speakers` name has fewer than N train rows (e.g. 20), and print all 49 counts. In `mix_v3`, fail if a requested speaker yields 0 rows, and warn about source speakers that were not requested. | [run]: 25 of 49 mock speakers had no rows and were accepted |
| 2 | **major** (needed before E1's pseudo rows) | `p11_pseudo_zh.py:98-99,113-115,121,126,145-146,158-161` | The script is single-speaker by construction: one reference list (`ref_kurisu.txt`), one vector, and `speaker: "kurisu"` written into every row (line 161). As written, the trainer would stop with `row speaker 'kurisu' has no embedding`, which at least fails loudly. A naive loop over speakers fails silently instead, in three ways:<br>(a) The similarity gate uses `p06_eval_v3.centroid()` (Kurisu's v2 test takes) and one global 25th-percentile floor (line 126).<br>(b) `best` is keyed by `utt` (line 145), so all speakers compete for each text and only one speaker per text survives.<br>(c) The file names `{u}_s{s}.wav` (line 112) and `pseudo_zh_{u}.wav` (line 158) are not speaker-qualified. If two speakers share a text, one wav overwrites the other, and rows labelled speaker A carry speaker B's audio. | Loop over speakers, giving each its own reference list and vector, its own ECAPA centroid, and its own similarity floor. Put the speaker in `pid`, `id` and the paths, key `best` by `(speaker, utt)`, and assign disjoint texts per speaker. Budget the work: the current 900 texts × 2 samples is for one voice, so a few hundred rows over 49 speakers needs roughly 10–20 texts per speaker. | [read] |
| 3 | minor | `train_sft.py:313-321`; guards at `:692-700` | This is harmless in the queue but becomes a real problem if you rerun by hand. `spk_embedding.pt` is matched on the list of names only. If you rerun in the same `--out` with the same names but edited `ref_*.txt` files, the old vectors are reused silently. The "refuse to mix runs" guard only looks for `state.pt`, and a `--stop_after 1.0` run never writes one: `StopTraining` is raised at `:1051`, before the epoch-end save at `:1079`. Such a rerun also appends to `train_log.jsonl` and `val_log.jsonl`. Reordering the names does fail loudly. `queue_v3.sh:35` runs `rm -rf` on the run dir first, so the queue path is safe. | Key the cache on a hash of (name, ordered ref paths, file size and mtime) and the init model. Refuse to start if the run dir already contains `checkpoint-step-*` or `train_log.jsonl`. | [run]: same names, new refs → reused |
| 4 | minor | `train_sft.py:323-329` | This is a diagnostic gap, but the design risk behind it is real. The speaker signal is one frozen x-vector per speaker, and those vectors sit close together. In Kurisu v3 the cosines were 0.981 (kurisu/amadeus) and 0.89–0.98 between unrelated AISHELL and Kurisu voices. Forty-nine voices from one franchise, mostly young women, are likely to cluster even tighter. Two failures would go unnoticed: a near-duplicate pair, or a copy-paste ref list (cosine 1.0). Only `names[:8]` is printed, and those are speakers 1–8 only. | Print each shipped speaker's nearest neighbour and cosine, and warn above about 0.99. Check whether any two ref lists share clips. | [log] (Kurisu cosine table) |
| 5 | minor | `train_sft.py:530-531` | The number of saved speakers is not checked until save time. With 73 or more, `weight[3000+i]` raises `IndexError` after the whole run. 49 fits: rows 3000–3048, with 23 spare. | Add `assert len(saved_speakers) <= vocab_size - SPK_ROW0` right after line 734. | [run] |
| 6 | minor | `mix_v3.py:116-121`; `p05_replay.py:860-863` | With p05's defaults, the replay pool is 1,713 train rows (1,200 zh + 513 en). With 9.8k JA rows, `--replay_share 0.15` asks for 1,729 rows and `SystemExit`s. 0.14 asks for 1,595 and passes, but stops working above about 10.5k JA rows. | Raise `--zh_hours` / `--en_hours` in p05 (e.g. 2.0 / 1.0) or keep the share at 0.14 or less. | [run]: arithmetic on the Kurisu `replay/report.json` |
| 7 | minor | `train_sft.py:552-571`, `:907-910` | There is one aggregate validation loss across 49 speakers plus replay, so a single speaker going wrong cannot be seen. Validation also always uses the explicit-language layout (`train=False`), so the Auto path is never validated. If `--max_val_batches` stays below the number of val batches, the batches are length-sorted (`shuffle=False`), so only the *shortest* clips are kept. The queue's value of 1000 hides this. With a 3% split, a 3-minute speaker has 1–2 val clips. | Log validation loss per speaker (frame-weighted) and per language. Shuffle before truncating. Guarantee at least 5 val clips per shipped speaker in the builder. | [read] |
| 8 | minor | `queue_v3.sh:32,41,50,55-62`; `mix_v3.py:73,78,88,108,143`; `p04_codes_v3.py:19-22`; `train_sft.py:689,1110`; `ft_config.py:16` | **Kurisu hard-codes that must change:**<br>- `queue_v3.sh` passes `--speaker_name kurisu`, so rows without a speaker field raise an error (loud).<br>- `queue_v3.sh` uses the `kurisu_v3.json` checkpoint marker and the Kurisu eval suite (`score`).<br>- `mix_v3`'s v2-id pinning (line 73) is Kurisu-only; do not pass `--v2_ids`.<br>- `mix_v3` reads `r["text_plain"]` when `--text plain`, so the BanG Dream builder must write that field or `mix_v3` raises `KeyError`.<br>- `mix_v3`'s literal `phone` checks are harmless.<br>- `mix_v3`'s summary shows only `most_common(12)` speakers.<br>- p04's `ja` source is `build_v3/kept.jsonl`.<br>- `KURISU_WORK` defaults to `/workspace/kurisu`.<br>- The report name is `ft_02_train_{speaker_name}-full-r32.json` for every run, so runs without `--name` overwrite each other's report.<br><br>p05 has no Kurisu or single-speaker assumption. Its aux names (`zh_ssb…`, `en_…`) cannot collide with band names, and duplicates are rejected at `:739`. | Parameterise the paths and the default speaker. Pass `--name`. Write `text_plain` in the builder. | [read] |
| 9 | minor (comment only) | `train_sft.py:483` | The comment "rows 3000+ are untrained in Base **and CustomVoice**" is false for CustomVoice. CustomVoice's rows 3010, 3061, 3065 and 3066 have norm 13.9–15.1 (serena/vivian/uncle_fu/ryan). In Base, all of rows 3000–3071 have norm 0.0216–0.0236. This only matters if the init is ever not Base. | Correct the comment, or assert at startup that rows `3000..3000+n` have norm below 0.1. | [run] |
| 10 | minor (design) | `train_sft.py:236-255`, `:899-901`, `:975` | There is no per-row or per-speaker weighting; every row is drawn exactly once per epoch. The 1-epoch recipe was tuned when one speaker had 76% of rows (kurisu, 8,333 of 10,970). In E1 a 3-minute speaker (~45 clips) contributes about 45 examples to the whole run, against about 900 for a 60-minute speaker. | See §5 below. | [read] / [log] |

## Answers to the eight checks

### 1. Speaker table capacity and layout [run]

**Capacity.** The codec vocabulary has 3072 rows (`talker_config.vocab_size`), so rows 3000–3071 give 72 slots. 49 shipped speakers use 3000–3048, leaving 23 spare. In Base, all 72 rows are untrained (norm about 0.022).

**Aux speakers get a vector only, never a codec row.** Every training row's speaker slot is filled the same way: the codec input is set to id 0 and masked (`collate` `:206,217`), then overwritten from the in-memory `spk_table` (`build_inputs` `:419`). Codec rows 3000+ are never read during training. `save_checkpoint` writes only `spk_table[0:len(saved_speakers)]`, and `specs` puts the saved speakers first (`:734-738`). In the run, row 3049 stayed untouched and no aux name was exported.

**No collisions.** The special codec ids span 2050–2157: language ids 2050–2071, then pad 2148, bos 2149, eos 2150, and the think tokens 2154–2157. `generate()` suppresses ids 2048–3071 except eos (`modeling_qwen3_tts.py:2059-2063`), so a speaker row can never be sampled as output.

**`spk_id` in the export.** It is `{name: 3000+i}` for the 49 names, lowercased at `:730`, and `spk_is_dialect` is all `False`. `Qwen3TTSConfig.from_pretrained` on the exported config resolves the names as follows:

- `ppp_kasumi`, `PPP_Kasumi` and `PPP_KASUMI` all pass both checks. `_validate_speakers` and `generate` both lowercase the name (`qwen3_tts_model.py:183`; `modeling:2091,2094,2120`).
- `ppp-kasumi` fails, because hyphens are not normalised to underscores. Underscores are fine.

### 2. Batching and conditioning [run]

**Lookup.** `spk_index` maps the lowercased name to its table row (`:738`). Each row stores its index (`:123`), `collate` carries it as `spk_idx` (`:221`), and `build_inputs` gathers it with `spk_table[spk_idx]` at `spk_pos`. The run confirmed that every row's index matched, including an upper-case row speaker.

**Unknown speakers.** A row whose speaker is not in the table raises `ValueError` while the dataset is built (`:115-116`). That covers the train and val sets and any default-speaker fallback. The reverse case is not caught: a speaker in the table with no rows (finding 1).

**Cache.** The speaker-table cache is keyed by the ordered list of names only (finding 3).

### 3. Language-id path [run + read]

The run rebuilt `generate()`'s non-streaming custom-voice input symbolically and compared it with `collate`, position by position. It checked the text and codec channels, the masks, the speaker slot, the labels, the padding and the `codec_mask` frame span. Both layouts match:

- **Explicit language:** `[think, think_bos, LANG, think_eos, SPK, pad]`, with the speaker at position 7 and text starting at 9.
- **Auto:** `[nothink, think_bos, think_eos, SPK, pad]`, with the speaker at 6 and text starting at 8.

Against the issues in the research notes:

- **#323:** the language ids come from the config (`japanese` 2058, `chinese` 2055, `english` 2050), not the bogus 4206, and the speaker slot is kept.
- **PR #361:** the Auto layout here is the 3-slot one that `generate()` builds, not #361's padded 4-slot prefix.
- **#337:** the sub-talker reads `hidden[:, :-1][codec_mask[:, 1:]]` (`:464`), which is the state that predicted frame *t*. This is [read], and it matches `forward()` feeding `past_hidden` at `modeling:1666`.

**Label shift.** The main-head labels go in unshifted. In transformers 4.57.3, `LOSS_MAPPING["ForConditionalGeneration"]` is `ForCausalLMLoss`, which does the shift itself [read].

**`p_auto`.** With `p_auto 0.25`, the observed Auto share was 0.248. The swap is applied to every explicit row in training, including zh/en replay and pseudo rows, and never to validation.

### 4. Schedule maths [run, matches log]

The settings are micro-batch 16 × accumulation 2, `--epochs 2`, warmup ratio 0.03 and `min_lr_ratio` 0.1.

| Train rows | Steps/epoch | Total steps | Warmup steps | Stop step |
|---|---|---|---|---|
| 10,970 (Kurisu final) | 343 | 686 | 20 | 343 (matches `train_final_s20261028.log`) |
| 11,400 | 357 | 714 | 21 | 357 |
| 11,700 | 366 | 732 | 21 | 366 |
| 12,000 | 375 | 750 | 22 | 375 |

**What the learning rate does.** It climbs linearly to 5e-6 over about 21 steps, then follows the cosine. `--stop_after 1.0` is always halfway through the 2-epoch cosine, so the last step runs at **2.87e-6 (57% of peak)**. The mean over the run is 4.13e-6. Both numbers are independent of the row count.

**Consistency with Kurisu.** About 11–12k rows give 357–375 steps, close to Kurisu's 343. Nothing in the trainer scales with the number of speakers: no learning-rate or loss scaling. Loss is a token mean per micro-batch, divided by the accumulation window size.

### 5. Sampling: no hook exists [read]

There is no `weight` field, sampler or temperature anywhere. Batches come from `length_batches` over all rows, once per epoch. Because the trainer has no DataLoader, `WeightedRandomSampler` does not plug in. Two correct places to add weighting:

**A. Mix stage (simplest, no trainer risk).** `mix_v3` already repeats rows (`event_repeat` / `phone_repeat`, `:105-110`). Add a per-speaker integer repeat, e.g. `round((n_max/n_s)^(1-α))` capped at about 4.

- Because the repeats are appended before the replay and pseudo targets are computed (`:110` comes before `:116,124`), **replay and pseudo shares are preserved automatically**.
- Validation is untouched.
- `steps_per_epoch` simply grows.
- **Caveat (finding 6):** repeats raise the JA row count that the replay target is computed from. At share 0.14 the current 1,713-row pool supports only about 10.5k JA rows, so even an average ×1.1 repeat over 9.8k rows makes `mix_v3` stop with `SystemExit`. Grow the replay pool in p05 first, or compute the replay target from the unique (pre-repeat) JA rows.

**B. In the trainer.**

- Store `w = row.get("weight", 1.0)` on each item.
- At `:899` and `:975`, build a fixed-size multiset of N = `len(dataset)` indices with **systematic sampling**: each row appears ⌊N·wᵢ/Σw⌋ or ⌈N·wᵢ/Σw⌉ times, seeded with `seed+epoch`.
- Pass the multiset to `length_batches` (add an `indices` argument).
- Avoid i.i.d. sampling with replacement: it leaves about 37% of rows unseen per epoch even with uniform weights.

Interactions to handle with option B:

- **Steps per epoch.** `len(batches)` depends only on the number of indices (64-batch windows), so a fixed N keeps `steps_per_epoch`, warmup and `stop_step` consistent. The planning call at `:899` must use the same multiset.
- **`--stop_after 1.0`.** One "epoch" now means N draws. Rows of small speakers repeat within the run; at ×4, each of a 45-clip speaker's clips is seen about 4 times at lr ≈ 4e-6, so watch for memorisation. Rows of big speakers are partly skipped.
- **Weights must be set per group.** A temperature taken over all 104 table speakers would inflate the 55 aux speakers (about 30 rows each) and break the 14% replay share set in `mix_v3`. Apply the temperature over the 49 shipped JA speakers only, then rescale so the JA, replay and pseudo totals keep their row shares.
- **Validation stays unweighted.** It uses a separate dataset and the `shuffle=False` path. The batch-size probe sorts `dataset.lengths` and is also unaffected; the queue passes `--batch_size 16` anyway.
- **Log exposure, not row counts.** `[data] speakers` would then misreport what each speaker actually trained on.

### 6. Memory and time [read + log]

**Per step.** The only per-step speaker cost is the `spk_table[spk_idx]` gather, which is O(batch). Nothing is O(n_speakers).

**Startup.**

- The speaker encoder runs one forward pass per reference clip, sequentially. Kurisu's final run did 1,247 clips over 58 speakers; E1 will be about 1,100 aux clips plus the 49 reference lists, which takes seconds to a few minutes on an H100.
- The table is 104 × 2048 fp32, under 1 MB.
- Tokenising about 11k rows took 4.4 s in Kurisu.

**VRAM.** Kurisu peaked at 36.6 GiB with a maximum sequence of 303 tokens, and VRAM does not depend on speaker count. `--batch_size 16` skips the worst-case probe. If BanG Dream clips run close to `--max_frames 400` (32 s), run one `--batch_size auto` smoke test or check the `[data] max` line first.

### 7. Checkpoint export [run; full model load by reading]

**Verified by running:**

- `model.safetensors` rows 3000–3048 equal `spk_table[0..48]`.
- The values are bf16-rounded, identical to what was trained on (`:743`).
- Aux speakers are excluded, and the `speaker_encoder.*` keys are dropped.
- `tts_model_type` is `custom_voice`.
- `Qwen3TTSConfig.from_pretrained` exposes the 49 speakers.

**By reading only:**

- `copytree` carries `speech_tokenizer/`, `generation_config.json` and the tokenizer files.
- With `custom_voice`, `speaker_encoder` is `None`, so the dropped weights are not expected keys.
- `Qwen3TTSModel.from_pretrained` + `generate_custom_voice` therefore takes the same path as the published 3-speaker Kurisu v3.
- `generate_custom_voice` defaults to `language="Auto"`.

### 8. Kurisu and single-speaker hard-codes [read]

These are covered in findings 1, 2, 6 and 8. In short:

- **`p11_pseudo_zh.py`:** must be rewritten per speaker (finding 2).
- **`mix_v3.py`:**
  - `--speakers` must list all 49 names exactly (finding 1).
  - Do not pass `--v2_ids`.
  - JA rows need a `text_plain` field.
  - Keep `--replay_share` at 0.14 or less, or grow the replay pool (finding 6).
- **`queue_v3.sh`:** change `--speaker_name`, the `kurisu_v3.json` marker and `score()`. The `rm -rf` before each run is worth keeping.
- **`p04_codes_v3.py` and `ft_config.py`:** change the source paths and `KURISU_WORK`.
- **`p05_replay.py`:** fine as is.
