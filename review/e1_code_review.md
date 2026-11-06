# E1 code review: pre-flight before the paid H100 run

Reviewed 2026-10-30. No reviewed file was modified.

**Scope.** `bangdream/e1/{bootstrap_e1.sh, queue_e1.sh, e1_build.py, e1_codes.py, e1_mix.py, e1_pseudo.py, e1_eval.py, bdscore.py, e1_select.py}` against `E1_PLAN.md`. Also checked: the reused `kurisu_v2/pod` code (`train_sft.py`, `p05_replay.py`, `p11_pseudo_zh.py`, `p04_codes_v3.py`, `accent_metrics.py`, `metrics.py`, `ft_config.py`), and `qwen_tts` 0.1.1 / `qwen_asr` 0.0.6 from the local site-packages. I also checked the upload artefacts in `bangdream/_upload/`.

**Tags.**
- **[run]**: checked with a small CPU-only script or shell test, using the `.venv-tts` python with `-I`.
- **[read]**: checked by reading the code.
- **[log]**: checked against the Kurisu v3 pod logs and file mtimes in `kurisu_v2/artifacts/v3/`.

## Bottom line

No bug was found that would silently produce wrong training data or corrupt the evaluation. Interfaces, data separation and the pairing between conditions all check out (see "Answers to the checks" below).

There are four problems with the run as queued:
1. The replay step will probably abort the bootstrap.
2. The run will take about twice the budgeted GPU time.
3. Peak disk use is about 85–90 GB.
4. One pre-registered data property (replay speakers far from the 49 rows) is not implemented.

## Findings

| # | Severity | Where | Failure scenario | Fix | How checked |
|---|---|---|---|---|---|
| 1 | **critical** (likely crash, early, cheap to fix) | `bootstrap_e1.sh:45`; `p05_replay.py:317-353, 893-897` | `--zh_hours 2.0` keeps the defaults `--zh_speakers 30` and `--max_clips_per_speaker 50`. `walk()` hard-caps each speaker at 50 kept clips, so zh can reach at most 30 × 50 × 4.36 s (Kurisu's measured mean) = **1.817 h**. p05 exits with `SystemExit` below 0.9 × 2.0 = **1.80 h**, which leaves about 14 clips of margin. The selection is deterministic (same seed, so the same 30 speakers as Kurisu, who got 41–42 clips each for 1.5 h). Any speaker that runs out of usable clips, or tail clips that are a little shorter, puts the run under the line. Unlike `bootstrap_v3.sh`, there is no `\|\| echo WARN`, so `set -e` kills the chain at step 4. | Use the defaults (`--zh_hours 1.5 --en_hours 0.8`): they give 1,713 train rows, against the 1,509 that `e1_mix` asks for (0.14/0.86 × 9,268). Or keep 2.0 h and add `--zh_speakers 40` (or `--max_clips_per_speaker 70`). | [run] arithmetic; [log] `artifacts/v3/replay/report.json` |
| 2 | **major** (time and cost) | `e1_eval.py:52-59`, `:127-138`; `e1_pseudo.py:74-84`; `E1_PLAN.md` "Compute" | `plan()` gives 294 generate batches and 4,356 renders per condition [run]. On the H100, Kurisu v3's eval took 12–20 min for 82 batches (964 rows) and about 30 min for 2,516 rows (render + ASR, from file mtimes) [log]. That is roughly 9–12 s per batch, because generation is bound by per-step overhead rather than by batch size. E1 therefore needs about **45–60 min per condition**, not 25. With 7 render conditions (6 checkpoints + floor) that is **5–7 h of eval**. Pseudo-zh adds 5,880 renders (294 batches) plus scoring, about **45–60 min**, which the plan does not count. With training (2 × ~18 min; Kurisu did 1.25 epochs in 6:37) and the bootstrap, the total is about **8–9 h, ≈ $33–36**, against ≈ 4–5 h. | Pick any of these; none changes the pre-registered texts or seeds:<br>(a) Run the 3 checkpoint evals of an arm, the floor and `e1_pseudo` as parallel processes. Training peaks at 36.9 GiB and an eval process needs about 10–15 GiB, so E1a's evals fit beside E1b's training.<br>(b) Batch across voices within a language, e.g. 48 items. Both `generate_custom_voice` and `generate_voice_clone` take per-item speaker/prompt lists. The seed then becomes a per-(batch) label, and must still be the same for every condition.<br>(c) Use 1 seed for ep1/ep2 as a screen. Re-estimate the cost before launching. | [run] `plan()`; [log] mtimes; [read] |
| 3 | **major** (crash mid-E1b, depending on volume size) | `queue_e1.sh:25,27`; `train_sft.py:1063-1088` | `--keep_last 0` disables pruning, so `checkpoint-epoch-0/1` (2 × 4.5 GB, duplicates of the step checkpoints) stay until the queue's `rm -rf`. Every epoch end also writes `state.pt` with the fp32 AdamW moments: 2 × 1.917 B × 4 B = **15.3 GB**. It is written as `state.pt.tmp` beside the old one, so the epoch-1 end briefly holds **30.7 GB**. That file is useless in full mode, because `--resume` is refused (`:926-930`). Peak during E1b:<ul><li>E1b transient: 4 × 4.5 + 30.7 ≈ 49 GB</li><li>E1a step checkpoints: 13.6 GB</li><li>smoke checkpoint, never deleted: 4.5 GB</li><li>HF cache: ≈ 10 GB</li><li>wav24, tar + extracted: 7.6 GB</li><li>replay + pseudo: ≈ 4 GB</li><li>E1a renders: ≈ 3 GB</li></ul>That is **≈ 85–90 GB**. On a smaller `/workspace`, `torch.save` or `save_file` raises `OSError` mid-run. The trainer's free-disk check (`:1065`) only skips the epoch checkpoint; it still writes `state.pt`. | Use a volume of at least 120 GB. Also, or instead:<ul><li>`rm -rf runs/smoke` at the end of the smoke stage;</li><li>delete `wav24.tar` after extracting it;</li><li>pass `--keep_last 1`;</li><li>optionally, add a trainer flag that skips `state.pt` in full mode.</li></ul> | [read]; sizes from local files and the Kurisu `[save]` lines |
| 4 | **major** (pre-registered property not implemented) | `E1_PLAN.md` Data table ("Replay speakers far from all 49 rows by centred cosine"); `p05_replay.py`, `e1_mix.py` | Nothing computes or filters replay speakers by distance to the 49 rows: p05 has no speaker encoder, and `e1_mix` takes a random 1,509 of the pool. Kurisu v3 saw cosines of 0.89–0.98 between AISHELL and Kurisu vectors (trainer review finding 4). The p05 pool is 70% female, aged 14–40. An AISHELL speaker whose vector sits next to a BanG Dream row teaches "this region + Chinese → that AISHELL timbre", which is the zh identity drift the arms are meant to measure. | Before `e1_mix`, embed the aux lists the way the trainer does. Drop aux speakers whose centred cosine to any row is above a threshold fixed now (e.g. the 49 rows' own minimum nearest-neighbour cosine), and log the distances. Or amend the plan text. | [read] |
| 5 | minor (time on any rerun) | `bootstrap_e1.sh:50`; `e1_pseudo.py` (no skip) | Pseudo runs before `e1_codes` and both `e1_mix` calls, so a failure there also blocks E1a. `e1_pseudo` has no done-marker, so rerunning the bootstrap after any later failure re-renders all 5,880 clips (≈ 45–60 min). It also rewrites the wavs, which invalidates the pseudo code signature. | Skip `e1_pseudo` if `pseudo/pseudo_rows.jsonl` exists. Move it after `e1_mix --arm E1a`, or run it alongside E1a training (see finding 2a). | [read] |
| 6 | minor (guard calibration) | `e1_select.py:55,82`; `E1_PLAN.md` guard table | The Japanese CER guard (≤ 0.035, "floor 0.031") was calibrated on E0's held-out lines only. `stats()` pools held-out and novel lines, and novel lines are 1,176 of 2,004 Japanese renders. If the 12 novel sentences have a systematically different ASR CER, every checkpoint can pass or fail on that text pool rather than on the model. The same concern is weaker for the leak baseline; the plan already says leak is re-measured. | Apply the ja-CER guard to `kind == "ja_heldout"` rows (as calibrated), or define it relative to the E1-measured floor. Report both pools either way. | [read] |
| 7 | minor (silent no-op) | `queue_e1.sh:33` | `for ck in $(ls -d … \| sort …)`: a failing command substitution in a `for` list does not trigger `set -e`. If no `checkpoint-step-*` exists (e.g. `train_sft` caught a `KeyboardInterrupt`, exited 0, and the queue then touched `DONE`), the loop runs zero times and `STAGE_DONE E1a` is printed with no eval. The numeric sort itself is right (339 < 678 < 1017). | `n=$(compgen -G "…/checkpoint-step-*" \| wc -l); [ "$n" -eq 3 ] \|\| exit 1` before the loop. | [run] shell test |
| 8 | minor (stage order) | `queue_e1.sh:69-73` | `refs` (floor + real) needs only E1a's first checkpoint, but it is meant to run after E1b. If E1b crashes, there is no floor and `e1_select` cannot run on E1a alone. `e1_select` also raises `FileNotFoundError` if any listed candidate lacks `scored.jsonl`. | Run the stages as `smoke E1a refs E1b select`. Pass `--cands` for the arms that finished. | [read] |
| 9 | minor | `e1_pseudo.py:93` | The length gate uses the raw render's `dur`, measured before `normalise_audio` trims silence and re-pads. A kept row can end up just under 2 s. It is harmless to training. | Re-check `2 ≤ len(new)/SR ≤ 12` after normalising, or document it. | [read] |
| 10 | nit | `E1_PLAN.md` Data table; `e1_build.py:73` | The plan says "Japanese train 9,379; val = 1 per row + 1% (≈ 140)". The code makes `max(2, round(1%))`, which is 111 val and **9,268** train. The replay target is therefore 1,509 rows, not 1,527. `DROP_IDS` is a no-op, because that clip is already `review`. | Fix the plan text. | [run] |

## Answers to the checks

### 1. Interfaces with the reused code

**`train_sft.py` CLI** [read]
- Every flag the queue passes exists.
- `--save_at 1.0,2.0,3.0 --stop_after 3.0 --epochs 3` gives `save_steps = {spe, 2·spe, 3·spe}` and `stop_step = 3·spe`.
- The final step goes through one `if (… or global_step in save_steps or stop_now)` at `:1044`. That is one save, then `StopTraining`, so there is no collision and exactly **three `checkpoint-step-*` dirs**.
- `steps_per_epoch` is the same every epoch: `length_batches` always yields 64·(n//W) + ⌈(n mod W)/bs⌉ batches. So steps spe and 2·spe are the last steps of epochs 0 and 1.
- The epoch-end code writes `checkpoint-epoch-0`/`-1` and `state.pt`. `checkpoint-epoch-2` is never written, because `StopTraining` comes first.
- The cleanup line `rm -rf checkpoint-epoch-* state.pt` is correct (disk: finding 3).

**`--speakers` string** [read + run]
- `"$(cat speakers.txt)"` is quoted, so it is one argv element (2.7 KB).
- It splits on `,` into 49 `name=/workspace/bd/bundle/refs/ref_<row>.txt` pairs. No path contains a comma or a space.
- The names are already lower-case, so `.lower()` changes nothing. The smoke assertion on `spk_id` order and the 3000–3048 range matches `save_checkpoint`.

**`p05_replay.py`** [read]
- With `KURISU_WORK=/workspace/bd` it writes `/workspace/bd/replay/{replay_rows.jsonl, aux_speakers.json, wavs/}`, which `e1_codes` and `e1_mix` read.
- `aux_speakers.json` is `{"zh_SSB…"/"en_…": [≤ 20 train wav paths]}`, the `{name: [paths]}` format that `--aux_speakers` expects.
- Rows carry `speaker` (same case as the keys), `language` `chinese`/`english` and `split` `train`/`val`.
- The trainer lowercases both sides; `e1_mix` compares in the original case, and the two agree.

**`p11_pseudo_zh.texts(n, seed)`** [read]
- It returns `[(utt, hanzi + "。"/"？")]`, unique texts, from AISHELL-3 *test*, not in train, 8–30 hanzi, ≤ 1 numeral.
- **Not verified:** that the test set holds ≥ 2,940 such texts. The content file is not local. If it does not, `e1_pseudo` exits before loading any model.

**`p04_codes_v3`** [read]
- `signature(items)`, `SPLIT` (`train/val/validation/test`) and `BATCH = 48` are used exactly as in p04.
- The output names `codes/{ja,replay,pseudo}_{train,val}_with_codes.jsonl` match `e1_mix`.
- `ja_rows` `dur` matches the wav length (200 sampled, 0 off by more than 0.05 s) [run], so the ±3-frame check will not trip.

**`speaker_embedding` / `ref_paths`** [read]
- `speaker_embedding` returns `[1, 2048]`, and `[0]` gives `(D,)`, as `VoiceClonePromptItem.ref_spk_embedding` documents.
- `generate()` does `.view(1,1,-1)`, so either shape works.
- `ref_paths(Path(...txt))` returns `list[Path]`.

**`accent_metrics.score_rows(rows, Path("/"), verbose=False)`** [read]
- Absolute `file` paths are kept.
- `pid` survives through `dict(r)`, so the join back by index is correct.
- `shared_positions` reaches all 16 zh probes [run].

**`generate_custom_voice` / `generate_voice_clone`** [read]
- `text`, `speaker`, `language`, `voice_clone_prompt` (a list of items), `non_streaming_mode` and `max_new_tokens` (via kwargs) all match qwen_tts 0.1.1.
- Speaker names are validated case-insensitively against `spk_id`.
- `spk_is_dialect` is all `False`, so Chinese takes no dialect override.

**`locked_vectors`** [run on the Base safetensors]
- `talker.model.codec_embedding.weight` exists, shape `[3072, 2048]`.
- `get_slice()[i:i+1][0]` returns a `(2048,)` bf16 tensor.
- This matches `save_checkpoint`: a single `model.safetensors`, rows 3000+i, and `config.json` `talker_config.spk_id`.
- Base ships an unsharded `model.safetensors`, so the `copytree` ignore rule covers it.

### 2. Data correctness

**No path for E0 held-out clips into training.**
- `e1_build` asserts this [run: 0 overlap].
- No ref clip is test, review or val [run: 3,011 refs].
- Replay is AISHELL-3 *train* + LibriTTS-R; pseudo is Base renders of AISHELL-3 *test* text.
- `e1_mix` re-checks ids against `eval_rows`.
- Held-out and novel Japanese texts appear in no training text, exactly or with punctuation stripped [run: 0/414, 0/12].
- Pseudo texts are asserted disjoint from all `probes_zh_v3` texts.

**Pseudo row fields** [read]
- `speaker` is the row id (in the shipped set), `language` is `chinese`, `source` is `pseudo_base`, and ids and paths are row-qualified.

**Gate direction** [read]
- For zh, `lid_ratio = logP(ja) − logP(zh)` (lower = more Mandarin) and `p_target = P(zh)`.
- `p_target ≥ 0.9 and lid_ratio ≤ −19` is the right direction.
- A missing key fails the gate (the default 0 fails both tests).

**Order dependence** [read]
- Each voice gets a disjoint slice `pool[vi·60:(vi+1)·60]`.
- Seeds are unique per (voice, sample, batch). Files go under `raw/<row>/`.
- `best` is built per row from `mine` only.
- `bdscore.score` mutates the same dicts in place, and qwen_asr `transcribe` preserves input order.
- So no text or sample can land under another voice. Trainer review finding 2 (single-speaker p11) is fully addressed.

**Arm identity** [read]
- `e1_mix` shuffles the replay pool with seed 7 before any arm-specific rows are added, and `want` depends only on the Japanese rows.
- So E1a and E1b get the **same** 1,509 replay rows, and the arms differ only by the pseudo rows.

### 3. Evaluation correctness

**Pairing** [run + read]
- `plan()` has 294 batches and unique seeds, and is identical for floor and checkpoints.
- The floor uses the E1a checkpoint's own bf16 speaker rows, which are the values the fine-tune was conditioned on.
- "Paired" means same text, batch composition and seed. Different models consume the RNG differently, so the samples are not coupled; this is the same as in E0.

**Real condition**
- It writes `cond ∈ {real_resynth, real_raw}` for the 414 held-out lines (seed 6000 only).
- `e1_select` reads leak and p_true from `real_resynth` and CER from `real_raw`.
- Codec resynthesis is the same `tok.decode(tok.encode(...))` as b07/b08, matching the probe's calibration.

**Aggregates**
- Pooled ja leak, the p_true tie-break, pooled zh SER over `m23_valid` rows, the per-voice zh SER maximum, zh lid, en WER, the runaway share over all renders, and the per-voice ja leak against the E1 floor all match the plan.
- Caveat: the ja CER pool (finding 6).
- E0 review findings C1 (codec-calibrated probe and resynth baselines), m1 (lowercase reading keys, `bdcommon.py:53-54`) and m7 (`m23_valid`) are addressed.

**Missing keys**
- `zh_ser`, `zh_ser_voice_max` and `zh_lid` set to `None` make the guard fail (the run is ineligible); there is no crash.
- `ja_cer` or `en_wer` would raise `TypeError` only if a candidate had no ja or en rows, which cannot happen with the full plan.

### 4. Robustness

**`set -euo pipefail`** [read + run]
- Every `python … | tee` and `| tail` is covered by `pipefail`.
- `ls | sort | head -1` cannot hit SIGPIPE at 3 lines of output.
- The exceptions are the `for`-list substitution (finding 7) and the bootstrap's `python … > log; tail` lines: on failure, the script exits before the tail prints, so read `logs/*.log` by hand.

**Resume**
- `train()` restarts an arm from scratch (full mode cannot resume); that costs about 18 min.
- `e1_eval` skips a condition with `summary.json` present, but re-renders a partly done condition in full.
- Pseudo has no resume (finding 5).
- Replay and codes reuse their caches and signatures.

**Model loads**
- There are no per-clip model reloads: ASR, the Base speaker encoder and the accent models each load once per `score()` call.
- The accent step costs about 0.016 s/clip [log].

### 5. Line endings and encoding [run]
- There are 0 CR bytes in all `.sh`, `.py`, refs and jsonl files, in `e1/`, `_upload/stage/` and the extracted `code_bundle.tgz`.
- The tarball and the staging folder are byte-identical to the current sources.
- `wav24.tar` extracts to `wav24/<row>/…`, so it goes under `/workspace/bd/data/`.
- `queue_e1.sh` contains UTF-8 Japanese/Chinese only inside a Python heredoc, which is fine.
