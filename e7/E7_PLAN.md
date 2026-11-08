# E7: BanG Dream v2, VC rows in the base recipe (pre-registered plan)

**Outcome (2026-11-08 13:37): v2 not adopted; v1 stays.** No E7a candidate passed: ep1 failed on ja CER and yomogi; ep2 on ja CER, yomogi and per-voice zh SER. E7d was not triggered. The pod cost ≈ $12.4. See `E7_RESULTS.md`. History: **Status: approved by the user 2026-11-08 ("proceed to E7, don't publish the dataset, make hf space for the models once finished").** This plan, the selection rule (§5) and the code (`e7/`, `e6/e6_eval.py --extra_on_seeds`) are fixed before the pod is launched. Results go to `E7_RESULTS.md`.

## 1. Why E7

E6 (`../e6/E6_RESULTS.md`) adopted nothing.
- Its rule required a passing E6a (no VC rows) before E6b (with VC rows) could ship.
- E6a failed G1: four of v1's weakest Mandarin voices (Eve, Hagumi, Rimi, Rinko) lost their Mandarin identity once 16 new voices were added.
- E6b, which carries E5's own-voice VC rows for those voices, passed G1 and G2 at epoch 1. It failed only G3: yomogi's per-row ja-mid leak was 0.475 on 40 renders.
- At epoch 2, E6b failed G2's per-row zh SER bound on 3 voices (0.163–0.174 against 0.15) and G3 on arale (0.25).

E7 makes the lesson the design: VC rows are part of the base recipe.

## 2. Arms

| Arm | Data | Runs |
|---|---|---|
| **E7a** | E6b's data exactly (§3) | always |
| **E7d** | E7a with replay share 25% instead of 14% | **only if** no E7a candidate passes, and one passes G3 and fails only accent bounds (§5) |

- **Hyperparameters:** E6's (E1b's): lr 5e-6, effective batch 32, fp32 master weights, `p_auto_lang` 0.25, 3-epoch cosine, checkpoints at epochs 1.0 and 2.0, stopped at 2.0. **Candidates: ep1, then ep2.**
- **New training seed:** E7 uses 20261108 (E6 used 20261028). E7a is therefore a second, independent sample of E6b's recipe, not a rerun. E6b's numbers are the prediction for it (§6).
- **Dropped from E6:**
  - E6c's `[nv]` arm: the token met its thresholds but didn't beat the literal `……っ` text in E6a.
  - The `real` and `floor_on` reference conditions: reported in E6, and unchanged.

## 3. Data (identical to E6b; no new data)

- **Japanese:** E6's bundle (`e6/bundle`, 65 rows, codec rows 3000–3064). 15,514 unique train rows; repeats are collapsed (E6 amendment 10).
- **Replay:** `p05_replay.py --zh_hours 5.5 --en_hours 2.5 --zh_speakers 70 --max_clips_per_speaker 70` (E6 amendment 9).
  - E7a draws 0.14/0.86 × the Japanese train rows (2,526). E7d draws 0.25/0.75 × the same (~5,171).
  - Both use E6's mix seed, 7.
- **Pseudo zh:** the E1b-recipe rows for the 37 kept v1 rows (374, identical to E1b), plus the Our Notes pseudo rows (`e6_pseudo_on.py`, bar 0.3896; 460 in E6). Both are regenerated on the pod with E6's code and seeds.
- **VC rows:** E5's 187 for the 13 retained v1 rows, with E5's listening outcome kept, plus E6's 36 for arale (20), nonoka (7), hotaru (8) and yomogi (1), from the user's 2026-11-08 listening verdict. 223 in total.
- **Mixes:**
  - E7a: `e6_mix.py --arm E7a --sources pseudo_v1,pseudo_on,vc`
  - E7d: `e6_mix.py --arm E7d --sources pseudo_v1,pseudo_on,vc --replay_share 0.25`

## 4. Evaluation

- **Conditions:** E7a_ep1/2, and E7d_ep1/2 if triggered. `e6/e6_eval.py` runs with E6's eval plan: same texts, probes and seed bases (6000, 11000), and the same scorer (`bdscore`, probeE6).
- **C0** (E1b_ep2, v1 as shipped) is not re-rendered. E6's C0 scores (`e6/results/eval/C0/`) are uploaded, since the checkpoint, probe, bundle, texts and seed bases are the same.
- **Change from E6 (precision):** `--extra_on_seeds` renders the Japanese held-out and novel lines of the 28 Our Notes rows with two more seed bases (16000, 21000).
  - This takes the per-row ja-mid count from 40 to 80 (1,120 extra renders per condition).
  - G3's ja-mid checks (pooled and per row) use all 80. Every other metric uses only E6's two seed bases, so G1, G2 and the C0 comparison are on E6's footing.
  - The base render layout is byte-identical to E6's (dry-tested 2026-11-08).
  - **Disclosure:** this was added after seeing E6's per-row spread. It narrows the per-row standard error at 0.2 from about 0.063 to about 0.045. It is a precision change, not a fix for any voice: it does not move a true leak of 0.3 under 0.2.

## 5. Guards and selection rule

**Guards: E6's G1–G3 (E6_PLAN.md §5), unchanged in content and bounds.** `e7/e7_select.py` calls `e6_select.guards` and `metrics` as they are.
- G1: non-inferiority to C0 on the 37 unchanged rows (pooled and per row).
- G2: absolute bounds, including ja CER ≤ 0.035 and every row's zh SER ≤ 0.15.
- G3: Our Notes coverage.
  - Pooled ja-mid leak ≤ 0.05.
  - Every row ≤ 0.20, except `mygo_raana`, which is reported. The Mutsumi↔Mortis and Sakiko↔Sakiko-tsuki pair confusions are excluded.
  - Pooled zh leak ≤ 0.15 over the 24 Base-routed rows.
- **Yomogi and arale stay gated.**

**Rule:**
1. The first E7a candidate (ep1, then ep2) passing G1–G3 is the release.
2. If none passes, but one passes G3 and fails **only** accent bounds, E7d runs in the same pod. Its candidates are judged by the same rule in E7a's place.
   - The accent bounds are E6's set: G1 zh SER, zh lid, en lid; G2 zh SER, zh lid and the per-row zh SER maximum.
3. Otherwise v2 is not adopted, v1 stays, and the results are reported.
4. **Listening before release** (a no-go check, as in E6 §6.4): the user hears 3 ja-short and 3 zh renders for each of the 16 new rows of the released candidate. Any row marked wrong-voice is listed as weak in the card.

**Reported, not gating:**
- C1, the short-line claim: the replaced rows' ja-short leak ≥ 0.10 below C0's.
- VC4 and not-VC4 zh leak.
- Per-row G3 values on 40 and on 80 renders.

## 6. Expected path (written before the run)

- **E7a ep1 ≈ E6b ep1** (G1 and G2 pass; G3 failed only on yomogi at 0.475).
  - If yomogi again exceeds 0.20 on 80 renders, ep1 fails G3. That is not accent, so it triggers nothing.
  - **A yomogi-only G3 failure means v1 stays.** Changing her status would be a decision for a plan after this one.
- **E7a ep2 ≈ E6b ep2** (G2 per-row zh SER on 3 voices; G3 on arale at 0.25 on 40 renders).
  - If arale is ≤ 0.20 on 80 renders, ep2 fails only on accent and E7d runs.
  - If arale still fails, ep2 fails G3 plus accent, and nothing is triggered.
- **G2 ja CER is the thinnest margin.** E6b ep1 passed it by 0.0003 (0.0347 against 0.035).
  - E6's high-CER Our Notes renders are rare names and coined words the ASR doesn't know (轟堂, 初華, 汐見, チュパサチュ) and interjections. They are not a text-normalisation bug, so the bound and the normaliser stay as they are.
  - A CER failure is a failure.
- **E7d** is expected to lower per-row zh SER (more replay). Its effect on Japanese coverage is unknown; there is less Japanese in the mix.

## 7. Pod, cost and outputs

- **Pod:** one Secure H100 (RunPod, $3.99/h), `e7/pod_up.sh all`, then `e7/pod_wait.sh` (keeps a dead-queue pod 60 min for a fix).
  - Upload (~75 min) → bootstrap (env, replay, pseudo, codes, mix) → queue (smoke → E7a train + eval → select → [E7d] → select) → download → stop.
  - The smoke stage also runs the extra-seed eval on 3 rows.
- **Estimate:** 5–7 h, **about $25–35**.
- **Outputs:** `e7/results/` (selection, per-condition summaries and scored files, logs, mixes). If a candidate is released: `models/<release>/` and its eval renders, for the listening check.

## 8. Release (if §5 releases a candidate)

1. The user's listening check (§5.4).
2. `release/MODEL_CARD_v2.md`, then `release/publish_hf_v2.py`, pointed at E7's selection, publishing to `starrydark/BanG_Dream_Qwen3_TTS`. The current main revision is tagged `v1` first.
   - Model only.
   - No audio, transcripts, reference-clip lists or RVC converters.
   - The v1 card's unofficial / AI-generated / takedown wording is kept.
3. **An HF Space for the models** (user, 2026-11-08): v1 always, and v2 if released.
   - Speaker picker plus text and language, using the checkpoints' built-in speaker rows (no reference audio).
   - The same wording as the card.
   - No game audio or transcripts.
4. **GitHub** (`starryark/bangdream-tts`, via `release/stage_github.sh`): E6 and E7 code, plans, results and numeric reports, whether or not E7 is adopted.
   - Never staged: audio, weights, transcripts, reference lists, per-clip scored files.
5. **The training data is never published** (user, 2026-11-06 and 2026-11-08).

If nothing is released: the Space serves v1, GitHub gets E6 and E7, and HF stays at v1.

## 9. Amendments

(none yet)
