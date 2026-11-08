# E7 results: VC rows in the base recipe (2026-11-08)

Tags: [L] measured, [I] inference. Plan: `E7_PLAN.md` (fixed before launch; no amendments).

**Verdict under the pre-registered rule: v2 is not adopted. v1 (E1b epoch 2, 49 voices) stays the model, and no new weights are published.** [L]
- Neither E7a candidate passes G1–G3.
- Epoch 2 fails Japanese CER and G3 as well as accent bounds, so E7d was not triggered (§5).

**Compute and cost.** [L]
- One Secure H100, pod `zct5j393jmix68` ($3.99/h), 2026-11-08 10:31 → 13:37, ≈ 3.1 h, **≈ $12.4** against the $25–35 estimate.
- The local machine reset at 10:37, during the upload. Bootstrap waits for `UPLOAD_DONE`, so nothing ran on partial data. `e7/upload_rest.sh` sent only the missing files (8,503, about 5 min), re-checked every size and set `UPLOAD_DONE` at 10:47.
- A second reset, at 11:24, only interrupted the local watcher, which was restarted.

**Where the numbers live.** `results/eval/selection.{json,md}` (from `e7_select.py`), `results/eval/E7a_ep{1,2}/{scored.jsonl,summary.json}`, `results/exp/E7a/mix.json` and `results/logs/`. No checkpoint was downloaded; all were deleted with the pod.

## What ran [L]

- **E7a = E6b's data exactly** (mix identical in size: 19,058 train rows, 223 VC rows), with the new training seed 20261108.
- **Smoke:** passed, including the extra-seed eval (40 + 40 ja-mid renders per Our Notes row, none for v1 rows).
- **E7a ep1 and ep2** were scored with `--extra_on_seeds`. C0 is E6's scored C0.

## Selection (§5) [L]

| cond | v1 ja-mid | v1 zh leak | v1 zh SER | v1 en lid | all ja CER | ON ja-mid (80) | ON zh (base-routed) | VC4 zh | replaced ja-short | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 (v1) | 0.010 | 0.160 | 0.074 | −9.11 | 0.032 | 0.206\* | 0.406\* | — | 0.333 | |
| E7a ep1 | 0.012 | 0.163 | 0.070 | −9.04 | **0.038** | 0.025 | 0.030 | 0.578 | 0.077 | G2 ja CER, G3 per-row |
| E7a ep2 | 0.010 | 0.110 | **0.085** | −8.63 | **0.036** | 0.023 | 0.046 | 0.445 | 0.093 | G1 zh SER, G2 per-row zh SER, G2 ja CER, G3 per-row |

\* C0 has only the 12 replaced voices.

**The failures** [L]:
- **G3, yomogi.** Ja-mid leak 0.35 (ep1) and 0.225 (ep2) on 80 renders, against 0.20. On E6's 40 renders alone she would be 0.50 and 0.15.
  - Arale passes at both epochs (0.125 / 0.15 on 80). Every other Our Notes voice is ≤ 0.20.
- **G2 ja CER ≤ 0.035 over all 65 voices.** 0.0381 (ep1) and 0.0356 (ep2).
  - The 37 v1 voices are at 0.025–0.027 (C0 0.025). The 28 Our Notes voices are at 0.051–0.055.
- **Accent (ep2 only):**
  - Pooled v1 zh SER 0.085 against C0 + 0.010 = 0.084 (G1).
  - Per-voice zh SER above 0.15: hhw_kokoro 0.157, pp_eve 0.168, ras_lock 0.163 (G2). These are the same three voices as in E6b ep2.
- **E7d not triggered:** ep1 fails CER and G3, which aren't accent bounds; ep2 fails CER and G3 besides accent.

## Against the expected path (§6) [L]

- **ep1 ≈ E6b ep1, as expected:** yomogi failed G3 again (0.50 on the same 40 renders as E6's 0.475).
  - The CER margin that E6b ep1 cleared by 0.0003 was lost: 0.0381, against E6b's 0.0347, on the same data with a different seed.
- **ep2 ≈ E6b ep2, as expected:** the same three voices over the per-row SER bound, plus G1 pooled SER this time.
  - Arale cleared G3 on 80 renders (0.15), as the plan allowed.
  - Yomogi failed on 80 renders (0.225) where 40 would have passed (0.15). The extra seeds cut both ways, as a precision change should.
  - Not decisive: ep2 also fails CER and SER.

## Calibration finding for any next plan [L]

**Real recordings don't clear G2's CER bound by much on this eval set.** E6's real reference condition, scored the same way (`e6/results/eval/real`), gives ja_heldout CER:

| | Our Notes lines | v1 lines | all |
|---|---|---|---|
| raw recordings | 0.038 | 0.021 | 0.028 |
| through the codec | 0.042 | 0.024 | 0.031 |

- The bound (0.035, set on E1's 49 v1 voices) leaves 0.004 of headroom over real audio through the codec.
- E7a is at the real-audio level on v1 lines (0.025–0.027). On Our Notes lines it is about 0.01 above real (0.051–0.055 against 0.042), and that gap alone breaks the pooled bound.
- [I] The ASR finds the Our Notes text hard even when the speaker is real: rare names and coined words, as noted in E7_PLAN §6.
  - A next plan would need to set the CER guard against real or C0 audio per voice set, not as one absolute number.
  - That has to be decided before a run, not applied to this one.

## What this means [I]

- With VC rows in the base, the Mandarin side of the data plan works: G1 zh leak holds at both epochs (0.163 / 0.110 against C0 0.160).
- The 28 Our Notes voices' coverage is near real-audio level (pooled ja-mid leak 0.023–0.025).
- Three things block a v2 under a rule of this shape:
  1. Yomogi's Japanese identity (2 runs × 2 epochs: 0.475, 0.125, 0.50/0.35, 0.15/0.225).
  2. An absolute CER bound that real audio barely clears on this text.
  3. Per-voice Mandarin SER on kokoro, eve and lock at epoch 2.
- Each needs a decision in a new plan (treat yomogi like Rāna, calibrate CER against real audio, or keep epoch 1's accent), not a reading of this one.

## Published

- **Hugging Face model:** unchanged at v1 (`starrydark/BanG_Dream_Qwen3_TTS`). No tag or new files.
- **Hugging Face Space:** `starrydark/BanG_Dream_Qwen3_TTS` (ZeroGPU), serving v1, made public after this run, as the user asked.
- **GitHub:** E6 and E7 code, plans and results (no transcripts, audio, per-clip files or weights).
- **The training data is not published.**
