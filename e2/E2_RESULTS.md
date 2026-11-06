# E2 results (and the E3 feasibility check), 2026-10-31

**Plans:** `E2_PLAN.md` and `../e3/E3_PLAN.md` (both pre-registered).

**Raw data:**
- `results/`: scored renders, the pseudo candidates and report, mix, logs.
- `results/e3/vc/`: the VC feasibility clips' scores.

**Compute:** one Secure H100, about 2.8 h, about $11. The pod was terminated; nothing was published.

**Verdict:** under the pre-registered rule, **E2 is not adopted. E1b epoch 2 (`models/E1b_ep2`) stays the model.** E3 (zero-shot VC) **failed its feasibility gate**, so nothing was trained on converted clips.

## E2 [L]

**Data.** 1,287 gated pseudo rows (10.7% of training):

| Language | From E1b | From Base | Total |
|---|---|---|---|
| zh | 373 | 485 | 858 |
| en | 352 | 121 | 473 |

They came from 13,860 candidates. Ten voices still got fewer than 10 Mandarin rows.

| | ja leak | ja p_true | zh SER | zh lid | **zh leak** | voices zh leak ≤ 0.15 | en lid | en leak | Eligible / adopted |
|---|---|---|---|---|---|---|---|---|---|
| Floor | 0.100 | 0.567 | 0.077 | −19.5 | 0.157 | 36/49 | −12.2 | 0.073 | |
| **E1b epoch 2 (current model)** | **0.014** | 0.802 | 0.069 | −18.6 | **0.155** | **34/49** | −9.3 | 0.040 | |
| E2 epoch 1 | 0.025 | 0.761 | 0.065 | −19.1 | 0.182 | 34/49 | −9.9 | 0.096 | eligible; not adopted |
| E2 epoch 2 | 0.021 | 0.786 | 0.063 | −19.7 | 0.186 | 34/49 | −10.1 | 0.110 | eligible; not adopted |

**What changed:**
- **Improved:** 月ノ森 Sakiko (zh leak 0.22 → 0.09) and Aya (0.16 → 0.03 at epoch 2). Both got 17–25 gated rows.
- **Worse:** Kasumi 0.75 → 1.00, Rinko 0.47 → 0.75, Lock 0.50 → 0.75, Hagumi 0.19 → 0.44. These are the voices with almost no rows.
- **English:** the accent improved a little (−9.3 → −10.1), but English and Japanese identity got slightly worse.

**Reading [I].** More synthetic rows help only when they are truly in the voice. 725 of the rows came from E1b itself, and for low-margin voices its timbre leans generic, so doubling the synthetic share (5% → 10.7%) diluted identity. The cheap route, sampling more from our own models, is exhausted.

## E3 feasibility (Seed-VC v1 zero-shot) [L]

Native E2 Mandarin renders were converted into Kasumi and Eve (20 each) with the voice's own Japanese reference.

| Voice | Assigned to own voice | zh lid, source → VC | SER, source → VC |
|---|---|---|---|
| Kasumi | 20/20 | −28 → −12 | 0.06 → 0.18 |
| Eve | 20/20 | −24 → −8 | 0.07 → 0.21 |

- Identity is fixed completely, but the Japanese accent comes back.
- Both accent metrics agree, including the human-validated SER.
- The converter takes articulation from the in-context Japanese reference.
- Gate pass rate: 0%. E3 stopped as pre-registered.

## Where this leaves the 49 voices

The current model, E1b epoch 2, is unchanged:
- **Japanese:** all 49 at real-recording identity.
- **Mandarin:** native-level accent for all; own voice for 34.
- **English:** intelligible, accented.

The ~10 voices whose Mandarin is generic (Kasumi, Rimi, Eve, Lock, Tomori, 月ノ森 Soyo, Kanon, Hagumi, Ako/demon princess, Rinko) need **native-accented speech that has their timbre**. None of our models or zero-shot VC produces that.

**Remaining options** (each a new decision, roughly in order of cost):
1. **Accept it.** Use these voices for Japanese, and for Mandarin take either the generic-timbre native output (E1b) or the own-voice accented output (E1a epoch 1, not kept locally; it would need retraining).
2. **RVC per voice, index ratio 0** (MIT). Train a converter per voice on its Japanese clips (~10–20 min each on an H100). Content comes from HuBERT features of the source, with no reference prompt in the decoder, so the source's native articulation should survive better than with Seed-VC. Untested; same feasibility gate first.
3. **Seed-VC fine-tuned per voice** (removes the long Japanese prompt). Also untested.
4. **Inference-time interpolation:** blend the speaker row toward the base default for Mandarin only. Cheap to try locally, but it trades identity for accent by construction.

The user chose option 2 (RVC) on 2026-10-31; see `../e4/E4_PLAN.md`.