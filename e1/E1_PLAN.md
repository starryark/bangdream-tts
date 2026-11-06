# E1 plan: first multi-speaker fine-tune (draft, pre-registered before any training)

**Status:** final, after the two code reviews in `bangdream/review/` (findings applied) and local logic tests (`e1/` scripts).

Tags: [L] local measurement, [I] inference.

## Question

Does one shared full SFT of Qwen3-TTS-12Hz-1.7B-Base (locked x-vector rows, language ids, native replay) fix identity for the voices that collapse zero-shot, while keeping Mandarin and English at the zero-shot floor? Does it also need Base-generated pseudo-Mandarin rows?

E0 established two things [L]:
- The Base floor already has Mandarin between Kurisu v3 and v2 (SER 0.077, lid −19.4).
- Identity is the gap. With the codec-calibrated probe, Japanese leak is 7.5% (CI 2.9–13%) against 1.2% for real clips, and p_true is 0.58 against 0.78. Kasumi, Eve and Lock fail clearly.

## Arms

Two training runs with identical data, except for the pseudo-Mandarin rows.

| Arm | Japanese rows | Replay | Pseudo-zh (Base, gated) | Why |
|---|---|---|---|---|
| **E1a** | all train clips, `japanese` | 14% share | **none** | Do language id + replay alone keep each voice's Mandarin at the floor? |
| **E1b** | same | same | up to 40 gated rows per voice | Does in-voice Mandarin help (Kurisu E4 adopted it)? How many rows does each voice actually pass? |

- **Why not 15 vs 60 rows.** E0 shows the floor Mandarin is already close to native, so the first question is whether pseudo rows are needed at all.
- **Collapsed voices.** They will pass few pseudo rows in E1b anyway, so E1b's pass-rate table doubles as the round-2 plan [I].

**Schedule (both arms).**
- Trainer, mode and optimiser: the Kurisu v3 settings (full, fp32 master, AdamW, lr 5e-6, effective batch 32).
- Length: **3-epoch cosine**, with checkpoints at 1.0, 2.0 and 3.0 epochs, so the epoch count is measured rather than assumed.
- Training seed 20261028.

**Why more than v3's 1 epoch.** v3's 1 epoch was tuned for one voice with 10 h. Here each voice gets 39–424 rows per epoch, against ~9k for Kurisu [L]. One epoch may under-train identity [I].

## Data

| Part | Rows | Notes |
|---|---|---|
| Japanese train | 9,268 (+ 111 val) | E0 train split, transcript verdict ok, minus 1 mismatched clip. The 414 E0 test clips and 11 review clips are excluded. Speaker = row id, language `japanese`. Val = max(2, 1%) per row, never a reference clip |
| Replay | ≈ 1,509 (14% share) | AISHELL-3 zh + LibriTTS-R en (p05: 2.0 h zh, 1.0 h en, ≤ 70 clips per speaker), each source speaker its own aux speaker. `e1_aux_filter.py` drops any replay speaker whose centred cosine to a shipped row exceeds 0.7 (pre-registered) |
| Pseudo-zh (E1b only) | ≤ 40 per voice | AISHELL-3 *test* texts with no overlap with the eval probes. Kept only if **all** gates pass (below) |

**Steps.** ≈ 10.8k rows / 32 ≈ 337 steps per epoch (E1a); E1b adds up to 40 rows per voice. The trainer reproduces Kurisu's 343 steps/epoch at this size; at the 3-epoch stop the LR is at the cosine floor.

**Small voices.** The smallest row (ro_ako_maou) has 39 train rows. There is no temperature sampling in E1; that is E2.

**Pseudo-zh gates** (all required). The source is Base x-vector-only from the row's locked vector, language Chinese, 2 samples per text; the most on-row sample of each text is kept.
- **Identity:** the codec-calibrated probe assigns it to its own row, and p_true ≥ 0.5. The probe was fit on Japanese, so correct-voice Mandarin may score lower; the gate is conservative, and its pass rate is reported.
- **Accent:** VoxLingua P(zh) ≥ 0.9 and lid_ratio ≤ −19 (the floor mean).
- **Text:** Qwen3-ASR CER ≤ 0.05.
- **Stability:** not a runaway; 2–12 s measured after the training audio policy (trim + pad).
- **Report:** the pass rate per voice, before training.
- **E1b runs only if ≥ 25 voices keep ≥ 15 gated rows** (pre-registered). Otherwise E1b ≈ E1a and is skipped; E1a's Mandarin guards then answer whether pseudo rows are needed.

## Evaluation (per checkpoint, plus the E0 floor and real clips as references)

**Texts per voice:**
- own held-out Japanese lines (4–10, paired with real audio);
- 12 shared novel Japanese sentences (`e1/probes_ja_novel.jsonl`);
- 16 zh `shared` probes and 8 en probes (the same as E0).

**Sampling:** 2 seeds. That gives ≈ 84 renders per voice, ≈ 4.1k per checkpoint.

**Inference:** CustomVoice path (`generate_custom_voice`, speaker = row id), explicit language id.
- The floor uses the Base `generate_voice_clone` x-vector path from the same speaker rows. That is a different inference prefix, so read the floor as "zero-shot from this vector", not "the same model minus training" (as in Kurisu E0).
- Batches of 48 mix voices. The smoke stage asserts that mixed-speaker CustomVoice batches keep each item's speaker.

**Scores** (b05 code, unchanged):
- Qwen3-ASR CER/WER;
- probe p_true / pred;
- accent (lid, shared-char SER);
- runaway.

## Selection rule (fixed now)

**Primary metric:** pooled Japanese probe leak over all 49 voices (codec-calibrated probe, `probe_codec.npz`), with mean Japanese p_true as the tie-break. E0 floor: leak 0.075, p_true 0.58. Real clips through the codec: 0.012 / 0.78. The E1 eval re-measures both on the larger probe set.

A checkpoint is **eligible** only if all of these hold (pooled unless stated):

| Guard | Threshold | Floor value |
|---|---|---|
| zh shared-char SER | ≤ 0.087 | 0.077 + 0.01 |
| zh lid | ≤ −18.0 | −19.4 |
| Any single voice's zh SER | ≤ 0.15 | — |
| Japanese CER | ≤ 0.035 | floor 0.031, real 0.023 |
| en WER | ≤ 0.02 | 0.001 |
| Runaway | ≤ 1% of renders | 0.4% |
| Any single voice's Japanese leak | ≤ its floor + 0.2 (no voice may get clearly worse) | — |

**Pick:** the eligible checkpoint with the lowest pooled Japanese leak. If none is eligible, nothing is adopted, and the failing guard decides the next step:
- Mandarin guards fail → E1b-style pseudo rows or a VC route;
- leakage fails → speaker-row or sampling changes.

**Reported regardless:**
- the per-voice table;
- the named-pair confusions: eve/lock, rimi/lock, hagumi/ako, himari/kasumi, sakiko/sakiko_tsuki, soyo/soyo_tsuki, ako/ako_maou, mutsumi/mortis, uika/umiri;
- leakage of Mujica voices toward game voices.

## Compute and privacy

- **Hardware:** one Secure H100 80 GB at $3.99/h.
- **Time estimate (revised after the E1 code review)** [I]:
  - bootstrap ≈ 1.2 h (environment, models, replay, pseudo-zh 5,880 renders + scoring, codes);
  - smoke ≈ 0.2 h;
  - training ≈ 15 min per arm (Kurisu: 343 steps in ~250 s);
  - evaluation ≈ 30–35 min per condition × 8 (6 checkpoints, floor, real). Batches of 48 mix voices, 94 batches per condition.
  - **Total ≈ 6–7 h, ≈ $25–30.** The run order is smoke → E1a → refs → E1b → select, so E1a's result exists early.
- **Disk:** 150 GB volume. Training peaks at ≈ 45 GB per run (three 4.2 GB checkpoints plus a 15 GB optimiser state); `--keep_last 1`, and epoch checkpoints and state are deleted after each run.
- **Data handling:** the audio goes to the pod over SSH only, never to HF. No HF token on the pod: every download is a public model or an ungated dataset, and p05 is anonymous.
- **Storage:** container disk only (no network volume), so nothing persists after the pod is terminated.
- **Deletion:** checkpoints are deleted from the pod after the chosen one is copied home. The pod is terminated at the end.

## Amendment A (2026-10-30, after E1a epoch-1 scores and the pseudo yield; before any E1b training)

**Disclosure.** Written after seeing (a) the pseudo-zh yield and (b) E1a epoch 1:
- **(a) Pseudo-zh yield.** 564 gated rows, but only 18 voices kept ≥ 15 (bar: 25). The identity gate (p_true ≥ 0.5) passes 30% of renders, while 81.5% are assigned to their own voice: the Japanese-fit probe is less confident on Mandarin.
- **(b) E1a epoch 1.** Japanese identity is fixed: leak 0.0095 against floor 0.075 and real 0.012. But Mandarin fails the guards: shared-char SER 0.177 (guard 0.087), lid −12.5 (guard −18). This is the Kurisu v2 failure (SER 0.170).

**Change.** E1b runs despite the yield bar.
- **Reason:** the plan's own rule for a Mandarin-guard failure is "E1b-style pseudo rows". In Kurisu, the same ingredient (≈ 5% in-voice pseudo-zh on top of language id + replay, E4) gained 4.82 M1 against a matched spread of 0.96 and was adopted. E1b's share is 5.0%, so it is not a no-op.
- **Data:** E1b uses the mix already built (`exp/E1b`, 564 rows); the gates are unchanged.

**Added pre-registered analysis (E1b only).** Mandarin shared-char SER and lid per voice, compared between two groups:
- voices with ≥ 15 gated pseudo rows (18 voices);
- voices with 0 rows (12 voices).

**Interpretation:**
- Improvement only in the first group means in-voice data is needed per voice.
- Improvement in both means "language id + some in-voice Mandarin" generalises across speakers.

**Unchanged:** selection rule, guards and pick.

**Amendment A, clarification (2026-10-30 21:25, written before any E1b score exists).**
- **Confound.** E1a shows the two groups differ before any pseudo training: E1a epoch 1 zh SER is 0.152 for the ≥ 15-row group against 0.241 for the 0-row group. (Floor: 0.081 / 0.082.) The pseudo gate selects voices the Base already clones well, and those are the voices the fine-tune changed least.
- **Analysis.** It is therefore a difference-in-differences: per voice, ΔSER = SER(E1b, epoch k) − SER(E1a, epoch k), and Δlid likewise, at the same epoch. Report:
  - the mean Δ of each group, with a bootstrap CI over voices;
  - the gap between the two groups.
- **Reading.**
  - Δ < 0 in both groups: in-voice Mandarin generalises across speakers.
  - Δ < 0 only in the ≥ 15 group: in-voice data is needed per voice.
- **Context [L].** In E1a the accent is universal: 43 of 49 voices are worse than the floor by > 0.05 SER. It is unrelated to minutes of data (Spearman 0.09), and worst for the voices whose zero-shot identity was weakest (lock, kokoro, eve, rimi, kanon, aya).
