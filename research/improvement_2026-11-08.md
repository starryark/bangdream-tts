# How to improve the BanG Dream! voice model, and how to judge it (2026-11-08)

Tags: [P] primary source (paper, official repo or standard), [S] secondary or community, [L] measured on our own data, [I] inference. The sources come from three research passes run on 2026-11-08; URLs are inline.

## 0. What v1 is

- **v1 = E1b epoch 2**, the model on Hugging Face (`starrydark/BanG_Dream_Qwen3_TTS`) and in the demo Space.
- **Voices:** 49, from about 22 h of Japanese [L]:
  - 42 from Girls Band Party game audio;
  - 6 Ave Mujica voices from anime audio, only 2.8–6 min each;
  - Sakiko from mixed game and anime audio.
- **No Our Notes data.** Our Notes arrived on 2026-11-07 and went only into E6 and E7, which were not released.
- **Mandarin and English:** language ids, native replay and in-voice pseudo-Mandarin.

## 1. Where our measurements fell short (the user is right)

E6 and E7 were decided by automatic gates alone. Nobody listened to a candidate against v1, and the checkpoints were deleted with the pods. Our own data shows where the gates cannot stand in for ears.

- **Short lines (0.7–1.6 s): the speaker probe can't judge them** [L].
  - On *real* recordings it gives 14–17% leak.
  - Nine Our Notes voices would fail our own 0.20 per-voice bound on genuine audio: ournotes_kanata 0.50–0.62, mujica_sakiko_tsuki 0.57, millsage_houka 0.44, millsage_mahoro 0.38–0.44, and others.
  - On mid-length lines it is reliable: real recordings leak 3%.
- **CER:** real recordings of the Our Notes text score 0.038 raw and 0.042 through the codec. That is above the 0.035 bound both E6 and E7 failed [L].
  - The ASR misreads rare names and coined words.
  - So the gate rejected the ground truth.
- **The per-voice bound is noisy:**
  - 8 of 40 renders has a 95% interval of 0.09–0.36 [I, exact binomial].
  - With 65 voices, a model whose true leak is 0.12 on every voice is rejected about 95% of the time at 40 renders and 58% at 80, if voices are independent [I].
  - The same recipe gave yomogi 0.475 → 0.125 (epoch 1 → 2) and, with another seed, 0.50 → 0.15 [L].
- **The metrics agree with listeners only moderately.** Correlations with human similarity ratings:

  | metric | correlation |
  |---|---|
  | ECAPA | r 0.65 |
  | WavLM | r 0.51 |
  | between listeners (ceiling) | 0.71 |

  Listeners called a clone and its real source different people 41% of the time (VIPBench, English, preprint) [P] https://arxiv.org/html/2611.05638.
- **TTSDS2:** speaker-embedding metrics track similarity ratings (ρ 0.47–0.82). MOS predictors fail out of domain (UTMOS ρ −0.12 to −0.26 on in-the-wild speech) [P] https://arxiv.org/html/2506.19441v2.
- **Accent metrics** have been validated on human L2 speech, not on TTS [P] https://arxiv.org/abs/2310.11004.
- **Language confound:** a speaker probe partly measures language [P] https://arxiv.org/html/2601.21960v1. So our Mandarin "leak" mixes identity with accent [I].
- **Goodhart:** optimising TTS on CER flattened prosody. A few hundred human preference pairs per round beat it [P] https://arxiv.org/html/2509.18531v2.

**Lesson.** Automatic metrics are good *screens* for catching failures cheaply. They are poor *judges* of which model sounds more like the character, above all for short lines, emotion and accent. The judge should be a blind listening comparison.

## 2. Evaluation protocol from now on

1. **Screens, not gates.** Keep the automatic metrics, but calibrate every bound against real audio or v1 on the same texts. Reject only clear failures.
   - **CER:** the candidate may be at most v1 + a margin, with names and coined words masked or handled through a lexicon.
   - **Leak:** a voice fails only if it is clearly worse than v1 on matched renders (binomial interval), not on a fixed 0.20.
   - **Accent:** gross collapse only.
2. **The judge is a blind paired comparison against v1** on the same text and voice, rated −3 to +3 ("which is better for this character?"; ITU-T P.800 CCR) [P] https://www.itu.int/rec/T-REC-P.800-199608-I/en.
   - Labels, side and order are randomised.
   - Both sides are loudness-normalised.
   - Takes are fixed by seed, not cherry-picked.
   - Optional flags: wrong voice, misreading, accent.
   - Analysis [P] https://arxiv.org/html/2503.03250v2:
     - Wilcoxon signed-rank test, or a binomial test for A/B choices;
     - bootstrap over voices, not over individual clips.
3. **Size for one listener.** About 92 trials, 2–3 sessions of 20 minutes or less, about 45 minutes in total.
   - Japanese: 16 voices × 3 texts (every flagged voice included; half the texts chosen where the two systems differ most).
   - Mandarin: 8 voices × 2 texts.
   - English: 8 voices × 2 texts.
   - About 8 repeated pairs with the sides swapped, and 4 identical pairs, to check consistency.
   - Power: about 0.84 to detect a true 70% preference with 60 decisive pairs [I]. Per-voice results act only as a veto, for example a voice flagged "wrong voice" twice.
4. **Decision rule** (fixed before the run):
   - pooled Japanese preference significantly above 0;
   - Mandarin not significantly worse;
   - no voice vetoed;
   - the consistency checks pass;
   - otherwise v1 stays.
   - This states one expert listener's taste; for a product with one arbiter, that is what we want [P] https://www.cstr.ed.ac.uk/downloads/publications/2015/wester:listeners:IS2015.pdf.
5. **Keep the candidates until listening is done.** Download each candidate checkpoint (about 4.3 GB) and its listening renders before deleting the pod.
6. **Tool:** a small blind A/B page (local HTML, Gradio, or a private artifact page) with replay, a 7-point scale, a "no difference" option, hidden repeats and a CSV log. BeaqleJS and webMUSHRA are ready-made alternatives [P] https://github.com/HSU-ANT/beaqlejs.

## 3. Things the research suggested that we already do [L]

Checked in `kurisu_v2/pod/train_sft.py`. Qwen's official fine-tuning script has open, unfixed bugs: labels shifted twice (#371/#278), a speaker encoder that decays under weight decay (#351), no language token in training (#361) and the wrong sub-talker hidden state (#337). [P] https://github.com/QwenLM/Qwen3-TTS/issues/371, https://github.com/QwenLM/Qwen3-TTS/pull/351. **Our trainer already fixes all four.**
- It shifts labels once.
- It aligns the sub-talker.
- It applies the text projection.
- It freezes the speaker encoder and uses one locked vector per voice, averaged from about 64 clips, in both training and the saved row.
- It trains with language-id prefixes, plus 25% "Auto".

Also already in place: native replay, gated in-voice pseudo-Mandarin, and RVC-converted rows gated by the probe and by listening.

## 4. Ways to improve, ranked by expected gain per cost

1. **Learnable speaker rows for crowded voice clusters** (problem: sibling confusion) [L + I].
   - Today each voice is a fixed Base speaker-encoder vector. The raw vectors are almost identical: the median nearest-neighbour cosine is 0.993.
   - After centring, yomogi's nearest voices are arale (0.68), raika (0.56) and chieri (0.53). Those are exactly the voices her renders leak to.
   - Other tight pairs: chieri–raika 0.77, nonoka–arale 0.71, yuno–nagi 0.68.
   - Fix: start each row from its centroid and let it train with its own learning rate (as the `[nv]` token did, 1e-4), so the model can push similar voices apart. Optionally use 10% row dropout to enable speaker CFG later [P] https://arxiv.org/html/2502.05236v2.
   - Cost: a small trainer change plus one run. Risk: rows drift away from the timbre. Listening decides.
2. **Checkpoint averaging and epoch interpolation** (seed variance, and the accent/identity trade-off).
   - Average the last K sub-epoch checkpoints.
   - Interpolate θ = α·ep2 + (1−α)·ep1 as a dial: epoch 2 has more identity, epoch 1 less accent.
   - Precedents: CosyVoice 2 averages its 5 best checkpoints, F5-TTS ships EMA weights, and "model soups" [P] https://github.com/FunAudioLLM/CosyVoice, https://arxiv.org/abs/2203.05482.
   - Cost: minutes of CPU, plus saving more checkpoints.
3. **Lower learning rate, saved more often, chosen by ear.**
   - We use 5e-6; Qwen's example uses 2e-6 at the same batch of 32 [P] https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md.
   - Community reports: lower rates kept timbre, and the loss did not track quality [S] https://github.com/QwenLM/Qwen3-TTS/issues/314.
   - "Use the earliest checkpoint that meets your requirements" (Fish-Speech) [P].
   - Cost: one or two arms in the same pod.
4. **Best-of-N at inference, reranked by probe then CER** (short lines, stability).
   - Llasa: hard-set WER 12.1 → 5.2 and SIM 0.64 → 0.75 with verifier search [P] https://arxiv.org/html/2502.04128.
   - Short lines are cheap to render, so N = 8 is affordable in the Space.
   - Also sweep the code-predictor (sub-talker) temperature separately from the talker's: timbre detail lives in the residual codebooks [I].
   - Cost: no training.
5. **Data work: short lines and interjections, and cleaner anime audio.**
   - Add hand-checked short lines and interjections per character, with the ー, っ and repetitions in transcripts restored, since ASR collapses them [P] https://huggingface.co/litagin/anime-whisper.
   - Separate background music from v1's anime-sourced Ave Mujica clips (MDX-Net, then DeEcho, as in GPT-SoVITS) [P] https://github.com/RVC-Boss/GPT-SoVITS.
   - Our Notes already gives Ave Mujica far more clean data than v1 had.
   - Avoid aggressive upsampling of small voices: it hurt the larger voices' F0 [P] https://ar5iv.labs.arxiv.org/html/1904.00771.
   - Cost: days of data and listening work.
6. **Preference tuning (DPO) on the model's own samples**, after a stable SFT.
   - Koel-TTS: CER 2.56 → 0.62 and similarity 0.60 → 0.65 [P] https://arxiv.org/html/2502.05236v2.
   - Rank pairs by probe identity, then CER, with a MOS-predictor guard.
   - Better still, use **the user's own A/B choices as the preference pairs**; a few hundred per round worked best in 2509.18531 [P].
   - Risks: reward hacking and flattened prosody. Never use ground truth as the "chosen" sample [P].
   - Cost: days of engineering plus a few GPU hours.
7. **Per-voice LoRA for the few voices that stay confused** [P] https://arxiv.org/html/2603.10904v1.
   - Raised similarity per speaker, but adds models to maintain and can bring accent back.
   - Last resort.

**Not recommended:**
- A speaker-classification auxiliary loss. YourTTS's reported gain was void: the gradient never flowed (erratum) [P] https://arxiv.org/html/2112.02418v4.
- RL on a speaker-similarity reward alone. CosyVoice 3 and 2509.18531 saw hacking and instability [P].

## 5. Suggested next step (not started; needs the user's go-ahead)

**One pod run of about 4 h, about $15–25, decided by listening.**
- **Arms:**
  - **A:** E7a's data, lr 5e-6, saving every 0.25 epoch.
  - **B:** the same with learnable speaker rows.
  - **C (optional):** lr 2e-6.
- **Screens:** recalibrated against v1 and real audio (§2.1).
- **Download** each arm's best and averaged checkpoints, and the listening renders. Nothing is deleted before listening.
- **Decision:** the blind paired comparison against v1 (§2.2–2.4), about 45 minutes of listening split over 2–3 sessions.
- **Infrastructure gap:** the local GPU currently drops off the PCIe bus. Rendering listening sets on the pod avoids depending on it.
