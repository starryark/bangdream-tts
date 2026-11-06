# Voice conversion and training-data prep for ~50 BanG Dream voices (2026-10-29)

Scope: (A) VC as a route to native-accent zh/en and as a pseudo-data generator; (B) cleaning and upgrading the training audio. This builds on `kurisu_v2/research/finetune_methods_2026-10.md` (VC-polyglot row, rank-4 pseudo-data plan, E0a/E4) and `final_report.md` §4. It does not repeat them.

**Tags:** [P] primary source I read (repo, paper, card, code, licence or API metadata). [S] secondary. [I] my inference. [U] unverified. [L] my own read-only measurement of the files in `E:\BaiduNetdiskDownload\bangdreamai\角色语音` (the scratch scripts are in my session scratchpad; no files were changed).

---

## 0. Decisions in brief

1. **Do not band-extend or "restore" the game mp3s by default.** They are LAME 3.99, mono, 32 kHz, VBR about 82–87 kbps, with a lowpass near **15 kHz** [L]. A 24 kHz target only needs content up to 12 kHz, so nothing is missing. The Qwen3-TTS tokenizer runs at 12.5 Hz × 16 codebooks of 2048 entries, about 2.2 kbps [P config], so in-band mp3 artefacts are very unlikely to survive tokenisation [I]. Decode each file once to 24 kHz WAV and train on that.
2. **For the anime clips, the separation model should be dialogue-aware.** Use BandIt Plus (speech/music/effects; DnR speech SDR 15.64; Apache-2.0 code) ahead of a music-vocal RoFormer. Add de-reverb only for clips flagged as reverberant. Gate on percentiles, not absolute MOS: DNSMOS loops 1–3 s clips up to 9 s [P code].
3. **Use VC as a pseudo-data generator, not as the deliverable.** The only 2026 head-to-head I found (2610.29351, Seed-VC) shows a TTS→VC cascade gives the most native accent but loses a lot of timbre: ECAPA 0.468 vs 0.601 for direct cloning, WER 3× higher [P].
4. **The VC output rate is a hard filter for pseudo-data.** X-VC, MeanVC/MeanVC2 and kNN-VC output 16 kHz. Pseudo-zh/en at 16 kHz next to ja at 24 kHz would teach "zh/en = muffled" [I]. Usable at ≥24 kHz: **RVC v2 (32/40/48k; MIT; multi-speaker training added 2026-08-04)**, Seed-VC 44.1k (GPL-3.0, archived), CosyVoice2/3 `inference_vc` (24k, Apache-2.0), Chatterbox VC (24k, MIT, adds a watermark) and Vevo (24k, CC-BY-NC).
5. **Speaker metric:** use litagin `anime_speaker_embedding` (MIT; ECAPA trained on 7,357 VN characters; `char` and `va` variants) for leakage checks and gating. Report WavLM-SV/ECAPA alongside it for comparability with the literature.
6. **Leakage is built into the data.** Several "sets" are the same character or the same voice actress: 豐川祥子 ×2 (+梦中情祥?), 長崎素世 ×2, 宇田川亞子/大魔姬亞子, and 若葉睦/墨提斯 [L, set names]. Decide whether to merge each group into one row or keep separate rows on purpose, before training.
7. **ASR checking:** no head-to-head between anime-whisper and the 2026 Qwen3-ASR fine-tunes exists. Run both and review only clips where both disagree with the filename transcript at kana level.

---

## 1. Facts about your files [L]

| Item | Measured |
|---|---|
| Set count | The archive holds **50 zips**: Afterglow 5, Morfonica 5, Mujica 8, MyGO!!!!! 6, Pastel＊Palettes 5, Poppin 5, RAS 5, Roselia 6, HHW 5. Plus `Poppin/ksm`, an unzipped folder of 258 mp3s with the same count as 戶山香澄.zip, so probably a duplicate. **This does not match the 52 in the brief.** Reconcile it before fixing the speaker-row count |
| Game sets (42 zips) | mp3, **LAME3.99** tag, mono, 32 kHz, VBR. Three probed files: 81.5–86.6 kbps |
| Spectrum (12 clips, `Poppin/ksm`) | Relative to 0.3–3 kHz: −30…−40 dB at 11 kHz, −33…−44 dB at 12 kHz, −38…−47 dB at 14 kHz, **≈ −93…−99 dB at 15.5 kHz**. The lowpass is ≈15–15.3 kHz |
| Anime sets (`Mujica/*.zip`, 8) | 44.1 kHz mono WAV. Medians 2.3–3.2 s; p10 0.96–1.7 s; WAV totals 2.4–12.5 min per set (57–203 WAV clips). **`豐川祥子.zip` is mixed: 57 WAV + 60 mp3.** The mp3 half was not measured, so its 2.4 min understates that set |
| Same-identity groups | 豐川祥子 + 豐川祥子（月ノ森限定） (+ 梦中情祥, probably Sakiko as well [U]); 長崎素世 + （月ノ森限定）; 宇田川亞子 + 大魔姬亞子 (Ako's persona); 若葉睦 + 墨提斯 (Mortis is Mutsumi's alter, same VA [S/I]) |

---

## 2. A1: VC state of the art usable on anime female voices

### 2.1 Candidates

| System | Code / weights licence | Output SR | Fine-tune | Maintenance (as of 2026-10) | Notes for this project |
|---|---|---|---|---|---|
| **Seed-VC** v1 offline (whisper-small, DiT) / v1 44.1k f0 (whisper-base) / v2 (hubert-bsqvae + optional AR) | GPL-3.0 code and weights (HF `Plachta/Seed-VC`; mirror `rtikw/Seed-VC` 2026-08-26) [P] | 22.05k / **44.1k** / 22.05k [P] | Yes: "minimum 1 utterance per speaker", "minimum 100 steps, 2 min on T4", clips 1–30 s [P] | **Archived 2025-11-21**; last push 2025-04-20 [P API]. No successor named; only small forks [S] | v2's AR mode does "accent & emotion conversion" [P]. **Keep it off**, because you want to *keep* the native source accent [I]. Used for the native anchors in 2610.29351 [P] |
| **RVC WebUI** v2 | MIT [P API + LICENSE] | 32k / 40k / 48k [P cli.md] | Per-speaker training; "≥10 min low-noise data" recommended [P]. **Multi-speaker training with per-speaker retrieval indices added 2026-08-04** [P commit, cli.md] | Active. Release 2.3.260718 (2026-07-21): "base model unchanged"; FCPE f0; UVR5 replaced by PyMSS (MSST) separation; CUDA-graph realtime 4.7× [P]. "RVCv3" base still only promised [P README] | The cheapest way to hold 50 voices in one model [P/I]. Content comes from HuBERT/ContentVec; source timbre can leak, which the index ratio controls [S/I] |
| **Applio** (RVC fork) | MIT [P API] | as RVC | as RVC | Very active: 3.7.1 on 2026-10-26 [P] | Maintained RVC front-end [P] |
| **X-VC** (2604.12456) | MIT code and weights [P] | **16 kHz** (SAC 16k codec, ERes2Net 16k) [P] | Training code, no speaker-FT recipe [P] | 85★, last push 2026-05-06 [P] | Trained on ~10k h **en+zh only** (Emilia DNSMOS > 3.45 + LibriTTS), with pairs generated by Seed-VC-small [P]. Offline RTF 0.014 [P] |
| **MeanVC** / **MeanVC2** (2606.09050) | Apache-2.0 [P] | **16 kHz** (Vocos) [P] | MeanVC2 supports speaker-specific FT [P] | MeanVC2 pushed 2026-09-13 [P] | Content is a WeNet Fast-U2++ streaming-ASR BNF; MeanVC v1's content encoder "only supports Chinese" per X-VC [P]. Streaming latency 110 ms [P] |
| **Vevo** / Vevo1.5 / **Vevo2** (Amphion) | Amphion code MIT; weights **CC-BY-NC-4.0** (Vevo), **CC-BY-NC-ND-4.0** (Vevo1.5, Vevo2) [P cards] | 24 kHz (Vevo vocoder config) [P] | Full training recipe (from scratch) [P] | Amphion pushed 2026-03-25 [P] | "Vevo-Timbre: style-preserved VC" is the mode that keeps the source accent [P README]. The Vevo2 card lists `ja` but is ND. A third-party "MIT" mirror claim is wrong [P card vs S] |
| **CosyVoice2/3 `inference_vc`** | Apache-2.0 [P] | 24 kHz (CV2/CV3) [I from configs] | Flow-matching module trainable with the repo recipe [I/U] | Repo last commit 2026-05-25 [P] | Source S3 speech tokens → flow conditioned on target prompt tokens + mel + x-vector [P code]. Inherited by CosyVoice2/3 classes [P]. Seed-VC EVAL: CosyVoice(v1) VC SECS 0.844 vs Seed-VC 0.868 [P] |
| **Chatterbox VC** | MIT [P] | 24 kHz (`S3GEN_SR`) [P code] | No recipe [I] | Pushed 2026-07-21 [P] | **Applies a Perth watermark to every output** [P code]. Whether a TTS trained on watermarked pseudo-data inherits anything is [U] |
| OpenVoice V2 | MIT [P] | 22.05 kHz [P config] | No (zero-shot tone-colour converter) [I] | Last push 2025-04-19 [P] | Weakest SECS in Seed-VC's table (0.755) [P] |
| kNN-VC / FreeVC | MIT [S] | 16 kHz (kNN-VC) [P README] | No training; reference pool | Stale [I] | Baselines only |
| Qwen3-TTS | — | — | — | — | **No VC mode** in the README [P grep] |
| Beatrice v2 (JP realtime VC) | MIT-style header in a trainer mirror [S] | [U] | Per-speaker training [S] | [U] | Not evaluated |

### 2.2 Cross-lingual and anime evidence

- **No source tests a Japanese-trained anime target speaking zh or en.** The closest evidence follows.
- **X-VC, Seed-TTS-Eval** [P]:
  - SIM falls from the same-language setting to the cross-lingual one.
  - Same-language, offline: X-VC en 0.63 / zh 0.73; Seed-VC-small 0.56 / 0.73.
  - Cross-lingual en→zh: X-VC 0.52, Seed-VC-small 0.52.
  - Cross-lingual zh→en: X-VC 0.49, Seed-VC-small 0.42, MeanVC 0.42.
  - **Cross-lingual VC costs roughly 0.1–0.2 SIM even between en and zh** [I]. This is rough: the same-language and cross-lingual rows use different source/reference pairings.
- **Seed-VC EVAL.md** [P]:
  - Setup: LibriTTS English sources converted to Uma Musume (Japanese anime female) targets, zero-shot, against So-VITS models trained on those targets.
  - Tokai Teio: SECS 0.890 vs 0.864, CER 4.66 vs 9.63.
  - Matikane Tannhäuser: 0.877 vs 0.859.
  - The metric is Resemblyzer, which separates anime voices poorly (§8). Treat these as optimistic.
- **2610.29351** (2026-10-21): Seed-VC converts four LibriTTS-R donors into each target and keeps the highest-SIM candidate as the anchor [P]. Selecting the best of several donors is a cheap, published trick [P].

### 2.3 Selection rules for pseudo-data [I]

1. Output ≥ 24 kHz.
2. A timbre-only, style-preserving mode, so that native accent and prosody come from the source.
3. Per-character fine-tuning, because 3–60 min is enough for RVC or a Seed-VC fine-tune.
4. A licence that allows tool use.

- **RVC v2 40k/48k multi-speaker** meets all four, and **Seed-VC 44.1k fine-tuned** is the higher-similarity alternative [I]. Vevo-Timbre is fine for a non-commercial run.
- X-VC and MeanVC2 are attractive for speed but fail rule 1.

---

## 3. A2: does VC pseudo-data fix accent for low-resource targets?

| Work | Setup | Target data | Result | Risk evidence |
|---|---|---|---|---|
| Huybrechts et al., ICASSP 2021 ([2011.05707](https://arxiv.org/abs/2011.05707)) [P] | CopyCat VC turns supporting speakers' expressive speech into the target; TTS trained on it + FT | 15–45 min real target style; 4–7 h synthetic | 11 of 12 evaluations improved, 8 significantly. 15 min, Female 2 style: 58.1 → 67.7 MUSHRA | Speaker embedding was moved to the phoneme encoder "to reduce occurrences of speaker leakage". **"For Female 2, (DR+VC+FT) never reaches the quality of (non-DR)"**: a ceiling. Multi-speaker naturalness 56.9–59.4 vs recordings 78.8–81.7 |
| Comini et al., Interspeech 2022 ([2207.14607](https://arxiv.org/abs/2207.14607)) [P abstract] | F0-conditioned VC augmentation + F0 predictor | 1 h conversational | "competitive in naturalness" | No numbers in the abstract |
| Ellinas et al. 2022 ([2210.17264](https://arxiv.org/abs/2210.17264), v2 withdrawn) [S] | One model: native-speaker TTS then flow VC to target | Low-resource speakers ≈7.5 min | Better target-language pronunciation; SIM close to baseline | — |
| Amazon VC-polyglot, ICONIP 2023 ([2309.08255](https://arxiv.org/abs/2309.08255)) [P full text] | 24 kHz flow VC; converted native data trains the target model | (not stated) | Naturalness MUSHRA ↑ in all tables, e.g. es-MX 69.60 vs 64.08; SIM "on par" (−3.5…+12.3%) | Authors blame SIM on the VC and suggest diffusion VC. GAN-vocoder artefacts on synthetic mels handled by feature normalisation |
| Closed-loop dual reference ([2610.29351](https://arxiv.org/abs/2610.29351), 2026-10) [P] | Seed-VC native anchor used at **inference** (not training), VoxCPM2 | 20 French→English speakers | See §4 | Anchor quality "depends on the zero-shot converter" |
| 2025–26 study of VC pseudo-data for **codec-LM TTS fine-tuning** | — | — | **None found** [U] | — |

### 3.1 Risks specific to this plan [I unless tagged]

1. **Timbre ceiling.** The cross-lingual SIM loss above (§2.2) carries into whatever the TTS learns from these rows. Keep pseudo-data at 5–10% of rows, as yesterday, so that real ja data keeps defining timbre.
2. **Prosody comes from the source.** VC keeps the donor's prosody; VC-polyglot attributes its naturalness gain to this [P]. A calm audiobook donor gives a calm Kasumi. Choose expressive, pitch-matched donors. One way: per character, Qwen3-TTS VoiceDesign/CustomVoice zh/en voices described like the character, then VC them.
3. **Channel ↔ language confound.** If only zh/en rows pass through VC, the model can bind the "VC sound" (vocoder colour, bandwidth) to the language id. Mitigations:
   - Also convert some *native Japanese* donor speech into each character, under the ja id.
   - Match the output SR and loudness to the real data.
4. **Source-speaker leakage and similar-character leakage.**
   - Reject a clip if its nearest character centroid (anime-ECAPA `char`) is not the target, or if its cosine is below the real-clip p25 for that character.
   - Use best-of-N donors, as in 2610.29351.
5. **Artefacts on expressive content.** Laughs, shouts and singing-like lines are where VC fails [S, community]. Skip non-verbal donor clips.
6. **Watermarks.** Chatterbox VC output is watermarked [P]. Avoid it for training data unless the watermark is confirmed harmless [U].
7. **Filters** (yesterday's, plus additions):
   - Unforced LID equals the target language, with Whisper P(lang) ≥ 0.9.
   - CER ≤ 5%.
   - Anime-ECAPA `char` cosine to the target centroid ≥ real p25, **and** the target is the nearest centroid.
   - Bandwidth check: energy above 8 kHz within the real-data range.

---

## 4. A3: cascade (native TTS → per-character VC) vs fine-tuning the TTS

**Direct evidence** (2610.29351, Table 2, 20 speakers, n = 1200 per arm, VoxCPM2 + Seed-VC) [P]:

| Arm | Residual source-LID P(fr) ↓ | ECAPA ↑ | WER ↓ | RTF ↓ | Notes |
|---|---|---|---|---|---|
| Direct cross-lingual cloning | 0.093 | **0.601** | 0.016 | 0.474 | |
| **TTS→VC cascade** | **0.001** | 0.468 | 0.047 | ≈0.80 | 3.2% of outputs had word repetition or insertions; DNSMOS highest (3.026); listeners preferred it on **pronunciation** (32 vs 13, p = 0.026) |
| Dual-reference guidance | 0.025 | 0.559 | 0.009 | 0.587 | Listeners preferred it on **speaker identity** (38 vs 5) |

**Older evidence:**
- Voice Filter (Amazon, ICASSP 2022, [2202.08164](https://arxiv.org/abs/2202.08164)): a VC post-module with **1 min** of target was "competitive with a TTS model trained on 30 times more data" [P abstract].
- Ellinas 2022: the cascade improved pronunciation with SIM near baseline [S].

**Latency** [P]:

| VC system | Offline RTF |
|---|---|
| Seed-VC tiny | 0.069 |
| Seed-VC small | 0.161 |
| MeanVC | 0.094 |
| X-VC | 0.014 |

- These are from X-VC Table 3; the GPU is not stated in my extract [U].
- RVC realtime has CUDA-graph acceleration [P].
- A cascade adds roughly 0.1–0.3× real time to TTS. That is fine for offline line generation and streamable with RVC [I].

**Engineering for 50 speakers** [P/I]:
- RVC multi-speaker gives one checkpoint plus 50 per-speaker indices, and inference selects the speaker by ID [P cli.md].
- Training on 3–60 min per character is hours, not days [S].

**Verdict** [I]:
- A cascade is a *realistic* deliverable and the strongest accent fix.
- Its ceiling is the timbre loss in the table above, plus prosody that belongs to the donor, not the character. Timbre is the product here.
- Use it as:
  - (a) the **pseudo-data generator** for the fine-tune (rank-4 / E4 route b);
  - (b) a **fallback "zh/en mode"** for characters whose fine-tuned row still has a Japanese accent.
- Decide per character with the same blind A/B used for Kurisu, on fixed zh/en probes: native-accent rating, plus anime-ECAPA SIM to the real-ja centroid.

---

## 5. B4: lossy mp3 training data, and whether to restore it

**Evidence that lossy sources train well:**
- LibriTTS was built from "chapter-level audio files (in 128kbps MP3 format)", from LibriVox originals "recorded at 44.1 or 32kHz" [P 1904.02882].
- Emilia comes from in-the-wild video audio standardised to 24 kHz WAV and lists no codec handling [P 2407.05361]. Most 2025–26 TTS, including probably Qwen3-TTS's pre-training [U], has seen such data.
- No study isolates the effect of mp3 bitrate on TTS training [U] (searched).

**Your files:** the lowpass is ≈15 kHz, above the 12 kHz Nyquist of a 24 kHz model [L]. The Qwen tokenizer bitrate is about 2.2 kbps (12.5 Hz × 16 codebooks of 2048) [P config]. The official recipe pre-extracts `audio_codes` with `Qwen3-TTS-Tokenizer-12Hz` (`prepare_data.py`) and then trains `sft_12hz.py` on those codes, so the tokenizer and decoder are not trained [P finetuning/README.md]. **So restoration can only matter if it changes the tokens** [I].

**Restoration evidence:**
- LibriTTS-R (Miipher) [P]:
  - TTS MOS rose from 2.5–3.4 to 3.7–4.3.
  - Its problems were noise, reverb and missing phoneme bands, not mp3 specifically.
  - SpkSim fell from 0.784 to 0.762.
  - 2 of 640 samples were distorted by restoration failures.
- Sidon ([2509.17052](https://arxiv.org/abs/2509.17052)) [P]:
  - Open Miipher-2 equivalent; code and weights MIT (`sarulab-speech/Sidon`, HF `sidon-v0.1`); 48 kHz output.
  - Trained on 104 languages including JSUT/JVS/HiFi-CAPTAIN. Simulated degradations include **mp3 at 65–245 kbps**.
  - SpkSim 0.961–0.979.
  - F5-TTS trained on TED-LIUM: original 3.25 → Demucs 3.27 → VoiceFixer 3.77 → Sidon 4.25 MOS.
- Generative SE hallucinates "phoneme omissions and speaker inconsistency", which quality metrics miss (2601.12254, ICASSP 2026) [P abstract].

| Tool | Licence | What it does | Fit here |
|---|---|---|---|
| Sidon v0.1 | MIT [P] | Restoration incl. mp3 artefacts → 48k | **The only one worth a pilot**; trained on read speech, so anime timbre risk [I] |
| LavaSR v2 (Interspeech 2026) | Apache-2.0 [P] | Fast enhancement + BWE, any 8–48k input | Pilot candidate; no primary SIM data read [U] |
| Resemble-Enhance | MIT [P] | Denoise + CFM enhance, 44.1k | Older (2023); hallucination risk [I] |
| AudioSR / FlashSR / NovaSR / AP-BWE | MIT / [U] / Apache-2.0 / MIT [P] | Bandwidth extension to 48k; NovaSR takes 16k input only [P] | **Not needed**: nothing to extend below 12 kHz |
| Apollo (MSST) | CC-BY-SA-4.0 [P] | Low-quality mp3 *music* restoration | Music-trained; skip |

**Recommendation** [I]:
1. Use **mp3 → 24 kHz mono WAV** via a high-quality resampler (soxr VHQ) on the game sets. No restoration.
2. **Cheap test before deciding otherwise:** on 50 clips, run Qwen3-TTS-Tokenizer encode→decode on (a) the mp3-decoded audio and (b) the Sidon output downsampled to 24k. Blind ABX the two resyntheses and diff the codes. If listeners cannot tell, restoration cannot help training.
3. If you pilot Sidon anyway, gate each restored clip. Keep it only if anime-ECAPA cosine to the *raw* clip is ≥ 0.9× the median raw-to-centroid cosine and kana-CER does not get worse. Then fine-tune A/B one character.

---

## 6. B5: cleaning the anime clips (8 Mujica sets, 2.4–12.5 min each)

**Separated audio is standard practice in TTS training data** (this partly resolves yesterday's [U]):
- Emilia uses UVR-MDX-Net Inst [P].
- X-VC, Vevo and MeanVC2 train on Emilia [P].
- But Demucs-only cleaning did not raise TTS MOS on TED-LIUM (3.254 → 3.265), while restoration did [P Sidon].

**Pipeline** (inference-only; MSST = ZFTurbo Music-Source-Separation-Training, MIT; or `python-audio-separator`, MIT):

| Step | Model | Licence | Rule |
|---|---|---|---|
| 1. Dialogue vs BGM vs SFX | **BandIt Plus** (speech/music/effects; DnR speech SDR 15.64) [P MSST docs] | Code Apache-2.0 [P]; weights [U] | Run first. Music-vocal models are not trained on SFX [I] |
| 2. Residual music (clips with songs or BGM under vocals) | Mel-Band RoFormer (KimberleyJensen, SDR 10.98), BS-RoFormer (viperx, 10.87), BS PolarFormer (11.00) [P MSST, Multisong] | Weights vary; Kim repo has no licence file [P/U] | Only if step 1 leaves audible music |
| 3. De-reverb / de-echo | anvuew dereverb Mel-RoFormer (**GPL-3.0**) or Sucial DeReverb-Echo (SDR 10.01, **CC-BY-NC-SA-4.0**) [P] | as listed | Only on clips flagged by listening or a C50 estimate. Over-dry versus the dry game sets is fine; uneven reverb across a set is not [I] |
| 4. Denoise | aufr33 Mel-RoFormer denoise (*normal*, not aggressive) [P], or DeepFilterNet3 / MossFormer2-SE-48k [S] | various | Usually unnecessary after step 1 [I] |
| 5. Format | 24 kHz mono; one loudness target (−20 dBFS ±3 dB like Emilia [P], or −23 LUFS) | | Lead silence 100–200 ms; tail ≤ 300 ms (yesterday) |
| 6. Gate | see below | | |

**Gating:**
- **DNSMOS P.835:**
  - DNSMOS runs at 16 kHz and **tiles clips shorter than 9.01 s** (`np.append(audio, audio)`) [P code]. So it is blind above 8 kHz and noisy on 1–3 s clips.
  - Use OVRL ≥ 3.0 (Emilia [P]) only as a hard floor.
  - Compute it on **both** the raw and the separated clip. MOSS-TTS scores before denoising: DNSMOS > 2.8 and Audiobox PQ > 6.5 [S].
- **Relative cut:** within each character, drop roughly the bottom 15% on combined ranks (DNSMOS, Audiobox-Aesthetics PQ, UTMOSv2). Raon-OpenTTS found a 15% cut beat 50% (average rank 3.40) [P]. MOS predictors are trained on non-anime speech, so absolute thresholds are not trustworthy here [I].
- **Bleed:** music+effects stem energy ≤ −30 dB relative to speech after step 1 [I]. Plus a listening pass; at about 800 clips this takes about 1 h.
- **Identity:** anime-ECAPA `char` cosine to the set centroid ≥ that set's p10, which catches crowd lines and other speakers.
- **Text:** kana-CER (§7).
- **Fallback:**
  - Four of the eight sets total about 3.1 min or less of WAV: 八幡海鈴 3.1, 梦中情祥 3.1, 若葉睦 2.7, 豐川祥子 2.4 (plus 60 unmeasured mp3s). After gating they may be too thin for their own row.
  - Options: merge each with its same-identity set (§1), or use those clips only as x-vector/ICL references [I].

---

## 7. B6: transcript checking for Japanese anime speech

**Benchmarks** (each from its own card; they use different test sets, so there is **no head-to-head**):

| Model | Licence | Anime-domain evidence |
|---|---|---|
| **litagin/anime-whisper** (kotoba-whisper-v2.0 + 5.3k h galgame; 2024-11; not updated since) | MIT [P] | 5 held-out VN games, `no_repeat_ngram_size=5`. Averages: anime-whisper ≈13.0 (yesterday) vs whisper-large-v3 16.5, v3-turbo 16.9, kotoba-v2.0 18.8, ReazonSpeech-nemo-v2 23.6 CER [P] |
| **Qwen3-ASR-1.7B** (2026-01-28) | Apache-2.0 [P] | No Japanese-specific numbers on the card [P]. Includes **Qwen3-ForcedAligner with Japanese** (42.2 ms AAS) [P] |
| **jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame** (2026-05-31, trained on the same galgame corpus) | Apache-2.0 [P] | 800 clips, strict CER. Base Qwen3-ASR → FT: Nekopara 0.280 → 0.228; anime-speech 0.109 → 0.080; overall 0.144 → 0.129 [P]. Not compared with anime-whisper |
| kotoba-whisper v2.2, ReazonSpeech v2, parakeet-tdt_ctc-0.6b-ja | Apache / Apache / CC-BY-4.0 [P] | Weaker on anime in litagin's table, or untested |

**Protocol** [I, tools P]:
1. The filename transcript is the reference. ASR only *verifies* it.
2. Run anime-whisper (no initial prompt, which the card warns causes hallucination [P]; `no_repeat_ngram_size` 5) and jaykwok Qwen3-ASR-1.7B.
3. Convert reference and hypotheses to katakana with `pyopenjtalk.g2p(text, kana=True)` [P]. Use a user dictionary (`mecab_dict_index` / `update_global_jtalk_with_user_dict` [P]) for character, band and song names.
4. Normalise: strip punctuation, ♪, ～ and …; unify ー with vowel lengthening; fold small kana. Then compute CER over mora.
5. Decide:

   | Condition | Action |
   |---|---|
   | Either ASR has kana-CER ≤ 5% | Accept |
   | Both > 15% | Review or drop |
   | Anything between | Review |
   | Clips < 2 s | Judge by edit distance ≤ 2 mora, not by CER |

6. Use Qwen3-ForcedAligner to trim long trailing breaths or silence (yesterday's ≤ 0.3 s tail rule).

---

## 8. B7: speaker similarity and leakage detection

| Encoder | Licence | Anime suitability |
|---|---|---|
| **litagin anime_speaker_embedding** `char` / `va` (2025-05 / 2025-06) | MIT [P] | ECAPA with GroupNorm, trained on ~7M VN clips. `char`: 7,357 characters and separates two characters voiced by one VA. `va`: 989 VAs, EER 0.41% on its own validation. Its t-SNEs show vanilla ECAPA, WeSpeaker-ResNet34 and Resemblyzer collapsing anime voices [P]. **Same-speaker cosines run low, so calibrate thresholds per model** [P] |
| WavLM-large SV (Seed-TTS-Eval SIM) | [U] | The literature's comparability metric; not anime-tuned [I] |
| SpeechBrain ECAPA VoxCeleb | Apache-2.0 [P] | Poor separation of anime voices in litagin's plots [P] |
| ERes2Net / CAM++ (3D-Speaker) | [U] | Used inside X-VC [P]; no anime evidence |

**Leakage protocol** [I]:
1. **Real baseline.** Embed all real clips with `char`. Build per-character centroids. Compute the real-clip nearest-centroid confusion matrix and each character's intra-similarity p10/p25.
2. **Generated.** For each speaker row, synthesise 30 held-out lines: ja, zh and en separately, with fixed seeds. Compute generated-clip nearest-centroid accuracy and the margin (cos to own centroid − max cos to another centroid).
3. **Flag** a row when generated accuracy falls more than 10 points below the real baseline, or when any pair's generated → other-centroid similarity exceeds that other character's real p25. Check the zh/en rows separately: leakage there means the model falls back to a "default" voice.
4. **Cross-check** with the `va` variant and WavLM-SV. Do blind ABX listening for flagged pairs within the same band.
5. **Same-identity groups (§1):**
   - Separate rows: treat them as styles. Expect high mutual similarity and exclude those pairs from the leakage alarm.
   - Merged rows: add a style tag.
   - Either way, decide before training.

---

## 9. Could not verify

- Any VC result for a Japanese-trained anime target speaking zh or en.
- Any 2025–26 study of VC pseudo-data for codec-LM TTS fine-tuning, or of artefact inheritance from it.
- The BandIt Plus weight licence; the KimberleyJensen Mel-RoFormer weight licence; the WavLM-SV and 3D-Speaker licences; Beatrice's licence.
- The CosyVoice2/3 VC output rate (24k is inferred from the model configs, not run). Whether the flow model can be fine-tuned per speaker without the LLM.
- The GPU behind X-VC's RTFs. The MOSS-TTS thresholds (search snippet only).
- DNSMOS/UTMOS bias on high-pitched anime voices (no study found).
- Whether Chatterbox's watermark survives into a TTS trained on its outputs.
- Whether 梦中情祥 is Sakiko, and the identity of 墨提斯's VA (from the set names and common knowledge, not checked).
- My bandwidth measurement covers one set (`Poppin/ksm`, 12 clips) of the **raw source files**. I did not locate your length-filtered 7–10 s processed set. The other 41 mp3 zips were not decoded. All mp3 sets, and your processed copies, presumably come from the same LAME pipeline [I].
- Why the archive has 50 sets versus the 52 in the brief.
- Qwen3-ASR Japanese accuracy versus anime-whisper on the same clips.

## 10. Sources (accessed 2026-10-29/30)

**VC**
- Seed-VC repo: https://github.com/Plachtaa/seed-vc (EVAL.md)
- Seed-VC weights: https://huggingface.co/Plachta/Seed-VC
- Seed-VC mirror: https://huggingface.co/rtikw/Seed-VC
- RVC repo: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI (release 2.3.260718; `docs/en/Changelog_EN.md`; `docs/en/cli.md`; commit of 2026-08-04)
- Applio: https://github.com/IAHispano/Applio
- X-VC: https://arxiv.org/abs/2604.12456 (HTML v3); https://github.com/Jerrister/X-VC; https://huggingface.co/chenxie95/X-VC
- MeanVC: https://github.com/ASLP-lab/MeanVC
- MeanVC2: https://github.com/ASLP-lab/MeanVC2 (arXiv 2606.09050)
- Vevo and Vevo2 weights: https://huggingface.co/amphion/Vevo ; https://huggingface.co/RMSnow/Vevo2
- Amphion Vevo code: https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo
- Vevo paper: https://arxiv.org/abs/2502.07243
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice (`cosyvoice/cli/cosyvoice.py`, `frontend.py`)
- Chatterbox: https://github.com/resemble-ai/chatterbox (`example_vc.py`, `vc.py`)
- OpenVoice V2: https://huggingface.co/myshell-ai/OpenVoiceV2
- kNN-VC: https://github.com/bshall/knn-vc

**Pseudo-data and cascade**
- Huybrechts 2021: https://arxiv.org/abs/2011.05707
- Comini 2022: https://arxiv.org/abs/2207.14607
- Ellinas 2022: https://arxiv.org/abs/2210.17264
- Voice Filter: https://arxiv.org/abs/2202.08164
- VC-polyglot: https://arxiv.org/abs/2309.08255
- Dual-reference guidance: https://arxiv.org/abs/2610.29351 (HTML v2)
- X-Voice: https://arxiv.org/abs/2605.05611

**Audio data**
- LibriTTS: https://arxiv.org/abs/1904.02882
- LibriTTS-R: https://arxiv.org/abs/2305.18802
- Emilia: https://arxiv.org/abs/2407.05361
- Sidon: https://arxiv.org/abs/2509.17052 ; https://github.com/sarulab-speech/Sidon
- Generative-SE confidence filtering: https://arxiv.org/abs/2601.12254
- Raon-OpenTTS: https://arxiv.org/abs/2605.20830
- MOSS-TTS: https://arxiv.org/abs/2603.18090 (search snippet only)
- In-the-wild preprocessing pipelines: https://arxiv.org/abs/2510.03111
- MSST pretrained models: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
- DNSMOS: https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
- NovaSR: https://github.com/ysharma3501/NovaSR
- LavaSR: https://github.com/ysharma3501/LavaSR
- LAME lowpass table (old mirror) [S]: https://bazaar.launchpad.net/~x3lectric/xbmc/svn-trunk/revision/22289/lib/liblame/libmp3lame/lame.c
- Qwen3-TTS tokenizer config: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/speech_tokenizer/config.json
- Qwen3-TTS fine-tuning README: https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md
  - Aside: it says the official recipe "currently supports single-speaker fine-tuning", with multi-speaker promised for "future releases" [P]. The 50-row plan therefore needs your own multi-speaker script, as you used for Kurisu.

**ASR and speaker embeddings**
- anime-whisper: https://huggingface.co/litagin/anime-whisper
- Qwen3-ASR-1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- jaykwok anime fine-tune: https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
- pyopenjtalk: https://github.com/r9y9/pyopenjtalk
- Anime speaker embedding: https://huggingface.co/litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm ; https://github.com/litagin02/anime_speaker_embedding
- PASQA (pitch-accent MOS for Japanese TTS output, Apache-2.0, Interspeech 2026; useful for evaluation, not gating): https://huggingface.co/ly-corporation/PASQA
