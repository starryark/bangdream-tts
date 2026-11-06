# One TTS model for ~50 BanG Dream voices (ja + zh + en): many-speaker fine-tuning, state as of 2026-10-29

This report is the delta to `kurisu_v2/research/sota_models_2026-10.md` and `finetune_methods_2026-10.md` (2026-10-28). Those reports cover the model landscape and the single-speaker (10 h) recipe, and this one does not repeat them. Its question is what changes with **52 voice sets, 3–60 min each, Japanese only**, when every voice must also speak native Mandarin and English.

**Tags:**
- **[P]** primary source read in this session, by me or by a research sub-agent that quoted it: repo code, commit, issue, HF/ModelScope API or file, arXiv, licence. §8 lists what I read myself.
- **[P\*]** a detail (usually a number from a paper body or a community card) taken from a sub-agent's reading that I did not re-check.
- **[S]** secondary source.
- **[I]** my inference.
- **[U]** unverified.
- **[L]** the user's local files and logs.

All sources were accessed on 2026-10-29.

---

## 0. Answer in brief

1. **Nobody has published a many-speaker, low-data-per-speaker, cross-lingual result on any of these bases.**
   - Precedents for multi-character game TTS on LM bases exist. The 62-speaker official Llasa Genshin model and a 40-character BanG Dream Qwen3-TTS LoRA on ModelScope both ship with **no evaluation, no per-speaker data counts and no user reports** [P].
   - Your Kurisu v3 numbers, a single-speaker run that also trained 55 replay speakers, are the strongest evidence available for this plan [L].
2. **Qwen3-TTS-1.7B-Base with your trainer is still the best fit, and it is the only base whose training code has a speaker table that holds fixed voices.**
   - Your trainer gives each voice one locked x-vector row (rows 3000+), a per-row language id, and aux speakers. That already scales to 52 shipped voices: v3 trained 58 vectors [L].
   - In the other bases, speaker identity comes from elsewhere:
     - **VoxCPM2:** reference audio, or one LoRA per voice (the maintainer's advice).
     - **CosyVoice3:** only a text instruct reaches the LLM.
     - **dots.tts and Confucius4:** a reference clip or x-vector only [P].
   - **Llasa:** a plain-text speaker name. It is CC-BY-NC and outputs 16 kHz audio [P].
3. **Data floor.** For a **shared multi-speaker SFT on a codec-LM**, every documented minimum is **≥30 min to ~1 h per speaker**:
   - CosyVoice maintainer: "at least 1 h per person".
   - CosyVoice2: 400 recordings.
   - vspeech Qwen trainer: "≥30 min per speaker for multi-speaker".
   - JoyVoice and GLM-TTS: about 1 h, or LoRA below 1 h [P].

   The documented sub-10-min figures are all **one model or adapter per voice**:
   - VoxCPM2 LoRA: "5–50 clips" (≈1–7 min) [P].
   - GPT-SoVITS: "1 minute" few-shot [P].

   No source measures 3–16 min per speaker inside a shared LM-TTS fine-tune [U]. With locked x-vectors a small speaker is never learned from scratch, because the Base already clones from its vector [I]. So **the per-speaker floor to beat is the Base's own zero-shot similarity from the same vector**, not an absolute minute count [I].
4. **Failure modes with evidence:**
   - **Learned speaker rows.** Training a new speaker row from scratch made the timbre change from one text to the next (Qwen #213) [P].
   - **No speaker key.** Fine-tuning with no speaker key on 174 low-data speakers averaged their voices (Greek Parler-TTS) [P].
   - **Monolingual SFT costs other languages.** A Chinese-only CosyVoice2 speaker's English WER rose from 2.57 to 4.26 after SFT [P].
   - **Short-utterance onset flips.** A 21-min Qwen LoRA gave the wrong timbre on about 15% of short utterances; it is likely the speaker-encoder decay bug, which your trainer already fixes [P/I].
   - **Leakage between similar voices.** No paper or issue measures it [U].
   - **Your x-vectors barely separate voices.** In your own log, raw cosine between unrelated women is 0.93–0.95, and Kurisu against Amadeus is 0.981 [L]. Persona variants will sit even closer [I].
5. **Cross-lingual retention.** No source tests per-speaker zh/en retention after a joint multi-speaker fine-tune on monolingual data [U].
   - The levers that worked for Kurisu all scale: language id, replay data, and Base-generated pseudo-Mandarin in each voice.
   - But **pseudo-Mandarin per speaker drops from 531 rows to about 20 at the same 5% share** [I]. Whether "language id ⇒ native phonetics" generalises across speakers at that dose is **the main open question**, and the first ablation should test it (§6).
6. **No relevant releases since 2026-10-28:**
   - Qwen PR #384 is still open with 0 reviews. No commits, no weights, no new `qwen-tts`.
   - VoxCPM is unchanged since 2026-10-08.
   - GPT-SoVITS V5 is on `main` but has **no release tag** (latest is `20250606v2pro`).
   - No new ja+zh+en TTS with training code since 2026-10-25. Yesterday's report missed one earlier model, Audio8-TTS (2026-10-14), whose Japanese is weaker [P].

---

## 1. Decision table

| Base (licence) | How a speaker is specified in fine-tuning code | Fits 52 fixed voices in one checkpoint? | Min-data evidence per speaker | Cross-lingual-after-FT evidence | Many-speaker precedent |
|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** (Apache-2.0) | Upstream: x-vector from the row's `ref_audio`. PR #384 saves the **first** vector seen per speaker to row 3000+idx [P]. Your trainer: **locked mean x-vector per speaker** plus language id [L] | **Yes.** `spk_id` rows 3000–3071 = 72 slots [P config / I] | vspeech trainer README: "≥30 min per speaker for multi-speaker" (claim, no data) [P/U]. #343: 21 min LoRA gives onset timbre flips (likely encoder decay) [P] | Tech report Table 9: one monolingual FT speaker keeps 10 languages by WER; recipe undisclosed [P]. Your v3: 5.00/5 Mandarin, 1 speaker, 1 rater [L] | ModelScope `HELPMEEADICE/BanGDream-Qwen3TTS-LoRA`: 40 BanG Dream characters, one shared LoRA plus a 40×2048 speaker table, no eval [P] |
| **VoxCPM2** (Apache-2.0) | No speaker field. Optional per-row `ref_audio` used as a prompt; `dataset_id` only balances datasets [P] | Only via reference clip, or **52 hot-swapped LoRAs** (`load_lora`/`set_lora_enabled`) [P] | Docs: "5–50 clips" for one-speaker LoRA [P]. Maintainer: a shared LoRA without refs "randomly assigns" a voice; train one LoRA per voice (#150) [P] | Best human-rated zero-shot cross-lingual pronunciation (prior report). No FT retention data [U] | None found |
| **Fun-CosyVoice3-0.5B-2512** (Apache-2.0) | LLM gets **no speaker vector**; only an instruct string ("You are Speaker B…"). Flow gets a CAM++ vector [P] | Yes via instruct text plus `spk2info` (none shipped; build it yourself) [P] | Maintainer: "每个人最低1h吧" (≥1 h per person) (#943) [P]. CosyVoice2: "as few as 400 audio recordings" [P] | **Negative**: CosyVoice2 Spk E (zh-only) English WER 2.57 → 4.26 after SFT [P]. CosyVoice3 polyglot recipe: no amounts [P] | #1514: "Speaker A" multi-speaker FT learns voices but "makes a lot of simple mistakes" [S] |
| **Confucius4-TTS** (Apache-2.0 HF tag) | TSV rows carry `ref_audio_paths`; one is drawn at random per step. `{lang}:` prefix in text; language-balanced sampler [P] | Only with a reference clip per character at inference [I] | None [U] | None after FT [U] | None |
| **dots.tts** (Apache-2.0) | x-vector computed from the **target clip itself** (frozen CAM++), with `xvec_drop_rate`; no speaker field [P] | Only with a reference clip or stored x-vector [I] | None (smoke config only) [P] | None [U] | None |
| **Llasa-1B-Multilingual** (CC-BY-NC-4.0; xcodec2 CC-BY-NC, 16 kHz) | **Plain-text name** in the assistant prefix: `Speaker {name}` [P code] | Yes (62 Genshin speakers) [P] | None stated. Genshin has hours per character [I] | **Not evidence**: the released script filters by speaker only, so each name pools 4 dubs with different performers [P code / I] | Official HKUST 62-speaker Genshin model, full FT, lr 1e-5, 1 epoch [P] |
| **Fish S2 Pro / Drama 3** (research NC / community) | Reference in context; `<\|speaker:0\|>`; FT code only at v2.0.0-beta (S2) or removed (Drama 3) [P] | Per-character LoRA plus reference [P] | Community per-character LoRAs with 3–5 h each [P] | None [U] | 5 per-character S2 models (`Skttttt/*`, 2026-10); 6 galgame adapters (TachibanaKimika) [P] |

---

## 2. Q1: Fitting dozens of speakers with little data each

### 2.1 What each base actually supports

**Qwen3-TTS**
- **Upstream PR #384** (2026-10-17, still open; 0 comments, 0 reviews, last update 2026-10-17) [P] https://github.com/QwenLM/Qwen3-TTS/pull/384
  - It adds a per-row `"speaker"` field. Each row's slot gets the x-vector of that row's `ref_audio`.
  - The saved row is the **first x-vector seen** for each speaker (sorted names, rows 3000+).
  - The README recommends "use the same `ref_audio` for all samples of a given speaker".
  - It has **no language-id path** and does not freeze the speaker encoder (the #351 decay bug).
  - **Your trainer is ahead of it on all three points**: mean vector, language id, frozen encoder [L].
- **vspeech/Qwen3-TTS-Train** (MIT, pushed 2026-03-18) [P] https://github.com/vspeech/Qwen3-TTS-Train
  - It has `--multi_speaker` and `--multi_language` options, and saves a running-mean x-vector per speaker.
  - Its README says "At least 30 minutes per speaker for multi-speaker" and "At least 1 hour of data for single speaker" (lines 174–175). It gives no evidence for either figure [P/U].
- **Row budget** [P config / I]:
  - Base `vocab_size` is 3072, and CustomVoice uses rows such as 2861–3066 for its 9 voices.
  - Your trainer's rows 3000–3071 give 72 slots: 52 BanG Dream voices plus any device or persona rows fit.
  - The rows only *store* vectors; at training time the slot is overwritten from the locked table [L].

**VoxCPM2**
- **Hot-swap LoRA is documented** (`load_lora`, `set_lora_enabled`, `unload_lora`; "compatible with `torch.compile`") [P] https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- **The maintainer advises one LoRA per voice** (#150, 2026-01-20) [P] https://github.com/OpenBMB/VoxCPM/issues/150
  - Quote: "train multiple LoRAs, hot-load at inference".
  - A multi-voice LoRA with no reference "should randomly assign" a voice.
- **#183 (closed 2026-03-14)** [P] https://github.com/OpenBMB/VoxCPM/issues/183
  - 800 min across 12 voices, full FT at lr 1e-5 for 3k steps: "stability did not improve".
  - The maintainer suggested about 20k steps and said "we haven't tested this".

**CosyVoice3**
- **The repo has moved** to `QwenAudio/CosyVoice` [P].
- **The LLM never receives a speaker embedding.** `Qwen2LM.inference` ignores `embedding`, and the yaml says `use_spk_embedding: False # change to True during sft` (flow only) [P] https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/llm/llm.py
- **So the LLM's prosody and timbre per character must come from the instruct text** ("You are Speaker B. Please speak German.", paper §2.6.1) or from a prompt clip [P].
- **#1514** (2025-08-08) [S] https://github.com/QwenAudio/CosyVoice/issues/1514: users report that "Speaker A<|endofprompt|>" multi-speaker FT learns each speaker but makes more content errors.

**Confucius4-TTS and dots.tts**
- Both condition only on a reference clip or an x-vector taken from one.
- They would work as a "fine-tuned cloner": a fixed reference clip is stored per character and shipped with the Space [P code / I].
- Neither has a multi-speaker FT report [P issues].
- Confucius4's sampler balances by language TSV, which would push zh/en replay to parity unless it is merged into one TSV [P\*/I].

**Llasa (official game-character precedent)**
- Data and selection [P] https://github.com/zhenye234/LLaSA_training/blob/main/finetune/online_finetune/tts_online_dataset_genshin.py ; https://huggingface.co/spaces/HKUST-Audio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko:
  - It filters `simon3000/genshin-voice` by a hard-coded set of character names, **with no language filter**.
  - It formats each row as `Speaker {speaker}` in the assistant turn.
- Training [P] config.json: full FT of `Llasa-1B-Multilingual`, lr 1e-5, 1 epoch, bf16, DeepSpeed ZeRO-2.
- Usage and reports [P]: the Space exposes 62 names. Model and Space have 0 discussions, and the model gets 21 downloads a month.
- The author on name versus prompt conditioning: "I think both methods can work well" (LLaSA_training #21, 2025-02) [P].
- **Caveat** [I]: the dataset holds four dubs, each voiced by a different performer per character. So "Speaker Paimon" conditions a (speaker × language) timbre mix, and the model shows nothing about keeping **one** timbre across languages.

### 2.2 The options compared for 52 voices

| Option | Evidence | Pros for this data | Cons |
|---|---|---|---|
| **A. Shared full SFT with locked x-vector rows** (your trainer) | v3: 58 vectors trained, 3 shipped, A5 device-voice separation works [L]. Seed-TTS: 5 speakers by speaker-index token, 1–10 h each, beat zero-shot (WER 2.83 vs 3.15, SIM 0.789 vs 0.779) [P\*] | Small speakers inherit the Base's zero-shot cloning from their vector. One checkpoint, no reference audio in the Space | Leakage between near-identical vectors (§2.3). Nothing tested at 3–16 min [U] |
| **B. Shared LoRA plus a speaker table** | BanG Dream ModelScope LoRA: 40 characters, r16/α32 on talker q/v/o, `config_patch` with `spk_id` 0–39 [P files]; no eval | Cheap; the same recipe on the same franchise | Unknown quality. The ~10 MB adapter barely moves the model, so timbre comes mostly from the zero-shot vector [I] |
| **C. One adapter per character** | VoxCPM maintainer advice [P]; Spark/Fish/Qwen-0.6B community packs [P]; NanoVoice: 40 per-speaker LoRAs trained batch-wise, "comparable" to baselines [P\*] | Isolates speakers, so no leakage | Each 3-min speaker trains alone with no shared support. 52 artefacts, and cross-lingual replay must be repeated per adapter [I] |
| **D. Prompt (in-context) conditioning after FT** | TachibanaKimika S2: LoRA plus reference beat LoRA alone and zero-shot in 6/6 (ERes2NetV2 SIM, n = 1 sentence each) [P]. VoxCPM docs: 30–50% `ref_audio` rows keep cloning [P] | Rescues weak speakers at inference | A reference clip carries its BGM and language accent [I]. Your trainer has no in-context (ICL) rows [L] |
| **E. Embedding-only / speaker inversion** | MiniMax PVC: embedding-only beats SFT/LoRA, no numbers [P\*]. Irodori: 16 learned tokens, lr 0.01, 3k steps, no data floor stated [P\*]. **Qwen #213: a learned row made timbre "change randomly for every text"** [P] | Only for later additions without retraining | Evidence on Qwen is negative for rows learned from scratch |

### 2.3 Minimum data and failure modes

**Documented minimums**

| Source | Minimum per speaker | Model | Tag |
|---|---|---|---|
| CosyVoice maintainer #943 | ≥1 h per person | CosyVoice | [P] |
| CosyVoice2 §4.6 | "as few as 400 audio recordings" (≈45–60 min at 7–10 s per clip [I]) | CosyVoice2 | [P] |
| vspeech README | ≥30 min per speaker for multi-speaker | Qwen3-TTS | [P] claim / [U] evidence |
| VoxCPM docs | "Clone a single speaker: 5–50 clips" (≈1–7 min), **one LoRA per voice** | VoxCPM2 | [P] (no metrics) |
| GPT-SoVITS README | "1 minute" few-shot, one GPT+SoVITS pair per voice | GPT-SoVITS | [P] (no metrics) |
| JoyVoice (2512.19090) | 2–3 h per speaker; "LoRA-SFT when … less than 1 hour" | CosyVoice2-style | [P\*] |
| GLM-TTS (2512.14291) | LoRA needs about 1 h | GLM-TTS | [P\*] |
| ForwardTacotron (2501.05976) | 5 min pooled with high-resource speakers already gave high similarity; 1 min needed ×6 upsampling | non-LM | [P\*] |
| MigoXV ZZZ Qwen-0.6B LoRAs | 100 clips per speaker, "no human listening score yet" | Qwen3-TTS-0.6B | [P\*] |
| Your v3 replay speakers | ~36 rows (AISHELL-3) / ~19 rows (LibriTTS-R) each; trained, timbre not evaluated | Qwen3-TTS-1.7B | [L] |

**Failure modes**

1. **Collapse onto a mean voice when the model is not told the speaker clearly.**
   - Greek Parler-TTS fine-tuned on 174 Common Voice speakers (17.5 h, "many speakers contribute only a few minutes each"). It "learns a diffuse, speaker-averaged representation" (§3.2) [P] https://arxiv.org/abs/2609.10022
   - Parler-TTS is conditioned on a text description, not a speaker vector, so this is the *no-key* case [I]. The fix there was a stage-2 LoRA on 3.5 h of one speaker (SIM ≈ 0.60) [P].
   - Locked x-vectors should guard against this. Your 52 vectors are close together, though (next item) [I].
2. **Your x-vector space is anisotropic.**
   - In `kurisu_v2/artifacts/v3/logs/queue_final.log` [L]:

     | Pair | Raw cosine |
     |---|---|
     | kurisu vs amadeus (same performer, device EQ) | 0.981 |
     | kurisu vs phone | 0.957 |
     | kurisu vs unrelated AISHELL-3 women | 0.927–0.955 |
     | AISHELL-3 speakers vs each other | 0.94–0.98 |

   - The model still separated amadeus and phone (A5 projection ≥ 1.0) [L]. But the separate **anime-Amadeus** vector failed ("its speaker vector cannot carry that sound") [L].
   - **Implications** [I]:
     - Persona variants (white/black) that differ mainly in acting will get near-identical vectors and will likely be merged, or be distinguished only weakly.
     - The 8 BGM-laden anime sets risk vectors that encode the room or music more than the voice.
3. **Onset timbre flips on short utterances.**
   - Qwen #343 (2026-07-10): a 21-min English LoRA flipped timbre or gender on about 15% of short utterances [P] https://github.com/QwenLM/Qwen3-TTS/issues/343
   - The workaround was to inject the Base-extracted vector at inference, which points to the upstream speaker-encoder decay bug (#351) [P/I].
   - Your trainer freezes the encoder and locks the vectors [L], so this probably does not apply. Still test short lines for the 1–3 s anime sets [I].
4. **Content errors with speaker tags.** CosyVoice #1514 [S]; CosyVoice2 mSFT; a maintainer says CER rising after SFT is "normal" (#787) [S].
5. **Weak training data.** In arXiv 2603.10904 (Qwen-0.5B LoRA) [P abstract / P\* body]:
   - Gains depend on the training data's acoustic variability.
   - DNS-MOS can degrade while loss improves.
   - Noisy references worsen SNR after FT.
6. **Leakage between similar voices.** No paper, issue or community report measures it for any LM-TTS [U]. You will have to measure it yourself (§6).

---

## 3. Q2: Cross-lingual retention with many monolingual speakers

**What exists**

- **Qwen3-TTS report §4.2.5 / Table 9** [P] https://arxiv.org/html/2601.15621
  - One speaker ("Aiden") was fine-tuned "exclusively on monolingual data" and keeps 10 languages by WER (zh 0.903, ja 4.924).
  - §3.2 says only "lightweight speaker fine-tuning". It gives no hours, replay or language-id details.
- **CosyVoice2 §2.7 / §4.7 Table 11** [P] https://arxiv.org/html/2412.10117
  - mSFT trains many speakers at once with `Speaker A<|endofprompt|>` tags (unlabelled rows get `unknown`).
  - **Spk E had only Chinese recordings.** After SFT its similarity rose, but SEED-en WER went **2.57 → 4.26**.
  - The paper calls it "still worse than the base model especially on English".
  - **This is the only direct measurement of monolingual-speaker SFT and other-language loss**, and it shows a loss.
- **CosyVoice3 §2.6.1 / §5.8** [P]
  - Polyglot recipe: auxiliary monolingual data from random speakers, with speaker and language in the instruction, and speaker prompts randomly masked.
  - **No speaker count, per-speaker amount or mix ratio is given.** No per-language similarity is reported.
- **Qwen-Audio-3.0-TTS §2.3** (closed) [P\*]
  - Stage 1: LM+FM on the target speakers plus a replay subset "matched by effective audio duration" (≈1:1 [I]).
  - Stage 2: FM only.
  - Four speakers; no cross-lingual results.
- **Adapter study on a small GAN TTS (2508.18006)** [P\*]
  - Own-language fine-tuning "results in severe degradation in accent nativeness".
  - Phoneme substitution rate (lower = more native): full FT 8.44, whole-model adapters 6.27, vocoder-only adapters 0.75. Similarity follows the opposite order: 0.70 / 0.78 / 0.62.
  - This is a timbre-versus-accent trade-off [P\*]. Transfer to codec LMs is [I].
- **Your v3** [L]:
  - Japanese language id, 15% native replay (30 AISHELL-3 + 25 LibriTTS-R speakers, each with its own vector), and about 5% Base-generated pseudo-Mandarin in the target voice.
  - zh CER 0.017, SER 0.060, against the native anchor's 0.048. Blind rating 5.00/5 (one rater).
  - Pseudo-Mandarin was the largest single accent gain.

**Not found** [U]:
- Any per-speaker zh/en accent retention after joint multi-speaker FT.
- Any replay-ratio ablation for TTS speaker FT.
- Any speaker–language adversarial loss in a 2025–26 LM-TTS fine-tune.
- Any game-character model trained on Japanese only and evaluated in zh/en.
- The Genshin "multilingual" models are trained on per-language performers (§2.1), so they do not count.

**How the v3 levers scale to 52 speakers** [I unless noted]

| Lever | v3 (1 voice) | 52 voices at the same share | Risk |
|---|---|---|---|
| Language id on every row | Yes; 25% Auto layout | Same, no cost | None new |
| Native replay ~14% | 1,566 rows from 2.3 h pools | ~3.9 h needed (0.143 / 0.81 × 21.6 h JA). Exceeds the v3 pools, but AISHELL-3 and LibriTTS-R are much larger [I] | Replay voices may sit near some characters in x-vector space. Prefer replay speakers far from all 52 by mean-centred cosine |
| Pseudo-Mandarin ~5% | **531 rows for one voice** | ~1.3 h total ≈ **15–25 rows per voice** | **Unknown whether the effect transfers across voices at this dose.** Low-data and BGM voices get worse Base renders to filter from |
| Pseudo-English | Not used (EN WER 0.002) | Optional | Not measured for accent in v3 |
| Base renders needed | 1,800 (900 texts × 2) | ×52 ≈ 94k renders for the same per-voice pool; ~25k if 250 texts per voice | Compute only |

---

## 4. Q3: Community precedents (2025–2026)

| Id (date) | Base | Voices | Data per voice | Conditioning | Languages | Metrics / reports | Licence |
|---|---|---|---|---|---|---|---|
| `HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko` (2025-02-13) | Llasa-1B-Multilingual | 62 | not stated (hours per character in the source) | Text `Speaker {name}` | 4 dubs pooled per name | None; 0 discussions [P] | CC-BY-NC-4.0 |
| `kafka23/LLASA-ZZZ-Genshin-HSR` Space (2025-06) | the model above | 11 | 1 reference clip | **ICL on the multi-speaker FT model**, applied to unseen characters | — | Space in BUILD_ERROR [P\*] | — |
| **ModelScope `HELPMEEADICE/BanGDream-Qwen3TTS-LoRA`** (2026-05-21) | Qwen3-TTS-1.7B-Base [P\* sibling train_args] | **40** (Poppin'Party, Afterglow, HHW, Morfonica, MyGO, PasPale, RAS, Roselia; no Ave Mujica) | unknown | One shared LoRA (r16/α32, talker q/v/o, peft 0.19.1) + `speaker_embedding.safetensors` (≈40×2048 bf16) + `config_patch` (`custom_voice`, `spk_id` 0–39) [P files] | ja presumably | Empty card [P] | labelled GPL-3.0 |
| ModelScope `HELPMEEADICE/MyGO-Qwen3TTS-LoRA` (2026-05-21) | Qwen3-TTS-1.7B-Base | 5 | ≈1.8k clips in total [I] | Same; lr 3e-5, 3 epochs, sub-talker weight 0.3 [P\*] | — | Loss logs only | GPL-3.0 |
| `KitsuneX07/Datasets_for_BanG_Dream` (2024-06-27) | dataset | 40 zips, PPP–MyGO | — | — | ja | — | CC-BY-NC-SA [P\*] |
| `rijulpaul/Qwen3-TTS-0.6B-*-LoRa-*` (2026-09) | Qwen3-TTS-0.6B | 1 per adapter (Genshin) | 273–881 clips | Per-character LoRA + baked x-vector | en | None [P\*] | MIT |
| `MigoXV/qwen3-tts-0.6b-lora-*` (2026-10-02) | Qwen3-TTS-0.6B | 1 per repo (ZZZ) | 100 clips | Per-speaker LoRA + fixed x-vector | zh | "no human listening score yet" [P\*] | Apache-2.0 |
| `wesjos/spark-tts-genshin-charactors` (2025-06-26) | Spark-TTS-0.5B | 18 models | — | One full FT per character | zh | Samples only [P\*] | Apache-2.0 |
| `Skttttt/Fishspeech-S2-Pro-*-V2` (2026-10-20/27) | Fish S2 Pro | 5 models | 4.2–5.3 h | Per-character merged model + reference | ja | "Blind listening tests found V2 a dramatic improvement" (no n or protocol) [P\*] | Fish research NC |
| `TachibanaKimika/fish-speech-lora` (2026-08-01) | Fish S2 Pro | 6 adapters (galgame) | — | Per-character LoRA r8, `<\|speaker:0\|>` | ja | LoRA + prompt beat zero-shot SIM in 6/6 (+0.02…+0.15); LoRA without prompt beat it in 4/6; n = 1 sentence [P\*] | Fish research NC |
| `OmniAICreator/Galgame-Llasa-*`, `NandemoGHS/Anime-Llasa-3B` (2025) | Llasa | thousands, no ids | — | Zero-shot prompt | ja only | None [P\*] | CC-BY-NC-4.0 |

**Takeaways**
- **Two patterns, and nobody compares them** [P/U]:
  - a shared model with a speaker key (HKUST Llasa; the BanG Dream Qwen LoRA);
  - one model or adapter per character (Spark, Fish, Qwen-0.6B packs).
- **The 40-character BanG Dream Qwen LoRA is the closest precedent** to this project [P]. It shows a single `custom_voice` checkpoint with a 40-row table is buildable on Qwen. It gives no evidence of quality, low-data behaviour or zh/en.
- **No precedent trains Japanese-only characters and reports zh/en in the same timbre** [U].
- **GPT-SoVITS V5** [P\*] https://github.com/RVC-Boss/GPT-SoVITS/issues/2843 ; https://github.com/RVC-Boss/GPT-SoVITS/releases :
  - It landed on `main` in `3b95794` (2026-10-06); the README points to branch `cuda_graph_accel_v5`. Weights are in `lj1995/GPT-SoVITS/gsv-v5-pretrained` (2026-10-05/06).
  - **There is still no release tag**; the latest is `20250606v2pro`.
  - V5 notes: "Significantly improved voice similarity without SoVITS fine-tuning", plus a new vocoder.
  - #2843 (2026-10-11): V5 SoVITS trains as a LoRA (r32, 48 kHz). Fine-tuning the GPT stage made Japanese timbre unstable.
  - The general README (not V5-specific) still claims 5 s zero-shot, 1 min few-shot, and en/ja/ko/yue/zh cross-lingual.
  - No V5-specific multilingual or few-shot data guidance exists [U].
  - It is still one GPT+SoVITS pair per character, and the wiki says single-language fine-tuning loses cross-lingual ability (prior report) [S].

---

## 5. Q4 and Q5: New since 2026-10-28, and tricks for low-data speakers

### 5.1 New since 2026-10-28

**Qwen3-TTS** [P]
- No commits since `022e286` (2026-03-17).
- PR #384 unchanged (open, mergeable, 0 reviews).
- Issues updated since 10-27 are bot pings only (#369, #350, #278, #118).
- The Qwen HF org's newest model is Qwen-Image-2.1-Turbo (2026-10-09).
- `qwen-tts` is still 0.1.1.

**VoxCPM** [P]
- Last commit `cce58a5` (2026-10-08, i18n and HF Hub training); `voxcpm` 2.0.3 (2026-05-11).
- No multi-speaker-LoRA feature beyond the documented hot-swap.

**New models**
- No new open ja+zh+en TTS with training code created on HF since 2026-10-25 [P listing].
- Missed by yesterday's report: **Edge0/Audio8-TTS-Realistic-0.6b** (2026-10-14) [P\*]. It is Apache-2.0, a Fish-style DualAR model with SFT code and no speaker field. Its CV3-ja CER is 7.51, worse than Qwen's. Not a candidate [I].

### 5.2 Tricks for low-data speakers, by evidence

| Trick | Evidence | Applies here? |
|---|---|---|
| **Locked encoder x-vector rather than a learned row** | Qwen #213: learned row → timbre "change randomly for every text" [P]. #343/#351: encoder decay breaks baked vectors [P]. Your v3 [L] | **Keep.** Already in your trainer |
| **Build the vector from a consistent subset** | CosyVoice2 §4.6: "unsupervised clustering on the speaker embeddings of the same speaker to ensure the stability of the speaker's timbre" [P] | **Yes** for BGM sets and persona variants. Cluster each set's clip vectors and average the dominant clean cluster [I] |
| **Clean before embedding** | Emilia-Pipe: UVR-MDX music removal + DNSMOS > 3.0 [P\*]. Fujita (NTT, 2401.05111): music/noise lowers similarity DMOS (3.28 → 2.18 at −5 dB); speech enhancement (SE) helps but "solely applying SE … would be insufficient" [P\*] | **Yes** for the 8 anime sets. v3 dropped anime clips below the 28.1 dB studio SNR floor [L] |
| **Upsample small speakers** | ForwardTacotron: ×6 weighting only for 1-min speakers; batch-binning ablation mixed [P\*]. No LM-TTS study [U]. Your E6 (tags ×3) cost similarity [L] | **Cautiously**, e.g. temperature sampling at α ≈ 0.5 so a 5-min voice is seen ~3× and a 60-min voice ~0.9× per epoch [I]. Ablate |
| **Shared stage, then per-speaker stage** | Qwen-Audio-3.0 (LM+FM, then FM only) [P\*]; jmtl Cantonese (language LoRA then 0.54 h speaker LoRA, CER 0.150) [S]; Greek Parler (stage-2 LoRA, 3.5 h) [P]; NanoVoice (40 per-speaker LoRAs batch-wise) [P\*] | **Fallback only**, for voices that fall below their Base zero-shot similarity after shared FT. Serving Qwen LoRAs per voice is undocumented [U] |
| **Prompt / ICL at inference after FT** | S2 6/6 SIM gain (n = 1) [P\*]; VoxCPM 30–50% ref rows [P]; Llasa author "both can work" [P] | Possible rescue for weak voices, but your trainer has no ICL rows [L]. Untested on Qwen [U] |
| **Freeze parts / embedding-only** | MiniMax PVC (claims only) [P\*]; Irodori inversion (no data floor) [P\*]; your E3f text-path freeze gave no gain [L] | No |
| **Initialising new speaker rows** (mean / nearest neighbour) | No study; only the KSA language-row-mean anecdote [S, prior report] | Moot with locked x-vectors [I] |

---

## 6. Recommended plan [I unless tagged]

1. **One shared full SFT of Qwen3-TTS-1.7B-Base with your trainer.**
   - **52 shipped rows** (one per voice set), each a locked mean x-vector.
   - **Language id on every row**, with the 25% Auto layout kept.
   - **~14% native replay**: ~3.9 h from AISHELL-3 and LibriTTS-R, choosing replay speakers far from all 52 by mean-centred cosine.
   - **~5% pseudo-Mandarin from Base, spread over all voices.**
   - **1 epoch.** About 2× v3's data, so roughly 700–800 steps at v3's batch size.
2. **Pre-flight checks, before any training:**
   - **Mean-centred cosine matrix** of the 52 vectors. Subtract the mean of the 52+55 vectors first, since raw cosine is 0.93–0.98 for unrelated voices [L].
   - **Per-voice Base zero-shot baseline**: similarity and Mandarin LID/SER from the same vector. This is each voice's floor.
   - **BGM sets**: source separation and a DNSMOS filter on the 8 anime sets, and build their vectors from the dominant clean cluster (CosyVoice2-style).
   - **Persona variants**: give them separate rows only if their centred cosine is clearly below that of same-performer, same-persona clip pairs. Otherwise merge them, and treat the persona as untrainable under locked vectors (the anime-Amadeus lesson [L]).
3. **First ablation (the open question):** pseudo-Mandarin at about 20 rows per voice (5% share) against about 100 rows per voice (≈20% share). Each arm is scored on:
   - per-voice zh SER/LID;
   - Japanese CER;
   - **per-voice similarity against the Base zero-shot floor**;
   - a **leakage metric**: nearest-centroid speaker ID over the 52 real-clip centroids, with the confusion rate of generated clips compared against real held-out clips.
4. **Second ablation:** temperature sampling (α ≈ 0.5) against natural proportions, scored on the 12 small sets.
5. **Fallback** for any voice still below its Base zero-shot floor:
   - Serve it with its Base x-vector instead of the fine-tuned row, i.e. pass the speaker embedding as the #343 workaround does [P].
   - Or train a stage-2 per-voice adapter.
   - Or, for the few tiny sets that still fail, use a **VoxCPM2 per-voice LoRA** (hot-swap). It is the only documented low-data, per-voice path (5–50 clips) and has the best human-rated zero-shot cross-lingual pronunciation. The cost is a second engine in the Space, with its own zh/en replay per adapter [P/I].
   - Do **not** move the whole project to VoxCPM2 or CosyVoice3. Neither has a fixed-voice table: VoxCPM2 would mean 52 adapters, and CosyVoice's own floor is about 1 h [P].
6. **Licence check** [P]:
   - Qwen3-TTS, AISHELL-3 (Apache-2.0) and LibriTTS-R (CC-BY-4.0) all allow public weights and a free Space.
   - If you ever use Llasa or the KitsuneX07 BanG Dream dataset, they carry CC-BY-NC(-SA), which still allows a non-commercial Space.

---

## 7. Could not verify

1. Any LM-TTS result for 3–16 min per speaker inside a *shared* multi-speaker fine-tune. Shared-SFT floors are ≥30 min to 1 h. The sub-10-min figures (VoxCPM2 5–50 clips, GPT-SoVITS 1 min) are per-voice claims without metrics, and ForwardTacotron's 5 min is non-LM.
2. Any measurement of leakage between similar voices in multi-speaker fine-tuning, for any base.
3. Any per-speaker zh/en retention or accent measurement after a joint multi-speaker fine-tune on monolingual speakers.
4. Whether the BanG Dream ModelScope LoRA's speaker table holds frozen x-vectors or learned rows; its data source, minutes per character and quality.
5. Which language tracks, and how many minutes per character, the released HKUST Genshin model used. The script has no language filter, but the actual run is not documented.
6. The basis for vspeech's "≥30 min per speaker" claim.
7. Whether pseudo-Mandarin's accent gain transfers across voices at ~20 rows per voice.
8. Whether the Qwen ICL (reference-codes) mode can be combined with fine-tuned `custom_voice` rows to rescue weak voices.
9. GPT-SoVITS V5's language coverage and data guidance. The V5 notes do not state them.
10. Bilibili and ModelScope coverage is partial. Web search returned nothing relevant, and ModelScope's name filter is fuzzy.

## 8. Verification notes

**Read directly by me:**
- the Llasa Genshin card, API, Space `app.py`, dataset loader, config and issues #21/#51;
- the HF API for the Llasa licences;
- the BanG Dream ModelScope file list, `config_patch.json`, `adapter_config.json` and README;
- Qwen #343;
- the vspeech README lines;
- CosyVoice2 §2.7/§4.6/§4.7;
- the Greek Parler-TTS paper, abstract and HTML;
- the 2603.10904 abstract;
- the Qwen Base/CustomVoice `config.json`;
- local logs and README_v3;
- the PR #384 diff: x-vector `.detach()`ed from each row's `ref_audio`; the first vector per speaker is kept (`if spk_key not in target_speaker_embeddings`) and saved to row `3000 + idx`; no language lines are added;
- VoxCPM #150 (maintainer, 2026-01-20): "如果要支持多个音色可以训练多个Lora，推理的时候可以热加载" ("to support several voices, train several LoRAs and hot-load them at inference");
- CosyVoice #943 (maintainer, 2025-02-06): "每个人最低1h吧" ("at least 1 h per person").

**Corrections from spot-checks:**
- **Greek Parler-TTS:** per-speaker minutes are not reported. The paper says 174 speakers, 17.5 h, and "many speakers contribute only a few minutes each" (≈6 min average [I]).
- **2603.10904:** the mixed-training similarity claim (SIM 0.721 from 11–22% of the data) is not in the abstract. It is a sub-agent full-text reading [P\*] and is left out of the tables.

**Everything else marked [P\*]** came from sub-agents reading the cited repos, issues and papers.
