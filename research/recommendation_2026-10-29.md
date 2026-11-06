# BanG Dream character voices (ja + zh + en): recommended approach (2026-10-29)

This is the synthesis. Evidence lives in four companion files in this folder:
- `data_facts_2026-10-29.md`: local measurements of the pack, including speaker separability on the GPU.
- `multispeaker_ft_2026-10-29.md`: many-speaker fine-tuning with little data per speaker, plus cross-lingual retention.
- `vc_and_data_2026-10-29.md`: voice conversion, audio restoration and cleaning, ASR checking, speaker metrics.
- `community_and_legal_2026-10-29.md`: existing BanG Dream work, data provenance, rules and law.

The model landscape is not repeated here; see `kurisu_v2/research/sota_models_2026-10.md`.

**Tags:**
- **[P]** primary source I read myself.
- **[P\*]** primary source read and quoted by a research sub-agent, not re-checked by me.
- **[S]** secondary source.
- **[L]** measured locally.
- **[I]** inference.
- **[U]** unverified.

**User requirements (2026-10-29):**
- Japanese, Mandarin and English.
- Packaging is open ("search in depth into the best approach").
- Ignore the bundled GPT-SoVITS pack.
- Publish on Hugging Face as with Kurisu.

---

## 0. Answer in brief

1. **Publishing on HF as planned conflicts with the rights holder's published rules.** These rules are explicit and cover non-commercial fan work.
   - The official BanG Dream! guideline prohibits「本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開。」(item 10: making, displaying, distributing or publishing derivative works that use the content's audio) [P, https://bang-dream.com/bdp-guideline/]. The guideline permits fan works only when they are personal and non-commercial, and item 10 sits in its list of prohibited uses.
   - The Girls Band Party posting rule (制定日 2026-09-24) bans videos made by altering game voices,「加工・改変することを目的として生成AIにボイス等を入力する行為を含みます」(including inputting voices into generative AI) [P, https://bang-dream.bushimo.jp/rule/].
   - The realistic options:
     - **(a)** Keep the models local or private.
     - **(b)** Publish only code, recipe and evaluation: no weights, no audio, no demo Space.
     - **(c)** Ask Bushiroad for permission.
   - Gating, disclaimers and a takedown contact reduce visibility. They do not resolve item 10 (§6).
   - **Superseded (user, 2026-11-06): the E1b epoch-2 weights were released publicly on HF (`starrydark/BanG_Dream_Qwen3_TTS`) with the code on GitHub; no audio, transcripts or converters. See `e5/E5_RESULTS.md`.**
   - **Decision (user, 2026-10-29): (a) local/private only.** Nothing is published: no weights, no Space, no dataset, no audio. Training on a rented RunPod pod is still fine, as long as checkpoints stay private and are deleted from the pod afterwards.
   - **What this changes:**
     - Licence no longer rules anything out. Non-commercial bases become admissible: Fish S2, Llasa, Higgs 3, OmniVoice, Vevo.
     - None of them changes the ranking, because each still lacks a fine-tuning path or fixed-voice table, or has weaker Japanese.
     - The HF card, Space and discovery work drop out of the plan.
2. **Technical recommendation: one shared full SFT of Qwen3-TTS-12Hz-1.7B-Base, using your Kurisu trainer unchanged in kind.** It uses one locked mean x-vector row per voice, a language id on every row, ~14% native zh/en replay, Base-generated pseudo-Mandarin in every voice, and 1 epoch.
   - All three research threads and both local GPU runs point here:
     - **Only Qwen has a fixed-voice speaker table.** It is the only candidate whose training code stores fixed voices in one checkpoint, with rows 3000–3071, so 72 slots [P\*]. VoxCPM2 needs a reference clip or one LoRA per voice; CosyVoice3's LM sees only an instruct string [P\*].
     - **Low-data voices inherit the Base clone.** With locked vectors, a small voice is never learned from scratch. It starts from the Base's own zero-shot clone of that vector [I]. Qwen #213 shows that speaker rows learned from scratch make the timbre "change randomly for every text" [P\*].
     - **Your recipe is the only primary evidence for native Mandarin from a Japanese-trained voice.** Kurisu v3 rated 5.00/5, equal to the native anchor (1 rater, 16 sentences) [L].
     - **The speaker vectors separate.** After mean-centring, different characters sit at a median cosine of −0.05 against 0.97 for same-set halves, and 94% of real clips land on their own centroid [L].
3. **Expectations by data bucket:**
   - **20–60 min (35 sets):** should beat zero-shot.
   - **10–20 min (5 sets):** probably beat zero-shot; measure.
   - **< 10 min (9 sets after dedup, mostly Ave Mujica):** the honest target is *not to fall below the Base's zero-shot clone from the same vector*. No source measures 3–16 min per speaker inside a shared LM-TTS fine-tune. Every documented shared-SFT floor is ≥ 30 min to 1 h per speaker [P\*].
4. **Cost.**
   - Kurisu v3's selected run trained 343 steps in **~250 s on one H100** [L `queue_final.log`].
   - This build has ~1.8× the audio, so about 8 min per epoch.
   - The pipeline cost is dominated by code extraction, pseudo-Mandarin generation and evaluating ~50 voices. That is a few hours on one Secure H100 at $3.99/h, roughly $15–40 per full iteration [I].
5. **The one open technical question is the pseudo-Mandarin dose per voice.**
   - Kurisu had 531 Base-generated Mandarin rows for one voice. At the same ~5% share, that becomes **~15 rows per voice** across ~50 speaker rows [I].
   - Nobody knows whether that is enough to keep the Mandarin native. It is the first ablation (§5).
6. **Not recommended:**
   - **Per-character GPT-SoVITS** (V5 still has no release tag) [P\*].
   - **VoxCPM2 per-voice LoRAs:** 50 adapters, and replay must be repeated per adapter.
   - **Irodori or Style-Bert-VITS2:** Japanese only.
   - **A native-TTS → voice-conversion cascade as the deliverable:** best accent, but similarity 0.601 → 0.468 and 3× WER [P\*].
   - **CosyVoice3:** its maintainer gives ≥ 1 h per person as the floor [P\*].
   - Each keeps a narrow fallback role (§2).

---

## 1. The data in five lines (details: `data_facts_2026-10-29.md`)

- **Size.** 50 zips = 49 distinct voice sets (梦中情祥 is a duplicate) = **45 characters** once the persona sets are merged (月ノ森祥子, 月ノ森素世, 大魔姬亞子, 墨提斯); 49 speaker rows (§4.1). 9,804 unique clips, 21.6 h, Japanese, with filenames as transcripts. (The bundled GPT-SoVITS pack has 52 *models* because it splits extra styles; there are 50 voice *sets*.) [L]
- **Two sources** [L]:
  - 42 game sets: 32 kHz mono mp3, 52–82 kbps by mutagen header; length-filtered to ~7–10 s.
  - 8 Ave Mujica sets: anime-style 44.1 kHz WAV, 1–3 s clips, ~8 dB quieter.
  - Every set is band-limited to ~15.3 kHz.
  - The game mp3s are the Bestdori files themselves, not re-encodes [P\*].
- **Buckets (unique minutes)** [L]:
  - < 10 min: 9 sets after dedup (10 counting the 梦中情祥 duplicate), including every Ave Mujica member.
  - 10–20 min: 5 sets.
  - 20–30 min: 16 sets.
  - 30–60 min: 19 sets.
- **Duplicates** [L]: 梦中情祥 is a byte-identical subset of 豐川祥子（月ノ森限定）. 115 byte-identical duplicate groups in total.
- **Separability** [L]:
  - Raw Qwen speaker-encoder cosines are ~0.98, but that is a shared offset. Centred, the closest pairs are 豐川祥子 ~ 月ノ森 0.857, 若宮伊芙 ~ 朝日六花 0.818 and 三角初華 ~ 八幡海鈴 0.814.
  - WavLM-base-plus-sv separates these characters on only 56% of clips, so it cannot be the metric.

---

## 2. Approaches compared for this data and goal

| Approach | Japanese | zh/en natively in the character's voice | Low-data (< 10 min) behaviour | Packaging for ~50 voices | Licence (public weights + free Space) | Verdict |
|---|---|---|---|---|---|---|
| **A. Shared full SFT, Qwen3-TTS-1.7B-Base, locked x-vector rows, language id, replay, pseudo-zh** (your trainer) | v3 JA CER 0.030, sim 0.671 (Kurisu) [L] | v3: 5.00/5 Mandarin, SER 0.060 vs native 0.048 [L]; untested with many speakers [U] | Starts from the Base zero-shot clone of its vector [I]; untested at 3–16 min [U] | **One checkpoint, ~50 rows ≤ 72 slots** [P\*] | Apache-2.0 [P\*] | **Recommended** |
| B. Shared LoRA + speaker table on Qwen (the ModelScope `HELPMEEADICE/BanGDream-Qwen3TTS-LoRA` pattern, 40 characters) | Unknown; its MyGO run was 110 steps, no eval [P\*] | No claim [P\*] | Unknown | One adapter | GPL-3.0 on that repo [P\*] | Precedent only. Your full-SFT path is already better tested |
| C. One VoxCPM2 LoRA per voice | MiniMax-ja SIM 0.828 zero-shot [P\*] | Best human-rated zero-shot cross-lingual pronunciation among fine-tunable models [P\*, prior report] | Documented "5–50 clips" per voice, no metrics [P\*] | 50 hot-swapped adapters; replay repeated per adapter [I] | Apache-2.0 | **Fallback for the tiny voices that fail A** |
| D. Per-character GPT-SoVITS (v2ProPlus / V5), Japanese reference → Chinese text | Community default | This is what Chinese creators do today, but nobody measures accent [P\*] | "1 minute" claim [P\*] | 50 GPT + SoVITS pairs | MIT | Not recommended. V5 has no tag, and the v2Pro notes say v3/v4 cannot handle average-quality data [P\*] |
| E. Irodori v4.1-Small speaker inversion / LoRA | Best published readings among Japanese-only models [P, prior] | **None (Japanese only)**; needs a second engine for zh/en, so two timbres [I] | Inversion is a 49 KB file per voice; a 9-character Umamusume Space ships this way [P\*] | 50 inversion files | MIT | Japanese-only alternative; not for this goal |
| F. Native zh/en TTS → per-character VC (RVC v2 multi-speaker, Seed-VC 44.1k) | — | Most native accent; timbre loss (ECAPA 0.601 → 0.468) and 3× WER [P\*, arXiv 2610.29351] | RVC recommends ≥ 10 min clean data per voice [P\*] | One TTS + one VC model | RVC MIT; Seed-VC GPL-3.0 [P\*] | **Pseudo-data generator only**, if the dose experiment fails |
| G. CosyVoice3 multi-speaker SFT (polyglot recipe) | CV3-ja 10.4 without RL [P, prior] | Polyglot recipe, no per-speaker amounts [P\*] | Maintainer: ≥ 1 h per person [P\*] | One model; speaker key via instruct text only | Apache-2.0 | Borrow the polyglot idea (already in A), not the model |

---

## 3. Ranking by bucket

| Bucket | Sets | Plan | Success criterion |
|---|---|---|---|
| **30–60 min** | 19 (e.g. 市ヶ谷有咲 59.2, 青葉摩卡 57.8, 丸山彩 50.4) | A | Sim above the Base zero-shot floor; JA CER ≤ v3's 0.03; zh SER near the native anchor |
| **20–30 min** | 16 | A | Same |
| **10–20 min** | 千早愛音 10.8, 月ノ森祥子 12.5, 要樂奈 12.6, 月ノ森素世 16.0, 湊友希那 18.4 | A, with temperature upsampling if E2 says so | Same; leakage no worse than on real clips |
| **< 10 min** | 若葉睦 2.7, 八幡海鈴 3.1, 墨提斯 5.0, 大魔姬亞子 5.2, 三角初華 5.5, 豐川祥子 6.1, 祐天寺若麥 6.5, 宇田川亞子 7.9, 長崎素世 9.5 | A after cleaning (§4.2). Fallbacks in order: (1) serve that voice with its Base vector, the #343 workaround [P\*]; (2) a VoxCPM2 LoRA for that voice | **Not below the Base zero-shot clone** of the same vector, on sim and leakage |

---

## 4. Recipe

### 4.1 Build

- **Dedup on audio hash.** Drop 梦中情祥 (keep its 69 ids as a style label on the 月ノ森 rows if wanted) [L].
- **Speaker rows (≈50)** [L/I]:
  - One row per character.
  - Separate rows for 墨提斯 (centred 0.237 vs 若葉睦), 大魔姬亞子 (0.197 vs 宇田川亞子) and 長崎素世（月ノ森限定） (0.593).
  - **豐川祥子 白/黒 at 0.857** are the closest pair. Even their real clips confuse with each other (59–69% nearest-centroid accuracy). Train two rows, but expect them to blend; merge if the leakage metric shows no separation beyond the real-clip baseline.
  - 祐天寺若麥's `nyamu0`/`nyamu1`: centred cosine **0.941**, against split-half 0.98 / 0.962 [L, `scripts/nyamu_check.py`]. The encoder barely tells them apart, so **merge them into one row**. The mode difference is acting, which a locked vector cannot carry (the anime-Amadeus lesson).
  - Result: **49 rows**, i.e. 45 characters plus 墨提斯, 大魔姬亞子, 長崎素世（月ノ森限定） and 豐川祥子（月ノ森限定）, within the 72 slots.
- **Transcripts** [L/P\*]:
  - Strip `(A)`; normalise full-width digits to readings in about 600 rows.
  - Fix one policy for `…`, `――`, `〜` and `♪`.
  - Check every row with litagin/anime-whisper (no initial prompt, `no_repeat_ngram_size` 5) and `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`, comparing at kana level. Accept a row if either model is within 5%; review it if both exceed 15%.
- **Audio** [L/P\*/I]:
  - Resample to 24 kHz. Normalise loudness to one target; the Mujica sets are ~8 dB low.
  - Pad the lead to 100–200 ms (the game clips are trimmed to 20 ms); trail ≤ 300 ms.
  - **No restoration or bandwidth extension.** 15.3 kHz exceeds the 12 kHz Nyquist of a 24 kHz model. Restoration lowered speaker similarity in LibriTTS-R and can hallucinate [P\*].
  - Optional check: ABX of tokenizer resyntheses, raw vs Sidon-restored, on 50 clips [P\*].
- **Language ids:** `japanese` on all pack rows, `chinese` / `english` on replay and pseudo rows. Keep v3's Auto-layout share [L].
- **Replay:** ~14% native zh/en, AISHELL-3 (Apache-2.0) and LibriTTS-R (CC-BY-4.0) [P\*]. Pick replay speakers far from all 50 by **centred** cosine [I].
- **Pseudo-Mandarin:** Base zero-shot from each voice's locked vector, with the same script as `kurisu_v2/pod/p11_pseudo_zh.py`. The dose is set by E1.

### 4.2 Cleaning the anime-sourced sets (all 8 Ave Mujica sets)

- **Separation:** a dialogue/music/effects separator first (BandIt Plus), then a music-vocal RoFormer only if music remains. De-reverb only flagged clips [P\*].
  - Their inter-word floors are already near digital silence (−80 to −107 dB) [L], so some sets may have been separated upstream. Check before re-separating.
- **Quality gate:**
  - Drop about the bottom 15% per character on DNSMOS/UTMOS.
  - Use percentiles, not an absolute 3.0, because DNSMOS loops 1–3 s clips [P\*].
- **Speaker vectors:** build each vector from the dominant clean cluster of that set's clip vectors (CosyVoice2 practice) [P\*].
- **Same gate for the game sets with low noise floors:** 大魔姬亞子, 珠手知由, 宇田川亞子, 桐谷透子 (20–45% of clips have a floor above −50 dB) [L]. Spot-listen.

### 4.3 Evaluation (reuse the Kurisu suite, plus two additions)

- **Reused:** JA kana-CER (anime-whisper), zh shared-char SER (human-validated, ρ = −0.84 against your ratings [L]), en WER and the runaway rate.
- **Added: per-voice similarity floor.** Render the same probe texts from the **Base zero-shot clone of each voice's locked vector**. That is each voice's floor.
- **Added: leakage.**
  - Fit a speaker-ID model on real held-out clips: litagin `anime_speaker_embedding` `char` variant (MIT, separates characters voiced by one actress) [P\*], or a linear probe on centred Qwen speaker-encoder vectors (94% real-clip accuracy) [L].
  - Score generated clips. Report the confusion rate **minus** the real-clip confusion rate per voice.
  - Do not use WavLM-SV (56%) [L].
- **Human rating:** blind Mandarin rating on a stratified subset (2 voices per bucket), as for Kurisu.

---

## 5. Experiment plan

| # | Experiment | Arms | Decides |
|---|---|---|---|
| E0 | Pre-flight, no training (local 5060 Ti) | Base zero-shot renders for all voices (floor); transcript check; Mujica cleaning A/B on vectors. *Done:* nyamu modes merge (0.941) | Rows, merges, floors |
| **E1** | **Pseudo-Mandarin dose** | ~15 rows per voice (~5% share) vs ~60 rows per voice (~20%) | Whether language id plus a small per-voice dose keeps native Mandarin across many speakers. Score per voice: zh SER, JA CER, sim against the floor, leakage |
| E2 | Small-voice weighting | Natural proportions vs temperature sampling α ≈ 0.5 (a 5-min voice seen ~3×, a 60-min voice ~0.9×) | Low-data bucket |
| E3 | If E1 fails on accent | Add 5–10% VC pseudo-data (native zh/en → RVC v2 48k multi-speaker; also convert some Japanese so "VC sound" is not tied to zh/en) [P\*/I] | Accent rescue |
| E4 | Fallbacks for voices below the floor | Base vector at inference; per-voice VoxCPM2 LoRA | Tiny voices |

**E0 outcome (2026-10-30, `e0/e0_results.md`).** E0 is done and changes three things:
- The Base zero-shot floor already gives near-native Mandarin from every voice: zh SER 0.077 against native 0.048, lid −19.4.
- Identity collapses for the bright game voices: leak 50–88% for kasumi, rimi, lock, eve, rinko, kanon and kokoro. The Mujica voices clone well. So E1's success criterion is identity (probe leak and p_true against real clips), with Mandarin held at the floor.
- Pseudo-Mandarin rows must pass the identity probe and the accent gate (lid ≤ −19, SER ≤ ~0.08). The collapsed voices get a second generation round from the round-1 fine-tune.
- The leakage pairs to watch are the bright game voices (eve/lock, rimi/lock, hagumi/ako, himari/kasumi) as well as the persona pairs.

**E1 outcome (2026-10-30, `e1/E1_RESULTS.md`).** A model now exists: E1b epoch 2.
- Japanese is done for all 49 voices.
- Mandarin accent is at about floor level for all of them; Mandarin identity is right for 34.
- English is mildly to clearly accented.
- E2 = in-voice Mandarin for the remaining 15 voices + English pseudo rows.

Selection follows `kurisu_v2/SELECTION_RULE.md`, extended per voice: a checkpoint ships only if no voice falls below its floor on sim or leakage, unless that voice is routed to a fallback.

---

## 6. Rights and publishing

**Rights-holder rules** [P, read 2026-10-29]:
- **Guideline item 10** (quoted in §0) covers characters' 「音声」 among the protected materials. Fan works are allowed only「個人により営利を目的としない場合に限り」(by individuals, non-commercially), and item 10 is excluded even then.
- **GBP posting rule, 2026-09-24:** explicitly names inputting voices into generative AI.
- **Bushiroad's own AI voicebank, 夢ノ結唱:** its terms forbid using its output to build speech models [P\*]. Do not use it for pseudo-data.

**Japanese law and industry, 2026** (these bear on the Kurisu release too, not only BanG Dream):
- **Ministry of Justice interpretive guidance** (2026-08-07, v1.1 2026-10-22). Voice is protected; naming the character supports identifying the performer; non-profit posting does not by itself make the use permissible [P\*].
- **Tokyo District Court, 2026-09-30:** first ruling that voice falls within publicity rights [P\*/S].
- **Nippairen 「音声に関する生成AI利用ガイドライン」, 2026-10-15** [S, several news reports, e.g. kumanichi.com/articles/2046861, prtimes.jp]:
  - written consent at each stage: training, generation and use;
  - 22 articles;
  - a voluntary standard with no legal force;
  - multilingual use needs new permission [P\*].
- **Correction to `kurisu_v2/research/final_report.md` §5 item 6:** 2024-10-15 is the launch of #NOMORE無断生成AI; the guideline itself is from 2026-10-15.

**Platform:**
- HF has no voice-specific policy; removal goes through DMCA [P\*].
- No BanG Dream voice repo has been taken down. Several, including some serving raw game audio, have been live since 2023 [P\*].
- That a repo has survived is not permission [I].

**If publishing is ever reconsidered** (not the current plan; the user chose local/private on 2026-10-29) [I]:
- Ship no game audio anywhere, including dataset, Space references and examples.
- Gate the weights.
- Label outputs as AI-generated and unofficial.
- Give a takedown contact.
- This lowers exposure, but item 10 still applies; only option (c), permission from Bushiroad, resolves it.

---

## 7. More data exists, with the same ceiling and the same rights problem

- **Scale:** Bestdori mirrors ~361k Japanese voice mp3s, about 375 h, or roughly 5–10 h per main character. They are the same 32 kHz LAME files as this pack, so quality cannot improve [P\*].
- **For quality:** using them would remove the data floor for every game character. Ave Mujica would still be small, because it joined the game later [P\*/I].
- **Against:** it is the same item-10 material, and the GBP terms ban reverse-engineering the client (Art. 6(17)) [P\*].
- This report does not recommend fetching it; it is your call.

---

## 8. Could not verify

1. Any LM-TTS result for 3–16 min per speaker inside a shared fine-tune, and leakage between similar voices in any such model.
2. Whether ~15 pseudo-Mandarin rows per voice keep native Mandarin (E1 decides).
3. Whether the litagin `char` embedding separates 豐川祥子 白/黒 better than the Qwen encoder does.
4. The full text of the MOJ report, the Tokyo judgment and the Nippairen guideline (summaries and news only).
5. Any BanG Dream cast or agency statement on AI voices (none found).
6. Where the "bangdreamai" pack came from.

---

## 9. Files

- `data_facts_2026-10-29.md`: data measurements, including §7 separability (raw and centred).
- `multispeaker_ft_2026-10-29.md`: options A–E, data floors, failure modes, precedents.
- `vc_and_data_2026-10-29.md`: VC candidates, mp3/restoration, anime cleaning, ASR, metrics.
- `community_and_legal_2026-10-29.md`: existing BanG Dream work, Bestdori provenance, rules and law.
- `scripts/`: `text_audit.py`, `audio_audit.py`, `floor_check.py`, `spk_sep.py`, `spk_centered.py`, plus outputs `spk_sep.json` and `spk_vecs.npz` (run the GPU ones with `kurisu_qwen3tts/finetune/.venv-tts`).
