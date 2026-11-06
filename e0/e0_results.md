# E0 pre-flight results (local RTX 5060 Ti, 2026-10-30)

Tags: [L] measured locally in this run; [I] inference.

Pipeline: `bangdream/pipeline/` (`run_e0.sh` runs b02 → b05; `b06_report.py` builds the tables; `b07_resynth_check.py` is the probe control). Full per-voice table: `e0/floor_table.md`.

## CORRECTION (2026-10-30, after code review): identity numbers re-measured with a codec-calibrated probe

The independent review (`review/e0_code_review.md`, finding C1) showed that the first probe (b03) was trained on real recordings only. The Qwen codec shifts every clip's speaker vector by a largely shared amount, towards `mujica_sakiko_tsuki`. Real clips survive that shift; low-margin renders do not.

b08 (`pipeline/b08_codec_probe.py`) refits the probe on codec-resynthesised train clips and takes its baseline from resynthesised held-out clips. **These numbers supersede every identity figure below** (leak, p_true, the "collapsed voices" list, the 145-clip `sakiko_tsuki` sink):

| | Japanese leak (95% CI, bootstrap over voices) | probe p_true |
|---|---|---|
| Real held-out, through the codec (baseline) | **0.012** (0.002–0.023) | 0.777 |
| Base floor, Japanese | **0.075** (0.029–0.130) | 0.581 |
| Base floor, Mandarin / English (no real baseline) | 0.154 / 0.094 | 0.414 / 0.481 |

Probe 5-fold CV on resynthesised train clips: 98.3%.

**Clear zero-shot identity failures (Japanese):**

| Voice | Japanese leak | p_true (real baseline) | Renders land on |
|---|---|---|---|
| ppp_kasumi | 8/10 | 0.12 (0.57) | sakiko_tsuki |
| pp_eve | 5/10 | 0.19 (0.78) | pareo |
| ras_lock | 5/10 | 0.16 (0.76) | moca |

**Mild** (about 2 of 6–10): mujica_sakiko, mygo_soyo_tsuki, hhw_hagumi, ppp_rimi, ro_rinko.

**Everything else** is 0–1 of 10. Even so, mean p_true sits well below the real clips (0.58 vs 0.78).

**Conclusion, revised [I]:**
- Identity is still the gap the fine-tune must close: leak 7.5% vs 1.2%, p_true 0.58 vs 0.78.
- "Collapse" applies to three voices, not ten. The leak destinations are now spread out rather than one sink.
- The claim that floor identity worsens with more data is **withdrawn**: ρ = 0.10 (Japanese), not significant (review M1).
- Mandarin and English results are unchanged; they were not measured with the probe.

**Other review fixes applied:**
- Latin-script band names are now read correctly in scoring (m1).
- b07 no longer copies stale fields (m2).
- b06 buckets use unpadded source minutes (m6).
- The misassigned count is 279, not 287 (m8).
- Five duplicate groups whose transcripts differ in words (m5) are left as known noise (5 of 9,379 rows).

## Headline (first pass; identity figures superseded by the correction above) [L]

1. **The Base zero-shot floor's Mandarin from every voice's locked vector sits between Kurisu v3 and v2, much closer to v3, and its English is clean:**
   - zh shared-char SER 0.077 (Kurisu: native anchor 0.048, v2 0.170, v3 0.060);
   - VoxLingua lid −19.4 (native −22.6, v3 −19.6);
   - zh CER 0.025;
   - en WER 0.001.

   That the Mandarin is "near-native" is an inference from where these numbers fall on the one-rater Kurisu scale [I]; listen before relying on it. The fine-tune must **keep** this rather than create it.
2. **Identity is the real problem, and it depends on voice type, not on minutes of data.**
   - **Collapsed voices.** Bright, high-pitched game voices clone poorly from the x-vector. The share of floor renders the probe assigns to another character (one seed; 95% Wilson CI):

     | Voice | Japanese (has a real baseline: 1%) | All languages |
     |---|---|---|
     | ppp_kasumi | 10/10 (0.72–1.00) | 30/34 (0.73–0.95) |
     | ras_lock | 10/10 (0.72–1.00) | 26/34 (0.60–0.88) |
     | hhw_hagumi | 7/8 (0.53–0.98) | 14/32 (0.28–0.61) |
     | ppp_rimi | 6/8 (0.41–0.93) | 27/32 (0.68–0.93) |
     | ras_pareo | 6/8 (0.41–0.93) | 8/32 (0.13–0.42) |
     | hhw_kokoro | 6/10 (0.31–0.83) | 17/34 (0.34–0.66) |
     | pp_eve | 5/10 (0.24–0.76) | 25/34 (0.57–0.85) |
     | ro_rinko, hhw_kanon, ag_himari | 3/10 each (0.11–0.60) | 21, 18, 12 of 34 |

     - **Japanese, pooled:** 74/414 = 17.9% (CI 14–22%), against 1% for real clips.
     - **Per voice:** with 4–10 Japanese clips per voice, the intervals are wide. Kasumi and Lock are certain collapses; the rest need the larger E1 probe set.
     - **Mandarin and English:** zh/en leak has no real-clip baseline. Part of it is language drift, so rinko, kanon and eve's high all-language numbers come mostly from zh/en.

   - **Voices that clone well.** The Ave Mujica voices and the lower, mature ones clone well: leak 0, probe p_true 0.65–0.82 against 0.6–0.9 for real clips.
   - **By bucket, floor identity gets worse as data grows:** leak 9% (< 10 min) → 21% (30–60 min). That is because the bright voices happen to be the well-recorded game characters.
   - **Consequence:** the fine-tune has the most work exactly where it has the most data. The < 10 min "risk bucket" from the research report starts from the best floor.
3. **The leakage is not a probe artefact.**
   - Real held-out clips passed through the Qwen codec keep 98.3% correct identity (leak 1.0% → 1.7%; p_true 0.81 → 0.71).
   - The floor's 18% leak is therefore real voice collapse.
   - The collapsed renders converge on one region the probe labels `mujica_sakiko_tsuki` (145 of 287 misassigned floor clips). That is probably the Base model's own default young-female voice [I].

## Status

| Step | State |
|---|---|
| b01 build (24 kHz, dedup, Kurisu audio policy) | done: 9,804 clips, 22.1 h, 49 rows, 46 duplicates dropped |
| b02 transcript check (Qwen3-ASR-1.7B + anime-whisper, kana CER) | done |
| b03 split, reference lists, locked vectors, probe | done |
| b04 floor renders (Base, x-vector only, the trainer's locked vector) | done: 49 rows, 1,590 clips. Claude Code stopped the wrapper script at 02:16 (low system memory); the render process finished on its own |
| b05 scoring | done (run 08:45 at the user's "check") |
| b07 codec-resynthesis control for the probe | done |

**Shared GPU.** Another job (`E:\New STEINS DataSet` curation ASR) shared the GPU throughout. Two CUDA processes on Windows WDDM slowed both, so ASR took 1 h 54 min.

**Bug fixed during the run.** In b01's dedup, byte-identical clips inside one set share an id and path, and the dedup step deleted the kept clip's file (46 files). b01 was fixed and re-run; no files are missing now.

## Transcripts (b02) [L]

- **Only 11 of 9,804 clips (0.1%) fail both ASR systems** against the filename text at kana level. The mean best-of-two kana CER is 0.011. **The filename transcripts are reliable.**
- Most of the 11 are non-verbal or hard to read rather than wrong:
  - sobbing and laughter (ぐすっ, ふえ〜ん, うふふ, ハァ, ふふん);
  - Chu²'s "Unstoppable";
  - 豊川 heard as 十河.
- One looks genuinely mismatched: `mygo_tomori_d788f622e9` (絆創膏/楽屋 vs パソコン/学園). Drop it.
- **Qwen3-ASR echoed its proper-noun context on 34 clips**, e.g. 「とがわサキコ」 → 「戸山香澄」. Kurisu saw 163 such echoes. Do not give Qwen3-ASR a name list for scoring; b05 scores without context.
- The BanG Dream readings table (`bdcommon.BD_READINGS`) is required. OpenJTalk misreads about 25 cast names (有咲 → ユーサキ, 燈 → アカリ, はぐみ → ワグミ, 祥子 → サチコ).

## Speaker vectors and leakage probe (b03) [L]

- **Reference lists:** `refs/ref_<row>.txt` holds up to 64 train clips per row, ≥ 2 s and with an ok transcript, with the 20% least typical (by centred cosine) removed.
  - The smallest are mujica_mutsumi (25 clips, 105 s), mujica_umiri (35, 104 s) and ro_ako_maou (30, 251 s).
  - Locked vectors are computed by `train_sft.speaker_embedding` itself and saved in `vectors/locked.pt`.
- **Locked vectors, centred:** median cosine between rows −0.029, p95 0.571, max 0.822.
- **Closest pairs:**

  | Pair | Centred cosine |
  |---|---|
  | pp_eve ~ ras_lock | 0.822 |
  | mujica_sakiko ~ mujica_sakiko_tsuki | 0.801 |
  | mujica_uika ~ mujica_umiri | 0.799 |
  | hhw_hagumi ~ ro_ako | 0.769 |
  | ag_himari ~ ppp_kasumi | 0.755 |

- **Leakage probe** (logistic regression on centred Qwen speaker-encoder clip vectors, 49 classes):
  - Real held-out clips: 99.0% accuracy.
  - 5-fold CV on train clips: **98.7%**.
  - Almost all real-clip errors are between personas of one character:

    | Row | CV accuracy |
    |---|---|
    | mujica_sakiko ↔ sakiko_tsuki | 89–91% |
    | mygo_soyo ↔ soyo_tsuki | 92–96% |
    | ro_ako ↔ ro_ako_maou | 92–96% |
    | mujica_mortis ↔ mutsumi | 93–97% |
    | ras_pareo → ag_tsugumi | 96% |

  - This is the real-clip baseline that rendered clips must be compared against. It replaces WavLM-SV, which separates these voices on only 56% of clips.
- **Mujica "dominant cluster vs all clips":** the vector built from the most typical 80% moves each Mujica row by at most ±0.03 against its nearest Mujica neighbour.
  - uika ~ umiri: 0.791 → 0.799.
  - sakiko ~ tsuki: 0.775 → 0.803.
  - mutsumi ~ sakiko: 0.205 → 0.152.
  - **Clustering does not separate them** [L]. Whatever the Mujica sets share is spread across all their clips, not carried by a few outliers. It is either genuine voice similarity or a common anime channel/mix [I].
  - Only a source-separation pass could tell those two apart. It is not worth doing before E1: the probe still tells real Mujica clips apart at ≥ 93%.

## Smoke test of the full chain (3 rows, 151 clips; `bangdream/_smoke/`) [L]

mujica_umiri has only 3.1 min of data. Its Base zero-shot floor from its locked vector:

| Language | CER / WER | Leakage | Accent | Sim (centred) |
|---|---|---|---|---|
| Japanese | 0.032 | 0 | — | 0.56 (real held-out clips: 0.71) |
| Mandarin | 0.021 | 0 | VoxLingua lid_ratio −20.5; **shared-char SER 0.045** | — |
| English | 0.000 (WER) | 0 | — | — |

The Kurisu native-Mandarin anchor was SER 0.048 and M1 −22.6. The full run confirms this for all 49 voices (see Headline).

## Listening check (≈ 3 min)

| What | Files |
|---|---|
| Collapsed voice | `e0/floor/ppp_kasumi/ja_*.wav` (renders) vs `data/wav24/ppp_kasumi/` (real) |
| Well-cloned low-data voice, for contrast | `e0/floor/mujica_umiri/ja_*.wav` vs `data/wav24/mujica_umiri/` |
| Mandarin floor, accent judgement | `e0/floor/<row>/zh3_shared_*.wav` |

All floor numbers come from **one sampling seed**.

## Floor, all 49 voices (b04/b05; one seed; 414 ja + 784 zh + 392 en renders) [L]

| Bucket | Rows | Floor leak (all languages) | Floor ja CER | zh SER | zh lid | en WER |
|---|---|---|---|---|---|---|
| < 10 min | 9 | 0.09 | 0.029 | 0.063 | −21.4 | 0.002 |
| 10–20 min | 5 | 0.15 | 0.054 | 0.075 | −19.2 | 0.000 |
| 20–30 min | 13 | 0.18 | 0.032 | 0.082 | −18.8 | 0.000 |
| 30–60 min | 22 | 0.21 | 0.029 | 0.080 | −19.0 | 0.001 |
| **All** | 49 | **ja 0.18 vs real ja 0.01** (probe p_true 0.44 vs 0.81); zh 0.19 / en 0.14 have **no real-clip baseline** (the same voice scores lower across languages, so part of it is language drift, not collapse) | 0.031 (real 0.023) | 0.077 | −19.4 | 0.001 |

Notes:
- Bucket minutes here include the 0.37 s of padding per clip from b01, so a few rows move up a bucket against `research/data_facts` (10/5/16/19 there).
- **Centred sim to the locked vector is biased.** Renders are generated *from* that vector, so for some rows they score above real clips (mujica_sakiko 0.62 vs real 0.51). Use the probe (p_true, leak) as the identity metric; use sim only within one row and one condition.
- **Mandarin outliers.**
  - The highest zh SERs (ag_ran 0.142, mo_rui 0.142, ro_sayo 0.127, hhw_misaki 0.122, ppp_tae 0.120, hhw_kaoru 0.116) all include one runaway clip.
  - 6 of 784 zh renders (0.8%) hit the 400-frame cap, 5 of them on the same sentence (`zh3_shared_11`, 「物理学的基本定律在任何地方都同样成立。」). Keep it as a stability probe.
  - The most Japanese-sounding Mandarin by lid: ras_lock −11.1, hhw_hagumi −12.2, pp_eve −13.9, mo_mashiro −14.2. These are also collapsed or weak-identity voices.
- **Japanese.** Floor CER 0.031 vs real 0.023. Outliers: mygo_anon 0.173, ag_moca 0.080, mo_nanami 0.077.

## What this changes in the E1 plan [I]

1. **The success criterion per voice is identity.**
   - Headline metric: **Japanese** probe leak and p_true against the real-clip baseline.
   - zh/en leak has no real baseline; compare it with the floor and between E1 arms only.
   - Mandarin SER and lid only need to stay at the floor (≤ ~0.08 / ≤ −19), not improve.
2. **Pseudo-Mandarin from Base must be gated by the probe.** E1 also needs ~20 shared novel Japanese texts per voice (×2 seeds) so per-voice Japanese leak has usable intervals.
   - For the collapsed voices, Base renders carry the generic default voice. Training a voice's row on them would teach "this character speaks Mandarin in someone else's voice".
   - Keep a pseudo-zh sample only if the probe assigns it to its own row (p_true above that row's real-clip 25th percentile). This replaces Kurisu p11's ECAPA-similarity gate.
   - **Expect few or no passing samples for kasumi, rimi, lock, eve, rinko, kanon and kokoro.** For them, generate pseudo-zh in a second round from the round-1 fine-tune, once their Japanese identity is learned.
   - Round-2 samples must pass **both** gates: the identity probe, and the accent gate at the floor values (lid ≤ −19, shared-char SER ≤ ~0.08). A round-1 model trained on Japanese-only rows for those voices may produce accented Mandarin, as Kurisu v2 did (SER 0.170), and ungated samples would teach that accent.
   - Report the pass rate per voice before training, so a voice with 0 passing rows is visible. Fall back to VC pseudo-data only if both rounds fail.
3. **The training-time leakage risk is among the bright game voices, not the Mujica ones.**
   - The closest locked-vector pairs are pp_eve ~ ras_lock 0.822, hhw_hagumi ~ ro_ako 0.769, ag_himari ~ ppp_kasumi 0.755 and ppp_rimi ~ ras_lock 0.733. These are the same voices that the Base maps onto one output today.
   - So the shared SFT has to separate near-identical vectors that are currently collapsed.
   - **Named pairs for the E1 leakage check:** eve/lock, rimi/lock, hagumi/ako, himari/kasumi, plus the persona pairs (sakiko/sakiko_tsuki, soyo/soyo_tsuki, ako/ako_maou, mutsumi/mortis) and uika/umiri.
   - The Mujica voices already clone at real-clip level. Check only that fine-tuning does not pull them toward the game voices.
4. **The dose ablation (E1) stays as planned**, but count only gated pseudo rows, and report the gate pass rate per voice.

