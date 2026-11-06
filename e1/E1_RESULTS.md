# E1 results: first 49-voice fine-tune (2026-10-30)

**Plan and amendments:** `E1_PLAN.md` (selection rule fixed before training; Amendment A and its clarification written before any E1b score).

**Raw data:** `results/` (scored renders, summaries, selection, logs, train logs, mix and pseudo reports). Analysis scripts: `e1_select.py`, `e1_did.py`.

**Compute:** one Secure H100, about 5 h, about $20.
- The pod was terminated after the picked checkpoint was copied home; its sha256 was verified.
- Nothing was published. The checkpoint is local only.

Tags: [L] measured on the pod / locally, [I] inference.

## Pick

**`E1b` epoch 2** (`runs/E1b/checkpoint-step-710`), now at `bangdream/models/E1b_ep2/` (sha256 `27652e20…`).
- Usage: `Qwen3TTSModel.from_pretrained(path)`, then `generate_custom_voice(speaker=<row id>, language="Japanese"|"Chinese"|"English")`.
- Row ids are in `pipeline/rows.json`.
- Listening samples (8 voices × ja/zh/en): `bangdream/samples/E1b_ep2/`.

## Results (pre-registered table) [L]

4,356 renders per condition: 49 voices × (held-out + 12 novel Japanese, 16 zh, 8 en) × 2 seeds. The probe is codec-calibrated.

| Condition | ja leak (95% CI, over voices) | ja p_true | ja CER (held-out) | zh SER | zh lid | zh leak | en WER | Eligible |
|---|---|---|---|---|---|---|---|---|
| Real clips through the codec | 0.012 | 0.777 | 0.023 | — | — | — | — | |
| Floor (Base zero-shot, same speaker rows) | 0.100 (0.052–0.154) | 0.567 | 0.032 | 0.077 | −19.5 | 0.157 | 0.001 | |
| E1a epoch 1 | **0.009** (0.003–0.018) | 0.810 | 0.027 | 0.177 | −12.6 | 0.008 | 0.006 | no |
| E1a epoch 2 | 0.014 | 0.816 | 0.028 | 0.200 | −11.4 | 0.011 | 0.006 | no |
| E1a epoch 3 | 0.013 | 0.807 | 0.032 | 0.215 | −11.1 | 0.008 | 0.008 | no |
| E1b epoch 1 | 0.014 | 0.796 | 0.028 | **0.069** | −19.9 | 0.174 | 0.002 | yes |
| **E1b epoch 2 (pick)** | 0.014 (0.005–0.025) | 0.802 | 0.028 | 0.069 | −18.6 | 0.155 | 0.003 | yes |
| E1b epoch 3 | 0.015 | 0.803 | 0.028 | 0.072 | −19.1 | 0.189 | 0.005 | yes |

Kurisu references: native Mandarin anchor SER 0.048, lid −22.6; Kurisu v3 0.060 / −19.6; Kurisu v2 0.170.

## What we learned

1. **The shared fine-tune fixes Japanese identity for all 49 voices.**
   - Leak goes from 10% (floor) to about 1%, the real-clip level, and p_true reaches real-clip level.
   - The named game-voice pairs (eve/lock, rimi/lock, hagumi/ako, himari/kasumi, uika/umiri) show 0–1 confusions per checkpoint (of about 100 renders). Only the persona pairs remain (Sakiko white/black 5–11, Ako/demon princess 1–6, Soyo 1–7 of about 100 renders), and nothing leaks from Mujica voices to game voices.
   - One epoch is enough; more epochs do not help identity [L].
2. **Japanese-only rows plus language id and 14% replay (E1a) give a Japanese accent in Mandarin and English.** This is the Kurisu v2 failure.
   - The accent is universal: 43 of 49 voices are worse than the floor by more than 0.05 SER.
   - It grows with epochs: SER 0.177 → 0.200 → 0.215.
   - It is unrelated to minutes of data (ρ = 0.09), and worst for the voices the fine-tune moved most (lock, kokoro, eve, rimi).
3. **About 5% gated in-voice pseudo-Mandarin (E1b) brings every voice's Mandarin accent to about the floor, including the 12 voices that had none.** The 0-row group lands slightly worse than the floor (SER 0.093 vs 0.082, lid −15.1 vs −17.0). Difference-in-differences, `results/did_amendA.json`, epoch 2:

   | Group | zh SER, E1a → E1b (floor) | zh lid | zh leak (identity) |
   |---|---|---|---|
   | ≥ 15 pseudo rows (18 voices) | 0.180 → **0.063** (0.081) | −14.2 → −20.4 (−20.9) | 0.002 → **0.050** (0.014) |
   | 0 pseudo rows (12 voices) | 0.266 → **0.093** (0.082) | −6.3 → −15.1 (−17.0) | 0.016 → **0.404** (0.544) |

   - **The accent fix generalises across speakers.**
   - **Identity in Mandarin does not generalise.** Voices without in-voice Mandarin fall back to the Base's zero-shot timbre when speaking Mandarin. For these voices (kasumi, rimi, eve, lock, rinko, kokoro, kanon, hagumi, himari, aya, tomori, soyo_tsuki) that timbre was already generic.
   - **Measured by pseudo-row count (E1b epoch 2, zh leak):**

     | Pseudo rows | Voices | zh leak |
     |---|---|---|
     | 0 | 12 | 0.404 |
     | 1–9 | 14 | 0.125 |
     | 10–14 | 5 | 0.019 |
     | ≥ 15 | 18 | 0.050 |

     **34 of 49 voices have own-voice Mandarin (zh leak ≤ 0.15; 31 at ≤ 0.10).** The 15 that do not are the 12 zero-row voices plus ro_ako_maou (0.69), mujica_sakiko_tsuki (0.22) and ras_masking (0.19).
   - **Japanese identity is unaffected in both groups.**
4. **English is only partly fixed.**
   - en lid: −11.3 for the ≥ 15 group (floor −13.5), a mild accent; −6.6 for the 0-row group (floor −10.3), a clear Japanese accent.
   - There were no English pseudo rows. WER is fine (0.003).
   - The plan had no English accent guard, so this passes the rule but is a known weakness [L].
5. **Gap in the pre-registered rule.** It does not guard Mandarin identity (zh leak), because zh had no real-clip baseline. The pick (E1b epoch 2) has zh leak 0.155 overall: fine for 34 voices, wrong for 15. Read the pick as:
   - Japanese: done for all 49 voices;
   - Mandarin: about floor-level accent for all, own voice for 34;
   - English: intelligible; mildly accented for well-cloned voices, clearly accented for the 12 zero-row voices.
6. **Epoch choice.** Epochs 1 and 2 tie on Japanese leak; the p_true tie-break (0.802 vs 0.796) is within noise. Epoch 1 has a slightly better Mandarin lid (−19.9 vs −18.6), epoch 2 slightly better Mandarin identity (0.155 vs 0.174). Either is fine.

## Next step (E2), in order of expected value [I]

1. **In-voice Mandarin for the 15 voices without own-voice Mandarin** (the 12 zero-row voices, plus ro_ako_maou, mujica_sakiko_tsuki and ras_masking).
   - Generate round-2 pseudo-zh from the E1b checkpoint. Its Mandarin is native for every voice, so the accent gate will pass. Keep only renders the probe assigns to their own voice: 60% of the 12 zero-row voices' E1b Mandarin renders already are (kasumi and rimi only 25%).
   - If that still yields too little, use a VC route: convert native Mandarin into each voice with a converter trained on its Japanese clips (`research/vc_and_data_2026-10-29.md`).
   - Then retrain with ≥ 15 rows per voice for all 49.
2. **English pseudo rows**, the same way (Base or E1b English, gated on accent and identity), to bring en lid to the floor.
3. **Add guards to the selection rule:** Mandarin identity (zh leak per voice ≤ floor + 0.1) and English accent (en lid ≤ floor + 2). Fix them before E2 runs.
4. **Persona pairs** (Sakiko, Ako, Soyo). If they matter to you, give each persona more distinct reference clips, or merge the pair.
