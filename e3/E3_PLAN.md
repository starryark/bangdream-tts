# E3 plan: voice-converted Mandarin and English for the voices sampling cannot fix (pre-registered 2026-10-31)

Tags: [L] measured, [I] inference. Builds on `e2/E2_PLAN.md`.

## Why

E2's pseudo generation measured the problem directly on 320 Mandarin candidates per hard voice (`pseudo/report.json` on the E2 pod) [L].

Two failure patterns:
- **Own voice, accented:** Eve has 246 of 320 renders in her own voice, at lid −6.7, and only 3 that are both own-voice and native. Lock: 73 own-voice, lid −7.3.
- **Native, wrong voice:** Kasumi and Rimi's native renders land on other voices, mostly the `mujica_sakiko_tsuki` region (Kasumi 71, Rimi 132 of 320).

So no model we have produces native Mandarin in these voices. That is the case for voice conversion: keep a native render's phonetics and prosody, and replace only its timbre (`research/vc_and_data_2026-10-29.md` §2–4).

## Target voices

The voices that kept fewer than 10 gated Mandarin pseudo rows in E2 (10):

| Voice | Rows kept |
|---|---|
| ppp_kasumi | 2 |
| ppp_rimi | 0 |
| pp_eve | 0 |
| ras_lock | 0 |
| mygo_tomori | 0 |
| mygo_soyo_tsuki | 2 |
| hhw_kanon | 2 |
| hhw_hagumi | 5 |
| ro_ako_maou | 1 |
| ro_rinko | 1 |

Plus every voice that kept fewer than 6 gated English rows in E2.

## Method

- **Converter:** Seed-VC v1 zero-shot, the default speech model (`seed-uvit-whisper-small-wavenet`, 22.05 kHz, timbre only, no F0 conditioning), with 25 diffusion steps and cfg 0.7. It runs in its own venv. It is GPL-3.0 and used as a tool; outputs are never published.
- **Reference per voice:** that voice's reference clips (`bundle/refs/ref_<row>.txt`, the trainer's own list), concatenated in list order until about 25 s.
- **Sources** (no new TTS rendering): E2 candidates of the same language that pass the **accent** and **text** gates (any identity). Order of preference:
  1. the voice's own texts;
  2. if fewer than 60, accent-passing candidates of other target voices' texts.

  At most 60 per voice and language, the most native first (lowest lid).
- **Output:** resampled to 24 kHz, then the Kurisu v3 audio policy.
- **Gates:** identical to E2: identity (pred = row, p_true ≥ 0.5 × real p_true), accent, text, length. All are re-scored on the converted audio with `bdscore`.
- **Caps:** combined with the voice's E2 rows, at most 25 zh and 12 en per voice (E2 rows are kept; VC rows fill up to the cap).

## Feasibility gate (before any training)

On 20 Kasumi and 20 Eve Mandarin sources, the converted clips' gate pass rate must be ≥ 20% for at least one of the two voices, and the converted accent must stay native (median lid ≤ −19).

If not, E3 stops; it is reported and nothing is trained.

## Training and evaluation

- **Training:** E2's command and schedule (full SFT from Base, stop at 2.0 of a 3-epoch cosine). Data = E2's mix + the VC rows.
- **Evaluation:** unchanged.

**Selection:** the E2 rule, with the reference being the better of E1b epoch 2 and the E2 pick (if adopted).

**Adoption requires all of:**
- every guard, including ja leak ≤ 0.03 and no English regression against the reference;
- pooled zh leak ≤ 0.08;
- at least 43 of 49 voices with zh leak ≤ 0.15.

Otherwise the reference model stays.

**Reported regardless:**
- the VC gate pass rates per voice;
- zh leak and lid for the 10 target voices;
- English lid by group.

## Feasibility result (2026-10-31): FAILED; E3 stops, nothing trained [L]

40 Seed-VC v1 conversions (20 Kasumi, 20 Eve; sources = native E2 candidates; reference = the voice's own ~25 s Japanese reference audio):

| Voice | Assigned to own voice | Identity gate | zh lid, source → VC | shared-char SER, source → VC | CER (VC) | Pass all gates |
|---|---|---|---|---|---|---|
| ppp_kasumi | 20/20 | 70% | −28.2 → −11.6 | 0.062 → 0.175 | 0.02 | 0% |
| pp_eve | 20/20 | 50% | −23.9 → −8.1 | 0.072 → 0.210 | 0.01 | 0% |

- **What worked:** VC fixes identity completely; every clip lands on its own voice.
- **What failed:** it re-imposes a Japanese accent, and both accent metrics agree (lid and the human-validated SER), while staying intelligible.
- **Interpretation [I]:** Seed-VC's DiT is conditioned in context on the reference mel, so it takes on the reference's articulation along with its timbre. With only Japanese references available, zero-shot Seed-VC cannot give native Mandarin in these voices.

**Options not tried** (each needs a new decision):
- **(a)** RVC per voice with index ratio 0: content comes from HuBERT features of the source, with no reference prompt in the decoder.
- **(b)** Seed-VC fine-tuned per voice, which removes the need for a long in-context reference.
- **(c)** Accept the current trade-off for these ~10 voices.
