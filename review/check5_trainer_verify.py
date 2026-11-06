"""CPU-only checks of train_sft.py index arithmetic for a 49 + 55 speaker run. No model weights loaded."""
import json, math, random, shutil, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

POD = Path(r"E:\STEINS;GATEVoice\kurisu_v2\pod")
BASE = Path(r"C:\Users\lyang\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-Base\snapshots\fd4b254389122332181a7c3db7f27e918eec64e3")
sys.path.append(str(POD))
import train_sft as T  # noqa: E402

cfg_json = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
tc = SimpleNamespace(**{k: v for k, v in cfg_json["talker_config"].items() if k != "code_predictor_config"})
config = SimpleNamespace(talker_config=tc, tts_pad_token_id=cfg_json["tts_pad_token_id"],
                         tts_bos_token_id=cfg_json["tts_bos_token_id"], tts_eos_token_id=cfg_json["tts_eos_token_id"])
IM_START, ASSIST, NL, IM_END = 151644, 77091, 198, 151645


def processor(text, return_tensors, padding):
    body = text.split("assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    ids = [IM_START, ASSIST, NL] + [1000 + (ord(c) % 500) for c in body] + [IM_END, NL, IM_START, ASSIST, NL]
    return {"input_ids": torch.tensor([ids])}


shipped = [f"band{b}_member{m}" for b in range(10) for m in range(5)][:49]
aux = [f"zh_ssb{i:04d}" for i in range(30)] + [f"en_{i}" for i in range(25)]
specs_names = shipped + aux
spk_index = {n: i for i, n in enumerate(specs_names)}
print(f"[1] speakers: {len(shipped)} shipped + {len(aux)} aux = {len(spk_index)} table rows; "
      f"codec rows used 3000..{3000 + len(shipped) - 1} of vocab {tc.vocab_size} "
      f"(capacity {tc.vocab_size - 3000})")
special = {"pad": tc.codec_pad_id, "bos": tc.codec_bos_id, "eos": tc.codec_eos_token_id, "think": tc.codec_think_id,
           "nothink": tc.codec_nothink_id, "think_bos": tc.codec_think_bos_id, "think_eos": tc.codec_think_eos_id,
           **{f"lang_{k}": v for k, v in tc.codec_language_id.items()}}
rows_used = set(range(3000, 3000 + len(shipped)))
print("    collision with special codec ids:", sorted(k for k, v in special.items() if v in rows_used) or "none",
      f"(special ids span {min(special.values())}..{max(special.values())})")
supp = [i for i in range(tc.vocab_size - 1024, tc.vocab_size) if i != tc.codec_eos_token_id]
print(f"    generate() suppresses ids {supp[0]}..{supp[-1]} except eos -> speaker rows never sampled: "
      f"{rows_used <= set(supp)}")

# ---- dataset + collate
rng = random.Random(0)
rows = []
for k in range(60):
    spk = rng.choice(specs_names)
    lang = "japanese" if spk in shipped else ("chinese" if spk.startswith("zh_") else "english")
    n = rng.randint(3, 20)
    rows.append({"audio": f"a{k}.wav", "text": "x" * rng.randint(2, 12), "language": lang, "speaker": spk.upper() if k == 0 else spk,
                 "audio_codes": np.random.randint(0, 2048, size=(n, 16)).tolist()})
ds = T.CodedDataset(rows, processor, config, 400, spk_index=spk_index, default_speaker="kurisu", p_auto=0.0)
assert all(ds.items[i]["spk"] == spk_index[rows[i]["speaker"].lower()] for i in range(len(rows)))
print("[2] spk_index lookup matches every row (incl. an UPPER-case row speaker):", True)
try:
    T.CodedDataset([{**rows[1], "speaker": "ppp_kasumii"}], processor, config, 400, spk_index=spk_index, default_speaker="kurisu")
except ValueError as e:
    print("    missing speaker ->", type(e).__name__, str(e)[:60])
try:
    T.CodedDataset([{k: v for k, v in rows[1].items() if k != "speaker"}], processor, config, 400, spk_index=spk_index, default_speaker="kurisu")
except ValueError as e:
    print("    row without speaker field + default 'kurisu' ->", type(e).__name__, str(e)[:60])
trained = {specs_names[it["spk"]] for it in ds.items}
untrained = [n for n in shipped if n not in trained]
print(f"    shipped speakers with ZERO train rows accepted silently: {len(untrained)} (e.g. {untrained[:3]})")


def reference_layout(lang, text_ids, codes):
    """Symbolic copy of Qwen3TTSForConditionalGeneration.generate (non-streaming, custom voice) + teacher-forced frames."""
    if lang == "auto":
        pre = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
    else:
        pre = [tc.codec_think_id, tc.codec_think_bos_id, tc.codec_language_id[lang], tc.codec_think_eos_id]
    codec = pre + ["SPK"] + [tc.codec_pad_id, tc.codec_bos_id]
    seq = [(t, None) for t in text_ids[:3]]
    L = len(codec)
    seq += [(config.tts_pad_token_id, c) for c in codec[:L - 2]] + [(config.tts_bos_token_id, codec[L - 2])]
    body = text_ids[3:]
    seq += [(t, tc.codec_pad_id) for t in body] + [(config.tts_eos_token_id, tc.codec_pad_id)]
    seq += [(config.tts_pad_token_id, tc.codec_bos_id)]
    seq += [(config.tts_pad_token_id, int(f[0])) for f in codes]
    labels = [None] * len(seq)  # label at position t = what logits at t-1 must predict
    first = len(seq) - len(codes)
    for j, f in enumerate(codes):
        labels[first + j] = int(f[0])
    labels.append(tc.codec_eos_token_id)
    seq.append((config.tts_pad_token_id, tc.codec_eos_token_id))  # eos position (input never seen at inference)
    return seq, labels


def check(batch, items, langs):
    ok = True
    for i, (it, lang) in enumerate(zip(items, langs)):
        seq, labels = reference_layout(lang, it["text_ids"][0].tolist(), it["codes"])
        n = len(seq)
        tmask = batch["text_embedding_mask"][i, :, 0]
        cmask = batch["codec_embedding_mask"][i, :, 0]
        for t, (txt, cod) in enumerate(seq):
            got_t = int(batch["input_ids"][i, t, 0]) if tmask[t] else None
            if cod == "SPK":
                got_c = "SPK" if (int(batch["spk_pos"][i]) == t and not cmask[t]) else ("BAD", int(batch["input_ids"][i, t, 1]))
            else:
                got_c = int(batch["input_ids"][i, t, 1]) if cmask[t] else None
            if got_t != txt or got_c != cod:
                ok = False
                print("   MISMATCH", lang, t, (got_t, got_c), (txt, cod))
                break
            lab = int(batch["codec_0_labels"][i, t])
            if (labels[t] if labels[t] is not None else -100) != lab:
                ok = False
                print("   LABEL MISMATCH", lang, t, lab, labels[t])
                break
        if bool(batch["attention_mask"][i, n:].any()) or bool(tmask[n:].any()) or bool(cmask[n:].any()):
            ok = False
            print("   padding leak", lang)
        # sub-talker: hidden[t-1] for frame at t
        cm = batch["codec_mask"][i]
        fr = torch.nonzero(cm).flatten().tolist()
        if fr != list(range(n - 1 - len(it["codes"]), n - 1)):
            ok = False
            print("   codec_mask wrong", lang)
    return ok


idx = list(range(8))
items = [ds.items[j] for j in idx]
b = ds.collate([ds[j] for j in idx], train=False)
print("[3] explicit-language layout == generate() non-streaming:", check(b, items, [it["language"] for it in items]))
ds.p_auto = 1.0
b = ds.collate([ds[j] for j in idx], train=True)
print("    Auto layout (p_auto forced 1.0) == generate() Auto:", check(b, items, ["auto"] * len(items)))
print("    spk_pos explicit/auto:", sorted({int(x) for x in b["spk_pos"]}), "; spk_idx carried:", b["spk_idx"].tolist()[:8])
ds.p_auto = 0.25
ds.rng = random.Random(1)
n_auto = 0
for _ in range(400):
    bb = ds.collate([ds[j] for j in idx], train=True)
    n_auto += int((bb["spk_pos"] == 6).sum())
print(f"    p_auto 0.25 -> observed Auto share {n_auto / (400 * len(idx)):.3f}")

# ---- schedule
print("[4] schedule (micro 16, accum 2, epochs 2, warmup_ratio 0.03, min_lr_ratio 0.1, lr 5e-6)")
for N in (10970, 11000, 11400, 11700, 12000):
    lengths = [random.Random(k).randint(30, 300) for k in range(N)]
    batches = T.length_batches(lengths, 16, 20260821, True)
    spe = max(1, math.ceil(len(batches) / 2))
    total = spe * 2
    warm = max(1, int(total * 0.03))

    def lr_at(s):
        if s < warm:
            return 5e-6 * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 5e-6 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
    stop = max(1, round(1.0 * spe))
    lrs = [lr_at(s) for s in range(stop)]
    print(f"    N={N}: {len(batches)} micro-batches, {spe} steps/epoch, total {total}, warmup {warm}, stop {stop}; "
          f"lr at last step {lrs[-1]:.3e} ({lrs[-1] / 5e-6:.1%} of peak), mean lr {np.mean(lrs):.3e}")

# ---- speaker_table cache staleness
tmp = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
cache = tmp / "spk_embedding.pt"
torch.save({"names": ["a", "b"], "table": torch.ones(2, 4)}, cache)
out = T.speaker_table(None, [("a", [Path("NEW_REFS_a.wav")]), ("b", [Path("NEW_REFS_b.wav")])], "cpu", torch.float32, cache)
print("[5] cache with same names but different ref lists -> silently reused:", bool((out == 1).all()))
try:
    T.speaker_table(None, [("b", []), ("a", [])], "cpu", torch.float32, cache)
except SystemExit as e:
    print("    reordered names -> SystemExit:", str(e)[:50])

# ---- export
base = tmp / "base"
base.mkdir()
shutil.copy(BASE / "config.json", base / "config.json")
names_mixed = shipped
sd = {"talker.model.codec_embedding.weight": torch.zeros(3072, 4), "speaker_encoder.x": torch.zeros(2)}
table = torch.arange(len(specs_names), dtype=torch.float32).unsqueeze(1).repeat(1, 4)
T.save_checkpoint(tmp / "ck", base, sd, table, names_mixed)
out_cfg = json.loads((tmp / "ck" / "config.json").read_text(encoding="utf-8"))["talker_config"]
from safetensors.torch import load_file
w = load_file(str(tmp / "ck" / "model.safetensors"))["talker.model.codec_embedding.weight"]
rowvals = [int(w[3000 + i, 0]) for i in range(len(shipped))]
print(f"[6] exported spk_id: {len(out_cfg['spk_id'])} entries, "
      f"{min(out_cfg['spk_id'].values())}..{max(out_cfg['spk_id'].values())}; aux names exported: "
      f"{[n for n in aux if n in out_cfg['spk_id']]}; rows hold table[0..48]: {rowvals == list(range(49))}; "
      f"row 3049 untouched: {float(w[3049].abs().sum()) == 0.0}; speaker_encoder dropped: "
      f"{'speaker_encoder.x' not in load_file(str(tmp / 'ck' / 'model.safetensors'))}")
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
C2 = Qwen3TTSConfig.from_pretrained(str(tmp / "ck"))
sup = {str(x).lower() for x in C2.talker_config.spk_id.keys()}
for q in ("band3_member2", "Band3_Member2", "BAND3_MEMBER2", "band3-member2"):
    ok_val = q.lower() in sup
    ok_gen = q.lower() in C2.talker_config.spk_id and C2.talker_config.spk_is_dialect[q.lower()] is False
    print(f"    generate_custom_voice(speaker={q!r}): validate {ok_val}, generate lookup {ok_gen}")
print("    tts_model_type:", C2.tts_model_type)

try:
    T.save_checkpoint(tmp / "ck2", base, {"talker.model.codec_embedding.weight": torch.zeros(3072, 4)},
                      torch.zeros(80, 4), [f"s{i}" for i in range(73)])
except IndexError as e:
    print("[7] 73 saved speakers -> IndexError at SAVE time (after training):", str(e)[:70])
shutil.rmtree(tmp, ignore_errors=True)
