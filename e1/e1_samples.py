"""Local check of the picked E1 checkpoint + listening samples (ja / zh / en) for a few voices.

  python -I e1_samples.py        (writes bangdream/samples/E1b_ep2/<voice>_<lang>.wav)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

CKPT = Path(r"E:\STEINS;GATEVoice\bangdream\models\E1b_ep2")
OUT = Path(r"E:\STEINS;GATEVoice\bangdream\samples\E1b_ep2")
VOICES = ["ppp_kasumi", "ras_lock", "pp_eve", "ro_yukina", "mujica_sakiko", "mujica_mortis", "mygo_tomori", "hhw_kokoro"]
TEXTS = {"Japanese": "今日の練習、思ったより早く終わったね。このあと少しだけ寄り道していかない？",
         "Chinese": "今天的练习比想象中结束得早。我们要不要顺便去附近逛一逛？",
         "English": "Practice ended earlier than I expected today. Want to take a little detour on the way home?"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    m = Qwen3TTSModel.from_pretrained(str(CKPT), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    for lang, text in TEXTS.items():
        torch.manual_seed(7)
        wavs, sr = m.generate_custom_voice(text=[text] * len(VOICES), speaker=VOICES, language=[lang] * len(VOICES),
                                           max_new_tokens=400, non_streaming_mode=True)
        for v, w in zip(VOICES, wavs):
            p = OUT / f"{v}_{lang[:2].lower()}.wav"
            sf.write(str(p), np.asarray(w, dtype=np.float32), sr)
            print(f"{p.name}: {len(w) / sr:.1f} s")


if __name__ == "__main__":
    main()
