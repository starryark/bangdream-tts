"""BanG Dream! character voices (unofficial fan demo) on Qwen3-TTS, for the checkpoints in
starrydark/BanG_Dream_Qwen3_TTS. Each checkpoint carries its voices as built-in speaker rows, so no
reference audio is used or shipped. MODEL_VERSIONS picks what is served, e.g. "v1=main" (one model) or
"v2=main,v1=v1" (v2 on main, v1 under its tag); the first entry is the default.

An entry can merge revisions with "+": "v1=main+ournotes" offers every voice of main, plus the voices of the
ournotes branch that main lacks, each served by the first revision that has it."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

LOGGER = logging.getLogger("bangdream_tts")

MODEL_REPO = os.getenv("MODEL_REPO", "starrydark/BanG_Dream_Qwen3_TTS").strip()
MODEL_VERSIONS = [(n.strip(), [r.strip() for r in revs.split("+") if r.strip()])
                  for n, revs in (v.split("=", 1) for v in os.getenv("MODEL_VERSIONS", "v1=main").split(",") if "=" in v)]
DEFAULT_SPEAKER = "ppp_kasumi"
MAX_TEXT_CHARACTERS = 200
MAX_NEW_TOKENS = 512
GPU_DURATION_SECONDS = 60

# Training transcripts carry no spaces next to Japanese characters, so the same normalisation is applied here.
_JA_CHAR = r"[　-〿぀-ヿ一-鿿＀-￯…―〜]"
_SPACE_NEAR_JA = re.compile(rf"(?<={_JA_CHAR})[ 　]+|[ 　]+(?={_JA_CHAR})")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")

NOTICE = (
    "**Unofficial fan demo.** Not affiliated with or endorsed by Bushiroad, Craft Egg, the BanG Dream! project "
    "or the voice cast. All audio produced here is **AI-generated**: label it as AI-generated wherever you share it, "
    "do not use it to impersonate anyone or to pass it off as official, and read the "
    "[BanG Dream! fan-work guideline](https://bang-dream.com/bdp-guideline/) before using outputs anywhere public "
    "(its item 10 prohibits derivative works that use the franchise's audio). "
    "Rights holders and cast members can ask for removal (Hugging Face: starrydark), and the demo and model "
    "will be taken down on request.\n\n"
    "**非官方粉丝自制演示**，与官方、运营及声优无关。生成的音频均为 **AI 合成**：发布时请标注“AI生成”"
    "（《人工智能生成合成内容标识办法》；bilibili 投稿请勾选 AI 标识），禁止商用，禁止冒充声优或冒充官方。"
    "权利方或声优要求时将下架。"
)
ABOUT = """
A full fine-tune of Qwen3-TTS-12Hz-1.7B-Base with one built-in speaker row per character voice
(persona variants such as 大魔姫あこ, 月ノ森 Soyo / Sakiko and Mortis have their own rows). Every voice
speaks Japanese, Mandarin and English; Japanese is the language the voices were trained in, and Mandarin
and English quality varies by voice (see the model card's limitations).

Training data, evaluation and the selection rules that picked each checkpoint are documented on the
[model card](https://huggingface.co/starrydark/BanG_Dream_Qwen3_TTS) and in the
[code repository](https://github.com/starryark/bangdream-tts). No game audio, transcripts or dataset are
distributed, here or there.
"""

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import spaces  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402


def load_version(revision: str) -> tuple[Qwen3TTSModel, dict]:
    source = Path(snapshot_download(repo_id=MODEL_REPO, revision=revision))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("tts_model_type") != "custom_voice":
        raise RuntimeError(f"{MODEL_REPO}@{revision} is not a Qwen3-TTS custom_voice checkpoint")
    voices = {}
    if (source / "voices.json").is_file():
        voices = {k: v for k, v in json.loads((source / "voices.json").read_text(encoding="utf-8")).items()
                  if isinstance(v, dict)}
    # ZeroGPU emulates CUDA at startup: place the model on CUDA at module scope, not per request.
    model = Qwen3TTSModel.from_pretrained(str(source), device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    return model, voices


def route(speakers_by_revision: list[tuple[str, list[str]]]) -> dict[str, str]:
    """speaker -> the first revision (in the listed order) that has it."""
    routes: dict[str, str] = {}
    for revision, speakers in speakers_by_revision:
        for s in speakers:
            routes.setdefault(s, revision)
    return routes


LOADED: dict[str, tuple[Qwen3TTSModel, dict]] = {}
ROUTES: dict[str, dict[str, str]] = {}      # version -> speaker -> revision
CHOICES: dict[str, list[tuple[str, str]]] = {}
for name, revisions in MODEL_VERSIONS:
    for revision in revisions:
        if revision not in LOADED:
            LOADED[revision] = load_version(revision)
    ROUTES[name] = route([(r, sorted({s.lower() for s in LOADED[r][0].get_supported_speakers()})) for r in revisions])
    label = lambda s, v: (f"{v[s]['zh'] + ' / ' if v[s].get('zh') and v[s]['zh'] != v[s]['character'] else ''}"  # noqa: E731
                          f"{v[s]['character']} ({v[s].get('band', '')}) · {s}") if s in v else s
    CHOICES[name] = [(label(s, LOADED[r][1]), s) for s, r in sorted(ROUTES[name].items())]
    LOGGER.warning("version %s: %s", name, {r: sum(v == r for v in ROUTES[name].values()) for r in revisions})
DEFAULT_VERSION = MODEL_VERSIONS[0][0]
MERGED = any(len(revisions) > 1 for _, revisions in MODEL_VERSIONS)
if MERGED:
    ABOUT += """
The voices from BanG Dream! Our Notes (16 new characters, and the Our Notes era of nine MyGO!!!!! / Ave Mujica
characters, ids ending in `_ournotes`) come from a second checkpoint, the model's `ournotes` branch; every
other voice comes from `main`. The status line names the checkpoint each clip came from.
"""


def validate_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to synthesize.")
    normalized = _SPACE_NEAR_JA.sub("", text.strip())
    if len(normalized) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"Text is too long ({len(normalized)} characters). Use at most {MAX_TEXT_CHARACTERS}.")
    return normalized


def pick_language(choice: str, text: str) -> str:
    """'Detect' -> the language the text is written in (the checkpoints were trained with language ids)."""
    if choice != "Detect":
        return choice
    if _KANA.search(text):
        return "Japanese"
    if _HAN.search(text):
        return "Chinese"
    if _LATIN.search(text):
        return "English"
    return "Auto"


def prepare_audio(wavs: object, sample_rate: object) -> tuple[int, np.ndarray]:
    if not isinstance(sample_rate, (int, np.integer)) or not 8_000 <= int(sample_rate) <= 192_000:
        raise RuntimeError("The model returned an invalid sample rate.")
    if not isinstance(wavs, (list, tuple)) or not wavs:
        raise RuntimeError("The model returned no audio.")
    waveform = np.asarray(wavs[0], dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("The model returned an empty or invalid waveform.")
    return int(sample_rate), waveform


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def synthesize(text: str, version: str, speaker: str, language: str):
    try:
        normalized = validate_text(text)
        if version not in ROUTES:
            raise ValueError(f"Unknown model version {version!r}.")
        if speaker not in ROUTES[version]:
            raise ValueError(f"Voice {speaker!r} is not in {version}.")
        revision = ROUTES[version][speaker]
        lang = pick_language(language, normalized)
        # ~1.6 codec frames per Japanese character at 12.5 Hz; ~2.5x that bounds a rare non-stopping sample.
        wavs, sample_rate = LOADED[revision][0].generate_custom_voice(
            text=normalized, speaker=speaker, language=lang, non_streaming_mode=True,
            max_new_tokens=min(MAX_NEW_TOKENS, 64 + 4 * len(normalized)))
        audio = prepare_audio(wavs, sample_rate)
        source = revision if MERGED else version
        return audio, f"AI-generated speech — {audio[1].size / audio[0]:.1f} s ({source}, {speaker}, {lang})."
    except ValueError as exc:
        return None, f"Input error: {exc}"
    except Exception:
        LOGGER.exception("Speech generation failed")
        return None, "Generation failed. Please try again with a shorter passage."


def voices_for(version: str):
    choices = CHOICES.get(version, [])
    keep = DEFAULT_SPEAKER if any(s == DEFAULT_SPEAKER for _, s in choices) else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=keep)


CSS = """
.gradio-container { max-width: 820px !important; margin: 0 auto !important; }
#generate-button { min-height: 48px; }
#status { min-height: 1.5rem; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="BanG Dream! AI Voices (unofficial) — Qwen3-TTS", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# BanG Dream! AI Voices (unofficial fan demo)\n"
                    "Type Japanese, Chinese or English and hear it in a BanG Dream! character's voice. "
                    "No reference audio is used; every voice is built into the model.\n\n"
                    "邦邦角色 AI 语音（非官方）：输入中文、日语或英语，选择角色即可合成，无需参考音频。"
                    "声线列表可直接输入中文名搜索（如 千早爱音、丰川祥子）。\n\n"
                    "バンドリ！キャラクターの AI 音声（非公式）。日本語・中国語・英語に対応。")
        gr.Markdown(NOTICE)
        text = gr.Textbox(label="Text", lines=4, max_lines=8, max_length=MAX_TEXT_CHARACTERS,
                          info=f"Up to {MAX_TEXT_CHARACTERS} characters.",
                          value="今日の練習、すごく楽しかったね！また明日もがんばろう。")
        with gr.Row():
            version = gr.Dropdown(label="Model", choices=[n for n, _ in MODEL_VERSIONS], value=DEFAULT_VERSION,
                                  visible=len(MODEL_VERSIONS) > 1)
            speaker = gr.Dropdown(label="Voice", choices=CHOICES[DEFAULT_VERSION],
                                  value=voices_for(DEFAULT_VERSION)["value"])
            language = gr.Dropdown(label="Language", choices=["Detect", "Japanese", "Chinese", "English", "Auto"],
                                   value="Detect", info="Detect reads the script of the text.")
        generate = gr.Button("Generate Speech", variant="primary", size="lg", elem_id="generate-button")
        audio = gr.Audio(label="Generated speech (AI-generated)", type="numpy", format="wav", interactive=False)
        status = gr.Markdown("Ready.", elem_id="status")
        gr.Examples(
            examples=[["今日の練習、すごく楽しかったね！また明日もがんばろう。", DEFAULT_VERSION, DEFAULT_SPEAKER, "Detect"],
                      ["今天的排练真开心，明天也一起加油吧！", DEFAULT_VERSION, DEFAULT_SPEAKER, "Detect"],
                      ["练习结束了。抹茶芭菲，再来一份。", DEFAULT_VERSION, "mygo_raana", "Detect"],
                      ["Thanks for coming to our live show tonight!", DEFAULT_VERSION, DEFAULT_SPEAKER, "Detect"]],
            inputs=[text, version, speaker, language], label="Examples")
        with gr.Accordion("About this model", open=False):
            gr.Markdown(ABOUT)
        version.change(voices_for, inputs=version, outputs=speaker)
        generate.click(fn=synthesize, inputs=[text, version, speaker, language], outputs=[audio, status],
                       api_name="generate_speech", show_progress="full", concurrency_limit=1, concurrency_id="bd-tts")
    return demo.queue(default_concurrency_limit=1, max_size=16)


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
