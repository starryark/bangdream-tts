# BanG Dream! in Chinese fan culture, the Chinese AI-voice scene, and how this project can be found (2026-11-09)

Why this note exists: the model was made with Chinese BanG Dream! fan creators in mind. This note covers four things:
- how that community talks;
- what AI-voice tools it already uses and how it finds them;
- how search engines and LLM assistants discover open models;
- the rules any 二创 made with this model must follow.

Method: desk research on 2026-11-09. Sources were bilibili (search and autocomplete APIs, video pages), Baidu (Baike and autocomplete), 萌娘百科 (through search extracts, because the site blocks fetching), Zhihu, news articles, the Hugging Face, ModelScope, GitHub and fish.audio APIs, platform docs and GEO papers. View counts were read on that date. "(unverified)" marks claims taken only from snippets or secondary write-ups.

## 1. Vocabulary

| Use | Meaning | Note |
|---|---|---|
| **邦邦**, BanG Dream! | The franchise | The default casual name in press and among fans; 邦多利 also occurs. Fans call themselves 邦友. |
| **BanG Dream! 少女乐团派对!** | Girls Band Party, Chinese server, run by bilibili since 2019 | The official word is 乐**团**; "少女乐队派对" is a common mistake. |
| **《迷途之子!!!!!》 / 《颂乐人偶》** | bilibili's official titles for the MyGO!!!!! and Ave Mujica anime | Short forms: MyGO / Mujica. MyGO is also 迷子; Ave Mujica is also 母鸡卡 (medium confidence). |
| **《BanG Dream! Our Notes》** (交织的乐章) | BanG Dream! Our Notes (2026-09-24; bilibili Games publishes it globally) | The launch coverage in Simplified Chinese uses the English title. 交织的乐章 appears in pre-registration coverage and on Baidu Baike; the Taiwan title is 交織的樂章. No mainland server has been announced. |
| **少女乐队 / 少女乐队二创** | A *genre* label: MyGO, Ave Mujica, 孤独摇滚, Girls Band Cry | "少女乐队二创" in practice means mostly MyGO/Mujica fan work, but used alone the term autocompletes to 少女乐队的呐喊 (Girls Band Cry). Pair it with 邦邦二创 / MyGO二创. |
| 二创, AI配音, AI翻唱, 名场面, 但是说中文 | Fan work, AI voice-over, AI singing covers, famous scenes, "…but in Chinese" | The vocabulary of the model-share and 二创 videos. |

**Avoid:**
- **木柜子 / 木柜子乐队:** a slur.
- **GO批:** half-derogatory.
- **少女乐队 alone as the franchise name.**
- **Anything that reads as official.** The official BanG Dream! AI singing voicebanks 夢ノ結唱 / 梦的结唱 (Bushiroad Music × CeVIO) exist, so "official" is easy to imply by accident.
- **The voice cast's real names as tags.**

**Character names:** official Chinese names, with fan nicknames in parentheses where they are well attested:
- 长崎爽世〔素世〕
- 祐天寺若麦〔喵梦〕
- 丰川祥子〔祥子、小祥〕
- 若叶睦〔睦、睦头人〕 and Mortis (left untransliterated in the game; 墨缇丝 is fan/media usage)
- 要乐奈〔乐奈、野猫〕
- 三角初华〔初华〕

People search with the **full Chinese name** (千早爱音ai配音, not 爱音ai配音). The full table, keyed by speaker id, is `release/zh_names.json`. Its GBP and MyGO names come from the CN server data; the Ave Mujica and Our Notes names come from the international server's Simplified Chinese text. 沢海奏多 is 泽海奏多: a millsage story character, not playable.

## 2. Memes (梗) that fans will recognise

These are context for writing copy, not material to quote.
- **为什么要演奏春日影 (MyGO ep 7–8).** Together with **组一辈子乐队** (灯, ep 1) and **惊世一跪** (爽世, ep 8), the press calls these the three signature MyGO memes.
- **Others:**
  - 那我呢 (爱音)
  - 抹茶芭菲 and 野猫 (乐奈)
  - 客服小祥 and the "落魄大小姐" cluster
  - 还在go
  - 睦头人 / Mortis
  - 小姨 and the 初华爬行 → Doro-lis mashup (Ave Mujica)
  - 假药, for wild weekly fan theories during the Ave Mujica broadcast
- **Older GBP-era community jokes** include 破琵琶, 夕阳红 and 摩卡 "0.5倍速".
- **Formats:**
  - subtitled screenshot stickers; fans have built search engines for them, e.g. MyGO-Searcher;
  - 名场面 clips;
  - 春日影 instrument covers;
  - 鬼畜/曲艺;
  - 填词;
  - fan-fiction videos;
  - WebGAL visual novels;
  - AI翻唱 and AI配音.

**Popularity on bilibili (2026-11-09):**
- 《迷途之子!!!!!》: about 1亿 plays, rated 9.7.
- 《颂乐人偶》: about 1.3亿 plays and 130万 弹幕.
- The MyGO!!!!!×Ave Mujica joint live added a Shanghai show (2025-10), which sold out on the day.

## 3. The AI-voice scene this model joins

**How models are shared:** mostly as bilibili videos titled `…语音模型分享` / `…模型发布`, with Quark or Baidu 网盘 links and a pointer to a GPT-SoVITS 整合包. Typical tags: mygo, GPT SoVITS, 模型分享, AI语音, 角色语音, BanG Dream!, plus the character names. The video description (简介) usually says 禁止商用 and includes an AI declaration.

| Kind | Examples (views / uses) |
|---|---|
| GPT-SoVITS character packs | BanG Dream 全员 GSV pack (56k), MyGO 8-character pack (46k), Mortis & 睦 (42k), MyGO tutorial (29k) |
| RVC speech models | MyGO RVC (56k), MyGO+Mujica RVC (24k); HF `Wanlau/RVC_BanGDream` (45 characters) |
| Bert-VITS2 | HF Space `Mahiruoshi/BangDream-Bert-VITS2` (2023). This is a direct predecessor: 2023 videos that made every MyGO member say the same line used it. |
| Online clone voices | fish.audio community voices: 千早爱音 81.7k uses, 丰川祥子(中文版) 22k, 长崎素世 13k. **Mandarin-tagged voices are used 5–30× more than Japanese ones.** |
| Apps | BandoriPet (GitHub, 514★): a desktop pet with a Qwen3-TTS backend (one LoRA per band plus Japanese reference audio). Its TTS client speaks a GPT-SoVITS-style HTTP API. |
| This model | HF main 54 downloads. A third-party GGUF conversion (`mradermacher/BanG_Dream_Qwen3_TTS-GGUF`) has 409. |

**What users ask for:**
- one-click Windows packages (整合包);
- online demos that work without a VPN (不用翻墙);
- Mandarin that sounds like the character ("…但是说中文" videos get 74k–157k views; a character ranting in English got 2.6M);
- whole-cast coverage (全员);
- no hassle with reference audio. "参考音频懒人包" videos exist; **this model needs no reference audio at all.**

**What they type into search** (autocomplete as a proxy for frequency):
1. `{full name}ai配音` / `ai语音` / `ai模型` / `ai翻唱`;
2. `mygo语音模型`, `mygo模型分享`, `mygoai配音`;
3. `{name}语音模型`;
4. `gptsovits模型分享`, `整合包`.

`mygo tts` and `ai说中文` are not real queries.

## 4. Access from mainland China

- **huggingface.co** has been blocked since 2023-05. hf.space is reported blocked on many networks since 2025-11 (unverified).
- **hf-mirror.com** (`HF_ENDPOINT=https://hf-mirror.com`) is how people download, but its robots.txt disallows crawling, so it does not help anyone *find* a model.
- **ModelScope (魔搭)** is the domestic hub: Qwen3-TTS-1.7B-Base has about 198k downloads there. It is reachable and crawlable, and it serves `llms.txt` and `.md` pages to agents. Character packs left with the default README get very few downloads, so a mirror needs a full Chinese README.
- **Chinese assistants each cite their own ecosystem's platforms:**
  - Doubao → Douyin/Toutiao;
  - Yuanbao → WeChat Official Accounts;
  - Kimi → Zhihu;
  - DeepSeek → mixed sources.

  This comes from practitioner reports. For Chinese discovery, posts on bilibili, Zhihu and WeChat matter more than repo metadata.

## 5. Rules for 二创 made with this model

- **BanG Dream! fan-work guideline** (<https://bang-dream.com/bdp-guideline/>), item 10: *prohibits* making, showing, distributing or publishing derivative works that use the content's audio (「本コンテンツのうち、音声を利用した二次的著作物の制作、展示、頒布、公開。」). A TTS model trained on the game's and anime's voice audio falls under it. This project is an unofficial fan work made despite that, and it will be taken down at a rights holder's or cast member's request.
- **China, 《人工智能生成合成内容标识办法》** (effective 2025-09-01; national standard GB 45438-2025):
  - AI-generated audio must carry an explicit label (a spoken or rhythmic cue, or on-screen text);
  - users must declare AI content and must not remove labels.
- **bilibili:**
  - The AI标识 option is now on the first upload page (2026-05).
  - Unlabelled AI content is relabelled, sent back or removed.
  - The 2026 「清朗」 notice names AI虚假配音 and AI impersonation specifically.
- **Voice rights:**
  - 民法典 art. 1023 protects a person's voice.
  - Beijing Internet Court (2024-04): an AI voice that ordinary listeners can identify engages the voice actor's rights, and licensing a recording is not licensing the voice.
  - Chinese voice actors issued collective statements in 2026-03 against unauthorised AI voices.
  - In Japan, 日本俳優連合 / #NOMORE無断生成AI.
  - So: no impersonation of the cast, no commercial use, label everything, and do not promote in Japanese voice-actor spaces.

## 6. Discovery checklist and what was done

| Item | Impact | Status |
|---|---|---|
| One canonical description, names first, reused verbatim (GitHub About, both HF cards, llms.txt; the Space uses a 60-character short form; future posts should reuse it) | high | done 2026-11-09 |
| Aliases in visible prose (邦邦, 邦多利, 少女乐团派对, 迷途之子, 颂乐人偶, 交织的乐章, バンドリ, ガルパ); Chinese names in the voice tables and the demo's voice list | high | done |
| GitHub topics (20), website set to the Space | high | done |
| A clear statement that this is not affiliated with Bushiroad / 夢ノ結唱 | high | done |
| Chinese README (`README.zh-CN.md`), FAQ, `llms.txt` | medium | done |
| Item 10 and the Chinese AI-labelling rules stated in Chinese | high | done |
| ModelScope mirror with a Chinese README | high | needs the owner's ModelScope account |
| bilibili model-share video or 专栏, Zhihu / WeChat post (earned mentions) | high | owner's decision; draft available |
| GPT-SoVITS-style HTTP API shim, so apps such as BandoriPet can use the model | medium | not started |
| llms.txt adoption by search engines | low | no engine has confirmed using it; kept for coding agents |

## Sources

**Vocabulary, memes and bilibili figures:**
- 萌娘百科 (春日影, 一辈子oo, 千早爱音, 若叶睦, BanG Dream!/相关梗及用语) and Baidu Baike (长崎爽世, 要乐奈, BanG Dream! 交织的乐章)
- bilibili 番剧 pages md23679123 and md23679586
- news.qq.com 20250516A05VO300 and 20250401A0AB6800
- gamersky 202501/1868990

**Rules:**
- bang-dream.com/bdp-guideline
- cac.gov.cn 2025-03/14 (标识办法)
- thepaper.cn 27130953 (AI-voice ruling)
- itmedia 2410/22 (#NOMORE)

**Ecosystem:**
- bilibili BV1sgbkzmELF, BV1PBTRzhEkq, BV18hQKYqEhz, BV1hBfdYhE2c, BV1XH4y137Lk, BV17Q4y1s7Xc
- fish.audio community voices
- github.com/HELPMEEADICE/BANDORI-PET-REV

**Discovery:**
- huggingface.co/docs/hub (search, model cards, Spaces config)
- GitHub docs (topics, repository search)
- llmstxt.org
- arXiv 2311.09735 (GEO), 2506.11097 (C-SEO Bench), 2509.08919 (earned-media bias)
- modelscope.cn/docs
