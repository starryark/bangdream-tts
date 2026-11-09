"""E8 (local): blind listening pages and their analysis (E8_PLAN.md §6). The pages and clips stay on this machine.

  python e8/e8_listen_page.py round1                  # A vs B          -> e8/listen/round1/index.html
  python e8/e8_listen_page.py round2 --winner A       # winner vs v1 (C0) + new-voice checks -> e8/listen/round2/s1.html, s2.html, new.html
  python e8/e8_listen_page.py analyze1 RESULTS.json   # round-1 rule
  python e8/e8_listen_page.py analyze2 R1.json R2.json NEW.json   # round-2 rule (the three downloaded result files)
  python e8/e8_listen_page.py dry                     # dry run on samples/E1b_ep2 (no E8 renders needed)

Pages: blind (systems never named), side and order randomised per trial (SystemRandom), both clips loudness-
normalised to -23 LUFS with peaks under -1 dBFS, random file names, unlimited replay, CCR -3..+3 (negative =
left better), optional flags per side, hidden repeats with sides swapped and null pairs (same clip twice).
Answers autosave to localStorage and download as JSON. The key (which side is which system) is written to
e8/listen/<round>_key.json, outside the page folder, and is never loaded by a page.
"""
from __future__ import annotations

import json
import random
import secrets
import shutil
import sys
from pathlib import Path

BD = Path(r"E:\STEINS;GATEVoice\bangdream")
E8 = BD / "e8"
LISTEN = E8 / "listen"
RENDERS = E8 / "listen_renders"
SR = random.SystemRandom()


def norm_copy(src: Path, dst_dir: Path) -> str:
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    y, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(-1)
    meter = pyloudnorm.Meter(sr)
    try:
        y = pyloudnorm.normalize.loudness(y, meter.integrated_loudness(y), -23.0) if len(y) > sr * 0.4 else y
    except Exception:
        pass
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.891:                      # -1 dBFS
        y = y * (0.891 / peak)
    name = secrets.token_hex(6) + ".wav"
    sf.write(str(dst_dir / name), y, sr, subtype="PCM_16")
    return f"audio/{name}"


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#fafafa;--fg:#1b1b1b;--card:#fff;--line:#ddd;--acc:#3b5bdb}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--card:#1e1e1e;--line:#333;--acc:#7c94ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}
.wrap{max-width:760px;margin:0 auto}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.row{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1;min-width:240px}audio{width:100%}
.scale{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.scale label{border:1px solid var(--line);border-radius:6px;padding:6px 8px;cursor:pointer}
.scale input{margin-right:4px}.flags label{margin-right:12px;font-size:13px}button{padding:8px 16px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}.muted{opacity:.7;font-size:13px}.text{font-size:17px}
</style></head><body><div class="wrap">
<h2>__TITLE__</h2><p class="muted">__INTRO__</p>
<div class="card"><div class="muted" id="prog"></div><div class="text" id="text"></div></div>
<div class="card" id="trial"></div>
<div class="row"><button id="back">Back</button><button class="primary" id="next">Next</button><span style="flex:1"></span><button id="dl">Download results</button></div>
<p class="muted" id="msg"></p></div>
<script>
const T=__TRIALS__, KEY="__STORE__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
st.answers=st.answers||{}; st.started=st.started||new Date().toISOString();
let i=0; while(i<T.length-1 && st.answers[T[i].tid]) i++;
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function el(h){const d=document.createElement('div');d.innerHTML=h;return d.firstElementChild}
function render(){
 const t=T[i], a=st.answers[t.tid]||{flags:[],plays:0}; st.answers[t.tid]=a; a.t0=a.t0||Date.now();
 document.getElementById('prog').textContent=`Trial ${i+1} of ${T.length}`+(t.lang?` · ${t.lang}`:'');
 document.getElementById('text').textContent=t.text||'';
 const box=document.getElementById('trial'); box.innerHTML='';
 if(t.type==='pair'){
  box.append(el(`<div class="row"><div class="col"><b>A</b><audio controls preload="auto" src="${t.a}"></audio></div><div class="col"><b>B</b><audio controls preload="auto" src="${t.b}"></audio></div></div>`));
  const labels=[[-3,'A much better'],[-2,'A better'],[-1,'A slightly better'],[0,'No difference'],[1,'B slightly better'],[2,'B better'],[3,'B much better']];
  const sc=el('<div class="scale"></div>'); labels.forEach(([v,l])=>{const x=el(`<label><input type="radio" name="s" value="${v}">${l}</label>`);if(a.score===v)x.querySelector('input').checked=true;sc.append(x)}); box.append(sc);
  const fl=el('<div class="flags" style="margin-top:10px"></div>'); [['wrong_voice','wrong voice'],['misreading','misreading'],['accent','accent'],['artefact','artefact']].forEach(([k,l])=>['A','B'].forEach(s=>{const id=`${k}_${s}`;const x=el(`<label><input type="checkbox" value="${id}">${s}: ${l}</label>`);if(a.flags.includes(id))x.querySelector('input').checked=true;fl.append(x)})); box.append(fl);
 } else {
  box.append(el(`<div class="row"><div class="col"><b>Real reference (the character)</b><audio controls preload="auto" src="${t.ref}"></audio></div><div class="col"><b>Generated</b><audio controls preload="auto" src="${t.a}"></audio></div></div>`));
  box.append(el('<p>Does the generated clip sound like this character?</p>'));
  const sc=el('<div class="scale"></div>'); [['yes','Yes'],['partly','Partly'],['no','No']].forEach(([v,l])=>{const x=el(`<label><input type="radio" name="s" value="${v}">${l}</label>`);if(a.score===v)x.querySelector('input').checked=true;sc.append(x)}); box.append(sc);
  const fl=el('<div class="flags" style="margin-top:10px"></div>'); [['misreading','misreading'],['accent','accent'],['artefact','artefact']].forEach(([k,l])=>{const x=el(`<label><input type="checkbox" value="${k}">${l}</label>`);if(a.flags.includes(k))x.querySelector('input').checked=true;fl.append(x)}); box.append(fl);
 }
 box.querySelectorAll('audio').forEach(au=>au.addEventListener('play',()=>{a.plays++;save()}));
 box.querySelectorAll('input[name=s]').forEach(r=>r.addEventListener('change',()=>{a.score=t.type==='pair'?Number(r.value):r.value;a.rt_ms=Date.now()-a.t0;save()}));
 box.querySelectorAll('.flags input').forEach(c=>c.addEventListener('change',()=>{a.flags=[...box.querySelectorAll('.flags input:checked')].map(x=>x.value);save()}));
 document.getElementById('back').disabled=i===0; document.getElementById('next').textContent=i===T.length-1?'Finish':'Next';
 const done=T.filter(x=>st.answers[x.tid]&&st.answers[x.tid].score!==undefined).length; document.getElementById('msg').textContent=`${done} of ${T.length} answered. Answers are kept in this browser; use "Download results" at the end.`;
}
document.getElementById('next').onclick=()=>{const a=st.answers[T[i].tid];if(a.score===undefined){alert('Choose an answer first.');return}
 if(i<T.length-1){i++;render();window.scrollTo(0,0)}else{st.finished=new Date().toISOString();save();download()}};
document.getElementById('back').onclick=()=>{if(i>0){i--;render()}};
function download(){const blob=new Blob([JSON.stringify({page:KEY,...st},null,1)],{type:'application/json'});const u=URL.createObjectURL(blob);const l=document.createElement('a');l.href=u;l.download=KEY+'_results.json';l.click();setTimeout(()=>URL.revokeObjectURL(u),2000)}
document.getElementById('dl').onclick=download; render();
</script></body></html>"""


def write_page(path: Path, title: str, intro: str, trials: list[dict], store: str) -> None:
    html = PAGE.replace("__TITLE__", title).replace("__INTRO__", intro).replace("__STORE__", store) \
               .replace("__TRIALS__", json.dumps(trials, ensure_ascii=False))
    path.write_text(html, encoding="utf-8")


def pair_trials(items: list[dict], sys_x: str, sys_y: str, out: Path, n_rep: int, n_null: int, src) -> tuple[list, dict]:
    """Main pairs (x vs y, random side), n_rep repeats of random main items with sides swapped, n_null nulls."""
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    trials, key = [], {}
    clips = {}
    for it in items:
        clips[it["id"]] = {s: norm_copy(src(s, it), audio) for s in (sys_x, sys_y)}
    mains = []
    for it in items:
        left = SR.choice([sys_x, sys_y]); right = sys_y if left == sys_x else sys_x
        tid = secrets.token_hex(4)
        trials.append({"tid": tid, "type": "pair", "a": clips[it["id"]][left], "b": clips[it["id"]][right], "text": it["text"], "lang": it["language"]})
        key[tid] = {"kind": "main", "item": it["id"], "voice": it["voice"], "lang": it["lang"], "ikind": it["kind"], "round": it["round"], "left": left, "right": right}
        mains.append((tid, it, left, right))
    for tid0, it, left, right in SR.sample(mains, n_rep):
        tid = secrets.token_hex(4)
        trials.append({"tid": tid, "type": "pair", "a": clips[it["id"]][right], "b": clips[it["id"]][left], "text": it["text"], "lang": it["language"]})
        key[tid] = {"kind": "repeat", "of": tid0, "item": it["id"], "voice": it["voice"], "lang": it["lang"], "ikind": it["kind"], "round": it["round"], "left": right, "right": left}
    for _, it, left, right in SR.sample(mains, n_null):
        s = SR.choice([sys_x, sys_y]); tid = secrets.token_hex(4)
        trials.append({"tid": tid, "type": "pair", "a": clips[it["id"]][s], "b": clips[it["id"]][s], "text": it["text"], "lang": it["language"]})
        key[tid] = {"kind": "null", "item": it["id"], "voice": it["voice"], "system": s}
    SR.shuffle(trials)
    return trials, key


def items_file() -> dict:
    return json.loads((E8 / "listen_items.json").read_text(encoding="utf-8"))


def render_path(system: str, it: dict) -> Path:
    p = RENDERS / system / f"{it['id']}.wav"
    if not p.is_file():
        raise SystemExit(f"missing render {p}")
    return p


INTRO_PAIR = ("Each trial plays the same text in two versions, A and B. Listen to both (replay as often as you like) and say which "
              "sounds better <b>for this character</b>: voice identity first, then naturalness, pronunciation and accent. "
              "Some trials repeat; answer each on its own. Keep a session to about 20 minutes.")


def round1() -> None:
    d = items_file()
    out = LISTEN / "round1"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    its = [it for it in d["items"] if it["round"] == 1]
    trials, key = pair_trials(its, "A", "B", out, 2, 1, render_path)
    write_page(out / "index.html", "E8 listening, round 1", INTRO_PAIR, trials, "e8_round1")
    (LISTEN / "round1_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"[page] round1: {len(trials)} trials -> {out / 'index.html'}")


def round2(winner: str) -> None:
    d = items_file()
    out = LISTEN / "round2"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    its = [it for it in d["items"] if it["round"] == 2]
    trials, key = pair_trials(its, winner, "C0", out, 8, 4, render_path)
    half = (len(trials) + 1) // 2
    write_page(out / "s1.html", "E8 listening, round 2 (part 1 of 3)", INTRO_PAIR, trials[:half], "e8_round2_s1")
    write_page(out / "s2.html", "E8 listening, round 2 (part 2 of 3)", INTRO_PAIR, trials[half:], "e8_round2_s2")
    news = [it for it in d["items"] if it["round"] == "new"]
    audio = out / "audio"
    refs = {v: norm_copy(Path(p), audio) for v, p in d["new_voice_refs"].items()}
    ntrials = []
    for it in news:
        tid = secrets.token_hex(4)
        ntrials.append({"tid": tid, "type": "abs", "a": norm_copy(render_path(winner, it), audio), "ref": refs[it["voice"]], "text": it["text"], "lang": it["language"]})
        key[tid] = {"kind": "new", "item": it["id"], "voice": it["voice"], "lang": it["lang"], "ikind": it["kind"], "system": winner}
    SR.shuffle(ntrials)
    write_page(out / "new.html", "E8 listening, round 2 (part 3 of 3): new voices",
               "Each trial plays a real clip of the character and a generated clip. Does the generated clip sound like this character?",
               ntrials, "e8_round2_new")
    (LISTEN / "round2_key.json").write_text(json.dumps({"winner": winner, "trials": key}, indent=1), encoding="utf-8")
    print(f"[page] round2: {len(trials)} pair trials in s1/s2, {len(ntrials)} new-voice trials -> {out}")


def load_answers(paths: list[str]) -> dict:
    ans = {}
    for p in paths:
        ans.update(json.loads(Path(p).read_text(encoding="utf-8"))["answers"])
    return ans


def pref(k: dict, a: dict, system: str) -> int:
    """Preference for `system` on a pair trial: the CCR score signed so that + favours `system`."""
    s = int(a["score"])
    return s if k["right"] == system else -s


def consistency(key: dict, ans: dict, system: str) -> dict:
    reps = [(k, t) for t, k in key.items() if k["kind"] == "repeat"]
    agree = 0
    for k, t in reps:
        p0, p1 = pref(key[k["of"]], ans[k["of"]], system), pref(k, ans[t], system)
        agree += (p0 > 0 and p1 > 0) or (p0 < 0 and p1 < 0) or (p0 == 0 and p1 == 0)
    nulls = [abs(int(ans[t]["score"])) for t, k in key.items() if k["kind"] == "null"]
    return {"repeats": len(reps), "repeats_agree": agree, "nulls": nulls, "nulls_within_1": all(n <= 1 for n in nulls)}


def analyze1(paths: list[str]) -> None:
    key = json.loads((LISTEN / "round1_key.json").read_text(encoding="utf-8"))
    ans = load_answers(paths)
    main = [(t, k) for t, k in key.items() if k["kind"] == "main"]
    missing = [t for t, _ in main if t not in ans or "score" not in ans[t]]
    if missing:
        raise SystemExit(f"{len(missing)} round-1 trials unanswered")
    pb = [pref(k, ans[t], "B") for t, k in main]
    wv = {s: sum(1 for t, k in main for f in ans[t].get("flags", []) if f == f"wrong_voice_{'A' if k['left'] == s else 'B'}") for s in ("A", "B")}
    mean_b = sum(pb) / len(pb)
    winner = "B" if mean_b > 0 and wv["B"] <= wv["A"] else "A"
    out = {"mean_pref_B": round(mean_b, 3), "n": len(pb), "wrong_voice_flags": wv, "winner": winner,
           "consistency": consistency(key, ans, "B"),
           "per_voice_pref_B": {v: sum(p for (t, k), p in zip(main, pb) if k["voice"] == v) for v in sorted({k["voice"] for _, k in main})}}
    (E8 / "results").mkdir(exist_ok=True)
    (E8 / "results" / "listening_round1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def analyze2(paths: list[str]) -> None:
    from scipy.stats import wilcoxon
    kk = json.loads((LISTEN / "round2_key.json").read_text(encoding="utf-8"))
    key, cand = kk["trials"], kk["winner"]
    ans = load_answers(paths)
    items = {it["id"]: it for it in items_file()["items"]}
    replaced = {"mygo_tomori", "mygo_anon", "mygo_taki", "mujica_uika", "mujica_mutsumi", "mujica_umiri", "mujica_nyamu", "mujica_sakiko"}
    main = [(t, k) for t, k in key.items() if k["kind"] == "main"]
    missing = [t for t, k in key.items() if k["kind"] != "null" and (t not in ans or "score" not in ans[t])]
    if missing:
        raise SystemExit(f"{len(missing)} round-2 trials unanswered")

    def group(k):
        if k["lang"] == "ja":
            return "ja_replaced" if k["voice"] in replaced else "ja_unchanged"
        return k["lang"]
    groups = {}
    for t, k in main:
        groups.setdefault(group(k), []).append(pref(k, ans[t], cand))

    def test(x, alt):
        nz = [v for v in x if v != 0]
        if len(nz) < 1:
            return 1.0
        return float(wilcoxon(x, alternative=alt).pvalue)
    res = {g: {"n": len(x), "mean": round(sum(x) / len(x), 3), "p_greater": test(x, "greater"), "p_two_sided": test(x, "two-sided")}
           for g, x in groups.items()}
    wrong = {}
    for t, k in main:
        side = "A" if k["left"] == cand else "B"
        if f"wrong_voice_{side}" in ans[t].get("flags", []):
            wrong[k["voice"]] = wrong.get(k["voice"], 0) + 1
    newv = {}
    for t, k in key.items():
        if k["kind"] == "new":
            newv.setdefault(k["voice"], []).append(ans[t]["score"])
    weak = sorted(v for v, s in newv.items() if sum(x == "no" for x in s) >= 2)
    cons = consistency({t: k for t, k in key.items() if k["kind"] in ("main", "repeat", "null")}, ans, cand)
    rules = {
        "ja_replaced_better": res["ja_replaced"]["mean"] > 0 and res["ja_replaced"]["p_greater"] < 0.05,
        "ja_unchanged_not_worse": res["ja_unchanged"]["p_two_sided"] >= 0.05 or res["ja_unchanged"]["mean"] >= 0,
        "zh_not_worse": res["zh"]["p_two_sided"] >= 0.05 or res["zh"]["mean"] >= 0,
        "no_shared_voice_wrong_twice": all(n < 2 for n in wrong.values()),
        "new_voices_weak_le_3": len(weak) <= 3,
        "consistent": cons["repeats_agree"] >= 6 and cons["nulls_within_1"],
    }
    out = {"candidate": cand, "groups": res, "wrong_voice_flags": wrong, "new_voice_answers": newv, "weak_new_voices": weak,
           "consistency": cons, "rules": rules, "adopt_by_listening": all(rules.values()),
           "note": "The screens (E8_PLAN §5) must also pass. If only 'consistent' fails, round 2 is repeated once (§6)."}
    (E8 / "results").mkdir(exist_ok=True)
    (E8 / "results" / "listening_round2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def dry() -> None:
    """Build a small page from samples/E1b_ep2 to check the generator end to end (not a real round)."""
    src_dir = BD / "samples" / "E1b_ep2"
    wavs = sorted(src_dir.glob("*_ja.wav"))[:4]
    fake = [{"id": f"d{i}", "voice": w.stem.rsplit("_", 1)[0], "lang": "ja", "language": "Japanese", "kind": "ja", "round": "dry",
             "text": f"(dry run) {w.stem}"} for i, w in enumerate(wavs)]
    out = LISTEN / "dry"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    trials, key = pair_trials(fake, "X", "Y", out, 1, 1, lambda s, it: src_dir / f"{it['voice']}_{'ja' if s == 'X' else 'en'}.wav")
    write_page(out / "index.html", "E8 listening, DRY RUN", INTRO_PAIR, trials, "e8_dry")
    (LISTEN / "dry_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"[page] dry: {len(trials)} trials -> {out / 'index.html'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "round1":
        round1()
    elif cmd == "round2":
        round2(sys.argv[sys.argv.index("--winner") + 1])
    elif cmd == "analyze1":
        analyze1(sys.argv[2:])
    elif cmd == "analyze2":
        analyze2(sys.argv[2:])
    elif cmd == "dry":
        dry()
    else:
        raise SystemExit(__doc__)
