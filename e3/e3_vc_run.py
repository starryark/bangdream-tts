"""E3 step 2 (pod, Seed-VC venv, cwd = /workspace/seed-vc): run every job in e3/vc/jobs.json.

  /workspace/vcenv/bin/python /workspace/bd/e3/e3_vc_run.py /workspace/bd/e3/vc/jobs.json

Seed-VC v1 zero-shot, default speech model (whisper-small content, 22.05 kHz, no F0
conditioning), 25 diffusion steps, cfg 0.7, fp16 - E3_PLAN.md. inference.main() reloads the
models on every call, so load_models is swapped for a cached copy (models load once). Output
names follow inference.main: vc_<source stem>_<ref stem>_1.0_25_0.7.wav in the job's out_dir.
Jobs whose output exists are skipped (resumable).
"""
import argparse
import json
import os
import sys
import time



def out_path(j):
    src = os.path.basename(j["src"]).split(".")[0]
    ref = os.path.basename(j["ref"]).split(".")[0]
    return os.path.join(j["out_dir"], f"vc_{src}_{ref}_1.0_25_0.7.wav")


def main():
    sys.path.insert(0, os.getcwd())
    import inference as inf  # Seed-VC; sets HF_HUB_CACHE=./checkpoints/hf_cache on import
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    args = argparse.Namespace(source=None, target=None, output=None, diffusion_steps=25, length_adjust=1.0,
                              inference_cfg_rate=0.7, f0_condition=False, auto_f0_adjust=False,
                              semi_tone_shift=0, checkpoint=None, config=None, fp16=True)
    cached = inf.load_models(args)
    inf.load_models = lambda a: cached
    t0, done = time.time(), 0
    for k, j in enumerate(jobs):
        p = out_path(j)
        if os.path.exists(p):
            continue
        args.source, args.target, args.output = j["src"], j["ref"], j["out_dir"]
        inf.main(args)
        if not os.path.exists(p):
            raise SystemExit(f"expected output missing: {p}")
        done += 1
        if done % 50 == 0:
            print(f"[e3_vc] {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[e3_vc] done {done} new conversions in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
