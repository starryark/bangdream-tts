"""E8 (pod or local): uniform weight average of checkpoints (E8_PLAN.md §3).

  python e8/e8_average.py OUT_DIR CKPT_1 CKPT_2 [CKPT_3 ...]

Every floating-point tensor in model.safetensors is averaged in fp32 and cast back to its stored dtype
(bf16). Non-float tensors must be identical across inputs. Everything else (config, tokenizer,
speech_tokenizer/, ...) is copied from the LAST checkpoint, so its config and speaker ids are the ones
shipped. Tensor by tensor, so memory stays near one checkpoint plus one fp32 accumulator.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out, ckpts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    if len(ckpts) < 2:
        raise SystemExit("need at least two checkpoints")
    spk = [json.loads((c / "config.json").read_text(encoding="utf-8"))["talker_config"].get("spk_id") for c in ckpts]
    if any(s != spk[-1] for s in spk):
        raise SystemExit("checkpoints have different speaker ids")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpts[-1], out, dirs_exist_ok=True, ignore=lambda d, names: {"model.safetensors"} if Path(d) == ckpts[-1] else set())
    handles = [safe_open(str(c / "model.safetensors"), "pt") for c in ckpts]
    keys = list(handles[-1].keys())
    for h in handles[:-1]:
        if set(h.keys()) != set(keys):
            raise SystemExit("checkpoints have different tensor sets")
    sd = {}
    for k in keys:
        ts = [h.get_tensor(k) for h in handles]
        if ts[-1].is_floating_point():
            acc = torch.zeros_like(ts[-1], dtype=torch.float32)
            for t in ts:
                acc += t.float()
            sd[k] = (acc / len(ts)).to(ts[-1].dtype).contiguous()
        else:
            if any(not torch.equal(t, ts[-1]) for t in ts):
                raise SystemExit(f"non-float tensor {k} differs across checkpoints")
            sd[k] = ts[-1].contiguous()
    meta = handles[-1].metadata()
    save_file(sd, str(out / "model.safetensors"), metadata=meta)
    (out / "average.json").write_text(json.dumps({"inputs": [str(c) for c in ckpts], "tensors": len(keys)}, indent=1), encoding="utf-8")
    print(f"[average] {out}: mean of {len(ckpts)} checkpoints, {len(keys)} tensors")


if __name__ == "__main__":
    main()
