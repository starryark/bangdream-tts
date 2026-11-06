#!/bin/bash
# E3 (pod): Seed-VC in its own venv (it pins torch 2.4 / transformers 4.46; the training env
# must not change). Tool use only (GPL-3.0); outputs stay on the pod / local machine.
set -euo pipefail
cd /workspace
[ -d seed-vc ] || git clone -q https://github.com/Plachtaa/seed-vc.git
(cd seed-vc && git log -1 --format="seed-vc %h %ci")
[ -x vcenv/bin/python ] || python -m venv vcenv
P=/workspace/vcenv/bin/pip
$P install -q --upgrade pip
$P install -q torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
$P install -q scipy==1.13.1 librosa==0.10.2 "huggingface-hub>=0.28.1" munch==4.0.0 einops==0.8.0 \
    descript-audio-codec==1.0.0 pydub==0.25.1 transformers==4.46.3 soundfile==0.12.1 numpy==1.26.4 \
    hydra-core==1.3.2 pyyaml python-dotenv accelerate
/workspace/vcenv/bin/python -c "import torch, transformers, librosa; print('vcenv ok', torch.__version__, transformers.__version__, torch.cuda.is_available())"
echo VCENV_OK
