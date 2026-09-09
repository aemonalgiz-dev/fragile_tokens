#!/bin/bash
# remote bootstrap: deps, layout, hardware check
set -u
mkdir -p ~/glitch && cd ~/glitch && tar xzf ~/glitch_sync.tgz && mkdir -p logs results
python3 -m pip install -q -U torch transformers accelerate safetensors scipy huggingface_hub 2>&1 | tail -1
echo "--- gpus ---"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--- disk ---"; df -h ~ | tail -1
echo "--- python ---"; python3 -c "import torch, transformers, accelerate; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'bf16', torch.cuda.is_bf16_supported())"
