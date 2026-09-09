#!/bin/bash
# remote bootstrap: layout, isolated python env, hardware check.
# Lambda Stack ships system torch/torchvision/sklearn/pandas in dist-packages that
# break one after another once a newer torch lands in user site -- so everything
# goes into ~/venv and drivers run with PATH=$HOME/venv/bin:$PATH.
set -u
mkdir -p ~/glitch && cd ~/glitch && tar xzf ~/glitch_sync.tgz && mkdir -p logs results results/compact docs
if [ ! -x "$HOME/venv/bin/python3" ]; then
  python3 -m venv ~/venv 2>/dev/null || {
    # container images (RunPod, Vast) often lack the venv module; root there, no sudo needed
    (command -v sudo >/dev/null && sudo apt-get install -y -q python3-venv || apt-get install -y -q python3-venv) >/dev/null 2>&1
    python3 -m venv ~/venv; }
fi
export PATH=$HOME/venv/bin:$PATH
python3 -m pip install -q -U pip 2>&1 | tail -1
python3 -m pip install -q -U torch transformers accelerate safetensors scipy numpy huggingface_hub 2>&1 | tail -1
# FP8 checkpoints under transformers >= 5 need the hub-kernels client at exactly this minor
python3 -m pip install -q "kernels==0.16.0" 2>&1 | tail -1
# transformers 5.16.1: update_tp_plan dereferences None for dense FP8 models -- guard it
F=$(python3 -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'quantizers', 'quantizer_finegrained_fp8.py'))")
sed -i "s/layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)$/layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl) or {}  # patched: None for dense models/" "$F"
echo "--- gpus ---"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--- disk ---"; df -h ~ | tail -1
echo "--- python ---"; python3 -c "import torch, transformers, accelerate; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'bf16', torch.cuda.is_bf16_supported(), 'cc', torch.cuda.get_device_capability(0))"
python3 -m py_compile src/cut/fragility.py src/cut/fragility_predict.py src/cut/reasoning_drift.py src/cut/specimens.py src/cut/interaction_screen.py src/cut/gate_summary.py src/cut/compact_pt.py src/cut/loadmodel.py && echo "--- compiles ---"
