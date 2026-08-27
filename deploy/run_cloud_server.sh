#!/bin/bash
set -e

echo "=========================================================================="
echo "          VISUOMOTOR HAND POLICY REMOTE SERVER INITIALIZATION             "
echo "=========================================================================="

# Check for NVIDIA GPU / CUDA availability
echo "[INFO] Checking Hardware Acceleration Environment..."

if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA Driver detected via nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "[WARNING] nvidia-smi not found. Checking PyTorch hardware probes..."
fi

# PyTorch CUDA verification probe
python3 -c "
import torch
print(f'[PROBE] PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'[PROBE] CUDA Available: True | Device Count: {torch.cuda.device_count()}')
    print(f'[PROBE] Active Device: {torch.cuda.get_device_name(0)}')
    print(f'[PROBE] CUDA Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('[PROBE] CUDA Available: False -> Running in optimized CPU Fallback Mode.')
"

# Parse custom environment variables or flags
PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"
CONFIG="${CONFIG:-config/system_config.yaml}"

echo "=========================================================================="
echo "[INFO] Launching WebSocket Inference Server on ws://${HOST}:${PORT}"
echo "=========================================================================="

exec python3 apps/remote_server.py --host "${HOST}" --port "${PORT}" --config "${CONFIG}"
