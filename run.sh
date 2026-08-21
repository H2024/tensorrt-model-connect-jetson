#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start an interactive, throwaway TRTMC container.
#
# Isolation notes:
#   --rm                 container disappears on exit
#   --name trtmc-lab     distinct name, will not collide with your other containers
#   -v trtmc-cache:...   a NAMED volume used only by this experiment
#   -v ./work:/work      the ONLY host path exposed, and it is a subfolder of this repo
#   --memory             a cgroup cap so a runaway build cannot OOM the whole board
#   default bridge net   not --network host
#   no --privileged, no --ipc=host, no host root mount, no writes to /usr/local/cuda
#
# Deliberately NOT used here:
#   -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu
#   -v /usr/local/cuda:/usr/local/cuda
# Those would drag the host's JetPack TensorRT 10.16 into a container built around
# TensorRT 11.1 and guarantee a symbol clash. The nvidia runtime already injects the
# driver, which is the only host piece that legitimately has to be shared.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-trtmc-orin:jp72}"
NAME="${NAME:-trtmc-lab}"
SM="${SM:-87}"
MEMORY="${MEMORY:-48g}"
mkdir -p work

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "a container named '${NAME}' already exists. Remove it or set NAME=..." >&2
    exit 1
fi

# GPU access. GPU_MODE=runtime (default) | cdi | gpus
#
# NVIDIA_VISIBLE_DEVICES is passed explicitly: the nvidia runtime treats "unset"
# exactly like runc and injects nothing at all. The image sets it too, but passing
# it here means an older image still works.
GPU_MODE="${GPU_MODE:-runtime}"
case "$GPU_MODE" in
    runtime)
        GPU_ARGS=(--runtime nvidia)
        if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
            echo "note: 'nvidia' docker runtime not found, falling back to --gpus all" >&2
            GPU_ARGS=(--gpus all)
        fi
        ;;
    cdi)
        # Requires: sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml
        GPU_ARGS=(--device nvidia.com/gpu=all)
        ;;
    gpus)
        GPU_ARGS=(--gpus all)
        ;;
    *)
        echo "unknown GPU_MODE=${GPU_MODE} (use runtime|cdi|gpus)" >&2; exit 1 ;;
esac
GPU_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)

exec docker run --rm -it \
    --name "$NAME" \
    "${GPU_ARGS[@]}" \
    --memory "$MEMORY" \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "TRTMC_SM=${SM}" \
    -e "HF_HOME=/cache/hf" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -v trtmc-cache:/cache \
    -v "$PWD/work:/work" \
    -w /work \
    "$IMAGE" \
    "${@:-bash}"
