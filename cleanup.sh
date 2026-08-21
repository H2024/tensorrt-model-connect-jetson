#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Remove every trace of this experiment. Nothing else is touched.
set -euo pipefail

IMAGE="${IMAGE:-trtmc-orin:jp72}"
NAME="${NAME:-trtmc-lab}"

echo "removing container ${NAME} (if running)"
docker rm -f "$NAME" 2>/dev/null || true

echo "removing image ${IMAGE}"
docker rmi "$IMAGE" 2>/dev/null || true

echo "removing volume trtmc-cache"
docker volume rm trtmc-cache 2>/dev/null || true

echo
echo "left alone on purpose: the ${IMAGE%%:*} build cache, the base ubuntu:24.04 image,"
echo "and the ./work folder. Reclaim build cache with:  docker builder prune"
