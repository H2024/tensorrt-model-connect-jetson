#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the isolated TRTMC experiment image.
#
# What this script touches on the host:
#   - the docker image store: adds ONE new image tagged $IMAGE (default trtmc-orin:jp72)
#   - the BuildKit layer cache under /var/lib/docker (disk space only)
#   - /tmp/.gh_token_$$, mode 0600, deleted on exit, ONLY if you set GITHUB_TOKEN
# What it does NOT touch:
#   - anything under /usr, /opt, /etc, /usr/local/cuda, the JetPack apt packages
#   - any existing container, volume, network, or image other than $IMAGE
#   - no sudo, no apt, no pip on the host - every install happens inside the build
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-trtmc-orin:jp72}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:24.04}"
SM="${SM:-87}"                       # Jetson AGX Orin = compute capability 8.7
TRT_VERSION="${TRT_VERSION:-11.1.0.106}"
TRTMC_RELEASE_TAG="${TRTMC_RELEASE_TAG:-latest}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
TRTMC_REQUIRED="${TRTMC_REQUIRED:-0}"   # 1 = fail the build if the wheel is not found
TRTMC_TRT_TAG="${TRTMC_TRT_TAG:-trt111}"  # must match TRT_VERSION: trt111 <-> 11.1.0.106
FORCE="${FORCE:-0}"

# --- guard 1: never silently overwrite an image tag you already use -----------
if docker image inspect "$IMAGE" >/dev/null 2>&1 && [ "$FORCE" != "1" ]; then
    echo "refusing to overwrite the existing image tag '${IMAGE}'." >&2
    echo "  rebuild it:      FORCE=1 $0" >&2
    echo "  or use another:  IMAGE=trtmc-orin:test $0" >&2
    exit 1
fi

# --- guard 2: show the docker state before and after, so you can diff it ------
snapshot() {
    echo "  images:     $(docker images -q | wc -l)"
    echo "  containers: $(docker ps -aq | wc -l)"
    echo "  volumes:    $(docker volume ls -q | wc -l)"
    echo "  disk:       $(docker system df --format '{{.Type}}={{.Size}}' 2>/dev/null | tr '\n' ' ')"
}
echo "== docker state BEFORE =="
snapshot

echo
echo "building ${IMAGE}  (base=${BASE_IMAGE}, SM=${SM}, TensorRT=${TRT_VERSION}, torch=${INSTALL_TORCH})"
echo

# GITHUB_TOKEN is optional; it only raises the GitHub API rate limit.
# Note the deliberate lack of --pull: if you already have a local ${BASE_IMAGE},
# it is reused as-is, so this build cannot re-point that tag at a newer digest
# and change what someone else's `docker build` would pick up later.
SECRET_ARGS=()
TOKEN_FILE=""
if [ -n "${GITHUB_TOKEN:-}" ]; then
    TOKEN_FILE="$(mktemp -t trtmc-gh-token.XXXXXX)"
    trap 'rm -f "$TOKEN_FILE"' EXIT
    chmod 600 "$TOKEN_FILE"
    printf '%s' "$GITHUB_TOKEN" > "$TOKEN_FILE"
    SECRET_ARGS=(--secret "id=gh_token,src=${TOKEN_FILE}")
fi

DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "SM=${SM}" \
    --build-arg "TRT_VERSION=${TRT_VERSION}" \
    --build-arg "TRTMC_RELEASE_TAG=${TRTMC_RELEASE_TAG}" \
    --build-arg "INSTALL_TORCH=${INSTALL_TORCH}" \
    --build-arg "TRTMC_REQUIRED=${TRTMC_REQUIRED}" \
    --build-arg "TRTMC_TRT_TAG=${TRTMC_TRT_TAG}" \
    "${SECRET_ARGS[@]}" \
    -t "${IMAGE}" \
    .

echo
echo "== docker state AFTER =="
snapshot
echo
echo "the only new tag should be: ${IMAGE}"
docker images "${IMAGE%%:*}"
echo
echo "built ${IMAGE}. Next: ./run.sh"
