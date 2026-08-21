# syntax=docker/dockerfile:1.7
#
# TensorRT Model Connect (TRTMC) experiment image for Jetson AGX Orin 64GB
# Host assumption: JetPack 7.2 / Jetson Linux r39.2, Orin on the SBSA CUDA path.
#
# Design goal: touch NOTHING on the host and nothing in your other containers.
#   * Base is plain Ubuntu 24.04 arm64 (glibc 2.39 -> satisfies TRTMC's >=2.39 rule).
#   * CUDA 13 userspace + TensorRT 11.1.0.106 come from pip wheels INSIDE the image,
#     so the host's JetPack TensorRT 10.16 is never read, linked, or modified.
#   * The only thing borrowed from the host is the GPU kernel driver, injected at
#     runtime by the NVIDIA container runtime. That is read-only and shared safely.
#
# THIS IS EXPERIMENTAL. NVIDIA's TensorRT docs state JetPack is not a supported
# platform for TensorRT 11.x. This image exists to find out empirically whether the
# SBSA aarch64 wheels load and emit SM 8.7 kernels on Orin.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# TensorRT cohort required by the TRTMC release wheels. Do not change one without
# the other: TRTMC refuses to mix TensorRT cohorts.
ARG TRT_VERSION=11.1.0.106
# Orin = compute capability 8.7. Thor would be 110, AGX Xavier 72.
ARG SM=87
# Optional: pin a specific TRTMC release tag instead of "newest with a matching asset".
ARG TRTMC_RELEASE_TAG=latest
# Set to 1 to also install a CUDA 13 PyTorch build (only needed if TRTMC's build path
# wants torch to read HF checkpoints; leave off first, add it if the build asks for it).
ARG INSTALL_TORCH=0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV=/opt/venv-trtmc \
    TRTMC_SM=${SM} \
    HF_HOME=/cache/hf \
    XDG_CACHE_HOME=/cache/xdg

# CRITICAL for a non-NVIDIA base image. The nvidia container runtime decides whether
# to inject the driver by reading NVIDIA_VISIBLE_DEVICES *from the image or the
# command line*. Per NVIDIA's docs, unset/empty/"void" means the runtime "has the
# same behavior as runc" - it silently runs as a plain container with no GPU at all.
# CUDA base images set this for you; ubuntu:24.04 obviously does not, which is why
# --runtime nvidia appeared to do nothing.
# capabilities: compute+utility only. "all" pulls in `graphics`, which has a known
# breakage on Jetson with container-toolkit 1.17.1+.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        unzip \
        libgomp1 \
        python3.12 \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv. Python 3.12 is one of the two interpreter versions TRTMC publishes
# wheels for (3.10 is the other).
RUN python3.12 -m venv "$VENV"
ENV PATH="/opt/venv-trtmc/bin:${PATH}"
RUN pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# Gate 1: TensorRT 11.1 for CUDA 13 on linux aarch64.
# If NVIDIA has not published an aarch64/SBSA wheel for this cohort, the build
# stops right here and the whole experiment is answered in 60 seconds.
# ---------------------------------------------------------------------------
RUN pip install --extra-index-url https://pypi.nvidia.com "tensorrt-cu13==${TRT_VERSION}"

# The CUDA runtime does NOT arrive with TensorRT, despite what the dependency graph
# claims. tensorrt_cu13_libs requires `nvidia-cuda-runtime-cu13`, which NVIDIA
# deprecated into a 1.4 kB stub that ships no libraries at all - so pip resolves the
# dependency "successfully" and libcudart.so.13 never lands on disk. See
# https://github.com/NVIDIA/TensorRT/issues/4614. `nvidia-cuda-runtime` is the real
# package. Pinned to the 13.2 series to match the JetPack 7.2 driver.
ARG CUDA_RUNTIME_VERSION=13.2.86
RUN pip install --extra-index-url https://pypi.nvidia.com \
        "nvidia-cuda-runtime==${CUDA_RUNTIME_VERSION}" \
 && echo "== libcudart on disk ==" \
 && (find "$VENV" -name 'libcudart.so*' -print || echo "STILL MISSING")
RUN echo "== resolved nvidia/tensorrt packages ==" ; \
    pip list --format=freeze | grep -Ei 'tensorrt|nvidia|cudnn|cublas' || true
# NOTE: no `import tensorrt` here on purpose. The driver is not present during
# `docker build`, so the import check belongs in verify.sh at runtime.

# Handy extras for pulling checkpoints.
RUN pip install "huggingface_hub[cli]" numpy

# ---------------------------------------------------------------------------
# Gate 2: the TRTMC wheel itself (an aarch64 asset from GitHub releases).
#
# NON-FATAL BY DEFAULT. Gate 1 above is the scientifically interesting one; if the
# wheel cannot be found, you still get an image that can answer "does TensorRT 11.1
# emit SM 8.7 kernels on Orin?" Re-run `install-trtmc` inside the container to retry
# without a rebuild, or set TRTMC_REQUIRED=1 to make the build fail here instead.
# ---------------------------------------------------------------------------
ARG TRTMC_REQUIRED=0
# Wheel cohort marker. Must match TRT_VERSION above: trt111 <-> 11.1.0.106,
# trt112 <-> 11.2.x. Mixing cohorts is explicitly unsupported by TRTMC.
ARG TRTMC_TRT_TAG=trt111

# Drop a .whl into ./wheels/ on the host to bypass the download entirely.
# The .keep file is what makes this COPY valid when the folder is empty.
COPY wheels/ /opt/local-wheels/
COPY docker/fetch-trtmc-wheel.sh /usr/local/bin/fetch-trtmc-wheel
COPY docker/install-trtmc.sh /usr/local/bin/install-trtmc
RUN chmod +x /usr/local/bin/fetch-trtmc-wheel /usr/local/bin/install-trtmc

RUN --mount=type=secret,id=gh_token,required=false \
    if GITHUB_TOKEN="$(cat /run/secrets/gh_token 2>/dev/null || true)" \
       TRTMC_RELEASE_TAG="${TRTMC_RELEASE_TAG}" \
       TRTMC_TRT_TAG="${TRTMC_TRT_TAG}" \
       install-trtmc ; then \
        echo "TRTMC wheel installed." ; \
    else \
        echo "############################################################" ; \
        echo "# TRTMC wheel NOT installed (see the listing above)." ; \
        echo "# The image is still useful:" ; \
        echo "#   ./run.sh verify   -> tests TensorRT 11.1 on SM 87" ; \
        echo "#   install-trtmc     -> retry the fetch inside the container" ; \
        echo "############################################################" ; \
        if [ "${TRTMC_REQUIRED}" = "1" ]; then exit 1 ; fi ; \
    fi

# Optional torch (CUDA 13 SBSA build). Off by default - it is a ~3GB download and
# TRTMC's whole point is not needing torch at runtime.
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cu130 torch ; \
    fi

# ---------------------------------------------------------------------------
# Make the pip-installed CUDA/TensorRT shared objects visible to the DYNAMIC
# LINKER, not just to Python.
#
# The pip wheels drop their .so files under site-packages/nvidia/... and
# site-packages/tensorrt_libs/. `import tensorrt` works because the package
# preloads them at import time - but the standalone C++ `trtmc` binary gets no
# such help, which is why it failed with "libcudart.so.13: cannot open shared
# object file". Registering the directories with ldconfig fixes every consumer.
# ---------------------------------------------------------------------------
RUN find "$VENV" -name '*.so*' \( -path '*nvidia*' -o -path '*tensorrt*' \) -printf '%h\n' \
        | sort -u > /etc/ld.so.conf.d/000-trtmc-pip.conf \
 && echo "== registered library dirs ==" && cat /etc/ld.so.conf.d/000-trtmc-pip.conf \
 && ldconfig \
 && echo "== sanity ==" \
 && (ldconfig -p | grep -E 'libcudart|libnvinfer' || echo "WARNING: still not in the cache")

# Belt and braces for anything that bypasses the ldconfig cache.
ENV LD_LIBRARY_PATH=/opt/venv-trtmc/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv-trtmc/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}

COPY docker/verify.sh /usr/local/bin/verify
COPY tools/trt_smoke.py /usr/local/bin/trt_smoke.py
COPY tools/trtinfo.py /usr/local/bin/trtinfo
RUN chmod +x /usr/local/bin/verify /usr/local/bin/trt_smoke.py /usr/local/bin/trtinfo

# Scratch + cache live on mounted volumes so the image itself stays disposable.
RUN mkdir -p /work /cache/hf /cache/xdg
WORKDIR /work

CMD ["bash"]
