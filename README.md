# TensorRT Model Connect on Jetson AGX Orin

Run **TensorRT 11.1** and **[TensorRT Model Connect][trtmc]** on a Jetson AGX Orin 64GB under
**JetPack 7.2**, in a container that leaves the host JetPack stack completely untouched.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Jetson%20AGX%20Orin%20%7C%20JetPack%207.2-76B900.svg)](#compatibility)
[![Status](https://img.shields.io/badge/status-works%2C%20unsupported%20by%20NVIDIA-orange.svg)](#the-caveat-that-matters)

> [!WARNING]
> **NVIDIA does not support this.** The TensorRT installation guide states plainly that
> JetPack is not a supported platform for TensorRT 11.x and that *"Jetson deployments must
> remain on a TensorRT 10.x release supported by their JetPack version."* This recipe works
> anyway - see [why](#why-this-works-at-all) - but you are off the supported path. Do not
> ship production robotics on it without understanding [what you give up](#known-limitations).

---

## Contents

- [What this is](#what-this-is)
- [Compatibility](#compatibility)
- [Why this works at all](#why-this-works-at-all)
- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Verify](#verify)
- [Use TRTMC](#use-trtmc)
- [Verified results](#verified-results)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [How it stays isolated](#how-it-stays-isolated)
- [Alternatives](#alternatives)
- [Report your results](#report-your-results)
- [License and credits](#license-and-credits)

---

## What this is

JetPack 7.2 ships **TensorRT 10.16**. TensorRT Model Connect's release wheels require
**TensorRT 11.1**. That single version gap is the whole problem, and this repository is the
resolution: a container that installs TensorRT 11.1 and TRTMC from pip wheels *inside* the
image, never reading or modifying JetPack's own CUDA and TensorRT installation.

You get:

| File | What it does |
|---|---|
| `Dockerfile` | `ubuntu:24.04` + pip CUDA 13 + TensorRT 11.1 + the TRTMC wheel |
| `build.sh` | Build wrapper with an image-overwrite guard and a before/after Docker state diff |
| `run.sh` | Throwaway container with the GPU flags that actually work on JetPack 7 |
| `docker/verify.sh` | Staged PASS/FAIL smoke test - the honest answer to "does this work here?" |
| `tools/trt_smoke.py` | Builds real fp32/fp16/bf16 engines against the TensorRT 11 API |
| `tools/trtinfo.py` | Platform, driver, CUDA and TensorRT report; runs on host **and** in container |

Everything is stdlib, ctypes and shell. No PyTorch, no pycuda, no cuda-python required to run the diagnostics.

## Compatibility

| | Status |
|---|---|
| **Verified on** | Jetson AGX Orin 64GB · JetPack 7.2 / L4T R39.2.0 · kernel 6.8.12-tegra · Ubuntu 24.04 · CUDA driver 13.2 · SM 8.7 |
| **Should work** | Jetson Orin NX / Orin Nano on JetPack 7.2 (same SM 8.7, less memory) - untested, [please report](#report-your-results) |
| **Should work** | Jetson AGX Thor on JetPack 7.x - SBSA native, SM 11.0; set `SM=110` |
| **Will not work** | JetPack 6.x or earlier. Orin only joined the SBSA CUDA path in JetPack 7.2; before that, the aarch64 wheels have nothing to bind to |
| **Will not work** | x86_64 - TRTMC publishes aarch64 release wheels only; x86 users must build from source |

## Why this works at all

Three facts have to line up, and as of JetPack 7.2 they finally do.

**1. Orin moved onto SBSA.** Historically Jetson ran a Tegra-specific CUDA stack that
generic aarch64 wheels could not target. JetPack 7.2 puts Orin on the Server Base System
Architecture path with CUDA 13.2 - the same target that datacenter ARM wheels are built
for. This is the change that makes everything below possible.

**2. The wheels are aarch64 and glibc-compatible.** TRTMC publishes
`…-py312-none-manylinux_2_39_aarch64.whl`. JetPack 7.2's Ubuntu 24.04 userspace *is*
glibc 2.39. The match is exact, not approximate.

**3. TensorRT 11 still has SM 8.7 kernels.** The support matrix lists compute capability
8.7 (Jetson AGX Orin) among supported architectures. NVIDIA's *installation* docs exclude
JetPack as a platform, but the kernels for Orin's GPU are in the build - which is why
containerising around the platform restriction works.

What NVIDIA is really saying with "not supported" is that they do not test or ship this
combination, not that it is technically impossible. Those are different claims, and this
repository documents the gap between them.

## Quick start

```bash
git clone https://github.com/H2024/tensorrt-model-connect-jetson
cd tensorrt-model-connect-jetson
chmod +x *.sh

screen -S trtmc          # important: see prerequisite 3 below
./build.sh
./run.sh verify
```

If `verify` is all green, you have TensorRT 11.1 building engines on Orin. Jump to
[Use TRTMC](#use-trtmc).

## Prerequisites

**1. JetPack 7.2 on a Jetson Orin.** Check:

```bash
cat /etc/nv_tegra_release      # expect "# R39 (release), REVISION: 2.x"
head -n1 /etc/os-release       # expect Ubuntu 24.04
```

JetPack 7.2 is a full platform reset from JetPack 6 (Ubuntu 22.04 → 24.04), so nothing
carries over from an older rootfs.

**2. NVIDIA Container Toolkit, working.**

```bash
docker info | grep -i -A3 runtimes    # expect an 'nvidia' runtime
```

If that comes back empty, fix it before going further. No container flag substitutes for a
missing runtime, and no bind-mount workaround is a substitute either - see
[pitfall 7](docs/troubleshooting.md#7-never-bind-mount-host-system-libraries).

**3. A terminal multiplexer.** BuildKit **cancels the build when its CLI client
disconnects**, so a dropped SSH session kills a 20-minute build. Use `screen`/`tmux`, or:

```bash
setsid nohup ./build.sh > build.log 2>&1 &
tail -f build.log
```

**4. About 10 GB free** in `/var/lib/docker` - `df -h /var/lib/docker`, `docker system df`.

## Install

```bash
./build.sh
```

The build has two deliberate gates, and *where* it stops is diagnostic information:

- **Gate 1 - `pip install tensorrt-cu13==11.1.0.106`.** If no aarch64 wheel exists for that
  cohort, you learn it in about a minute and the wheel path is closed.
- **Gate 2 - the TRTMC wheel.** Non-fatal by default: if discovery fails you still get an
  image that can answer the TensorRT question, and `install-trtmc` retries inside the
  running container without a rebuild.

Build options:

```bash
FORCE=1 ./build.sh                                    # overwrite an existing image tag
SM=110 ./build.sh                                     # Thor instead of Orin
TRT_VERSION=11.2.1.2 TRTMC_TRT_TAG=trt112 ./build.sh  # the other TensorRT cohort
TRTMC_RELEASE_TAG=trtmc-nightly-… ./build.sh          # pin an exact TRTMC release
TRTMC_WHEEL_URL="https://…/….whl" ./build.sh          # skip discovery entirely
cp yourwheel.whl wheels/ && ./build.sh                # offline / air-gapped
GITHUB_TOKEN=ghp_… ./build.sh                         # avoid GitHub API rate limits
INSTALL_TORCH=1 ./build.sh                            # add CUDA 13 PyTorch (~3 GB)
TRTMC_REQUIRED=1 ./build.sh                           # make gate 2 fatal again
```

> [!TIP]
> Docker images are immutable - a rebuild creates a new image and moves the tag, orphaning
> the old one. Before rebuilding, `docker tag trtmc-orin:jp72 trtmc-orin:jp72-prev` keeps a
> rollback and an A/B baseline instead of a dangling `<none>` image.

## Verify

```bash
./run.sh verify
```

| Stage | What it proves |
|---|---|
| 0 | aarch64, glibc ≥ 2.39, Python 3.12 - the wheel's own requirements |
| 1 | The nvidia runtime actually injected a driver (device nodes + `libcuda` + `cuDeviceGetCount`) |
| 1b | `libcudart.so.13` is resolvable by the **dynamic linker**, not just by Python |
| 2 | `import tensorrt` succeeds |
| 3 | **The real test** - builds and deserializes fp32/fp16/bf16 engines on this GPU |
| 4 | The `trtmc` binary runs, not merely exists on `PATH` |

Stage 3 is the one that matters. A version string proves nothing; an engine that builds and
round-trips proves TensorRT emitted kernels for your silicon.

For a fuller picture of the stack:

```bash
./run.sh trtinfo --probe          # inside the container: TensorRT 11.1
python3 tools/trtinfo.py --probe  # on the host: JetPack's TensorRT 10.16
```

`trtinfo` reports the Jetson model, L4T release, CUDA driver, GPU compute capability and
what precisions the *silicon* supports, then the TensorRT version, its backing `.so` files,
DLA core count and plugin registry. Diff the two stacks:

```bash
python3 tools/trtinfo.py --json > host.json
./run.sh trtinfo --json > container.json
diff <(jq -S . host.json) <(jq -S . container.json)
```

## Use TRTMC

```bash
./run.sh bash

trtmc build Qwen/Qwen3-0.6B --precision bf16 --max-cache-length 16384 --output /work/qwen3-0.6b.bundle
trtmc inspect /work/qwen3-0.6b.bundle
trtmc run /work/qwen3-0.6b.bundle --prompt "What is the capital of France? Answer in one word."
```

Two habits worth forming on this platform:

**Run `inspect` before `run`.** It reports the bundle's declared runtime dependencies,
which is where a profile expecting FP8 or FP4 hardware announces itself - before it fails
confusingly at execution on a GPU that has neither.

**Stick to bf16 / fp16 / int8.** TRTMC's headline profiles assume Blackwell low-precision
paths. Orin is SM 8.7: fp16 and bf16 yes, fp8 and fp4 never. Expect a usable subset of the
76 model families, not all of them.

`/work` is the repo's `work/` directory, so bundles survive container exit.

## Verified results

Measured on Jetson AGX Orin 64GB / JetPack 7.2. Full detail and reproduction steps in
[docs/results.md](docs/results.md).

| Test | Result | Evidence |
|---|---|---|
| `import tensorrt` 11.1.0.106 | works | - |
| fp32 engine build + deserialize | works | 12,876 bytes |
| fp16 engine build + deserialize | works | 9,780 bytes |
| bf16 engine build + deserialize | works | 9,588 bytes |
| `trtmc` CLI, `TRT support` | works | 0.1.0+trt111 |
| DLA cores visible to TensorRT | **lost** | reports 0 of 2 |
| fp8 / fp4 | **n/a** | no SM 8.7 hardware |

The engines *shrink* as precision drops, which is how you know TensorRT genuinely selected
fp16 and bf16 kernels rather than silently falling back to fp32.

## Known limitations

**You lose both DLAs.** Orin's deep learning accelerators appear as
`/dev/nvhost-ctrl-nvdla0` and `…nvdla1` inside the container, yet TensorRT 11.1 reports
`num_DLA_cores = 0`; JetPack's TensorRT 10.16 reports 2. The SBSA build appears to have no
Tegra DLA backend. If your workload uses DLA, this route is the wrong one.

**No fp8 or fp4.** SM 8.7 hardware limitation, not a software gap. No TensorRT version
changes it.

**TRTMC is a public preview.** Version 0.1.0, shipped as nightly prereleases. The API,
bundle format and profile list will move.

**Unsupported means unsupported.** No NVIDIA support channel will help you with this
configuration, and a future JetPack or TensorRT release can break it without notice.

**Untested beyond one board.** Everything here is verified on exactly one Jetson AGX Orin
64GB. Please [report what you find](#report-your-results) on other hardware.

## Troubleshooting

Seven traps, each of which cost real debugging time. Full symptom-indexed detail in
[docs/troubleshooting.md](docs/troubleshooting.md).

| Symptom | Cause |
|---|---|
| Build dies when SSH drops | BuildKit cancels on client disconnect - build in `screen` |
| No matching wheel found | Assets are tagged `py312`, not `cp312`; `trt111` and `trt112` cohorts both exist |
| `libcuda.so.1: cannot open shared object file` | `NVIDIA_VISIBLE_DEVICES` unset → the nvidia runtime behaves exactly like `runc` |
| `NVIDIA_VISIBLE_DEVICES=void` inside the container | Not an error - the runtime rewrites it after injecting |
| `trtmc: libcudart.so.13: cannot open shared object file` | Upstream bug: TensorRT depends on a deprecated empty stub package |
| `AttributeError: … 'platform_has_fast_fp16'` | Removed in TensorRT 11.0 along with weak typing and `IPluginV2` |
| GPU works but the host feels broken afterwards | Host library bind-mounts are read-write; the runtime writes symlinks into them |

## How it stays isolated

The design goal is that running this cannot disturb your JetPack install or any other
container. See [docs/how-it-works.md](docs/how-it-works.md) for the full accounting -
including what *is* genuinely shared, because "cannot affect anything" would be
overclaiming.

In short: stock `ubuntu:24.04` base, never `l4t-*`; all CUDA and TensorRT userspace inside
`/opt/venv-trtmc`; no `sudo`, `apt` or `pip` on the host; no `--pull`, so your local base
image is never re-pointed at a new digest; one host path mounted (`work/`); named volume for
caches. Removal is `./cleanup.sh`.

## Alternatives

Be honest with yourself about whether you need TRTMC specifically:

- **[TensorRT Edge-LLM][edgellm]** - NVIDIA's supported C++ LLM/VLM runtime for embedded,
  documented for the Orin family on JetPack 7.2. If you want fast local inference rather
  than TRTMC in particular, start here.
- **JetPack's own TensorRT 10.16** - supported, DLA-capable, already installed. For most
  Jetson inference work this remains the right answer.
- **[jetson-containers][jc]** - tested containers for JetPack 6.2 and 7, including Ubuntu
  24.04 CUDA images.
- **TRTMC from source** - the repo ships `Dockerfile.dev.aarch64` and builds per-SM, which
  compiles kernels for exactly SM 8.7. More work, closer to NVIDIA's own recipe.

## Report your results

The most useful contribution is a data point from hardware that isn't mine. Open a
[result report](.github/ISSUE_TEMPLATE/result-report.yml) with the output of:

```bash
./run.sh trtinfo --json
./run.sh verify
```

Particularly wanted: Orin NX / Orin Nano, Thor, JetPack 7.2.1 and later, and anyone who
gets DLA working - that last one would change a documented limitation into a solved problem.

## License and credits

Apache-2.0. See [LICENSE](LICENSE).

This is an independent community guide. It is not affiliated with or endorsed by NVIDIA.
NVIDIA, Jetson, Orin, JetPack, CUDA and TensorRT are trademarks of NVIDIA Corporation.

Sources and further reading:

- [TensorRT Model Connect][trtmc] - [system requirements][trtmc-req] · [installation][trtmc-install] · [source build][trtmc-src]
- [TensorRT 11.0 release notes - removed APIs][trt11]
- [TensorRT support matrix][trtmatrix] · [installation guide][trtinstall]
- [NVIDIA_VISIBLE_DEVICES and driver capabilities][ndv]
- [NVIDIA/TensorRT#4614 - deprecated `nvidia-cuda-runtime-cu13`][issue4614]
- [JetPack SDK downloads and component versions][jetpack]

[trtmc]: https://github.com/NVIDIA/TensorRT-Model-Connect
[trtmc-req]: https://nvidia.github.io/TensorRT-Model-Connect/getting-started/environment-and-repro
[trtmc-install]: https://nvidia.github.io/TensorRT-Model-Connect/getting-started/installation
[trtmc-src]: https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build
[trt11]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html
[trtmatrix]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
[trtinstall]: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
[ndv]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
[issue4614]: https://github.com/NVIDIA/TensorRT/issues/4614
[jetpack]: https://developer.nvidia.com/embedded/jetpack/downloads
[edgellm]: https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
[jc]: https://github.com/dusty-nv/jetson-containers




