This document helps reviewers build the artifact from source code, understand its code footprint, and run the included tests and evaluation scripts.

## Introduction

This artifact supports approximate nearest-neighbor search experiments for the RIVER paper. It implements MPMI as a common compact scalar-quantizer representation, provides an RVV-specialized tile-wise recovery and distance path, and implements ROrder as a post-build HNSW layout pass. The regular Knowhere HNSW-SQ build path accepts `sq_type=mpmi` and `graph_layout=rorder`; standalone programs additionally exercise the paper-facing S/V/M/F configurations, numerical agreement, and graph-layout baselines.

## Artifact Notes

This artifact includes the optimized implementation, patches, and test programs used for evaluation. The SLOC counting policy excludes third-party code under `thirdparty/` and counts nonblank, non-comment source lines across the released artifact. Under this policy, the artifact contains 58,082 SLOC: 54,460 lines of core implementation code, and 3,622 lines of build, configuration, CI, scripts, and artifact metadata.

The main artifact-related files are:

- `thirdparty/faiss/faiss/impl/ScalarQuantizer.h`, `ScalarQuantizer.cpp`, and `IndexScalarQuantizer.cpp`: define the MPMI type, offline residual selection, compact row format, and dynamic code size.
- `thirdparty/faiss/faiss/impl/ScalarQuantizerCodec_rvv.h` and `ScalarQuantizerDC_rvv.cpp`: widen the dense base and packed FP16 residuals within each recovery tile before RVV distance accumulation. Other CPU backends use the common MPMI representation with the existing dispatched FP32 distance primitive.
- `thirdparty/faiss/faiss/IndexHNSW.h` and `IndexHNSW.cpp`: keep `perm[new_id]=old_id` and support unordered, BFS, RCM, RabbitOrder, GOrder-only, and full ROrder layouts. ROrder uses the same static topology-derived global permutation as GOrder and additionally normalizes level-0 adjacency order.
- `src/index/refine/refine_utils.cc`, `src/index/hnsw/faiss_hnsw_config.h`, and `faiss_hnsw.cc`: expose MPMI and the static layout choice through the normal Knowhere HNSW-SQ build path, preserve original IDs for search and filtering, and serialize the mapping used by Milvus-facing calls.
- `tests/rvanns/test_mpmi_cq1.cpp`: runs S=Scalar+FP32, V=SIMD+FP32, M=SIMD+MPMI, and F=MPMI+ROrder while reporting the effective backend, quantizer, layout, and RVV LMUL settings.
- `tests/rvanns/test_mpmi_correctness.cpp`, `test_hnsw_reorder_micro.cpp`, and `test_knowhere_mpmi_rorder.cpp`: check MPMI numerical agreement, low-level graph-layout contracts, and the normal Knowhere build/search/filter/serialization path.
- `patches/mpmi.patch`: records the MPMI source delta separately from the complete artifact patch delivered with the release.
- `scripts/`, `ci/`, `CMakeLists.txt`, and `conanfile.py`: provide dependency installation, build, coverage, and CI entry points used by the artifact.

The paper-facing standalone programs cover:

- HNSW baseline sweeps over datasets and `efSearch` values.
- HNSW batch-search performance runs with perf counter collection.
- the four S/V/M/F paths with quantizer- and layout-qualified index caches;
- MPMI portable-reference versus selected-backend distance checks for L2 and inner product; and
- unordered, BFS, RCM, RabbitOrder, GOrder, and ROrder under one HNSW query path.

Each graph layout is a one-shot post-build choice. The direct-Faiss experiment
drivers cache the unordered index, reapply the selected layout after loading,
and use `perm[new_id]=old_id` to return results to the original ID space because
the added mapping is not part of upstream Faiss serialization. The normal
Knowhere HNSW-SQ `Build` path instead stores that mapping in its existing index
wrapper header, so a serialized ROrder index preserves Milvus-visible IDs and
bitset semantics after loading. Both paths reject adding vectors after layout;
rebuild from the pre-layout data when the graph must be extended. The normal
Knowhere hook is deliberately limited to one non-refined, non-partitioned
HNSW-SQ index, matching the paper's static post-build evaluation path. Invoke
the combined `Build` API for a non-`unordered` layout; a separate `Train` then
`Add` sequence intentionally remains growable and does not trigger the pass.
The existing `trace_visit` debugging output is rejected after layout because
its records use Faiss-internal IDs rather than the persisted external-ID map.

The regular unit tests are built when `with_ut=True`/`WITH_UT=ON` is enabled and run through the generated unit-test binary under `tests/ut/`. Faiss tests can be enabled with `with_faiss_tests=True`/`WITH_FAISS_TESTS=ON`. The Python tests live under `tests/python/`, and CI/E2E coverage is described by the Jenkins/Groovy files under `ci/`.

## System Requirements

All Linux distributions are available for development. However, the artifact has primarily been exercised on Ubuntu/CentOS-style Linux environments, with additional Mac coverage for both x86_64 and Apple Silicon.

Here is the verified OS list used by the upstream build configuration:

- Ubuntu 20.04 x86_64
- Ubuntu 20.04 Aarch64
- MacOS (x86_64)
- MacOS (Apple Silicon)

## Building The Artifact From Source Code

#### Install Dependencies

```bash
$ sudo apt install build-essential libopenblas-openmp-dev libaio-dev python3-dev python3-pip
$ pip3 install conan==1.61.0 --user
$ export PATH=$PATH:$HOME/.local/bin
```

#### Build From Source Code

* Ubuntu 20.04

```bash
$ mkdir build && cd build
#add conan remote
$ conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
#DEBUG CPU
$ conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Debug
#RELEASE CPU
$ conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#DEBUG GPU
$ conan install .. --build=missing -o with_ut=True -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s build_type=Debug
#RELEASE GPU
$ conan install .. --build=missing -o with_ut=True -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s build_type=Release
#DISKANN SUPPORT
$ conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Debug/Release
#build with conan
$ conan build ..
#verbose
export VERBOSE=1
```

* MacOS

```bash
#RELEASE CPU
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libc++ -s build_type=Release
#DEBUG CPU
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libc++ -s build_type=Debug
#build with conan
conan build ..
```

#### Build and run the RIVER checks

Enable the standalone artifact targets through the Conan option when creating a
Release build:

```bash
$ conan install .. --build=missing -o with_river_artifact_tests=True \
    -s compiler.libcxx=libstdc++11 -s build_type=Release
$ conan build ..
$ ctest --output-on-failure -R '^river_'
```

The four registered tests are self-contained: `river_mpmi_correctness`
validates the compact row format and selected distance backend;
`river_knowhere_mpmi_rorder` validates the normal Knowhere build, original-ID
filtering, and serialization round trip; `river_hnsw_layout_contract` validates
all six low-level layout paths; and `river_hnsw_layout_singleton` covers the
one-node GOrder/ROrder edge case. The dataset-driven S/V/M/F program is built as `test_mpmi_cq1`; run
`test_mpmi_cq1 --help` for its arguments.

On a RISC-V build, CMake applies
`-march=rv64gcv_zvfhmin -mabi=lp64d`; the Release configuration adds `-O3`.
The RVV MPMI defaults are decode `LMUL=4` and accumulation `LMUL=2`. Sensitivity
builds can override `RVANNS_RVV_DECODE_LMUL`, `RVANNS_RVV_ACC_LMUL`, and
`RVANNS_RVV_LMUL_TAG` as CMake cache variables.

The checked-in paper-facing configuration selects `ceil(1% * D)` FP32 residual
dimensions, then `ceil(10% * D)` disjoint FP16 residual dimensions, and uses a
GOrder layout window of 5. The S/V/M/F driver reproduces these defaults. The
other residual-budget and layout-window points in the paper are separate
source-level sensitivity builds, not runtime options in this minimal artifact
driver; change the constants in `ScalarQuantizer.cpp` and `IndexHNSW.cpp`,
respectively, and rebuild before regenerating those points.

#### Running Unit Tests

```bash
# in build directories
#Debug
$ ./Debug/tests/ut/*_tests
#Release
$ ./Release/tests/ut/*_tests
```

#### Clean up

```bash
$ git clean -fxd
```

## GEN PYTHON WHEEL(NEED RELEASE BUILD)

install dependency:

```
sudo apt install swig python3-dev
pip3 install bfloat16
```

after building the artifact:

```bash
cd python
python3 setup.py bdist_wheel
```

install the generated wheel:

```bash
pip3 install dist/*.whl
```

clean

```bash
cd python
rm -rf build
rm -rf dist
rm -rf *.egg-info
```

## Contributing

### Pre-Commit

Before submitting a pull request, please make sure running pre-commit checks locally to ensure the code is ready for review. Use the following command to install pre-commit checks:

```bash
pip3 install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push

# If clang-format and clang-tidy not already installed:
# linux
apt install clang-format clang-tidy
# mac
brew install llvm
ln -s "$(brew --prefix llvm)/bin/clang-format" "/usr/local/bin/clang-format"
ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"
```
