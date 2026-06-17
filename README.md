This document helps reviewers build the artifact from source code, understand its code footprint, and run the included tests and evaluation scripts.

## Introduction

This artifact supports approximate nearest neighbor search experiments on vector databases. The repository keeps a C++ vector search engine structure, adds post-build HNSW graph reordering support through `IndexHNSW::reorder_graph_after_build`, and provides standalone evaluation programs for HNSW, HNSW-SQ/MPMI, and graph-layout-aware batch-search experiments. The added programs load vector datasets, build or reuse cached indexes, compute or read ground truth, sweep search parameters, and write recall, QPS, and latency results into `vectors_out/results`.

## Artifact Notes

This artifact includes the optimized implementation, patches, and test programs used for evaluation. The SLOC counting policy excludes third-party code under `thirdparty/` and includes tests and build scripts. Under this policy, the artifact contains 57,289 SLOC.

The main artifact-related files are:

- `thirdparty/faiss/faiss/IndexHNSW.h` and `thirdparty/faiss/faiss/IndexHNSW.cpp`: add post-build graph reordering APIs, keep the `perm[new_id]=old_id` mapping, and support BFS and `gorder` traversal methods.
- `patches/mpmi.patch`: stores the architecture-specific MPMI source patch used to reproduce the corresponding implementation changes.
- the dedicated evaluation-test directory under `tests/`: contains five standalone C++ test and performance programs, about 3K non-empty source lines in total.
- `scripts/`, `ci/`, `CMakeLists.txt`, and `conanfile.py`: provide dependency installation, build, coverage, and CI entry points used by the artifact.

The standalone evaluation programs cover:

- HNSW baseline sweeps over datasets and `efSearch` values.
- HNSW batch-search performance runs with perf counter collection.
- HNSW-SQ/MPMI evaluation with optional graph reordering and CSV output under `vectors_out/results`.
- MPMI-focused performance runs.
- HNSW-SQ/MPMI batch-search performance runs with graph-layout-aware traversal.

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
