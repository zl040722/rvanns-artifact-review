This document helps reviewers build the RVANNS artifact from source code, understand the code footprint, and run the included tests and evaluation scripts.

## Introduction

RVANNS is an artifact built on top of Knowhere/Faiss for approximate nearest neighbor search experiments. The repository keeps the upstream C++ vector search library structure, adds HNSW graph reordering support through `IndexHNSW::reorder_graph_after_build`, and provides standalone RVANNS evaluation programs for HNSW, HNSW-SQ/MPMI, and ROrder-style graph layout experiments. The added programs load vector datasets, build or reuse cached indexes, compute or read ground truth, sweep search parameters, and write recall/QPS/latency results into `vectors_out/results`.

## Artifact Notes

This artifact includes the optimized implementation, patches, and test programs used for evaluation. The SLOC counting policy excludes third-party code under `thirdparty/` and includes tests and build scripts. Under this policy, the artifact contains 57,289 SLOC.

The main RVANNS-related files are:

- `thirdparty/faiss/faiss/IndexHNSW.h` and `thirdparty/faiss/faiss/IndexHNSW.cpp`: add post-build graph reordering APIs, keep the `perm[new_id]=old_id` mapping, and support BFS and `gorder` traversal methods.
- `tests/rvanns/`: contains five standalone C++ test and performance programs, about 3K non-empty source lines in total.
- `scripts/`, `ci/`, `CMakeLists.txt`, and `conanfile.py`: provide dependency installation, build, coverage, and CI entry points used by the artifact.

The RVANNS test programs are:

- `test_hnsw_224.cpp`: HNSW baseline sweep over datasets and `efSearch` values.
- `test_hnsw_rorder_perf.cpp`: HNSW batch-search performance test with perf counter collection.
- `test_mpmi.cpp`: HNSW-SQ/MPMI evaluation with optional graph reordering and CSV output to `vectors_out/results/hnsw_efs_rorder_sweep.csv`.
- `test_mpmi_perf.cpp`: MPMI-focused performance test.
- `test_vec_mpmi_rorder_perf.cpp`: HNSW-SQ/MPMI plus ROrder batch-search performance test.

The regular Knowhere unit tests are built when `with_ut=True`/`WITH_UT=ON` is enabled and run through `knowhere_tests`. Faiss tests can be enabled with `with_faiss_tests=True`/`WITH_FAISS_TESTS=ON`. The Python tests live under `tests/python/`, and CI/E2E coverage is described by the Jenkins/Groovy files under `ci/`.

## Building Knowhere Within Milvus

If you wish to only use Knowhere within Milvus without changing any of the Knowhere source code, we suggest that you move to the [Milvus main project](https://github.com/milvus-io/milvus) and build Milvus directly, where Knowhere is then built implicitly during Milvus build.

## System Requirements

All Linux distributions are available for Knowhere development. However, a majority of our contributor worked with Ubuntu or CentOS systems, with a small portion of Mac (both x86_64 and Apple Silicon) contributors. If you would like Knowhere to build and run on other distributions, you are more than welcome to file an issue and contribute!

Here's a list of verified OS types where Knowhere can successfully build and run:

- Ubuntu 20.04 x86_64
- Ubuntu 20.04 Aarch64
- MacOS (x86_64)
- MacOS (Apple Silicon)

## Building Knowhere From Source Code

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
$ ./Debug/tests/ut/knowhere_tests
#Release
$ ./Release/tests/ut/knowhere_tests
```

#### Clean up

```bash
$ git clean -fxd
```

## GEN PYTHON WHEEL(NEED REALSE BUILD)

install dependency:

```
sudo apt install swig python3-dev
pip3 install bfloat16
```

after build knowhere:

```bash
cd python
python3 setup.py bdist_wheel
```

install knowhere wheel:

```bash
pip3 install dist/pyknowhere-0.0.0-cp38-cp38-linux_x86_64.whl
```

clean

```bash
cd python
rm -rf build
rm -rf dist
rm -rf knowhere.egg-info
rm knowhere/knowhere_wrap.cpp
rm knowhere/swigknowhere.py
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
