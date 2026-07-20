#!/usr/bin/env python3
"""Fast source-level consistency checks for the RIVER review artifact.

This script is deliberately independent of the target ISA.  It catches accidental
packaging regressions before architecture-specific compilation or benchmark runs.
It complements, rather than replaces, the C++ numerical tests.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path.read_text(encoding="utf-8", errors="strict")


def require(text: str, needle: str, location: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{location}: missing {needle!r}")


def reject(text: str, needle: str, location: str, failures: list[str]) -> None:
    if needle.lower() in text.lower():
        failures.append(f"{location}: forbidden stale wording {needle!r}")


def main() -> int:
    failures: list[str] = []

    try:
        sq_header = read("thirdparty/faiss/faiss/impl/ScalarQuantizer.h")
        sq_source = read("thirdparty/faiss/faiss/impl/ScalarQuantizer.cpp")
        sq_codec = read("thirdparty/faiss/faiss/impl/ScalarQuantizerCodec.h")
        sq_codec_rvv = read(
            "thirdparty/faiss/faiss/impl/ScalarQuantizerCodec_rvv.h"
        )
        hnsw_header = read("thirdparty/faiss/faiss/IndexHNSW.h")
        hnsw_source = read("thirdparty/faiss/faiss/IndexHNSW.cpp")
        cq1 = read("tests/rvanns/test_mpmi_cq1.cpp")
        layout_test = read("tests/rvanns/test_hnsw_reorder_micro.cpp")
        correctness = read("tests/rvanns/test_mpmi_correctness.cpp")
        knowhere_wiring_test = read(
            "tests/rvanns/test_knowhere_mpmi_rorder.cpp"
        )
        hnsw_config = read("src/index/hnsw/faiss_hnsw_config.h")
        hnsw_wiring = read("src/index/hnsw/faiss_hnsw.cc")
        refine_utils = read("src/index/refine/refine_utils.cc")
        index_params = read("include/knowhere/comp/index_param.h")
        legacy_mpmi = read("tests/rvanns/test_mpmi.cpp")
        legacy_mpmi_rorder = read(
            "tests/rvanns/test_vec_mpmi_rorder_perf.cpp"
        )
        tests_cmake = read("tests/rvanns/CMakeLists.txt")
        root_cmake = read("CMakeLists.txt")
        conanfile = read("conanfile.py")
    except (FileNotFoundError, UnicodeError) as exc:
        failures.append(f"artifact file error: {exc}")
        sq_header = sq_source = sq_codec = sq_codec_rvv = ""
        hnsw_header = hnsw_source = tests_cmake = root_cmake = conanfile = ""
        cq1 = layout_test = correctness = knowhere_wiring_test = ""
        hnsw_config = hnsw_wiring = refine_utils = index_params = ""
        legacy_mpmi = legacy_mpmi_rorder = ""

    require(sq_header, "QT_HYBRID_FP8_16_32", "ScalarQuantizer.h", failures)
    require(sq_source, "QT_HYBRID_FP8_16_32", "ScalarQuantizer.cpp", failures)
    for token in (
        "std::ceil(0.01",
        "std::ceil(0.10",
        "fp32_bitmap",
        "fp16_bitmap",
        "set_derived_sizes()",
    ):
        require(sq_source, token, "ScalarQuantizer.cpp", failures)
    for token in (
        "struct QuantizerHybrid",
        "dense unsigned 8-bit affine base",
        "residual_component",
        "hybrid_code_size_from_trained",
        "HybridDistanceComputer",
    ):
        require(sq_codec, token, "ScalarQuantizerCodec.h", failures)
    for token in (
        "RVANNS_RVV_DECODE_LMUL",
        "RVANNS_RVV_ACC_LMUL",
        "decode_tile",
        "packed_fp16",
        "__riscv_vfwcvt_f_f_v_f32",
        "HybridDistanceComputer_rvv",
        "tile_size",
    ):
        require(sq_codec_rvv, token, "ScalarQuantizerCodec_rvv.h", failures)

    for method in ("rorder", "gorder", "bfs", "rcm", "rabbitorder"):
        require(hnsw_source, f'"{method}"', "IndexHNSW.cpp", failures)
    require(hnsw_source, "optimize_neighbors_by_id", "IndexHNSW.cpp", failures)
    require(
        hnsw_source,
        "Cannot add vectors after a post-build HNSW layout pass",
        "IndexHNSW.cpp",
        failures,
    )
    require(
        hnsw_source,
        "layout pass may be applied only once",
        "IndexHNSW.cpp",
        failures,
    )
    require(hnsw_header, "generateGorderPermutation", "IndexHNSW.h", failures)
    rorder_start = hnsw_source.find('method_equals(method, "rorder")')
    rorder_global = hnsw_source.find("generateGorderPermutation()", rorder_start)
    rorder_local = hnsw_source.find("optimize_neighbors_by_id()", rorder_start)
    if not (0 <= rorder_start < rorder_global < rorder_local):
        failures.append(
            "IndexHNSW.cpp: ROrder must apply the GOrder-style permutation "
            "before level-0 adjacency normalization"
        )

    for label in ("Scalar+FP32", "SIMD+FP32", "SIMD+MPMI", "MPMI+ROrder"):
        require(cq1, label, "test_mpmi_cq1.cpp", failures)
    require(cq1, "QT_HYBRID_FP8_16_32", "test_mpmi_cq1.cpp", failures)
    require(cq1, '"rorder"', "test_mpmi_cq1.cpp", failures)
    require(cq1, "fmt_river2", "test_mpmi_cq1.cpp", failures)
    require(cq1, "base_file_identity", "test_mpmi_cq1.cpp", failures)
    reject(cq1, "EnablePatchForComputeFP32AsBF16", "test_mpmi_cq1.cpp", failures)

    for method in ("unordered", "bfs", "rcm", "rabbitorder", "gorder", "rorder"):
        require(layout_test, method, "test_hnsw_reorder_micro.cpp", failures)
    require(layout_test, "all_level_edges", "test_hnsw_reorder_micro.cpp", failures)
    require(
        layout_test,
        "verify_payload_permutation",
        "test_hnsw_reorder_micro.cpp",
        failures,
    )
    require(
        layout_test,
        "verify_node_metadata_permutation",
        "test_hnsw_reorder_micro.cpp",
        failures,
    )
    require(
        layout_test,
        "repeated post-build layout pass was accepted",
        "test_hnsw_reorder_micro.cpp",
        failures,
    )
    reject(layout_test, "faiss::storage_idx_t", "test_hnsw_reorder_micro.cpp", failures)

    for case in ("L2", "inner product", "partial tile", "FP16", "FP32"):
        require(correctness, case, "test_mpmi_correctness.cpp", failures)
    require(correctness, "decode_format_oracle", "test_mpmi_correctness.cpp", failures)
    require(
        correctness,
        "restored MPMI code_size",
        "test_mpmi_correctness.cpp",
        failures,
    )
    require(
        correctness,
        "final partial tile has no selected residual",
        "test_mpmi_correctness.cpp",
        failures,
    )

    for token in ("mpmi", "graph_layout", "rorder"):
        require(hnsw_config, token, "faiss_hnsw_config.h", failures)
    require(
        refine_utils,
        '{"mpmi", faiss::ScalarQuantizer::QT_HYBRID_FP8_16_32}',
        "refine_utils.cc",
        failures,
    )
    require(index_params, "HNSW_GRAPH_LAYOUT", "index_param.h", failures)
    for token in (
        "ApplyGraphLayout",
        "reorder_graph_after_build",
        "label_to_internal_offset",
        "indexes.size() > 1 || !labels.empty()",
        "Can not add data after a post-build HNSW layout pass",
        "trace_visit is not supported after a post-build graph layout",
    ):
        require(hnsw_wiring, token, "faiss_hnsw.cc", failures)
    for token in (
        "INDEX_HNSW_SQ",
        'SQ_TYPE, "mpmi"',
        'HNSW_GRAPH_LAYOUT, "rorder"',
        "Serialize(MPMI+ROrder)",
        "filtered Knowhere search",
    ):
        require(
            knowhere_wiring_test,
            token,
            "test_knowhere_mpmi_rorder.cpp",
            failures,
        )
    require(legacy_mpmi, "QT_HYBRID_FP8_16_32", "test_mpmi.cpp", failures)
    reject(
        legacy_mpmi,
        "faiss::ScalarQuantizer::QT_8bit, name",
        "test_mpmi.cpp",
        failures,
    )
    require(
        legacy_mpmi_rorder,
        "QT_HYBRID_FP8_16_32",
        "test_vec_mpmi_rorder_perf.cpp",
        failures,
    )
    require(tests_cmake, "add_test", "tests/rvanns/CMakeLists.txt", failures)
    require(
        tests_cmake,
        "river_knowhere_mpmi_rorder",
        "tests/rvanns/CMakeLists.txt",
        failures,
    )
    require(root_cmake, "RIVER_BUILD_ARTIFACT_TESTS", "CMakeLists.txt", failures)
    require(
        conanfile,
        "with_river_artifact_tests",
        "conanfile.py",
        failures,
    )

    for path, text in (
        ("IndexHNSW.h", hnsw_header),
        ("IndexHNSW.cpp", hnsw_source),
        ("test_mpmi_cq1.cpp", cq1),
        ("faiss_hnsw.cc", hnsw_wiring),
        ("test_knowhere_mpmi_rorder.cpp", knowhere_wiring_test),
    ):
        for stale in (
            "trace-aware",
            "execution trace",
            "query trace collector",
            "co-visitation collector",
        ):
            reject(text, stale, path, failures)

    if failures:
        print("RIVER artifact consistency check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("RIVER artifact source-level consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
