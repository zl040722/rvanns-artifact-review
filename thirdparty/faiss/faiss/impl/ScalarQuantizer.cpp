/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ScalarQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#include <faiss/impl/platform_macros.h>
#include <omp.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/FaissHook.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/ScalarQuantizerOp.h>
#include <faiss/impl/ScalarQuantizerCodec.h>

#if defined(__riscv_vector)
#include <faiss/impl/ScalarQuantizerDC_rvv.h>
#endif

namespace faiss {

using QuantizerType = ScalarQuantizer::QuantizerType;
using RangeStat = ScalarQuantizer::RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * ScalarQuantizer implementation
 *
 * The main source of complexity is to support combinations of 4
 * variants without incurring runtime tests or virtual function calls:
 *
 * - 4 / 8 bits per code component
 * - uniform / non-uniform
 * - IP / L2 distance search
 * - scalar / AVX distance computation
 *
 * The appropriate Quantizer object is returned via select_quantizer
 * that hides the template mess.
 ********************************************************************/

#ifdef __AVX2__
#ifdef __F16C__
#define USE_F16C
#else
#warning \
        "Cannot enable AVX optimizations in scalar quantizer if -mf16c is not set as well"
#endif
#endif

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d, QuantizerType qtype)
        : Quantizer(d), qtype(qtype) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer() {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
        case QT_8bit_direct_signed:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_bf16:
            code_size = d * 2;
            bits = 16;
            break;
        case QT_1bit_direct:
            code_size = (d + 7) / 8;
            bits = 1;
            break;
        case QT_HYBRID_FP8_16_32:
            bits = 0;
            code_size = trained.empty()
                    ? 0
                    : hybrid_code_size_from_trained(d, trained);
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    int bit_per_dim = qtype == QT_4bit_uniform ? 4
            : qtype == QT_4bit                 ? 4
            : qtype == QT_6bit                 ? 6
            : qtype == QT_8bit_uniform         ? 8
            : qtype == QT_8bit                 ? 8
            : qtype == QT_1bit_direct          ? 1
                                               : -1;

    switch (qtype) {
        case QT_4bit_uniform:
        case QT_8bit_uniform:
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_4bit:
        case QT_8bit:
        case QT_6bit:
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_fp16:
        case QT_8bit_direct:
        case QT_bf16:
        case QT_8bit_direct_signed:
        case QT_1bit_direct:
            // no training necessary
            break;
        case QT_HYBRID_FP8_16_32: {
            const size_t bitmap_words = (d + 31) / 32;
            std::vector<float> base_min(d, 0.0f);
            std::vector<float> base_diff(d, 1.0f);

            if (n != 0) {
                std::vector<float> affine;
                train_NonUniform(
                        rangestat,
                        rangestat_arg,
                        n,
                        d,
                        256,
                        x,
                        affine);
                FAISS_THROW_IF_NOT_MSG(
                        affine.size() == 2 * d,
                        "MPMI affine training returned invalid metadata");
                std::copy(affine.begin(), affine.begin() + d, base_min.begin());
                std::copy(affine.begin() + d, affine.end(), base_diff.begin());
                for (size_t i = 0; i < d; ++i) {
                    if (!std::isfinite(base_min[i])) {
                        base_min[i] = 0.0f;
                    }
                    if (!std::isfinite(base_diff[i]) || base_diff[i] <= 0.0f) {
                        base_diff[i] = 0.0f;
                    }
                }
            }

            std::vector<double> residual_sse(d, 0.0);
            if (n != 0) {
                for (size_t row = 0; row < n; ++row) {
                    for (size_t col = 0; col < d; ++col) {
                        const float value = x[row * d + col];
                        uint8_t base_code = 0;
                        if (base_diff[col] > 0.0f) {
                            float normalized =
                                    (value - base_min[col]) / base_diff[col];
                            normalized = std::max(
                                    0.0f, std::min(1.0f, normalized));
                            base_code = static_cast<uint8_t>(
                                    std::lround(normalized * 255.0f));
                        }
                        const float reconstructed = base_min[col] +
                                (static_cast<float>(base_code) / 255.0f) *
                                        base_diff[col];
                        const double residual =
                                static_cast<double>(value) - reconstructed;
                        residual_sse[col] += residual * residual;
                    }
                }
            }

            std::vector<std::pair<double, size_t>> ranked(d);
            for (size_t i = 0; i < d; ++i) {
                ranked[i] = {
                        n == 0 ? 0.0 : residual_sse[i] / static_cast<double>(n),
                        i};
            }
            std::sort(
                    ranked.begin(),
                    ranked.end(),
                    [](const auto& lhs, const auto& rhs) {
                        if (lhs.first != rhs.first) {
                            return lhs.first > rhs.first;
                        }
                        return lhs.second < rhs.second;
                    });

            // The pools are disjoint: the highest-error ceil(1%) use FP32,
            // and the next ceil(10%) use FP16.
            size_t fp32_dims = d == 0 || n == 0
                    ? 0
                    : static_cast<size_t>(
                              std::ceil(0.01 * static_cast<double>(d)));
            size_t fp16_dims = d == 0 || n == 0
                    ? 0
                    : static_cast<size_t>(
                              std::ceil(0.10 * static_cast<double>(d)));
            fp32_dims = std::min(fp32_dims, d);
            fp16_dims = std::min(fp16_dims, d - fp32_dims);

            std::vector<uint32_t> fp16_bitmap(bitmap_words, 0);
            std::vector<uint32_t> fp32_bitmap(bitmap_words, 0);
            for (size_t rank = 0; rank < fp32_dims; ++rank) {
                const size_t dim = ranked[rank].second;
                fp32_bitmap[dim / 32] |= uint32_t(1) << (dim % 32);
            }
            for (size_t rank = fp32_dims;
                 rank < fp32_dims + fp16_dims;
                 ++rank) {
                const size_t dim = ranked[rank].second;
                fp16_bitmap[dim / 32] |= uint32_t(1) << (dim % 32);
            }

            trained.assign(2 + 2 * d + 2 * bitmap_words, 0.0f);
            trained[0] = static_cast<float>(bitmap_words);
            trained[1] = static_cast<float>(bitmap_words);
            std::copy(base_min.begin(), base_min.end(), trained.begin() + 2);
            std::copy(
                    base_diff.begin(),
                    base_diff.end(),
                    trained.begin() + 2 + d);
            if (bitmap_words != 0) {
                std::memcpy(
                        trained.data() + 2 + 2 * d,
                        fp16_bitmap.data(),
                        bitmap_words * sizeof(uint32_t));
                std::memcpy(
                        trained.data() + 2 + 2 * d + bitmap_words,
                        fp32_bitmap.data(),
                        bitmap_words * sizeof(uint32_t));
            }
            set_derived_sizes();
            break;
        }
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
    if (qtype == QT_HYBRID_FP8_16_32) {
        return new QuantizerHybrid(d, trained);
    }
    /* use hook to decide use AVX512 or not */
    return sq_sel_quantizer(qtype, d, trained);
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for if (n > 1)
    for (int64_t i = 0; i < n; i++)
        squant->encode_vector(x + i * d, codes + i * code_size);
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for if (n > 1)
    for (int64_t i = 0; i < n; i++)
        squant->decode_vector(codes + i * code_size, x + i * d);
}

void ScalarQuantizer::decode_portable_hybrid(
        const uint8_t* codes,
        float* x,
        size_t n) const {
    FAISS_THROW_IF_NOT_MSG(
            qtype == QT_HYBRID_FP8_16_32,
            "decode_portable_hybrid requires the MPMI quantizer type");
    QuantizerHybrid quantizer(d, trained);
#pragma omp parallel for if (n > 1)
    for (int64_t i = 0; i < n; ++i) {
        quantizer.decode_vector(codes + i * code_size, x + i * d);
    }
}

SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT ||
            metric == METRIC_Hamming || metric == METRIC_Jaccard);
    if (qtype == QT_HYBRID_FP8_16_32) {
#if defined(__riscv_vector)
        return sq_get_distance_computer_rvv(metric, qtype, d, trained);
#else
        return select_hybrid_distance_computer(metric, d, trained);
#endif
    }
    /* use hook to decide use AVX512 or not */
    if (metric == METRIC_Hamming) {
        assert(qtype == QT_1bit_direct);
        return sq_get_hamming_distance_computer(metric, qtype, d, trained);
    }
    if (metric == METRIC_Jaccard) {
        assert(qtype == QT_1bit_direct);
        return sq_get_jaccard_distance_computer(metric, qtype, d, trained);
    }
    return sq_get_distance_computer(metric, qtype, d, trained);
}

size_t ScalarQuantizer::cal_size() const {
    return sizeof(*this) + trained.size() * sizeof(float);
}

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
    if (qtype == QT_HYBRID_FP8_16_32) {
#if defined(__riscv_vector)
        return sq_select_inverted_list_scanner_rvv(
                mt, this, quantizer, d, store_pairs, sel, by_residual);
#else
        if (mt == METRIC_L2) {
            return sel1_InvertedListScanner<SimilarityL2<1>>(
                    this, quantizer, store_pairs, sel, by_residual);
        }
        if (mt == METRIC_INNER_PRODUCT) {
            return sel1_InvertedListScanner<SimilarityIP<1>>(
                    this, quantizer, store_pairs, sel, by_residual);
        }
        FAISS_THROW_MSG("MPMI supports only L2 and inner product scanners");
#endif
    }
    /* use hook to decide use AVX512 or not */
    return sq_sel_inv_list_scanner(mt, this, quantizer, d, store_pairs,
                                   sel, by_residual);
}

} // namespace faiss
