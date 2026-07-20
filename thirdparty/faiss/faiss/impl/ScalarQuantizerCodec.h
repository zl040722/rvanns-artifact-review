/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizerOp.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/ScalarQuantizerScanner.h>
#include "simd/hook.h"

namespace faiss {

using QuantizerType = ScalarQuantizer::QuantizerType;
using RangeStat = ScalarQuantizer::RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;
using SQuantizer = ScalarQuantizer::SQuantizer;

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        code[i] = (int)(255 * x);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (code[i] + 0.5f) / 255.0f;
    }
};

struct Codec4bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        code[i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }
};

struct Codec6bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                code[0] |= bits;
                break;
            case 1:
                code[0] |= bits << 6;
                code[1] |= bits >> 2;
                break;
            case 2:
                code[1] |= bits << 4;
                code[2] |= bits >> 4;
                break;
            case 3:
                code[2] |= bits << 2;
                break;
        }
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        uint8_t bits = 0x00;
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                bits = code[0] & 0x3f;
                break;
            case 1:
                bits = code[0] >> 6;
                bits |= (code[1] & 0xf) << 2;
                break;
            case 2:
                bits = code[1] >> 4;
                bits |= (code[2] & 3) << 4;
                break;
            case 3:
                bits = code[2] >> 2;
                break;
        }
        return (bits + 0.5f) / 63.0f;
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

enum class QuantizerTemplateScaling {
    UNIFORM = 0,
    NON_UNIFORM = 1
};

template <class Codec, QuantizerTemplateScaling SCALING, int SIMD>
struct QuantizerTemplate {};

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> : SQuantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained[0]), vdiff(trained[1]) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff != 0) {
                xi = (x[i] - vmin) / vdiff;
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin + xi * vdiff;
    }
};

template <class Codec>
struct QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> : SQuantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff[i] != 0) {
                xi = (x[i] - vmin[i]) / vdiff[i];
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin[i] + xi * vdiff[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin[i] + xi * vdiff[i];
    }
};

/*******************************************************************
 * MPMI hybrid quantizer
 *
 * Every row starts with a dense unsigned 8-bit affine base. The two
 * compact pools that follow contain residuals for the dimensions selected
 * during training. Quantizer-wide affine parameters and selection bitmaps
 * live in ScalarQuantizer::trained.
 *******************************************************************/

struct QuantizerHybrid : SQuantizer {
    const size_t d;
    const float* vmin = nullptr;
    const float* vdiff = nullptr;
    std::vector<uint8_t> fp16_bitmap;
    std::vector<uint8_t> fp32_bitmap;
    std::vector<uint32_t> fp16_prefix;
    std::vector<uint32_t> fp32_prefix;
    size_t fp16_count = 0;
    size_t fp32_count = 0;
    size_t code_size = 0;

    QuantizerHybrid(size_t d, const std::vector<float>& trained) : d(d) {
        const size_t header_size = 2 + 2 * d;
        FAISS_THROW_IF_NOT_MSG(
                trained.size() >= header_size,
                "MPMI metadata is shorter than its affine header");

        const auto read_word_count = [](float value) -> size_t {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(value) && value >= 0.0f &&
                            std::floor(value) == value,
                    "MPMI bitmap word count is invalid");
            return static_cast<size_t>(value);
        };

        const size_t fp16_words = read_word_count(trained[0]);
        const size_t fp32_words = read_word_count(trained[1]);
        const size_t required_words = (d + 31) / 32;
        FAISS_THROW_IF_NOT_MSG(
                fp16_words == required_words && fp32_words == required_words,
                "MPMI bitmap dimensions do not match the quantizer");
        FAISS_THROW_IF_NOT_MSG(
                trained.size() >= header_size + fp16_words + fp32_words,
                "MPMI metadata is shorter than its bitmap payload");

        vmin = trained.data() + 2;
        vdiff = vmin + d;

        std::vector<uint32_t> fp16_words_data(fp16_words, 0);
        std::vector<uint32_t> fp32_words_data(fp32_words, 0);
        if (fp16_words != 0) {
            std::memcpy(
                    fp16_words_data.data(),
                    trained.data() + header_size,
                    fp16_words * sizeof(uint32_t));
            std::memcpy(
                    fp32_words_data.data(),
                    trained.data() + header_size + fp16_words,
                    fp32_words * sizeof(uint32_t));
        }

        fp16_bitmap.resize(d, 0);
        fp32_bitmap.resize(d, 0);
        fp16_prefix.resize(d, 0);
        fp32_prefix.resize(d, 0);
        for (size_t i = 0; i < d; ++i) {
            const uint32_t mask = uint32_t(1) << (i % 32);
            fp16_bitmap[i] =
                    (fp16_words_data[i / 32] & mask) != 0 ? 1 : 0;
            fp32_bitmap[i] =
                    (fp32_words_data[i / 32] & mask) != 0 ? 1 : 0;
            FAISS_THROW_IF_NOT_MSG(
                    !(fp16_bitmap[i] && fp32_bitmap[i]),
                    "MPMI FP16 and FP32 residual pools must be disjoint");
            fp16_prefix[i] = static_cast<uint32_t>(fp16_count);
            fp32_prefix[i] = static_cast<uint32_t>(fp32_count);
            fp16_count += fp16_bitmap[i];
            fp32_count += fp32_bitmap[i];
        }

        code_size = d + fp16_count * sizeof(uint16_t) +
                fp32_count * sizeof(float);
    }

    FAISS_ALWAYS_INLINE uint8_t encode_base(float value, size_t i) const {
        if (!(vdiff[i] > 0.0f) || !std::isfinite(vdiff[i])) {
            return 0;
        }
        float normalized = (value - vmin[i]) / vdiff[i];
        normalized = std::max(0.0f, std::min(1.0f, normalized));
        return static_cast<uint8_t>(std::lround(normalized * 255.0f));
    }

    FAISS_ALWAYS_INLINE float decode_base(
            const uint8_t* code,
            size_t i) const {
        if (!(vdiff[i] > 0.0f) || !std::isfinite(vdiff[i])) {
            return vmin[i];
        }
        return vmin[i] + (static_cast<float>(code[i]) / 255.0f) * vdiff[i];
    }

    FAISS_ALWAYS_INLINE float residual_component(
            const uint8_t* code,
            size_t i) const {
        if (fp16_bitmap[i]) {
            uint16_t encoded;
            const size_t offset = d + fp16_prefix[i] * sizeof(uint16_t);
            std::memcpy(&encoded, code + offset, sizeof(encoded));
            return decode_fp16(encoded);
        }
        if (fp32_bitmap[i]) {
            float residual;
            const size_t offset = d + fp16_count * sizeof(uint16_t) +
                    fp32_prefix[i] * sizeof(float);
            std::memcpy(&residual, code + offset, sizeof(residual));
            return residual;
        }
        return 0.0f;
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(
            const uint8_t* code,
            size_t i) const {
        return decode_base(code, i) + residual_component(code, i);
    }

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; ++i) {
            const uint8_t base_code = encode_base(x[i], i);
            code[i] = base_code;
            const float residual = x[i] - decode_base(code, i);
            if (fp16_bitmap[i]) {
                const uint16_t encoded = encode_fp16(residual);
                const size_t offset = d + fp16_prefix[i] * sizeof(uint16_t);
                std::memcpy(code + offset, &encoded, sizeof(encoded));
            } else if (fp32_bitmap[i]) {
                const size_t offset = d + fp16_count * sizeof(uint16_t) +
                        fp32_prefix[i] * sizeof(float);
                std::memcpy(code + offset, &residual, sizeof(residual));
            }
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; ++i) {
            x[i] = reconstruct_component(code, i);
        }
    }
};

inline size_t hybrid_code_size_from_trained(
        size_t d,
        const std::vector<float>& trained) {
    return QuantizerHybrid(d, trained).code_size;
}

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16 {};

template <>
struct QuantizerFP16<1> : SQuantizer {
    const size_t d;

    QuantizerFP16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_fp16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_fp16(((uint16_t*)code)[i]);
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16 {};

template <>
struct QuantizerBF16<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    QuantizerBF16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_bf16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_bf16(((uint16_t*)code)[i]);
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return decode_bf16(((uint16_t*)code)[i]);
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect {};

template <>
struct Quantizer8bitDirect<1> : SQuantizer {
    const size_t d;

    Quantizer8bitDirect(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return code[i];
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned {};

template <>
struct Quantizer8bitDirectSigned<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)(x[i] + 128);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i] - 128;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return code[i] - 128;
    }
};

/*******************************************************************
 * 1bit_direct quantizer
 *
 * Note: The 1bit_direct quantizer currently does not support the
 *`reconstruct_component` method and does not provide SIMDWIDTH support.
 *******************************************************************/

struct Quantizer1bitDirect : SQuantizer {
    const size_t d;

    Quantizer1bitDirect(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        size_t code_size = (d + 7) / 8;
        for (size_t i = 0; i < code_size; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        size_t code_size = (d + 7) / 8;
        for (size_t i = 0; i < code_size; i++) {
            x[i] = (float)code[i];
        }
    }
};

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<Codec8bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<Codec6bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<Codec4bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<Codec8bit, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<Codec4bit, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_1bit_direct:
            return new Quantizer1bitDirect(d, trained);
        case ScalarQuantizer::QT_HYBRID_FP8_16_32:
            return new QuantizerHybrid(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template <int SIMDWIDTH>
struct SimilarityL2 {};

template <>
struct SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}

    /******* scalar accumulator *******/

    float accu;

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

template <int SIMDWIDTH>
struct SimilarityIP {};

template <>
struct SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        accu += *yi++ * x;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        accu += x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(q, code);
    }
};

/** Portable MPMI fallback.
 *
 * The compact row is reconstructed into distance-computer-local scratch and
 * then consumed by the platform's existing FP32 distance hook. This keeps the
 * representation portable without implying that non-RVV backends implement
 * the RVV-specific fused recovery path.
 */
template <class Similarity>
struct HybridDistanceComputer : SQDistanceComputer {
    using Sim = Similarity;

    QuantizerHybrid quant;
    mutable std::vector<float> decoded;
    mutable std::vector<float> decoded2;

    HybridDistanceComputer(size_t d, const std::vector<float>& trained)
            : quant(d, trained), decoded(d), decoded2(d) {}

    void set_query(const float* x) final {
        q = x;
    }

    float compute_distance(const float* x, const uint8_t* code) const {
        quant.decode_vector(code, decoded.data());
        if (Similarity::metric_type == METRIC_L2) {
            return fvec_L2sqr(x, decoded.data(), quant.d);
        }
        return fvec_inner_product(x, decoded.data(), quant.d);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(q, code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        quant.decode_vector(codes + i * code_size, decoded.data());
        quant.decode_vector(codes + j * code_size, decoded2.data());
        if (Similarity::metric_type == METRIC_L2) {
            return fvec_L2sqr(decoded.data(), decoded2.data(), quant.d);
        }
        return fvec_inner_product(decoded.data(), decoded2.data(), quant.d);
    }
};

inline SQDistanceComputer* select_hybrid_distance_computer(
        MetricType metric,
        size_t d,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        return new HybridDistanceComputer<SimilarityL2<1>>(d, trained);
    }
    if (metric == METRIC_INNER_PRODUCT) {
        return new HybridDistanceComputer<SimilarityIP<1>>(d, trained);
    }
    FAISS_THROW_MSG("MPMI supports only L2 and inner product distances");
    return nullptr;
}

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec6bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate<QuantizerBF16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte<Sim, SIMDWIDTH>(d, trained);
            } else {
                return new DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate<
                    Quantizer8bitDirectSigned<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_HYBRID_FP8_16_32:
            return new HybridDistanceComputer<Sim>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

// This wrapper adapts Jaccard and Hamming binary computers to the
// SQDistanceComputer interface
template <class BinaryComputerType>
struct BinarySQDistanceComputerWrapper : SQDistanceComputer {
    BinaryComputerType binary_computer;
    size_t code_size;
    std::vector<uint8_t> tmp;

    BinarySQDistanceComputerWrapper(size_t code_size, const std::vector<float>&)
            : code_size(code_size), tmp(code_size) {}

    void set_query(const float* x) final {
        for (size_t i = 0; i < code_size; ++i) {
            tmp[i] = (uint8_t)x[i];
        }
        binary_computer.set(tmp.data(), code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return binary_computer.compute(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const uint8_t* code_i = codes + i * code_size;
        const uint8_t* code_j = codes + j * code_size;

        BinaryComputerType temp_computer;
        temp_computer.set(code_i, code_size);
        return temp_computer.compute(code_j);
    }
};

SQDistanceComputer* select_hamming_distance_computer(
    size_t d,
    const std::vector<float>& trained);

SQDistanceComputer* select_jaccard_distance_computer(
        size_t d,
        const std::vector<float>& trained);

template <class DCClass, int use_sel>
InvertedListScanner* sel3_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (DCClass::Sim::metric_type == METRIC_L2) {
        return new IVFSQScannerL2<DCClass, use_sel>(
                sq->d,
                sq->trained,
                sq->code_size,
                quantizer,
                store_pairs,
                sel,
                r);
    } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
        return new IVFSQScannerIP<DCClass, use_sel>(
                sq->d, sq->trained, sq->code_size, store_pairs, sel, r);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (sel) {
        if (store_pairs) {
            return sel3_InvertedListScanner<DCClass, 2>(
                    sq, quantizer, store_pairs, sel, r);
        } else {
            return sel3_InvertedListScanner<DCClass, 1>(
                    sq, quantizer, store_pairs, sel, r);
        }
    } else {
        return sel3_InvertedListScanner<DCClass, 0>(
                sq, quantizer, store_pairs, sel, r);
    }
}

template <class Similarity, class Codec, QuantizerTemplateScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate<Codec, SCALING, SIMDWIDTH>;
    using DCClass = DCTemplate<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec8bit, QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec4bit, QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit:
            return sel12_InvertedListScanner<Similarity, Codec8bit, QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit:
            return sel12_InvertedListScanner<Similarity, Codec4bit, QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_6bit:
            return sel12_InvertedListScanner<Similarity, Codec6bit, QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_fp16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerFP16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_bf16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerBF16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner<
                        DistanceComputerByte<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner<DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner<DCTemplate<
                    Quantizer8bitDirectSigned<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_HYBRID_FP8_16_32:
            return sel2_InvertedListScanner<
                    HybridDistanceComputer<Similarity>>(
                    sq, quantizer, store_pairs, sel, r);
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner<SimilarityL2<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner<SimilarityIP<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // namespace faiss
