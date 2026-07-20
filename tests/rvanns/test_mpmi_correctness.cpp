// Standalone correctness checks for the common MPMI representation and the
// selected distance backend.  On RVV builds, get_distance_computer() selects
// the fused RVV implementation; on other ISAs it selects the portable path.

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CaseSpec {
    const char* name;
    size_t dim;
    size_t training_rows;
    bool expect_residuals;
    bool expect_tail_residual;
};

size_t
expected_code_size(size_t dim, bool with_residuals) {
    if (!with_residuals) {
        return dim;
    }
    const size_t fp32_dims = static_cast<size_t>(std::ceil(0.01 * dim));
    // FP16 is the next 10% after the disjoint FP32 pool, rather than the
    // remainder of a combined 10% high-precision pool.
    const size_t fp16_dims = std::min(
            static_cast<size_t>(std::ceil(0.10 * dim)), dim - fp32_dims);
    return dim + 2 * fp16_dims + 4 * fp32_dims;
}

std::vector<float>
make_training(size_t rows, size_t dim) {
    std::vector<float> values(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t j = 0; j < dim; ++j) {
            const float phase = static_cast<float>((row + 1) * (j + 3));
            float value = std::sin(phase * 0.071f) +
                    0.35f * std::cos(phase * 0.013f);
            // Give a deterministic subset a wider, non-linear range so the
            // offline error ranking has meaningful FP16 and FP32 candidates.
            if (j + 1 == dim) {
                // Make the last dimension a deterministic high-error choice.
                // For dim=257 this forces recovery in the one-element tile
                // following the first 256 dimensions.
                value += 50.0f * static_cast<float>(row * row);
            } else if (j % 29 == 0) {
                value += 0.02f * static_cast<float>(row * row);
            } else if (j % 11 == 0) {
                value += 0.11f * static_cast<float>(row);
            }
            values[row * dim + j] = value;
        }
    }
    return values;
}

std::vector<float>
make_vector(size_t dim, float shift) {
    std::vector<float> values(dim);
    for (size_t j = 0; j < dim; ++j) {
        values[j] = std::sin((static_cast<float>(j) + shift) * 0.17f) +
                0.25f * std::cos((static_cast<float>(j) - shift) * 0.031f);
    }
    return values;
}

float
reference_distance(
        faiss::MetricType metric,
        const std::vector<float>& query,
        const std::vector<float>& decoded) {
    double result = 0.0;
    for (size_t j = 0; j < query.size(); ++j) {
        if (metric == faiss::METRIC_L2) {
            const double diff = static_cast<double>(query[j]) - decoded[j];
            result += diff * diff;
        } else {
            result += static_cast<double>(query[j]) * decoded[j];
        }
    }
    return static_cast<float>(result);
}

void
require_close(
        const std::string& label,
        float expected,
        float actual,
        float relative_tolerance = 2.0e-4f) {
    const float tolerance = relative_tolerance *
            std::max(1.0f, std::max(std::fabs(expected), std::fabs(actual)));
    if (!std::isfinite(actual) || std::fabs(expected - actual) > tolerance) {
        throw std::runtime_error(
                label + ": expected " + std::to_string(expected) +
                ", got " + std::to_string(actual) +
                ", tolerance " + std::to_string(tolerance));
    }
}

std::vector<float>
decode_format_oracle(
        const faiss::ScalarQuantizer& sq,
        const std::vector<float>& source,
        const std::vector<uint8_t>& code,
        bool expect_residuals) {
    const size_t dim = source.size();
    const size_t bitmap_words = (dim + 31) / 32;
    const size_t header_floats = 2 + 2 * dim;
    if (sq.trained.size() != header_floats + 2 * bitmap_words ||
        static_cast<size_t>(sq.trained[0]) != bitmap_words ||
        static_cast<size_t>(sq.trained[1]) != bitmap_words) {
        throw std::runtime_error("MPMI metadata layout does not match dim");
    }

    std::vector<uint32_t> fp16_words(bitmap_words, 0);
    std::vector<uint32_t> fp32_words(bitmap_words, 0);
    if (bitmap_words != 0) {
        std::memcpy(
                fp16_words.data(),
                sq.trained.data() + header_floats,
                bitmap_words * sizeof(uint32_t));
        std::memcpy(
                fp32_words.data(),
                sq.trained.data() + header_floats + bitmap_words,
                bitmap_words * sizeof(uint32_t));
    }

    const auto selected = [](const std::vector<uint32_t>& words, size_t i) {
        return (words[i / 32] & (uint32_t(1) << (i % 32))) != 0;
    };
    size_t fp16_count = 0;
    size_t fp32_count = 0;
    for (size_t i = 0; i < dim; ++i) {
        const bool use_fp16 = selected(fp16_words, i);
        const bool use_fp32 = selected(fp32_words, i);
        if (use_fp16 && use_fp32) {
            throw std::runtime_error("FP16 and FP32 residual pools overlap");
        }
        fp16_count += use_fp16;
        fp32_count += use_fp32;
    }

    const size_t expected_fp32 = expect_residuals
            ? static_cast<size_t>(std::ceil(0.01 * dim))
            : 0;
    const size_t expected_fp16 = expect_residuals
            ? std::min(
                      static_cast<size_t>(std::ceil(0.10 * dim)),
                      dim - expected_fp32)
            : 0;
    if (fp16_count != expected_fp16 || fp32_count != expected_fp32) {
        throw std::runtime_error("MPMI residual-pool counts do not match policy");
    }

    const float* vmin = sq.trained.data() + 2;
    const float* vdiff = vmin + dim;
    const size_t fp16_begin = dim;
    const size_t fp32_begin = fp16_begin + fp16_count * sizeof(uint16_t);
    size_t fp16_rank = 0;
    size_t fp32_rank = 0;
    std::vector<float> reconstructed(dim, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
        uint8_t expected_base_code = 0;
        if (vdiff[i] > 0.0f && std::isfinite(vdiff[i])) {
            float normalized = (source[i] - vmin[i]) / vdiff[i];
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            expected_base_code = static_cast<uint8_t>(
                    std::lround(normalized * 255.0f));
        }
        if (code[i] != expected_base_code) {
            throw std::runtime_error("dense affine base code mismatch");
        }

        const float base = !(vdiff[i] > 0.0f) || !std::isfinite(vdiff[i])
                ? vmin[i]
                : vmin[i] +
                        (static_cast<float>(code[i]) / 255.0f) * vdiff[i];
        float residual = 0.0f;
        if (selected(fp16_words, i)) {
            uint16_t stored = 0;
            std::memcpy(
                    &stored,
                    code.data() + fp16_begin +
                            fp16_rank * sizeof(uint16_t),
                    sizeof(stored));
            const uint16_t expected = faiss::encode_fp16(source[i] - base);
            if (stored != expected) {
                throw std::runtime_error("FP16 residual-pool encoding mismatch");
            }
            residual = faiss::decode_fp16(stored);
            ++fp16_rank;
        } else if (selected(fp32_words, i)) {
            float stored = 0.0f;
            std::memcpy(
                    &stored,
                    code.data() + fp32_begin + fp32_rank * sizeof(float),
                    sizeof(stored));
            require_close(
                    "FP32 residual-pool encoding",
                    source[i] - base,
                    stored,
                    1.0e-6f);
            residual = stored;
            ++fp32_rank;
        }
        reconstructed[i] = base + residual;
    }
    return reconstructed;
}

void
run_case(const CaseSpec& spec) {
    faiss::ScalarQuantizer sq(
            spec.dim, faiss::ScalarQuantizer::QT_HYBRID_FP8_16_32);

    std::vector<float> training;
    if (spec.training_rows == 0) {
        // n=0 is the explicit dense-base-only metadata path.
        sq.train(0, nullptr);
    } else {
        training = make_training(spec.training_rows, spec.dim);
        sq.train(spec.training_rows, training.data());
    }

    const size_t expected_size =
            expected_code_size(spec.dim, spec.expect_residuals);
    if (sq.code_size != expected_size) {
        throw std::runtime_error(
                std::string(spec.name) + ": code_size expected " +
                std::to_string(expected_size) + ", got " +
                std::to_string(sq.code_size));
    }

    // Faiss deserialization restores d, qtype, and trained metadata before
    // recomputing derived sizes.  Exercise that path explicitly because MPMI
    // has a metadata-dependent row size.
    faiss::ScalarQuantizer restored;
    restored.d = sq.d;
    restored.qtype = sq.qtype;
    restored.trained = sq.trained;
    restored.set_derived_sizes();
    if (restored.code_size != sq.code_size) {
        throw std::runtime_error(
                std::string(spec.name) +
                ": restored MPMI code_size does not match trained metadata");
    }

    if (spec.expect_tail_residual) {
        const size_t bitmap_words = (spec.dim + 31) / 32;
        const size_t header_floats = 2 + 2 * spec.dim;
        std::vector<uint32_t> fp16_words(bitmap_words, 0);
        std::vector<uint32_t> fp32_words(bitmap_words, 0);
        std::memcpy(
                fp16_words.data(),
                sq.trained.data() + header_floats,
                bitmap_words * sizeof(uint32_t));
        std::memcpy(
                fp32_words.data(),
                sq.trained.data() + header_floats + bitmap_words,
                bitmap_words * sizeof(uint32_t));
        const size_t tail = spec.dim - 1;
        const uint32_t mask = uint32_t(1) << (tail % 32);
        if ((fp16_words[tail / 32] & mask) == 0 &&
            (fp32_words[tail / 32] & mask) == 0) {
            throw std::runtime_error(
                    std::string(spec.name) +
                    ": final partial tile has no selected residual");
        }
    }

    const std::vector<float> source = make_vector(spec.dim, 0.75f);
    const std::vector<float> query = make_vector(spec.dim, 2.25f);
    std::vector<uint8_t> code(sq.code_size, 0);
    std::vector<float> decoded(spec.dim, 0.0f);
    sq.compute_codes(source.data(), code.data(), 1);
    sq.decode_portable_hybrid(code.data(), decoded.data(), 1);

    const std::vector<float> oracle =
            decode_format_oracle(sq, source, code, spec.expect_residuals);
    for (size_t i = 0; i < spec.dim; ++i) {
        require_close("portable decode/format oracle", oracle[i], decoded[i]);
    }

    if (spec.expect_residuals &&
        std::none_of(
                code.begin() + static_cast<std::ptrdiff_t>(spec.dim),
                code.end(),
                [](uint8_t byte) { return byte != 0; })) {
        throw std::runtime_error(
                std::string(spec.name) + ": residual pools were not populated");
    }

    for (float value : decoded) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                    std::string(spec.name) + ": decode produced non-finite data");
        }
    }

    // The same encoded payload must agree for both L2 and inner product.
    for (faiss::MetricType metric :
         {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
                sq.get_distance_computer(metric));
        if (!dc) {
            throw std::runtime_error(
                    std::string(spec.name) + ": no distance computer");
        }
        dc->set_query(query.data());
        const float expected = reference_distance(metric, query, oracle);
        const float actual = dc->query_to_code(code.data());
        require_close(
                std::string(spec.name) +
                        (metric == faiss::METRIC_L2 ? "/L2" : "/IP"),
                expected,
                actual);
    }

    std::cout << "PASS " << spec.name << " dim=" << spec.dim
              << " code_size=" << sq.code_size
              << " residuals=" << (spec.expect_residuals ? "fp16+fp32" : "none")
#if defined(__riscv_vector)
              << " backend=rvv-fused"
#else
              << " backend=portable"
#endif
              << '\n';
}

} // namespace

int
main() {
    try {
        // 64 dimensions with n=0 exercises the dense 8-bit base without
        // residual pools. 257 dimensions guarantees non-empty FP16 and FP32
        // pools and a selected residual in the one-element final tile. 131 is
        // also non-divisible by the fixed recovery-tile capacity.
        const CaseSpec cases[] = {
                {"dense-base-only", 64, 0, false, false},
                {"fp16-fp32-and-tail-residual", 257, 96, true, true},
                {"partial-tile", 131, 64, true, false},
        };
        for (const CaseSpec& spec : cases) {
            run_case(spec);
        }
        std::cout << "All MPMI correctness cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MPMI correctness failure: " << error.what() << '\n';
        return 1;
    }
}
