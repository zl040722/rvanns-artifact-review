// End-to-end wiring check for the normal Knowhere HNSW-SQ build path used by
// Milvus: sq_type=mpmi plus the one-shot graph_layout=rorder pass.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/metric.h"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

namespace {

constexpr int64_t kRows = 96;
constexpr int64_t kDim = 64;

void
require_status(const char* operation, knowhere::Status status) {
    if (status != knowhere::Status::success) {
        throw std::runtime_error(
                std::string(operation) + " failed with status " +
                std::to_string(static_cast<int>(status)));
    }
}

knowhere::Index<knowhere::IndexNode>
make_index() {
    const auto version =
            knowhere::Version::GetCurrentVersion().VersionNumber();
    auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_HNSW_SQ, version);
    if (!created.has_value()) {
        throw std::runtime_error("failed to create Knowhere HNSW-SQ index");
    }
    return created.value();
}

knowhere::Json
make_config() {
    return {
            {knowhere::meta::DIM, kDim},
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::indexparam::HNSW_M, 16},
            {knowhere::indexparam::EFCONSTRUCTION, 96},
            {knowhere::indexparam::EF, 96},
            {knowhere::meta::TOPK, 1},
            {knowhere::indexparam::SQ_TYPE, "mpmi"},
            {knowhere::indexparam::HNSW_GRAPH_LAYOUT, "rorder"},
    };
}

std::vector<float>
make_base() {
    std::vector<float> base(kRows * kDim);
    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t j = 0; j < kDim; ++j) {
            const uint64_t mixed =
                    (static_cast<uint64_t>(row + 1) * 0x9e3779b185ebca87ULL) ^
                    (static_cast<uint64_t>(j + 3) * 0xc2b2ae3d27d4eb4fULL);
            base[row * kDim + j] =
                    static_cast<float>(static_cast<int32_t>(mixed >> 40)) /
                    65536.0f;
        }
        // Keep every source vector well separated from the others while the
        // pseudo-random dimensions produce a nontrivial HNSW topology.
        base[row * kDim + (row % kDim)] +=
                32.0f + static_cast<float>(row) * 0.125f;
    }
    return base;
}

uint32_t
choose_original_id(const std::vector<uint32_t>& new_to_original) {
    std::vector<uint8_t> seen(new_to_original.size(), 0);
    uint32_t moved = 0;
    bool found_moved = false;
    for (size_t new_id = 0; new_id < new_to_original.size(); ++new_id) {
        const uint32_t old_id = new_to_original[new_id];
        if (old_id >= new_to_original.size() || seen[old_id] != 0) {
            throw std::runtime_error("Knowhere post-build ID map is not bijective");
        }
        seen[old_id] = 1;
        if (!found_moved && old_id != new_id) {
            moved = old_id;
            found_moved = true;
        }
    }
    return found_moved ? moved : 0;
}

void
require_filtered_search_returns(
        const knowhere::Index<knowhere::IndexNode>& index,
        const knowhere::DataSetPtr& query,
        const knowhere::Json& config,
        uint32_t allowed_original_id) {
    std::vector<uint8_t> filter((kRows + 7) / 8, 0xff);
    filter[allowed_original_id >> 3] &=
            static_cast<uint8_t>(~(uint8_t(1) << (allowed_original_id & 7)));
    const knowhere::BitsetView bitset(
            filter.data(), kRows, static_cast<size_t>(kRows - 1));
    auto result = index.Search(query, config, bitset);
    if (!result.has_value() || result.value()->GetIds() == nullptr ||
        result.value()->GetIds()[0] != allowed_original_id) {
        throw std::runtime_error(
                "filtered Knowhere search did not preserve the original ID");
    }
}

} // namespace

int
main() {
    try {
        std::vector<float> base = make_base();
        const auto base_dataset =
                knowhere::GenDataSet(kRows, kDim, base.data());
        const knowhere::Json config = make_config();

        auto index = make_index();
        require_status("Build(MPMI+ROrder)", index.Build(base_dataset, config, false));

        const auto mapping = index.Node()->GetInternalIdToExternalIdMap();
        if (mapping == nullptr || mapping->size() != static_cast<size_t>(kRows)) {
            throw std::runtime_error("Knowhere did not expose the ROrder ID map");
        }
        const uint32_t allowed_original_id =
                choose_original_id(*mapping);
        const auto query = knowhere::GenDataSet(
                1,
                kDim,
                base.data() + static_cast<size_t>(allowed_original_id) * kDim);
        require_filtered_search_returns(
                index, query, config, allowed_original_id);

        knowhere::BinarySet binary_set;
        require_status("Serialize(MPMI+ROrder)", index.Serialize(binary_set));
        auto restored = make_index();
        require_status(
                "Deserialize(MPMI+ROrder)",
                restored.Deserialize(binary_set, config));

        const auto restored_mapping =
                restored.Node()->GetInternalIdToExternalIdMap();
        if (restored_mapping == nullptr || *restored_mapping != *mapping) {
            throw std::runtime_error(
                    "serialized Knowhere index lost the ROrder ID map");
        }
        require_filtered_search_returns(
                restored, query, config, allowed_original_id);

        if (restored.Add(base_dataset, config, false) ==
            knowhere::Status::success) {
            throw std::runtime_error(
                    "deserialized post-build layout accepted incremental Add");
        }

        std::cout << "PASS Knowhere HNSW-SQ sq_type=mpmi "
                     "graph_layout=rorder build/search/serialization contract\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Knowhere MPMI+ROrder wiring failure: "
                  << error.what() << '\n';
        return 1;
    }
}
