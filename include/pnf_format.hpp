#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pnf {

/**
 * @struct PNFMetadata
 * @brief Metadata for Parsed Numeric File (PNF) format.
 *
 * PNF Format:
 *   Text metadata section (key=value, line-delimited), then binary float data.
 *
 *   VERSION=1
 *   source_path=<absolute path>
 *   num_sequences=<N>
 *   num_features=<F>
 *   seq_ids=<id0>;<id1>;...;<idN-1>
 *   feature_labels=<lab0>;<lab1>;...;<labF-1>
 *   END_METADATA
 *   [binary: N x F x sizeof(float) bytes, row-major]
 *
 * Row-major layout: seq0_feat0, seq0_feat1, ..., seq0_featF-1, seq1_feat0, ...
 */
struct PNFMetadata {
    std::string source_path;
    uint32_t num_sequences = 0;
    uint32_t num_features  = 0;
    std::vector<std::string> seq_ids;
    std::vector<std::string> feature_labels;
    uint64_t data_offset = 0;  ///< Byte offset to binary section

    uint64_t get_data_size() const {
        return static_cast<uint64_t>(num_sequences) * num_features * sizeof(float);
    }
};

} // namespace pnf
