#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace encoder {

struct AlignmentResult {
    std::string stem;
    std::vector<std::vector<uint8_t>> columns;  // columns[j][i]: j-th encoded col, i-th sequence
    std::vector<std::pair<uint32_t, char>> map; // (original_pos, allele) per encoded column
    std::vector<std::string> missing_sequences; // "stem\tseq_id" for each missing sequence
    bool failed = false;
    std::string error_msg;
};

// Encode a single PFF file into a one-hot matrix.
// hyp_seq_names: sequence names from hypothesis file in output order (zeros already removed)
// min_minor: minimum total count of non-major, non-indel characters required to keep a position
// drop_major: if true, skip building a feature column for the most frequent allele at each position
// dropout_labels: set of "{stem}_{pos}_{allele}" labels to exclude (empty = include all)
// skip_x: if true, treat 'X' (unknown amino acid / nucleotide) like '-' and '?' (not counted
//         towards allele totals); set for protein/nucleotide datatypes, not for universal
AlignmentResult encode_pff(
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major = false,
    const std::unordered_set<std::string>& dropout_labels = {},
    bool skip_x = false
);

// Alternate encoder using direct lookup tables (DLT) in place of character equality comparisons.
// Produces identical output to encode_pff; replaces std::map allele counting and per-allele
// sequence passes with 256-entry arrays and a single sequence pass per position.
AlignmentResult encode_pff_dlt(
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major = false,
    const std::unordered_set<std::string>& dropout_labels = {},
    bool skip_x = false
);

} // namespace encoder
