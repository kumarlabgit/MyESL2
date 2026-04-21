#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Forward declare arma::fmat to avoid pulling in full Armadillo header
namespace arma { template<typename eT> class Mat; typedef Mat<float> fmat; }

namespace encoder {

struct AlignmentResult {
    std::string stem;
    std::vector<std::vector<uint8_t>> columns;  // columns[j][i]: j-th encoded col, i-th sequence
    std::vector<std::pair<uint32_t, char>> map; // (original_pos, allele) per encoded column
    std::vector<std::string> missing_sequences; // "stem\tseq_id" for each missing sequence
    std::vector<bool> col_is_minor; // parallel to columns: true if column is a non-major allele
    bool failed = false;
    std::string error_msg;
    size_t var_site_count = 0;  // count of variable sites (for PSC group penalty)
};

// Lightweight result from pass-1 of two-pass encoding: metadata only, no column data.
struct AlignmentMeta {
    std::string stem;
    size_t num_cols = 0;
    std::vector<std::pair<uint32_t, char>> map; // (original_pos, allele) per encoded column
    std::vector<std::string> missing_sequences;
    std::vector<bool> col_is_minor;
    bool has_minor_col = false;  // true if any col_is_minor entry is true
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

// Encode one gene from in-memory sequences (PSC mode).
// sequences[i] = sequence bytes for species i (may contain gaps from cancellation)
// gene_name: used as the stem for feature labels
// min_minor: minimum non-major, non-gap count to keep a position (default 2)
AlignmentResult encode_raw_sequences(
    const std::string& gene_name,
    const std::vector<std::vector<uint8_t>>& sequences,
    int min_minor = 2);

// ---- Two-pass encoding (eliminates intermediate column storage) ----

// Pass 1: Count columns and collect metadata (map, col_is_minor, missing).
// No column data is allocated — only the lightweight AlignmentMeta is returned.
AlignmentMeta count_pff_columns(
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major = false,
    const std::unordered_set<std::string>& dropout_labels = {},
    bool skip_x = false,
    bool minor_column = false
);

// Pass 2: Re-read PFF and encode directly into a pre-allocated features matrix.
// Writes columns starting at features(0, col_offset).
// If minor_column is true and the gene has minor alleles, writes an extra column at
// col_offset + num_regular_cols containing the OR of all minor-allele columns.
// Returns false on failure (same gene that would set failed=true in pass 1).
bool encode_pff_into(
    arma::fmat& features,
    uint64_t col_offset,
    const std::filesystem::path& pff_path,
    const std::vector<std::string>& hyp_seq_names,
    int min_minor,
    bool drop_major = false,
    const std::unordered_set<std::string>& dropout_labels = {},
    bool skip_x = false,
    bool minor_column = false,
    bool has_minor_col = false
);

} // namespace encoder
