#pragma once
#include <armadillo>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include "regression.hpp"

namespace fs = std::filesystem;

namespace pipeline {

struct EncodeResult {
    arma::fmat   features;                      // N × C, fp32
    arma::frowvec responses;                    // 1 × N, fp32
    arma::mat    alg_table;                     // 3 × n_groups
    std::vector<std::string> seq_names;         // sample names (post-balancing)
    std::vector<float>       hyp_values;        // raw labels (post-balancing)
    std::vector<std::string> all_stems_ordered; // gene order for GSS/PSS
    bool         is_overlapping  = false;
    uint32_t     N               = 0;
    uint64_t     total_cols      = 0;
    regression::Precision precision = regression::Precision::FP32;
    fs::path     combined_map_path;
    fs::path     alignment_table_path;
    fs::path     field_path;          // empty if not overlapping
    std::map<std::string, std::string> extra_params;  // e.g. sWeight path
    std::string  datatype;            // passed through for train/evaluate
};

struct EncodeOptions {
    fs::path     output_dir;
    fs::path     hyp_path;
    int          min_minor        = -1;    // -1 → use value from preprocess_config
    double       auto_bit_ct      = -1.0;
    bool         drop_major       = false;
    bool         minor_column     = false;
    std::string  class_bal;       // "", "up", "down", "weighted"
    std::string  write_features_path;
    std::string  write_features_transposed_path;
    std::unordered_set<std::string> dropout_labels;
    regression::Precision precision = regression::Precision::FP32;
    uint64_t max_mem    = uint64_t(8) * 1024 * 1024 * 1024;  // abort if estimated matrix exceeds this (bytes); default 8 GB
    bool     disable_mc = false; // if true, warn instead of throwing on max_mem exceeded
};

EncodeResult encode(const EncodeOptions& opts);

// Returns a map of stem -> encoded feature column count.
// Runs the per-file encoder but discards column data; writes nothing to disk.
std::map<std::string, uint64_t> encode_sizes(const EncodeOptions& opts);

} // namespace pipeline
