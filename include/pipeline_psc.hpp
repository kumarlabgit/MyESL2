#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pipeline::psc {

// ── Data types ────────────────────────────────────────────────────────────────

struct GeneAlignment {
    std::string name;                                       // gene name (file stem)
    uint32_t seq_len = 0;                                   // alignment length
    std::unordered_map<std::string, std::vector<uint8_t>> seqs; // species -> uppercase sequence
};

struct ComboJob {
    size_t index = 0;
    std::string combo_label;       // "combo_0", "combo_1", ...
    std::string combo_tag;         // species-based identifier
    std::vector<std::string> species;
    std::vector<double> y_raw;     // +1/-1 binary labels
};

struct GeneAggregate {
    std::string name;
    double single_highest_gss = 0.0;
    int single_best_rank = -1;      // -1 = unranked
    double highest_ever_gss = 0.0;
    int best_ever_rank = -1;        // -1 = unranked
    size_t num_combos_ranked = 0;
    size_t num_combos_ranked_top = 0;
    std::map<size_t, double> selected_sites;  // position -> max PSS
};

struct PredictionRow {
    std::string species_combo;
    double lambda1 = 0, lambda2 = 0, penalty_term = 0;
    size_t num_genes = 0;
    double input_rmse = 0;
    std::string species;
    double sps = 0;
    std::string true_phenotype; // empty = not available
    bool has_true_phenotype = false;
};

// Internal per-gene encoding result
struct GeneEncoded {
    std::string name;
    size_t var_site_count = 0;
    std::vector<std::vector<uint8_t>> columns;  // [feature_col][species] = 0 or 1
    std::vector<std::pair<uint32_t, char>> map;  // (position, allele) per column
};

// Feature metadata (for predictions)
struct FeatureMeta {
    std::string label;
    size_t gene_idx = 0;
    uint32_t position = 0;
    uint8_t aa = 0;
};

// Gene metadata (for group penalties)
struct GeneMeta {
    std::string name;
    size_t feature_start = 0;
    size_t feature_end = 0;
    size_t var_site_count = 0;
};

// Prediction design
struct PredictionDesign {
    std::vector<std::string> species;
    std::vector<std::vector<size_t>> feature_hit_rows; // [feature] -> list of pred species indices
    std::vector<double> true_values; // NaN = not available
};

// ── Options ───────────────────────────────────────────────────────────────────

struct PscOptions {
    // Input paths
    std::filesystem::path alignments_dir;
    std::filesystem::path output_dir;
    std::string output_base_name;

    // Species contrast source (exactly one required)
    std::filesystem::path species_groups_file;
    std::filesystem::path response_file;
    std::filesystem::path response_dir;
    std::filesystem::path auto_pairs_tree;

    // Lambda grid
    double initial_lambda1 = 0.01;
    double final_lambda1 = 0.99;
    double initial_lambda2 = 0.01;
    double final_lambda2 = 0.99;
    double lambda_step = 0.05;
    bool use_logspace = false;
    size_t num_log_points = 20;

    // Group penalty
    std::string group_penalty_type = "median";  // median, linear, sqrt, std
    double initial_gp_value = 1.0;
    double final_gp_value = 1.0;
    double gp_step = 1.0;
    bool use_default_gp = false;

    // Gap cancellation
    bool use_uncanceled_alignments = false;
    bool cancel_only_partner = false;
    bool cancel_tri_allelic = false;
    bool nix_full_deletions = false;
    std::string outgroup_species;
    size_t min_pairs = 2;

    // Solver
    std::string method = "sg_lasso";
    std::string precision_str = "fp32";
    int maxiter = 100;
    unsigned int threads = 0;
    std::map<std::string, std::string> params;

    // Output/prediction
    std::filesystem::path prediction_alignments_dir;
    std::filesystem::path species_pheno_path;
    bool no_pred_output = false;
    bool no_genes_output = false;
    bool show_selected_sites = false;
    double top_rank_frac = 0.01;
    std::filesystem::path limited_genes_list;

    // Null models
    bool make_null_models = false;
    bool make_pair_randomized_null_models = false;
    size_t num_randomized_alignments = 10;

    // Auto-pairs
    std::string auto_pairs_method = "simple_deterministic";
    int auto_pairs_num_alternates = 0;
    int auto_pairs_max_combinations = 1;
};

// ── Entry point ───────────────────────────────────────────────────────────────

void run_psc(const PscOptions& opts);

} // namespace pipeline::psc
