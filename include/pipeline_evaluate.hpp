#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace pipeline {

struct AimFeatureInfo {
    std::string label;
    double      signed_weight;
};

struct EvaluateResult {
    std::vector<AimFeatureInfo> ranked_features;  // populated if aim_window > 0
    double       intercept     = 0.0;
    double       hss           = 0.0;             // sum of all GSS values (DrPhylo)
    fs::path     gene_pred_path;
    fs::path     svg_path;
    int tp=0, tn=0, fp=0, fn=0;
    double tpr=0.0, tnr=0.0, fpr=0.0, fnr=0.0;
};

struct EvaluateOptions {
    // Required for direct evaluation:
    fs::path     weights_path;   // path to weights.txt
    fs::path     list_path;      // path to list.txt
    fs::path     output_file;    // path to write eval.txt (gene_predictions goes next to it)
    // Optional:
    fs::path     hyp_path;       // hypothesis file for metrics
    bool         no_visualize    = false;
    bool         m_grid          = false;
    std::string  datatype        = "universal";
    unsigned int num_threads     = 0;
    fs::path     cache_dir;
    // AIM feature ranking: if aim_window > 0, read ranked features from weights_path parent dir
    int          aim_window      = 0;
};

// Direct evaluate: run prediction from weights_path + list_path.
// Writes: output_file, gene_predictions.txt, SPS_SPP.txt, SVG (unless no_visualize).
// Populates ranked_features if aim_window > 0.
EvaluateResult evaluate(const EvaluateOptions& opts);

// DrPhylo aggregation: given run_dir containing lambda_N/eval_gene_predictions.txt files,
// aggregate qualifying lambda models and write:
//   run_dir/eval_gene_predictions.txt
//   run_dir/eval.txt
//   run_dir/eval_SPS_SPP.txt
//   run_dir/eval.svg
// Returns hss (sum of all per-lambda GSS values from run_dir/lambda_N/gss.txt).
struct DrPhyloAggResult {
    double hss = 0.0;
    bool   ok  = false;
};
DrPhyloAggResult evaluate_drphylo_aggregate(
    const fs::path& run_dir,
    double          grid_rmse_cutoff,
    double          grid_acc_cutoff);

} // namespace pipeline
