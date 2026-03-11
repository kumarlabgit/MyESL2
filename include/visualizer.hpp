#pragma once

#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <cmath>

namespace viz {

struct VizOptions {
    int gene_limit    = 100;
    int species_limit = 100;
    double ssq_threshold = 0.0;
    bool m_grid = false;  // show only positive-class samples (DrPhylo mode)
};

struct GenePredictionsTable {
    std::vector<std::string>          seq_ids;
    std::vector<double>               responses;
    std::vector<double>               predictions;
    double                            intercept = 0.0;
    std::vector<std::string>          gene_names;
    std::vector<std::vector<double>>  gene_scores; // [gene_idx][seq_idx]
};

// Read a gene_predictions.txt file (as written by evaluate)
GenePredictionsTable read_gene_predictions(const std::filesystem::path& path);

// Write an SVG heatmap from a gene predictions table
void write_svg(const GenePredictionsTable& table,
               const std::filesystem::path& out,
               const VizOptions& opts = {});

// ─── AIM visualization ────────────────────────────────────────────────────────

struct AimAccuracyPoint { double tpr, tnr, acc; };

struct AimVizData {
    std::vector<std::string> feature_labels;      // resorted, up to aim_window
    std::vector<std::string> seq_ids;             // hypothesis species (non-zero)
    std::vector<double>      responses;           // +1 / -1 per species
    // contributions[feat_idx][seq_idx] = weight * x_j_s (signed contribution)
    std::vector<std::vector<double>> contributions;
    std::vector<AimAccuracyPoint> curve;          // k=0 (intercept) .. k=W
    int cutoff_idx = -1;                          // first k meeting cutoff; -1 if none
};

// Write a two-panel SVG: top = feature heatmap (pos-class rows), bottom = TPR/TNR/Acc curves
void write_aim_svg(const AimVizData& data, const std::filesystem::path& out);

} // namespace viz
