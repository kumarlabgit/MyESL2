#pragma once

// Shared group-penalty weight computation for sparse group lasso.
//
// Used by both the PSC pipeline and the train/drphylo/aim pipelines.
// The solver reads alg_table row 2 as per-group penalty weights;
// these functions compute that row from gene/group metadata.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace group_penalty {

inline double round_to(double x, int digits) {
    double p = std::pow(10.0, digits);
    return std::round(x * p) / p;
}

inline std::vector<double> linear_space(double start, double end, double step) {
    if (step <= 0) throw std::runtime_error("step must be > 0");
    std::vector<double> out;
    double val = start;
    while (val <= end + 1e-12) {
        out.push_back(round_to(val, 6));
        val += step;
    }
    return out;
}

// Build the list of penalty terms to iterate over.
//
// - "median": returns a single value = floor(median(var_site_counts > 0)).
// - all others: returns linear_space(initial, final_val, step).
//
// For "std" and "sqrt" modes the penalty_term is not used by
// compute_group_weights, but the loop still runs once (the default
// linear_space(1,1,1) produces {1.0}).
inline std::vector<double> build_penalty_terms(
    const std::string& kind,
    double initial, double final_val, double step,
    const std::vector<size_t>& var_site_counts)
{
    std::string k = kind;
    for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (k == "median") {
        std::vector<size_t> vars;
        for (auto v : var_site_counts)
            if (v > 0) vars.push_back(v);
        if (vars.empty()) vars.push_back(1);
        std::sort(vars.begin(), vars.end());
        double median;
        if (vars.size() % 2 == 1) {
            median = static_cast<double>(vars[vars.size() / 2]);
        } else {
            size_t hi = vars.size() / 2;
            median = static_cast<double>(vars[hi - 1] + vars[hi]) / 2.0;
        }
        return {std::floor(median)};
    }

    return linear_space(initial, final_val, step);
}

// Compute per-group penalty weights for a given penalty term.
//
// Weight modes:
//   "std"              sqrt(feature_length)
//   "linear" / "median"  var_site_count + penalty_term
//   "sqrt"             sqrt(var_site_count)
//   (fallback)         sqrt(feature_length)
//
// feature_lengths[i] = number of encoded feature columns in group i
// var_site_counts[i] = number of polymorphic positions in group i
//                      (for numeric input, this equals feature_lengths[i])
inline std::vector<double> compute_group_weights(
    const std::string& kind,
    double penalty_term,
    const std::vector<size_t>& feature_lengths,
    const std::vector<size_t>& var_site_counts)
{
    std::string k = kind;
    for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    std::vector<double> weights;
    weights.reserve(feature_lengths.size());
    for (size_t i = 0; i < feature_lengths.size(); ++i) {
        double w;
        if (k == "std")
            w = std::sqrt(static_cast<double>(feature_lengths[i]));
        else if (k == "linear" || k == "median")
            w = static_cast<double>(var_site_counts[i]) + penalty_term;
        else if (k == "sqrt")
            w = std::sqrt(static_cast<double>(var_site_counts[i]));
        else
            w = std::sqrt(static_cast<double>(feature_lengths[i]));
        weights.push_back(w);
    }
    return weights;
}

} // namespace group_penalty
