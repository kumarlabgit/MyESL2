#include "pipeline_train.hpp"
#include "group_penalty.hpp"
#include "pipeline_utils.hpp"
#include "model_log.hpp"
#include "process_log.hpp"
#include "regression.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>

namespace pipeline {

namespace {

static void validate_lambda_in_open_unit(double v) {
    if (!(v > 0.0 && v < 1.0))
        throw std::runtime_error(
            "Lambda value " + std::to_string(v) +
            " is outside (0,1); all lambda values must be in the open interval (0,1)");
}

// Post-hoc emulation of sequential --min-groups skip-ahead.
//
// Walks the grid in original λ₁-outer/λ₂-inner order (the harness's natural
// lambda_list.txt order), maintaining the same `max_lambda2` ratchet and
// early-break as the sequential loop in train(). For each index, decides
// whether that point would have been skipped in a single-threaded run; if
// so, deletes the contents of lambda_<idx>/ (preserving the directory
// itself as a drphylo aggregation sentinel) and records the index.
//
// Preconditions: every actually-solved grid point has its gene_count stored
// at gene_counts[idx] (>= 0). Indices where gene_counts[idx] < 0 are
// treated as not-solved (their lambda dir is pruned, matching
// sequential-mode skipped semantics).
//
// Returns the sorted list of pruned indices.
std::vector<size_t> prune_skipped_lambdas(
    const std::vector<std::array<double, 2>>& lambdas,
    const std::vector<int>& gene_counts,
    const fs::path& output_dir,
    int min_groups,
    double min_lambda2)
{
    std::vector<size_t> pruned;
    double max_lambda2 = std::numeric_limits<double>::infinity();
    bool broke_early = false;

    for (size_t idx = 0; idx < lambdas.size(); ++idx) {
        const auto& lam = lambdas[idx];
        const bool would_skip =
            broke_early
            || (lam[1] > max_lambda2)
            || (idx < gene_counts.size() && gene_counts[idx] < 0);

        if (would_skip) {
            fs::path lam_dir = output_dir / ("lambda_" + std::to_string(idx));
            if (fs::exists(lam_dir) && fs::is_directory(lam_dir)) {
                std::error_code ec;
                for (auto& entry : fs::directory_iterator(lam_dir, ec))
                    fs::remove_all(entry.path(), ec);
            }
            pruned.push_back(idx);
            continue;
        }

        int gc = (idx < gene_counts.size()) ? gene_counts[idx] : -1;
        if (gc >= 0 && gc <= min_groups) {
            if (lam[1] == min_lambda2) broke_early = true;
            else max_lambda2 = std::min(max_lambda2, lam[1]);
        }
    }
    return pruned;
}

// Build an expanded map file for ol_sg_lasso: one line per field position,
// with label = combined.map[field[j]-1].  Returns the path to expanded.map.
static fs::path generate_expanded_map(const fs::path& output_dir, const arma::mat& alg_table) {
    fs::path exp_path = output_dir / "expanded.map";

    // Check if existing expanded.map already has the Group column
    if (fs::exists(exp_path)) {
        std::ifstream check(exp_path);
        std::string header;
        if (std::getline(check, header) && header.find("Group") != std::string::npos)
            return exp_path;
        // Stale 2-column file — regenerate
    }

    // Load field.txt (1-based indices, one row CSV)
    fs::path field_path = output_dir / "field.txt";
    arma::rowvec field;
    if (!field.load(field_path.string(), arma::csv_ascii))
        throw std::runtime_error("Failed to load field file: " + field_path.string());

    // Build group assignment: expanded position -> group index
    auto group_of = pipeline_utils::build_group_assignment(alg_table, field.n_cols);

    // Load combined.map lines (skip header)
    std::vector<std::string> map_lines;
    {
        std::ifstream mf(output_dir / "combined.map");
        std::string line;
        std::getline(mf, line); // header
        while (std::getline(mf, line))
            map_lines.push_back(line);
    }

    // Write expanded.map with Group column
    {
        std::ofstream ef(exp_path);
        ef << "Position\tLabel\tGroup\n";
        for (size_t j = 0; j < field.n_cols; ++j) {
            int phys = static_cast<int>(field(j)) - 1;
            if (phys >= 0 && phys < (int)map_lines.size())
                ef << j << '\t' << map_lines[phys].substr(map_lines[phys].find('\t') + 1)
                   << '\t' << group_of[j] << '\n';
            else
                ef << j << '\t' << "unknown_" << phys << '\t' << group_of[j] << '\n';
        }
    }
    return exp_path;
}

// Write weights_grouped.txt: label\tweight\tgroup for each non-zero expanded parameter.
// Uses getParameters() from the solver and the augmented expanded.map (Position\tLabel\tGroup).
static void write_grouped_weights(const regression::RegressionAnalysis& regr,
                                  const fs::path& expanded_map_path,
                                  const fs::path& output_path)
{
    arma::vec params = regr.getParameters();
    double intercept = regr.getInterceptValue();

    // Read augmented expanded.map (Position\tLabel\tGroup)
    std::vector<std::pair<std::string, int>> label_group; // label, group per expanded position
    {
        std::ifstream mf(expanded_map_path);
        std::string line;
        std::getline(mf, line); // skip header
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            auto t1 = line.find('\t');
            if (t1 == std::string::npos) continue;
            auto t2 = line.find('\t', t1 + 1);
            std::string label = (t2 != std::string::npos)
                ? line.substr(t1 + 1, t2 - t1 - 1)
                : line.substr(t1 + 1);
            int group = (t2 != std::string::npos) ? std::stoi(line.substr(t2 + 1)) : -1;
            label_group.push_back({label, group});
        }
    }

    std::ofstream wf(output_path);
    wf << std::setprecision(17) << std::scientific;
    size_t n = std::min(static_cast<size_t>(params.n_elem), label_group.size());
    for (size_t i = 0; i < n; ++i) {
        if (params(i) == 0.0) continue;
        wf << label_group[i].first << '\t' << params(i) << '\t' << label_group[i].second << '\n';
    }
    wf << "Intercept\t" << intercept << "\t-1\n";
}

// Sort key for the position token of a PSS entry. FASTA tokens are bare integers
// ("137"); VCF tokens are "{chrom}:{pos}" ("chr1:100"), which must order by
// chromosome and then numerically by coordinate. Parsing a VCF token with stoul
// would throw, so both forms go through this. Non-numeric tails fall back to a
// plain lexicographic compare of the whole token.
std::pair<std::string, unsigned long long> pss_pos_key(const std::string& token) {
    size_t colon = token.rfind(':');
    std::string head = (colon == std::string::npos) ? std::string() : token.substr(0, colon);
    std::string num  = (colon == std::string::npos) ? token : token.substr(colon + 1);
    unsigned long long pos = 0;
    if (num.empty()) return {token, 0ULL};
    for (char c : num) {
        if (c < '0' || c > '9') return {token, 0ULL};
        pos = pos * 10ULL + static_cast<unsigned long long>(c - '0');
    }
    return {head, pos};
}

} // anonymous namespace

TrainResult train(const EncodeResult& enc, const TrainOptions& opts_in) {
    // Step 1: Merge enc.extra_params into a local copy of opts.params
    // extra_params takes precedence (e.g. sWeight path set by class balancing)
    TrainOptions opts = opts_in;
    for (auto& [k, v] : enc.extra_params)
        opts.params[k] = v;

    const fs::path& output_dir      = opts.output_dir;
    const std::string& method       = opts.method;
    const arma::fmat&  features     = enc.features;
    const arma::frowvec& responses  = enc.responses;
    arma::mat          alg_table    = enc.alg_table; // mutable copy; row 2 may be rewritten per penalty term
    uint32_t N                      = enc.N;

    // Build per-group feature lengths from alg_table rows 0/1 for group penalty computation
    std::vector<size_t> group_feature_lengths;
    group_feature_lengths.reserve(alg_table.n_cols);
    for (arma::uword gi = 0; gi < alg_table.n_cols; ++gi) {
        auto start = static_cast<size_t>(alg_table(0, gi));
        auto end   = static_cast<size_t>(alg_table(1, gi));
        group_feature_lengths.push_back(end >= start ? end - start + 1 : 0);
    }

    auto penalty_terms = group_penalty::build_penalty_terms(
        opts.group_penalty_type,
        opts.initial_gp_value, opts.final_gp_value, opts.gp_step,
        enc.group_var_site_counts.empty() ? group_feature_lengths : enc.group_var_site_counts);

    const bool multi_penalty = penalty_terms.size() > 1;

    TrainResult result;
    result.output_dir = output_dir;

    // Step 2: If method is empty, write lambda_list.txt with default lambda and return.
    if (method.empty()) {
        fs::path gen_path = output_dir / "lambda_list.txt";
        {
            std::ofstream f(gen_path);
            f << opts.lambda[0] << " " << opts.lambda[1] << "\n";
        }
        return result;
    }

    // Step 3: Build lambdas vector
    auto load_lambda_list = [](const fs::path& p) {
        std::vector<std::array<double, 2>> list;
        std::ifstream f(p);
        if (!f) throw std::runtime_error("Cannot open lambda file: " + p.string());
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            double l1, l2;
            if (!(ss >> l1 >> l2))
                throw std::runtime_error("Bad lambda line: " + line);
            list.push_back({l1, l2});
        }
        if (list.empty())
            throw std::runtime_error("Lambda file contains no valid pairs");
        return list;
    };

    std::vector<std::array<double, 2>> lambdas;
    if (!opts.lambda_file_path.empty()) {
        lambdas = load_lambda_list(opts.lambda_file_path);
        for (const auto& lp : lambdas) {
            validate_lambda_in_open_unit(lp[0]);
            validate_lambda_in_open_unit(lp[1]);
        }
    } else if (opts.lambda_grid_set) {
        auto parse_spec = [](const std::string& spec, double& vmax_out) {
            std::vector<double> vals;
            double vmin, vmax, vstep;
            char c1, c2;
            std::istringstream ss(spec);
            if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || c1 != ',' || c2 != ',')
                throw std::runtime_error("--lambda-grid spec must be 'min,max,step': " + spec);
            if (vstep <= 0.0) throw std::runtime_error("lambda-grid step must be > 0");
            for (double v = vmin; v <= vmax + vstep * 1e-9; v += vstep)
                vals.push_back(v);
            vmax_out = vmax;
            return vals;
        };
        double vmax1 = 0.0, vmax2 = 0.0;
        auto v1 = parse_spec(opts.lambda_grid_specs[0], vmax1);
        auto v2 = parse_spec(opts.lambda_grid_specs[1], vmax2);
        for (double v : v1) validate_lambda_in_open_unit(v);
        for (double v : v2) validate_lambda_in_open_unit(v);
        if (opts.use_logspace) {
            // Anchor logspace endpoints to [vmin, vmax_effective] where vmax_effective is the
            // largest sweep value strictly < vmax AND < 1. Indices map uniformly onto [0,1] so
            // the largest sweep value projects to vmax_effective exactly.
            auto project = [](std::vector<double>& vals, double vmax, const char* which) {
                // Collect the anchorable values separately rather than erasing in
                // place. A single-point spec (min == max) has nothing strictly
                // below vmax, so an in-place erase would empty the sweep and the
                // size guard below could no longer put it back -- leaving an empty
                // Cartesian product and a baffling "no valid pairs" error.
                std::vector<double> kept;
                kept.reserve(vals.size());
                for (double v : vals)
                    if (v < vmax && v < 1.0) kept.push_back(v);

                if (kept.size() < 2) {
                    // Nothing to anchor to; leave the linear sweep untouched. A
                    // deliberately fixed lambda is normal usage and stays quiet;
                    // a real sweep that could not be projected is worth flagging.
                    if (vals.size() > 1)
                        std::cerr << "Note: --use-logspace left the " << which
                                  << " sweep linear; fewer than two of its values fall"
                                     " below its max.\n";
                    return;
                }

                const double vmin_eff = kept.front();
                const double vmax_eff = kept.back();
                const double ratio = vmax_eff / vmin_eff;
                const size_t N = kept.size();
                for (size_t i = 0; i < N; ++i) {
                    double t = static_cast<double>(i) / static_cast<double>(N - 1);
                    kept[i] = vmin_eff * std::pow(ratio, t);
                }
                vals = std::move(kept);
            };
            project(v1, vmax1, "lambda1");
            project(v2, vmax2, "lambda2");
        }
        fs::path gen_path = output_dir / "lambda_list.txt";
        {
            std::ofstream f(gen_path);
            for (double l1 : v1)
                for (double l2 : v2)
                    f << l1 << " " << l2 << "\n";
        }
        lambdas = load_lambda_list(gen_path);
        std::cout << "  Lambda grid: " << v1.size() << " x " << v2.size()
                  << " = " << lambdas.size() << " pairs -> "
                  << gen_path.string() << "\n";
    } else {
        validate_lambda_in_open_unit(opts.lambda[0]);
        validate_lambda_in_open_unit(opts.lambda[1]);
        fs::path gen_path = output_dir / "lambda_list.txt";
        {
            std::ofstream f(gen_path);
            f << opts.lambda[0] << " " << opts.lambda[1] << "\n";
        }
        lambdas = load_lambda_list(gen_path);
    }

    process_log::Section plog(opts.output_dir / "process_log.txt", "train");
    plog.param("method",    opts.method.empty() ? std::string("(none)") : opts.method)
        .param("precision", std::string(opts.precision == regression::Precision::FP64 ? "fp64" : "fp32"));
    if (opts.lambda_grid_set)
        plog.param("lambda_grid", opts.lambda_grid_specs[0] + " / " + opts.lambda_grid_specs[1]);
    else if (!opts.lambda_file_path.empty())
        plog.param("lambda_file", opts.lambda_file_path);
    else
        plog.param("lambda", std::to_string(opts.lambda[0]) + " " + std::to_string(opts.lambda[1]));
    plog.param("use_logspace", opts.use_logspace);
    plog.param("lambdas_count", (int)lambdas.size());
    if (opts.nfolds > 0)      plog.param("nfolds",     opts.nfolds);
    if (opts.min_groups > 0)  plog.param("min_groups", opts.min_groups);
    for (auto& [k, v] : opts.params) plog.param("param_" + k, v);

    try {

    // Step 4: Print Phase 3 header
    std::cout << "\n--- Phase 3: Regression ---\n";
    std::cout << "  Method:  " << method << "\n";
    std::cout << "  Lambdas: " << lambdas.size() << " pair(s)\n";
    if (opts.nfolds > 0)
        std::cout << "  K-fold CV: " << opts.nfolds << " folds\n";
    if (!opts.params.empty()) {
        std::cout << "  Params:\n";
        for (auto& [k, v] : opts.params)
            std::cout << "    " << k << " = " << v << "\n";
    }

    auto regr_start = std::chrono::steady_clock::now();

    // Step 5: Read combined.map to build label_to_col map
    pipeline_utils::log_rss("train: before label_to_col");
    std::unordered_map<std::string, uint64_t> label_to_col;
    {
        std::ifstream map_f(output_dir / "combined.map");
        std::string line;
        std::getline(map_f, line); // skip header
        uint64_t col = 0;
        while (std::getline(map_f, line)) {
            if (line.empty()) continue;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            label_to_col[line.substr(tab + 1)] = col++;
        }
    }
    pipeline_utils::log_rss("train: after label_to_col");

    // Step 6: Build xval_idxs if nfolds > 0
    //   priority: --cv-assignments file > --cv-seed shuffle > legacy i % nfolds
    arma::rowvec xval_idxs(N);
    if (opts.nfolds > 0) {
        if (!opts.cv_assignments_path.empty()) {
            std::ifstream af(opts.cv_assignments_path);
            if (!af) throw std::runtime_error(
                "Cannot open --cv-assignments file: " + opts.cv_assignments_path);

            // Parse header to locate SequenceID and Fold columns. If no header,
            // assume two-column (SequenceID, Fold).
            std::string line;
            int seq_col = 0, fold_col = 1;
            std::streampos data_start = af.tellg();
            if (std::getline(af, line)) {
                if (line.find("SequenceID") != std::string::npos
                        && line.find("Fold") != std::string::npos) {
                    seq_col = -1; fold_col = -1;
                    std::stringstream ss(line);
                    std::string tok;
                    int idx = 0;
                    while (std::getline(ss, tok, '\t')) {
                        if (tok == "SequenceID") seq_col = idx;
                        else if (tok == "Fold")  fold_col = idx;
                        ++idx;
                    }
                    if (seq_col < 0 || fold_col < 0)
                        throw std::runtime_error(
                            "--cv-assignments file header must contain both 'SequenceID' and 'Fold' columns");
                    data_start = af.tellg();
                }
            }
            af.clear();
            af.seekg(data_start);

            std::unordered_map<std::string, int> assigns;
            while (std::getline(af, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::vector<std::string> cols;
                std::stringstream ss(line);
                std::string tok;
                while (std::getline(ss, tok, '\t')) cols.push_back(tok);
                if (static_cast<int>(cols.size()) <= std::max(seq_col, fold_col)) continue;
                int fold;
                try { fold = std::stoi(cols[fold_col]); }
                catch (...) { throw std::runtime_error(
                    "--cv-assignments: cannot parse Fold value '" + cols[fold_col] +
                    "' for sample '" + cols[seq_col] + "'"); }
                if (fold < 0 || fold >= opts.nfolds)
                    throw std::runtime_error(
                        "--cv-assignments: Fold " + std::to_string(fold) + " for sample '" +
                        cols[seq_col] + "' is out of range [0, " + std::to_string(opts.nfolds - 1) + "]");
                assigns[cols[seq_col]] = fold;
            }

            std::vector<std::string> missing;
            for (uint32_t i = 0; i < N; ++i) {
                auto it = assigns.find(enc.seq_names[i]);
                if (it == assigns.end()) { missing.push_back(enc.seq_names[i]); continue; }
                xval_idxs(i) = static_cast<double>(it->second);
            }
            if (!missing.empty()) {
                std::string msg = std::to_string(missing.size())
                    + " sample(s) missing from --cv-assignments file:\n";
                for (auto& m : missing) msg += "  " + m + "\n";
                throw std::runtime_error(msg);
            }
            std::cout << "CV assignments loaded from " << opts.cv_assignments_path
                      << " (" << N << " samples, " << opts.nfolds << " folds)\n";
        } else if (opts.cv_seed >= 0) {
            std::vector<uint32_t> perm(N);
            std::iota(perm.begin(), perm.end(), 0u);
            std::mt19937 rng(static_cast<uint32_t>(opts.cv_seed));
            std::shuffle(perm.begin(), perm.end(), rng);
            for (uint32_t i = 0; i < N; ++i)
                xval_idxs(perm[i]) = static_cast<double>(i % opts.nfolds);
            std::cout << "CV fold assignment: shuffled round-robin (seed=" << opts.cv_seed << ")\n";
        } else {
            for (uint32_t i = 0; i < N; ++i)
                xval_idxs(i) = static_cast<double>(i % opts.nfolds);
        }
    }

    // Step 7: Build sorted_stems_desc for numeric longest-prefix matching
    std::vector<std::string> sorted_stems_desc = enc.all_stems_ordered;
    std::sort(sorted_stems_desc.begin(), sorted_stems_desc.end(),
        [](const std::string& a, const std::string& b){ return a.size() > b.size(); });

    // VCF labels are "{stem}_{chrom}:{pos}_{allele}". Resolve the stem by longest
    // prefix match rather than walking back from the end as the FASTA parser does:
    // chromosome names routinely contain underscores (chr1_KI270706v1_random), and
    // rfind-twice would split inside the chromosome and scatter GSS across phantom
    // genes. The site key is then everything before the final underscore, which is
    // safe because the allele is sanitised to contain none.
    auto split_vcf_label = [&sorted_stems_desc](const std::string& label,
                                                std::string& stem,
                                                std::string& pos_str) -> bool {
        for (const auto& s : sorted_stems_desc) {
            if (label.size() > s.size() + 1 &&
                label.compare(0, s.size(), s) == 0 &&
                label[s.size()] == '_') {
                stem = s;
                std::string rest = label.substr(s.size() + 1);
                size_t us = rest.rfind('_');
                pos_str = (us == std::string::npos) ? std::string() : rest.substr(0, us);
                return true;
            }
        }
        return false;
    };

    // Step 8: Helper lambda — compute and write GSS/PSS, return nonzero gene count
    auto compute_sig_scores = [&](const fs::path& wpath, const fs::path& lam_dir,
                                  std::ostringstream& sout, int lam_idx) -> int {
        bool is_numeric_mode = (enc.datatype == "numeric");
        bool is_vcf_mode     = (enc.datatype == "vcf");
        bool is_olsg = (method == "olsg_lasso_logisticr" || method == "olsg_lasso_leastr");
        std::map<std::string, double> gss; // stem -> sum(|w|)
        // PSS key = "stem\tpos_str" -> sum(|w|)
        std::map<std::string, double> pss;
        // OSS: per-group sum of |w| (olsg_lasso methods only)
        std::map<int, double> oss;
        double hss = 0.0;

        // Build col->group lookup for olsg_lasso OSS
        std::vector<int> col_group;
        if (is_olsg && alg_table.n_cols > 0) {
            size_t n_cols = static_cast<size_t>(alg_table(1, alg_table.n_cols - 1));
            col_group.assign(n_cols, -1);
            for (arma::uword gi = 0; gi < alg_table.n_cols; ++gi) {
                int start = static_cast<int>(alg_table(0, gi)) - 1; // 1-based to 0-based
                int end   = static_cast<int>(alg_table(1, gi)) - 1;
                for (int j = start; j <= end && j < static_cast<int>(n_cols); ++j)
                    col_group[j] = static_cast<int>(gi);
            }
        }

        std::ifstream wf(wpath);
        std::string wline;
        while (std::getline(wf, wline)) {
            if (wline.empty()) continue;
            auto tab = wline.find('\t');
            if (tab == std::string::npos) continue;
            std::string label = wline.substr(0, tab);
            double w = std::stod(wline.substr(tab + 1));
            if (label == "Intercept") continue;
            double aw = std::abs(w);
            hss += aw;

            // Accumulate per-group OSS for olsg_lasso methods
            if (is_olsg && !col_group.empty()) {
                auto cit = label_to_col.find(label);
                if (cit != label_to_col.end() && cit->second < col_group.size()) {
                    int gi = col_group[cit->second];
                    if (gi >= 0) oss[gi] += aw;
                }
            }

            if (is_numeric_mode) {
                // Longest-prefix match against known stems
                for (auto& s : sorted_stems_desc) {
                    if (label.size() > s.size() + 1 &&
                        label.compare(0, s.size(), s) == 0 &&
                        label[s.size()] == '_') {
                        gss[s] += aw;
                        break;
                    }
                }
            } else if (is_vcf_mode) {
                std::string stem, pos_str;
                if (!split_vcf_label(label, stem, pos_str)) continue;
                gss[stem] += aw;
                if (!pos_str.empty()) pss[stem + "\t" + pos_str] += aw;
            } else {
                // Check for {stem}_minor label (gene-level, no position)
                if (label.size() > 6 && label.compare(label.size() - 6, 6, "_minor") == 0) {
                    std::string stem = label.substr(0, label.size() - 6);
                    gss[stem] += aw;
                    continue;
                }
                // FASTA: {stem}_{pos}_{allele}
                size_t us2 = label.rfind('_');
                if (us2 == std::string::npos || us2 == 0) continue;
                size_t us1 = label.rfind('_', us2 - 1);
                if (us1 == std::string::npos) continue;
                std::string stem    = label.substr(0, us1);
                std::string pos_str = label.substr(us1 + 1, us2 - us1 - 1);
                gss[stem] += aw;
                pss[stem + "\t" + pos_str] += aw;
            }
        }

        // Write gss.txt: gene\tsum(|w|), sorted desc
        {
            std::vector<std::pair<double, std::string>> sorted_gss;
            sorted_gss.reserve(gss.size());
            for (auto& [g, v] : gss) sorted_gss.push_back({v, g});
            std::sort(sorted_gss.rbegin(), sorted_gss.rend());
            std::ofstream gf(lam_dir / "gss.txt");
            gf << std::setprecision(15);
            for (auto& [v, g] : sorted_gss) gf << g << '\t' << v << '\n';
        }

        // Write pss.txt (FASTA only): gene_pos\tsum(|w|)
        if (!is_numeric_mode && !pss.empty()) {
            std::vector<std::pair<std::string, double>> pss_entries(pss.begin(), pss.end());
            std::sort(pss_entries.begin(), pss_entries.end(),
                [](const auto& a, const auto& b) {
                    auto ta = a.first.find('\t'), tb = b.first.find('\t');
                    std::string sa = a.first.substr(0, ta), sb = b.first.substr(0, tb);
                    if (sa != sb) return sa < sb;
                    return pss_pos_key(a.first.substr(ta + 1))
                         < pss_pos_key(b.first.substr(tb + 1));
                });
            std::ofstream pf(lam_dir / "pss.txt");
            pf << std::fixed << std::setprecision(15);
            for (auto& [key, v] : pss_entries) {
                auto tp = key.find('\t');
                pf << key.substr(0, tp) << '_' << key.substr(tp + 1) << '\t' << v << '\n';
            }
        }

        // Write oss.txt (olsg_lasso methods): group\tsum(GSS), sorted by group index
        if (!oss.empty()) {
            std::ofstream of(lam_dir / "oss.txt");
            of << std::setprecision(15);
            for (auto& [gi, v] : oss) of << gi << '\t' << v << '\n';
        }

        sout << "  [" << lam_idx << "] HSS=" << std::fixed << std::setprecision(4) << hss << "\n";
        return static_cast<int>(gss.size());
    };

    // Step 8b: Helper — grouped sig scores for ol_sg_lasso methods
    auto compute_sig_scores_grouped = [&](const fs::path& wg_path, const fs::path& lam_dir,
                                          std::ostringstream& sout, int lam_idx) -> int {
        bool is_numeric_mode = (enc.datatype == "numeric");
        bool is_vcf_mode     = (enc.datatype == "vcf");

        // Flat aggregates (backward-compat gss.txt / pss.txt)
        std::map<std::string, double> gss;
        std::map<std::string, double> pss;
        // Grouped aggregates
        std::map<std::pair<std::string, int>, double> gss_grp; // (stem, group) -> sum(|w|)
        std::map<std::pair<std::string, int>, double> pss_grp; // (stem_pos key, group) -> sum(|w|)
        std::map<int, double> oss;                              // group -> sum(|w|)
        double hss = 0.0;

        std::ifstream wf(wg_path);
        std::string wline;
        while (std::getline(wf, wline)) {
            if (wline.empty()) continue;
            auto tab1 = wline.find('\t');
            if (tab1 == std::string::npos) continue;
            std::string label = wline.substr(0, tab1);
            if (label == "Intercept") continue;
            auto tab2 = wline.find('\t', tab1 + 1);
            double w = std::stod(wline.substr(tab1 + 1, tab2 == std::string::npos ? std::string::npos : tab2 - tab1 - 1));
            int group = (tab2 != std::string::npos) ? std::stoi(wline.substr(tab2 + 1)) : -1;
            double aw = std::abs(w);
            hss += aw;

            std::string stem, pos_str;
            if (is_numeric_mode) {
                for (auto& s : sorted_stems_desc) {
                    if (label.size() > s.size() + 1 &&
                        label.compare(0, s.size(), s) == 0 &&
                        label[s.size()] == '_') {
                        stem = s;
                        break;
                    }
                }
                if (stem.empty()) continue;
            } else if (is_vcf_mode) {
                if (!split_vcf_label(label, stem, pos_str)) continue;
            } else {
                // Check for {stem}_minor label
                if (label.size() > 6 && label.compare(label.size() - 6, 6, "_minor") == 0) {
                    stem = label.substr(0, label.size() - 6);
                    gss[stem] += aw;
                    gss_grp[{stem, group}] += aw;
                    oss[group] += aw;
                    continue;
                }
                // Check for tiered minor: {stem}_tminor_Xpct
                if (label.find("_tminor_") != std::string::npos) {
                    auto tpos = label.find("_tminor_");
                    stem = label.substr(0, tpos);
                    gss[stem] += aw;
                    gss_grp[{stem, group}] += aw;
                    oss[group] += aw;
                    continue;
                }
                // FASTA: {stem}_{pos}_{allele}
                size_t us2 = label.rfind('_');
                if (us2 == std::string::npos || us2 == 0) continue;
                size_t us1 = label.rfind('_', us2 - 1);
                if (us1 == std::string::npos) continue;
                stem    = label.substr(0, us1);
                pos_str = label.substr(us1 + 1, us2 - us1 - 1);
            }

            gss[stem] += aw;
            gss_grp[{stem, group}] += aw;
            oss[group] += aw;
            if (!is_numeric_mode && !pos_str.empty()) {
                std::string pss_key = stem + "\t" + pos_str;
                pss[pss_key] += aw;
                pss_grp[{pss_key, group}] += aw;
            }
        }

        // Write gss.txt (flat, backward compat)
        {
            std::vector<std::pair<double, std::string>> sorted_gss;
            sorted_gss.reserve(gss.size());
            for (auto& [g, v] : gss) sorted_gss.push_back({v, g});
            std::sort(sorted_gss.rbegin(), sorted_gss.rend());
            std::ofstream gf(lam_dir / "gss.txt");
            gf << std::setprecision(15);
            for (auto& [v, g] : sorted_gss) gf << g << '\t' << v << '\n';
        }

        // Write gss_grouped.txt: gene\tgroup\tvalue
        {
            std::vector<std::tuple<double, std::string, int>> sorted_gss_g;
            sorted_gss_g.reserve(gss_grp.size());
            for (auto& [key, v] : gss_grp) sorted_gss_g.push_back({v, key.first, key.second});
            std::sort(sorted_gss_g.rbegin(), sorted_gss_g.rend());
            std::ofstream gf(lam_dir / "gss_grouped.txt");
            gf << std::setprecision(15);
            for (auto& [v, g, gi] : sorted_gss_g) gf << g << '\t' << gi << '\t' << v << '\n';
        }

        // Write pss.txt (flat, backward compat)
        if (!is_numeric_mode && !pss.empty()) {
            std::vector<std::pair<std::string, double>> pss_entries(pss.begin(), pss.end());
            std::sort(pss_entries.begin(), pss_entries.end(),
                [](const auto& a, const auto& b) {
                    auto ta = a.first.find('\t'), tb = b.first.find('\t');
                    std::string sa = a.first.substr(0, ta), sb = b.first.substr(0, tb);
                    if (sa != sb) return sa < sb;
                    return pss_pos_key(a.first.substr(ta + 1))
                         < pss_pos_key(b.first.substr(tb + 1));
                });
            std::ofstream pf(lam_dir / "pss.txt");
            pf << std::fixed << std::setprecision(15);
            for (auto& [key, v] : pss_entries) {
                auto tp = key.find('\t');
                pf << key.substr(0, tp) << '_' << key.substr(tp + 1) << '\t' << v << '\n';
            }
        }

        // Write pss_grouped.txt: gene_pos\tgroup\tvalue
        if (!is_numeric_mode && !pss_grp.empty()) {
            std::vector<std::tuple<std::string, int, double>> pss_g_entries;
            pss_g_entries.reserve(pss_grp.size());
            for (auto& [key, v] : pss_grp) {
                auto tp = key.first.find('\t');
                std::string label = key.first.substr(0, tp) + "_" + key.first.substr(tp + 1);
                pss_g_entries.push_back({label, key.second, v});
            }
            std::sort(pss_g_entries.begin(), pss_g_entries.end());
            std::ofstream pf(lam_dir / "pss_grouped.txt");
            pf << std::fixed << std::setprecision(15);
            for (auto& [label, gi, v] : pss_g_entries) pf << label << '\t' << gi << '\t' << v << '\n';
        }

        // Write oss.txt: group\tvalue
        {
            std::vector<std::pair<double, int>> sorted_oss;
            sorted_oss.reserve(oss.size());
            for (auto& [gi, v] : oss) sorted_oss.push_back({v, gi});
            std::sort(sorted_oss.rbegin(), sorted_oss.rend());
            std::ofstream of(lam_dir / "oss.txt");
            of << std::setprecision(15);
            for (auto& [v, gi] : sorted_oss) of << gi << '\t' << v << '\n';
        }

        sout << "  [" << lam_idx << "] HSS=" << std::fixed << std::setprecision(4) << hss << "\n";
        return static_cast<int>(gss.size());
    };

    // Step 9: Validate skip-ahead ordering if min_groups > 0
    bool skip_ahead_valid = true;
    if (opts.min_groups > 0 && lambdas.size() > 1) {
        for (size_t i = 1; i < lambdas.size(); ++i) {
            bool same_l1 = (lambdas[i][0] == lambdas[i-1][0]);
            bool l1_incr = (lambdas[i][0] >  lambdas[i-1][0]);
            bool l2_incr = (lambdas[i][1] >  lambdas[i-1][1]);
            if (!same_l1 && !l1_incr) { skip_ahead_valid = false; break; }
            if ( same_l1 && !l2_incr) { skip_ahead_valid = false; break; }
        }
        if (!skip_ahead_valid)
            std::cerr << "Warning: lambda list is not in lambda1-outer/lambda2-inner order; "
                         "--min-groups skip-ahead disabled.\n";
    }

    double min_lambda2 = std::numeric_limits<double>::infinity();
    for (auto& lam : lambdas) min_lambda2 = std::min(min_lambda2, lam[1]);

    // Effective thread count for the grid loop. Force single-thread when nfolds>0
    // (CV + grid + solver threads is three-level nested parallelism — out of scope
    // for this change).
    unsigned int n_threads = opts.threads == 0 ? 1 : opts.threads;
    if (opts.nfolds > 0 && n_threads > 1) {
        std::cerr << "Note: --threads > 1 ignored when --nfolds > 0 "
                     "(grid-level parallelism only supported for non-CV mode).\n";
        n_threads = 1;
    }
    if (n_threads > static_cast<unsigned int>(lambdas.size()))
        n_threads = static_cast<unsigned int>(lambdas.size());

    // Hoist the idempotent "field" param assignment out of the per-point body so
    // concurrent workers don't race on map insertion. The conditional depends
    // only on enc + method, so the value would be the same at every grid point.
    if (enc.is_overlapping
        && (method == "olsg_lasso_leastr" || method == "olsg_lasso_logisticr"
            || method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
        && opts.params.find("field") == opts.params.end()) {
        opts.params["field"] = enc.field_path.string();
    }
    // ol_sg_lasso on non-overlapping input: generate identity field (1,2,...,n)
    // so the solver runs correctly. With identity field, output is bit-for-bit
    // identical to sg_lasso (0+x = x in IEEE 754).
    if ((method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr") && !enc.is_overlapping
        && opts.params.find("field") == opts.params.end()) {
        fs::path id_field = output_dir / "field.txt";
        if (!fs::exists(id_field)) {
            std::ofstream ff(id_field);
            for (arma::uword j = 0; j < features.n_cols; ++j) {
                if (j > 0) ff << ',';
                ff << (j + 1);  // 1-based
            }
            ff << '\n';
        }
        opts.params["field"] = id_field.string();
    }
    pipeline_utils::log_rss("train: before lambda loop");

    // ── Model generation log ────────────────────────────────────────────────────
    // Enumerate every model this run intends to produce BEFORE solving any of
    // them, then update each row in place as it completes, is skipped or fails.
    // A run cancelled partway through therefore leaves an accurate record of
    // which models finished, which `evaluate --from-run` scores and a future
    // resume can use to pick up the still-pending points.
    // --cv-scores promotes each fold model to a separately scoreable model, so
    // a CV lambda contributes nfolds rows instead of one.
    const bool   cv_scores      = (opts.nfolds > 0 && opts.cv_scores);
    const size_t rows_per_lambda = cv_scores ? static_cast<size_t>(opts.nfolds) : 1;

    model_log::Log mlog(output_dir / "models.tsv");
    for (size_t pi = 0; pi < penalty_terms.size(); ++pi) {
        const std::string pen_prefix =
            multi_penalty ? "penalty_" + std::to_string(pi) + "/" : "";
        for (size_t li = 0; li < lambdas.size(); ++li) {
            const std::string lam_prefix = pen_prefix + "lambda_" + std::to_string(li) + "/";
            if (cv_scores) {
                for (int k = 0; k < opts.nfolds; ++k)
                    mlog.add(pi, penalty_terms[pi], li, k, lambdas[li][0], lambdas[li][1],
                             lam_prefix + "weights_fold_" + std::to_string(k) + ".txt");
            } else {
                // Plain k-fold CV writes only weights_fold_N.txt and reports the
                // pooled held-out prediction, so there is no single scoreable
                // model and the row carries no weights path.
                std::string rel;
                if (opts.nfolds == 0) rel = lam_prefix + "weights.txt";
                mlog.add(pi, penalty_terms[pi], li, -1, lambdas[li][0], lambdas[li][1], rel);
            }
        }
    }
    mlog.flush();

    // Row index for one (penalty, lambda, fold). fold < 0 selects the lambda's
    // first row, which is its only row when --cv-scores is off.
    auto gid_of = [&](size_t pi, size_t li, int fold) {
        return (pi * lambdas.size() + li) * rows_per_lambda
             + (fold < 0 ? 0u : static_cast<size_t>(fold));
    };
    // Apply a status to every row belonging to one lambda.
    auto mark_lambda = [&](size_t pi, size_t li, model_log::Status st, const std::string& d) {
        for (size_t k = 0; k < rows_per_lambda; ++k)
            mlog.set(gid_of(pi, li, static_cast<int>(k)), st, d);
    };
    std::cout << "  Model log: " << mlog.size() << " model(s) -> "
              << (output_dir / "models.tsv").string() << "\n";

    // ── Penalty loop ────────────────────────────────────────────────────────────
    for (size_t pi = 0; pi < penalty_terms.size(); ++pi) {
        double penalty = penalty_terms[pi];

        // Rewrite alg_table row 2 with group weights for this penalty term.
        // Skip for the default single-term "std" case (row 2 already correct from encode).
        if (opts.group_penalty_type != "std" || multi_penalty) {
            auto weights = group_penalty::compute_group_weights(
                opts.group_penalty_type, penalty,
                group_feature_lengths,
                enc.group_var_site_counts.empty() ? group_feature_lengths : enc.group_var_site_counts);
            for (size_t gi = 0; gi < weights.size(); ++gi)
                alg_table(2, gi) = weights[gi];
        }

        // When multiple penalty terms exist, nest output under penalty_N/
        fs::path pen_dir = multi_penalty
            ? output_dir / ("penalty_" + std::to_string(pi))
            : output_dir;
        if (multi_penalty) {
            fs::create_directories(pen_dir);
            std::cout << "  ── Penalty term " << pi << " = " << penalty << " ──\n";
        }

    // Per-grid-index gene_count captured during solve; -1 = not solved / errored.
    // Used by the parallel post-hoc pruner to replay sequential skip-ahead logic.
    std::vector<int> gene_counts(lambdas.size(), -1);

    // Step 10: Lambda loop — branches on thread count.
    if (n_threads <= 1) {
        // ---- Sequential path (unchanged semantics for golden-output parity) ----
        double max_lambda2 = std::numeric_limits<double>::infinity();

        for (int idx = 0; idx < (int)lambdas.size(); ++idx) {
            auto& lam = lambdas[idx];
            fs::path lam_dir = pen_dir / ("lambda_" + std::to_string(idx));
            fs::create_directories(lam_dir); // always create dir (sentinel for drphylo)

            const size_t gid = gid_of(pi, static_cast<size_t>(idx), -1);

            if (skip_ahead_valid && opts.min_groups > 0 && lam[1] > max_lambda2) {
                std::cout << "  [" << idx << "] lambda=[" << lam[0] << "," << lam[1]
                          << "] Skipping (gene count threshold)\n";
                mark_lambda(pi, static_cast<size_t>(idx),
                            model_log::Status::Skipped, "min-groups skip-ahead");
                continue;
            }

            std::ostringstream out;
            out << std::fixed << std::setprecision(4);

            // Wraps both the single-model and CV branches below so a solver
            // failure lands in models.tsv before it unwinds. Left at this
            // indentation deliberately to keep the branch bodies unchanged.
            try {
            if (opts.nfolds == 0) {
                // Single model
                auto regr = regression::createRegressionAnalysis(
                    method, features, responses, alg_table.t(), opts.params, lam, opts.precision);
                {
                    std::ofstream wo(lam_dir / "weights.txt");
                    fs::path map_path = (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                        ? generate_expanded_map(output_dir, alg_table)
                        : (output_dir / "combined.map");
                    std::ifstream mi(map_path);
                    regr->writeSparseMappedWeightsToStream(wo, mi);
                }
                if (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                    write_grouped_weights(*regr, output_dir / "expanded.map", lam_dir / "weights_grouped.txt");
                out << "  [" << idx << "] lambda=[" << lam[0] << ","
                    << lam[1] << "] -> " << (lam_dir / "weights.txt").string() << "\n";
                int gene_count = (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                    ? compute_sig_scores_grouped(lam_dir / "weights_grouped.txt", lam_dir, out, idx)
                    : compute_sig_scores(lam_dir / "weights.txt", lam_dir, out, idx);
                out << "  [" << idx << "] Non-zero gene count: " << gene_count << "\n";
                std::cout << out.str();

                result.weights_paths.push_back(lam_dir / "weights.txt");
                result.lambdas_used.push_back(lam);
                result.penalties_used.push_back(penalty);
                gene_counts[idx] = gene_count;
                mlog.set(gid, model_log::Status::Complete);

                if (skip_ahead_valid && opts.min_groups > 0 && gene_count <= opts.min_groups) {
                    if (lam[1] == min_lambda2) break; // even min lambda2 is too sparse
                    max_lambda2 = std::min(max_lambda2, lam[1]);
                }
            } else {
                // K-fold CV
                std::vector<double> cv_preds(N, 0.0);
                for (int k = 0; k < opts.nfolds; ++k) {
                    auto regr = regression::createRegressionAnalysisXVal(
                        method, features, responses, alg_table.t(), opts.params, lam,
                        xval_idxs, k, opts.precision);
                    fs::path fw = lam_dir / ("weights_fold_" + std::to_string(k) + ".txt");
                    {
                        std::ofstream wo(fw);
                        fs::path map_path = (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                            ? generate_expanded_map(output_dir, alg_table)
                            : (output_dir / "combined.map");
                        std::ifstream mi(map_path);
                        regr->writeSparseMappedWeightsToStream(wo, mi);
                    }
                    if (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                        write_grouped_weights(*regr, output_dir / "expanded.map",
                            lam_dir / ("weights_fold_" + std::to_string(k) + "_grouped.txt"));

                    if (cv_scores) {
                        // The fold model is fully written, so publish it as a
                        // scoreable model in its own right. Phase 4 picks it up
                        // from weights_paths and evaluates it against the full
                        // dataset, exactly as it would an ordinary model.
                        result.weights_paths.push_back(fw);
                        result.lambdas_used.push_back(lam);
                        result.penalties_used.push_back(penalty);
                        mlog.set(gid_of(pi, static_cast<size_t>(idx), k),
                                 model_log::Status::Complete);
                    }

                    // Parse fold weights
                    double fold_intercept = 0.0;
                    std::vector<std::pair<uint64_t, double>> fold_weights;
                    {
                        std::ifstream wf(fw);
                        std::string line;
                        while (std::getline(wf, line)) {
                            if (line.empty()) continue;
                            auto tab = line.find('\t');
                            if (tab == std::string::npos) continue;
                            std::string label = line.substr(0, tab);
                            double w = std::stod(line.substr(tab + 1));
                            if (label == "Intercept") { fold_intercept = w; continue; }
                            auto it = label_to_col.find(label);
                            if (it != label_to_col.end())
                                fold_weights.emplace_back(it->second, w);
                        }
                    }

                    // Predict held-out
                    int held_out = 0;
                    for (uint32_t i = 0; i < N; ++i) {
                        if (xval_idxs(i) != static_cast<double>(k)) continue;
                        ++held_out;
                        double pred = fold_intercept;
                        for (auto& [col, w] : fold_weights)
                            pred += w * static_cast<double>(features(i, col));
                        cv_preds[i] = pred;
                    }

                    out << "  [" << idx << "] fold " << k
                        << ": held-out=" << held_out
                        << ", non-zero weights=" << fold_weights.size() << "\n";
                }

                // Write cv_predictions.txt
                {
                    std::ofstream cv_out(lam_dir / "cv_predictions.txt");
                    cv_out << std::fixed << std::setprecision(6);
                    cv_out << "SequenceID\tPredictedValue\tTrueValue\tFold\n";
                    for (uint32_t i = 0; i < N; ++i)
                        cv_out << enc.seq_names[i] << '\t'
                               << cv_preds[i] << '\t'
                               << enc.hyp_values[i] << '\t'
                               << static_cast<int>(xval_idxs(i)) << '\n';
                }

                // Classification metrics
                int tp = 0, tn = 0, fp = 0, fn = 0;
                for (uint32_t i = 0; i < N; ++i) {
                    bool pred_pos = cv_preds[i] > 0.0;
                    bool true_pos = enc.hyp_values[i] > 0.0f;
                    if      ( true_pos &&  pred_pos) ++tp;
                    else if (!true_pos && !pred_pos) ++tn;
                    else if (!true_pos &&  pred_pos) ++fp;
                    else                              ++fn;
                }
                double tpr = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
                double tnr = (tn + fp) > 0 ? static_cast<double>(tn) / (tn + fp) : 0.0;
                double fpr = (tn + fp) > 0 ? static_cast<double>(fp) / (tn + fp) : 0.0;
                double fnr = (tp + fn) > 0 ? static_cast<double>(fn) / (tp + fn) : 0.0;

                out << "  [" << idx << "] lambda=[" << lam[0] << ","
                    << lam[1] << "] CV -> " << lam_dir.filename().string() << "/\n";
                out << "    TP=" << tp << " TN=" << tn
                    << " FP=" << fp << " FN=" << fn << "\n";
                out << "    TPR=" << tpr << " TNR=" << tnr
                    << " FPR=" << fpr << " FNR=" << fnr << "\n";
                std::cout << out.str();
                if (!cv_scores)
                    mlog.set(gid, model_log::Status::Complete, "cross-validation");
            }
            } catch (const std::exception& e) {
                // Record the failure before unwinding so the log survives the
                // abort; rethrow preserves the existing whole-run-aborts
                // semantics rather than quietly continuing the grid.
                mark_lambda(pi, static_cast<size_t>(idx), model_log::Status::Failed, e.what());
                throw;
            } catch (...) {
                mark_lambda(pi, static_cast<size_t>(idx),
                            model_log::Status::Failed, "unknown error");
                throw;
            }
        }

        // The skip-ahead ratchet can break out of the grid early; anything
        // still pending for this penalty term was deliberately skipped, not
        // merely unattempted. No-op when the loop ran to completion.
        mlog.set_pending_in_penalty(pi, model_log::Status::Skipped,
                                    "min-groups skip-ahead (early break)");
    } else {
        // ---- Parallel path (nfolds==0 only; no skip-ahead during loop) ----
        const bool prune_enabled =
            opts.prune_skipped_lambda && opts.min_groups > 0 && skip_ahead_valid;
        std::cout << "  Grid loop parallelism: " << n_threads << " worker(s), "
                  << lambdas.size() << " point(s) — skip-ahead disabled in-loop"
                  << (prune_enabled ? " (post-hoc prune enabled)\n" : "\n");
        if (opts.prune_skipped_lambda && !prune_enabled)
            std::cerr << "Note: --prune-skipped-lambda is a no-op unless --min-groups > 0 "
                         "and the grid is lambda1-outer/lambda2-inner ordered.\n";

        // Pre-create all sentinel dirs up front (avoids directory-create contention).
        for (size_t idx = 0; idx < lambdas.size(); ++idx)
            fs::create_directories(pen_dir / ("lambda_" + std::to_string(idx)));

        // Pre-sized per-index result slots; workers write without locks.
        std::vector<fs::path> per_idx_weights(lambdas.size());
        std::vector<std::array<double, 2>> per_idx_lambdas(lambdas.size());
        std::vector<char> per_idx_solved(lambdas.size(), 0);

        std::atomic<size_t> next_lambda{0};
        std::mutex log_mutex;
        std::mutex err_mutex;
        std::exception_ptr first_error;

        auto worker = [&]() {
            while (true) {
                {
                    std::lock_guard<std::mutex> lk(err_mutex);
                    if (first_error) break;
                }
                size_t idx = next_lambda.fetch_add(1);
                if (idx >= lambdas.size()) break;
                try {
                    const auto& lam = lambdas[idx];
                    fs::path lam_dir = pen_dir / ("lambda_" + std::to_string(idx));

                    std::ostringstream out;
                    out << std::fixed << std::setprecision(4);

                    auto regr = regression::createRegressionAnalysis(
                        method, features, responses, alg_table.t(),
                        opts.params, lam, opts.precision);
                    fs::path wpath = lam_dir / "weights.txt";
                    {
                        std::ofstream wo(wpath);
                        fs::path map_path = (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                            ? generate_expanded_map(output_dir, alg_table)
                            : (output_dir / "combined.map");
                        std::ifstream mi(map_path);
                        regr->writeSparseMappedWeightsToStream(wo, mi);
                    }
                    if (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                        write_grouped_weights(*regr, output_dir / "expanded.map", lam_dir / "weights_grouped.txt");
                    out << "  [" << idx << "] lambda=[" << lam[0] << ","
                        << lam[1] << "] -> " << wpath.string() << "\n";
                    int gene_count = (method == "ol_sg_lasso_logisticr" || method == "ol_sg_lasso_leastr")
                        ? compute_sig_scores_grouped(lam_dir / "weights_grouped.txt", lam_dir, out, (int)idx)
                        : compute_sig_scores(wpath, lam_dir, out, (int)idx);
                    out << "  [" << idx << "] Non-zero gene count: " << gene_count << "\n";

                    per_idx_weights[idx] = wpath;
                    per_idx_lambdas[idx] = lam;
                    per_idx_solved[idx]  = 1;
                    gene_counts[idx]     = gene_count;
                    mlog.set(gid_of(pi, idx, -1), model_log::Status::Complete);

                    std::lock_guard<std::mutex> lk(log_mutex);
                    std::cout << out.str();
                } catch (const std::exception& e) {
                    mlog.set(gid_of(pi, idx, -1), model_log::Status::Failed, e.what());
                    std::lock_guard<std::mutex> lk(err_mutex);
                    if (!first_error) first_error = std::current_exception();
                } catch (...) {
                    mlog.set(gid_of(pi, idx, -1), model_log::Status::Failed, "unknown error");
                    std::lock_guard<std::mutex> lk(err_mutex);
                    if (!first_error) first_error = std::current_exception();
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (unsigned i = 0; i < n_threads; ++i)
            workers.emplace_back(worker);
        for (auto& t : workers) t.join();
        pipeline_utils::log_rss("train: after all lambda pairs");

        if (first_error) std::rethrow_exception(first_error);

        // Post-hoc pruning (optional, parallel mode only).
        std::unordered_set<size_t> pruned_set;
        if (prune_enabled) {
            auto pruned = prune_skipped_lambdas(
                lambdas, gene_counts, pen_dir, opts.min_groups, min_lambda2);
            pruned_set.insert(pruned.begin(), pruned.end());
            // The pruner deleted these lambda dirs' contents, so the models are
            // gone; demote them from complete so evaluate --from-run skips them.
            for (size_t pruned_idx : pruned)
                mlog.set(gid_of(pi, pruned_idx, -1), model_log::Status::Skipped,
                         "pruned for single-threaded skip-ahead parity");
            std::cout << "  Pruned " << pruned.size() << " of " << lambdas.size()
                      << " lambda point(s) to match single-threaded skip-ahead semantics.\n";
        }

        // Merge into result in grid order, skipping pruned indices.
        for (size_t idx = 0; idx < lambdas.size(); ++idx) {
            if (!per_idx_solved[idx]) continue;
            if (pruned_set.count(idx)) continue;
            result.weights_paths.push_back(per_idx_weights[idx]);
            result.lambdas_used.push_back(per_idx_lambdas[idx]);
            result.penalties_used.push_back(penalty);
        }
    }

    // Step 11: Grid median (if lambdas.size() > 1 && nfolds == 0)
    if (lambdas.size() > 1 && opts.nfolds == 0) {
        // Read all lambda gss.txt files (only non-zero entries appear in files)
        std::unordered_map<std::string, std::vector<double>> gss_all;
        for (size_t li = 0; li < lambdas.size(); ++li) {
            fs::path gss_path = pen_dir / ("lambda_" + std::to_string(li)) / "gss.txt";
            std::ifstream gf(gss_path);
            if (!gf) continue;
            std::string line;
            while (std::getline(gf, line)) {
                if (line.empty()) continue;
                auto tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string gene = line.substr(0, tab);
                double val = std::stod(line.substr(tab + 1));
                if (val != 0.0) gss_all[gene].push_back(val);
            }
        }
        {
            std::vector<std::pair<double, std::string>> med_gss;
            for (auto& [g, v] : gss_all) {
                double med = pipeline_utils::median_nonzero(v);
                if (med != 0.0) med_gss.push_back({med, g});
            }
            std::sort(med_gss.rbegin(), med_gss.rend());
            std::ofstream mf(pen_dir / "gss_median.txt");
            mf << std::fixed << std::setprecision(6);
            for (auto& [val, g] : med_gss) mf << g << '\t' << val << '\n';
            std::cout << "  gss_median.txt written (" << med_gss.size() << " genes)\n";
        }

        // PSS median (FASTA only)
        if (enc.datatype != "numeric") {
            std::unordered_map<std::string, std::vector<double>> pss_all;
            for (size_t li = 0; li < lambdas.size(); ++li) {
                fs::path pss_path = pen_dir / ("lambda_" + std::to_string(li)) / "pss.txt";
                std::ifstream pf(pss_path);
                if (!pf) continue;
                std::string line;
                while (std::getline(pf, line)) {
                    if (line.empty()) continue;
                    auto tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    std::string key = line.substr(0, tab);
                    double val = std::stod(line.substr(tab + 1));
                    if (val != 0.0) pss_all[key].push_back(val);
                }
            }
            {
                std::vector<std::pair<std::string, double>> med_pss;
                for (auto& [k, v] : pss_all) {
                    double med = pipeline_utils::median_nonzero(v);
                    if (med != 0.0) med_pss.push_back({k, med});
                }
                std::sort(med_pss.begin(), med_pss.end(),
                    [](const auto& a, const auto& b) {
                        auto ua = a.first.rfind('_'), ub = b.first.rfind('_');
                        std::string ga = a.first.substr(0, ua), gb = b.first.substr(0, ub);
                        if (ga != gb) return ga < gb;
                        return pss_pos_key(a.first.substr(ua + 1))
                             < pss_pos_key(b.first.substr(ub + 1));
                    });
                std::ofstream mf(pen_dir / "pss_median.txt");
                mf << std::fixed << std::setprecision(6);
                for (auto& [k, val] : med_pss) mf << k << '\t' << val << '\n';
                std::cout << "  pss_median.txt written (" << med_pss.size() << " positions)\n";
            }
        }

        // BSS median — aggregate raw weights across all lambda weights.txt files
        {
            std::unordered_map<std::string, std::vector<double>> bss_all;
            std::vector<std::string> bss_order; // first-appearance order
            for (size_t li = 0; li < lambdas.size(); ++li) {
                fs::path w_path = pen_dir / ("lambda_" + std::to_string(li)) / "weights.txt";
                std::ifstream wf(w_path);
                if (!wf) continue;
                std::string line;
                while (std::getline(wf, line)) {
                    if (line.empty()) continue;
                    auto tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    std::string label = line.substr(0, tab);
                    if (label == "Intercept") continue;
                    double val = std::stod(line.substr(tab + 1));
                    if (val != 0.0) {
                        if (bss_all.find(label) == bss_all.end())
                            bss_order.push_back(label);
                        bss_all[label].push_back(val);
                    }
                }
            }
            std::ofstream mf(pen_dir / "bss_median.txt");
            mf << std::fixed << std::setprecision(6);
            size_t bss_written = 0;
            for (const auto& label : bss_order) {
                double med = pipeline_utils::median_nonzero(bss_all[label]);
                if (med != 0.0) {
                    mf << label << '\t' << med << '\n';
                    ++bss_written;
                }
            }
            std::cout << "  bss_median.txt written (" << bss_written << " weights)\n";
        }

        // oss_median.txt for olsg_lasso methods (oss.txt exists but no gss_grouped.txt)
        if (!fs::exists(pen_dir / "lambda_0" / "gss_grouped.txt")
            && fs::exists(pen_dir / "lambda_0" / "oss.txt")) {
            std::unordered_map<std::string, std::vector<double>> oss_all;
            for (size_t li = 0; li < lambdas.size(); ++li) {
                fs::path p = pen_dir / ("lambda_" + std::to_string(li)) / "oss.txt";
                std::ifstream of(p);
                if (!of) continue;
                std::string line;
                while (std::getline(of, line)) {
                    if (line.empty()) continue;
                    auto tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    std::string group = line.substr(0, tab);
                    double val = std::stod(line.substr(tab + 1));
                    if (val != 0.0) oss_all[group].push_back(val);
                }
            }
            std::vector<std::pair<double, std::string>> med;
            for (auto& [g, v] : oss_all) {
                double m = pipeline_utils::median_nonzero(v);
                if (m != 0.0) med.push_back({m, g});
            }
            std::sort(med.rbegin(), med.rend());
            std::ofstream mf(pen_dir / "oss_median.txt");
            mf << std::fixed << std::setprecision(6);
            for (auto& [val, g] : med) mf << g << '\t' << val << '\n';
            std::cout << "  oss_median.txt written (" << med.size() << " groups)\n";
        }

        // Grouped medians (only if gss_grouped.txt exists, i.e. ol_sg_lasso methods)
        if (fs::exists(pen_dir / "lambda_0" / "gss_grouped.txt")) {
            // gss_median_grouped.txt: median per (gene, group)
            {
                // key = "gene\tgroup" -> vector of values
                std::unordered_map<std::string, std::vector<double>> gss_g_all;
                for (size_t li = 0; li < lambdas.size(); ++li) {
                    fs::path p = pen_dir / ("lambda_" + std::to_string(li)) / "gss_grouped.txt";
                    std::ifstream gf(p);
                    if (!gf) continue;
                    std::string line;
                    while (std::getline(gf, line)) {
                        if (line.empty()) continue;
                        auto t1 = line.find('\t');
                        if (t1 == std::string::npos) continue;
                        auto t2 = line.find('\t', t1 + 1);
                        if (t2 == std::string::npos) continue;
                        std::string key = line.substr(0, t2); // "gene\tgroup"
                        double val = std::stod(line.substr(t2 + 1));
                        if (val != 0.0) gss_g_all[key].push_back(val);
                    }
                }
                std::vector<std::pair<double, std::string>> med;
                for (auto& [k, v] : gss_g_all) {
                    double m = pipeline_utils::median_nonzero(v);
                    if (m != 0.0) med.push_back({m, k});
                }
                std::sort(med.rbegin(), med.rend());
                std::ofstream mf(pen_dir / "gss_median_grouped.txt");
                mf << std::fixed << std::setprecision(6);
                for (auto& [val, k] : med) {
                    auto t = k.find('\t');
                    mf << k.substr(0, t) << '\t' << k.substr(t + 1) << '\t' << val << '\n';
                }
                std::cout << "  gss_median_grouped.txt written (" << med.size() << " entries)\n";
            }

            // pss_median_grouped.txt: median per (gene_pos, group)
            if (enc.datatype != "numeric") {
                std::unordered_map<std::string, std::vector<double>> pss_g_all;
                for (size_t li = 0; li < lambdas.size(); ++li) {
                    fs::path p = pen_dir / ("lambda_" + std::to_string(li)) / "pss_grouped.txt";
                    std::ifstream pf(p);
                    if (!pf) continue;
                    std::string line;
                    while (std::getline(pf, line)) {
                        if (line.empty()) continue;
                        auto t1 = line.find('\t');
                        if (t1 == std::string::npos) continue;
                        auto t2 = line.find('\t', t1 + 1);
                        if (t2 == std::string::npos) continue;
                        std::string key = line.substr(0, t2);
                        double val = std::stod(line.substr(t2 + 1));
                        if (val != 0.0) pss_g_all[key].push_back(val);
                    }
                }
                std::vector<std::pair<std::string, double>> med;
                for (auto& [k, v] : pss_g_all) {
                    double m = pipeline_utils::median_nonzero(v);
                    if (m != 0.0) med.push_back({k, m});
                }
                std::sort(med.begin(), med.end());
                std::ofstream mf(pen_dir / "pss_median_grouped.txt");
                mf << std::fixed << std::setprecision(6);
                for (auto& [k, val] : med) {
                    auto t = k.find('\t');
                    mf << k.substr(0, t) << '\t' << k.substr(t + 1) << '\t' << val << '\n';
                }
                std::cout << "  pss_median_grouped.txt written (" << med.size() << " entries)\n";
            }

            // oss_median.txt: median per group
            {
                std::unordered_map<std::string, std::vector<double>> oss_all;
                for (size_t li = 0; li < lambdas.size(); ++li) {
                    fs::path p = pen_dir / ("lambda_" + std::to_string(li)) / "oss.txt";
                    std::ifstream of(p);
                    if (!of) continue;
                    std::string line;
                    while (std::getline(of, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        std::string group = line.substr(0, tab);
                        double val = std::stod(line.substr(tab + 1));
                        if (val != 0.0) oss_all[group].push_back(val);
                    }
                }
                std::vector<std::pair<double, std::string>> med;
                for (auto& [g, v] : oss_all) {
                    double m = pipeline_utils::median_nonzero(v);
                    if (m != 0.0) med.push_back({m, g});
                }
                std::sort(med.rbegin(), med.rend());
                std::ofstream mf(pen_dir / "oss_median.txt");
                mf << std::fixed << std::setprecision(6);
                for (auto& [val, g] : med) mf << g << '\t' << val << '\n';
                std::cout << "  oss_median.txt written (" << med.size() << " groups)\n";
            }
        }
    }

    } // end penalty loop

    double regr_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - regr_start).count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Phase 3 (regression): " << regr_elapsed << "s\n";

    // Step 12: Return result
    std::ostringstream plog_m;
    plog_m << "lambdas_run = " << result.weights_paths.size() << "\n";
    plog.finish(plog_m.str());
    return result;

    } catch (const std::exception& e) {
        plog.fail(e.what());
        throw;
    }
}

} // namespace pipeline
