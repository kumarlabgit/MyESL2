#include "pipeline_train.hpp"
#include "pipeline_utils.hpp"
#include "process_log.hpp"
#include "regression.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <cmath>

namespace pipeline {

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
    const arma::mat&   alg_table    = enc.alg_table;
    uint32_t N                      = enc.N;

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
    } else if (opts.lambda_grid_set) {
        auto parse_spec = [](const std::string& spec) {
            std::vector<double> vals;
            double vmin, vmax, vstep;
            char c1, c2;
            std::istringstream ss(spec);
            if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || c1 != ',' || c2 != ',')
                throw std::runtime_error("--lambda-grid spec must be 'min,max,step': " + spec);
            if (vstep <= 0.0) throw std::runtime_error("lambda-grid step must be > 0");
            for (double v = vmin; v <= vmax + vstep * 1e-9; v += vstep)
                vals.push_back(v);
            return vals;
        };
        auto v1 = parse_spec(opts.lambda_grid_specs[0]);
        auto v2 = parse_spec(opts.lambda_grid_specs[1]);
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

    // Step 6: Build xval_idxs if nfolds > 0
    arma::rowvec xval_idxs(N);
    if (opts.nfolds > 0)
        for (uint32_t i = 0; i < N; ++i)
            xval_idxs(i) = static_cast<double>(i % opts.nfolds);

    // Step 7: Build sorted_stems_desc for numeric longest-prefix matching
    std::vector<std::string> sorted_stems_desc = enc.all_stems_ordered;
    std::sort(sorted_stems_desc.begin(), sorted_stems_desc.end(),
        [](const std::string& a, const std::string& b){ return a.size() > b.size(); });

    // Step 8: Helper lambda — compute and write GSS/PSS, return nonzero gene count
    auto compute_sig_scores = [&](const fs::path& wpath, const fs::path& lam_dir,
                                  std::ostringstream& sout, int lam_idx) -> int {
        bool is_numeric_mode = (enc.datatype == "numeric");
        std::map<std::string, double> gss; // stem -> sum(|w|)
        // PSS key = "stem\tpos_str" -> sum(|w|)
        std::map<std::string, double> pss;
        double hss = 0.0;

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
            } else {
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
                    uint32_t pa = static_cast<uint32_t>(std::stoul(a.first.substr(ta + 1)));
                    uint32_t pb = static_cast<uint32_t>(std::stoul(b.first.substr(tb + 1)));
                    return pa < pb;
                });
            std::ofstream pf(lam_dir / "pss.txt");
            pf << std::fixed << std::setprecision(15);
            for (auto& [key, v] : pss_entries) {
                auto tp = key.find('\t');
                pf << key.substr(0, tp) << '_' << key.substr(tp + 1) << '\t' << v << '\n';
            }
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
    double max_lambda2 = std::numeric_limits<double>::infinity();

    // Step 10: Lambda loop
    for (int idx = 0; idx < (int)lambdas.size(); ++idx) {
        auto& lam = lambdas[idx];
        fs::path lam_dir = output_dir / ("lambda_" + std::to_string(idx));
        fs::create_directories(lam_dir); // always create dir (sentinel for drphylo)

        if (skip_ahead_valid && opts.min_groups > 0 && lam[1] > max_lambda2) {
            std::cout << "  [" << idx << "] lambda=[" << lam[0] << "," << lam[1]
                      << "] Skipping (gene count threshold)\n";
            continue;
        }

        std::ostringstream out;
        out << std::fixed << std::setprecision(4);

        if (enc.is_overlapping
            && (method == "olsg_lasso_leastr" || method == "olsg_lasso_logisticr")
            && opts.params.find("field") == opts.params.end()) {
            opts.params["field"] = enc.field_path.string();
        }

        if (opts.nfolds == 0) {
            // Single model
            auto regr = regression::createRegressionAnalysis(
                method, features, responses, alg_table.t(), opts.params, lam, opts.precision);
            {
                std::ofstream wo(lam_dir / "weights.txt");
                std::ifstream mi(output_dir / "combined.map");
                regr->writeSparseMappedWeightsToStream(wo, mi);
            }
            out << "  [" << idx << "] lambda=[" << lam[0] << ","
                << lam[1] << "] -> " << (lam_dir / "weights.txt").string() << "\n";
            int gene_count = compute_sig_scores(lam_dir / "weights.txt", lam_dir, out, idx);
            out << "  [" << idx << "] Non-zero gene count: " << gene_count << "\n";
            std::cout << out.str();

            result.weights_paths.push_back(lam_dir / "weights.txt");
            result.lambdas_used.push_back(lam);

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
                    std::ifstream mi(output_dir / "combined.map");
                    regr->writeSparseMappedWeightsToStream(wo, mi);
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
                cv_out << "SequenceID\tPredictedValue\tTrueValue\n";
                for (uint32_t i = 0; i < N; ++i)
                    cv_out << enc.seq_names[i] << '\t'
                           << cv_preds[i] << '\t'
                           << enc.hyp_values[i] << '\n';
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
        }
    }

    // Step 11: Grid median (if lambdas.size() > 1 && nfolds == 0)
    if (lambdas.size() > 1 && opts.nfolds == 0) {
        // Read all lambda gss.txt files (only non-zero entries appear in files)
        std::unordered_map<std::string, std::vector<double>> gss_all;
        for (size_t li = 0; li < lambdas.size(); ++li) {
            fs::path gss_path = output_dir / ("lambda_" + std::to_string(li)) / "gss.txt";
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
            std::ofstream mf(output_dir / "gss_median.txt");
            mf << std::fixed << std::setprecision(6);
            for (auto& [val, g] : med_gss) mf << g << '\t' << val << '\n';
            std::cout << "  gss_median.txt written (" << med_gss.size() << " genes)\n";
        }

        // PSS median (FASTA only)
        if (enc.datatype != "numeric") {
            std::unordered_map<std::string, std::vector<double>> pss_all;
            for (size_t li = 0; li < lambdas.size(); ++li) {
                fs::path pss_path = output_dir / ("lambda_" + std::to_string(li)) / "pss.txt";
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
                        uint32_t pa = static_cast<uint32_t>(std::stoul(a.first.substr(ua + 1)));
                        uint32_t pb = static_cast<uint32_t>(std::stoul(b.first.substr(ub + 1)));
                        return pa < pb;
                    });
                std::ofstream mf(output_dir / "pss_median.txt");
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
                fs::path w_path = output_dir / ("lambda_" + std::to_string(li)) / "weights.txt";
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
            std::ofstream mf(output_dir / "bss_median.txt");
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
    }

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
