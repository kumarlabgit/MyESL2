#include "pipeline_evaluate.hpp"
#include "process_log.hpp"
#include "fasta_parser.hpp"
#include "numeric_parser.hpp"
#include "pff_format.hpp"
#include "visualizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>

namespace pipeline {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static std::unordered_set<char> load_datatype_chars_inline(
    const fs::path& ini_path, const std::string& section)
{
    std::ifstream f(ini_path);
    std::string line, cur;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { cur = line.substr(1, line.find(']') - 1); continue; }
        if (cur == section) {
            auto eq = line.find('=');
            if (eq != std::string::npos && line.substr(0, eq) == "chars") {
                std::unordered_set<char> s;
                for (char c : line.substr(eq + 1)) s.insert(c);
                return s;
            }
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// evaluate()
// ---------------------------------------------------------------------------

EvaluateResult evaluate(const EvaluateOptions& opts)
{
    const fs::path& weights_path = opts.weights_path;
    const fs::path& list_path    = opts.list_path;
    const fs::path& output_path  = opts.output_file;
    const fs::path& hyp_path     = opts.hyp_path;
    const std::string& datatype  = opts.datatype;
    bool no_visualize            = opts.no_visualize;

    unsigned int num_threads = opts.num_threads;
    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    fs::path cache_dir = opts.cache_dir;
    if (cache_dir.empty()) cache_dir = fs::current_path() / "pff_cache";

    EvaluateResult result;

    fs::path eval_log_dir = opts.output_file.parent_path();
    if (eval_log_dir.empty()) eval_log_dir = ".";
    process_log::Section plog(eval_log_dir / "process_log.txt", "evaluate");
    plog.param("weights_path", opts.weights_path)
        .param("list_path",    opts.list_path)
        .param("output_file",  opts.output_file)
        .param("datatype",     opts.datatype);
    if (!opts.hyp_path.empty()) plog.param("hyp_path", opts.hyp_path);

    try {

    double intercept_val = 0.0;
    std::map<std::string, double> species_sums;
    std::vector<std::string> eval_gene_order;
    std::map<std::string, std::map<std::string, double>> eval_gene_scores;

    // -------------------------------------------------------------------------
    // Numeric branch
    // -------------------------------------------------------------------------
    if (datatype == "numeric") {

        // --- Parse raw weights (label → weight) ---
        std::map<std::string, double> raw_weights;
        {
            std::ifstream wf(weights_path);
            if (!wf) throw std::runtime_error("Cannot open weights file: " + weights_path.string());
            std::string line;
            while (std::getline(wf, line)) {
                if (line.empty()) continue;
                size_t tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string label = line.substr(0, tab);
                double w = std::stod(line.substr(tab + 1));
                if (label == "Intercept") { intercept_val = w; continue; }
                raw_weights[label] = w;
            }
        }

        // --- Load list.txt ---
        std::vector<fs::path> all_numeric_paths;
        {
            std::ifstream list_file(list_path);
            if (!list_file) throw std::runtime_error("Cannot open list file: " + list_path.string());
            fs::path list_dir = list_path.parent_path();
            std::string line;
            while (std::getline(list_file, line)) {
                if (line.empty()) continue;
                for (char& c : line) if (c == '\\') c = '/';
                all_numeric_paths.push_back(list_dir / line);
            }
        }
        std::map<std::string, fs::path> stem_to_numeric;
        for (auto& p : all_numeric_paths)
            stem_to_numeric[p.stem().string()] = p;

        // --- Match weight labels to stems (longest-prefix first) ---
        struct NumericEntry { std::string stem, feature; double weight; };
        std::vector<NumericEntry> num_entries;
        {
            std::vector<std::string> sorted_stems;
            for (auto& [s, _] : stem_to_numeric) sorted_stems.push_back(s);
            std::sort(sorted_stems.begin(), sorted_stems.end(),
                [](const std::string& a, const std::string& b){ return a.size() > b.size(); });
            for (auto& [label, w] : raw_weights) {
                for (auto& s : sorted_stems) {
                    if (label.size() > s.size() + 1 &&
                        label.compare(0, s.size(), s) == 0 &&
                        label[s.size()] == '_') {
                        num_entries.push_back({s, label.substr(s.size() + 1), w});
                        break;
                    }
                }
            }
        }

        std::set<std::string> model_stems;
        for (auto& e : num_entries) model_stems.insert(e.stem);
        std::cout << "Model: " << num_entries.size() << " weight(s), "
                  << "intercept=" << intercept_val << ", "
                  << model_stems.size() << " file(s)\n";

        // --- Verify all model files present in list ---
        {
            std::vector<std::string> missing;
            for (auto& s : model_stems)
                if (!stem_to_numeric.count(s)) missing.push_back(s);
            if (!missing.empty()) {
                std::string msg = std::to_string(missing.size()) +
                    " file(s) referenced in the model are missing from the list:\n";
                for (auto& m : missing) msg += "  " + m + "\n";
                throw std::runtime_error(msg);
            }
        }
        std::cout << "All model files present in list.\n";

        // --- Phase 1: Convert to PNF as needed ---
        fs::create_directories(cache_dir);
        for (auto& stem : model_stems) {
            fs::path txt_path = stem_to_numeric.at(stem);
            fs::path pnf_path = cache_dir / (stem + ".pnf");
            fs::path err_path = cache_dir / (stem + ".err");
            if (fs::exists(err_path)) {
                std::cerr << "Warning: " << stem << " has a prior conversion error, skipping\n";
                continue;
            }
            bool needs_convert = true;
            if (fs::exists(pnf_path)) {
                try {
                    auto meta = numeric::read_pnf_metadata(pnf_path);
                    if (meta.source_path == fs::absolute(txt_path).string())
                        needs_convert = false;
                } catch (...) {}
            }
            if (needs_convert) {
                try {
                    numeric::tabular_to_pnf(txt_path, pnf_path);
                    std::cout << "Converted: " << stem << "\n";
                } catch (const std::exception& ex) {
                    std::ofstream ef(err_path);
                    if (ef) ef << ex.what() << "\n";
                    throw std::runtime_error("Conversion failed for " + stem + ": " + ex.what());
                }
            }
        }

        // --- Phase 2: Accumulate predictions from PNF ---
        std::map<std::string, std::vector<size_t>> stem_to_idxs;
        for (size_t i = 0; i < num_entries.size(); ++i)
            stem_to_idxs[num_entries[i].stem].push_back(i);

        // Build gene order from model_stems preserving first-appearance in entries
        {
            std::set<std::string> seen;
            for (auto& e : num_entries) {
                if (!seen.count(e.stem)) { eval_gene_order.push_back(e.stem); seen.insert(e.stem); }
            }
        }
        for (auto& [stem, idxs] : stem_to_idxs) {
            fs::path pnf_path = cache_dir / (stem + ".pnf");
            if (!fs::exists(pnf_path)) {
                std::cerr << "Warning: no PNF for " << stem << ", skipping\n";
                continue;
            }
            auto meta = numeric::read_pnf_metadata(pnf_path);
            auto data = numeric::read_pnf_data(pnf_path, meta);

            std::unordered_map<std::string, uint32_t> feat_idx;
            for (uint32_t f = 0; f < meta.num_features; ++f)
                feat_idx[meta.feature_labels[f]] = f;

            for (auto& id : meta.seq_ids) {
                species_sums.emplace(id, 0.0);
                eval_gene_scores[stem].emplace(id, 0.0);
            }

            for (size_t idx : idxs) {
                auto& e = num_entries[idx];
                auto it = feat_idx.find(e.feature);
                if (it == feat_idx.end()) continue;
                uint32_t f = it->second;
                for (uint32_t si = 0; si < meta.num_sequences; ++si) {
                    double contrib = e.weight * data[si][f];
                    species_sums[meta.seq_ids[si]] += contrib;
                    eval_gene_scores[stem][meta.seq_ids[si]] += contrib;
                }
            }
        }

    } else {
        // -------------------------------------------------------------------------
        // FASTA branch
        // -------------------------------------------------------------------------

        // --- Parse weight labels as <stem>_<pos>_<allele> ---
        struct ModelEntry {
            std::string stem;
            uint32_t    pos;
            char        allele;
            double      weight;
        };
        std::vector<ModelEntry> model_entries;
        {
            std::ifstream wf(weights_path);
            if (!wf) throw std::runtime_error("Cannot open weights file: " + weights_path.string());
            std::string line;
            while (std::getline(wf, line)) {
                if (line.empty()) continue;
                size_t tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string label = line.substr(0, tab);
                double w = std::stod(line.substr(tab + 1));
                if (label == "Intercept") { intercept_val = w; continue; }
                size_t us2 = label.rfind('_');
                if (us2 == std::string::npos || us2 + 1 >= label.size()) continue;
                char allele = label[us2 + 1];
                size_t us1 = label.rfind('_', us2 - 1);
                if (us1 == std::string::npos) continue;
                uint32_t pos = static_cast<uint32_t>(
                    std::stoul(label.substr(us1 + 1, us2 - us1 - 1)));
                model_entries.push_back({label.substr(0, us1), pos, allele, w});
            }
        }

        std::set<std::string> model_stems;
        for (auto& e : model_entries) model_stems.insert(e.stem);
        std::cout << "Model: " << model_entries.size() << " weight(s), "
                  << "intercept=" << intercept_val << ", "
                  << model_stems.size() << " alignment(s)\n";

        // --- Load list.txt ---
        std::vector<fs::path> all_fasta_paths;
        {
            std::ifstream list_file(list_path);
            if (!list_file) throw std::runtime_error("Cannot open list file: " + list_path.string());
            fs::path list_dir = list_path.parent_path();
            std::string line;
            while (std::getline(list_file, line)) {
                if (line.empty()) continue;
                for (char& c : line) if (c == '\\') c = '/';
                all_fasta_paths.push_back(list_dir / line);
            }
        }

        std::map<std::string, fs::path> stem_to_fasta;
        for (auto& p : all_fasta_paths)
            stem_to_fasta[p.stem().string()] = p;

        // --- Verify all model alignments present in list ---
        {
            std::vector<std::string> missing;
            for (auto& stem : model_stems)
                if (!stem_to_fasta.count(stem))
                    missing.push_back(stem);
            if (!missing.empty()) {
                std::string msg = std::to_string(missing.size()) +
                    " alignment(s) referenced in the model are missing from the list:\n";
                for (auto& m : missing) msg += "  " + m + "\n";
                throw std::runtime_error(msg);
            }
        }
        std::cout << "All model alignments present in list.\n";

        // --- Load allowed chars ---
        std::unordered_set<char> allowed_chars;
        if (datatype != "universal") {
            // Search for data_defs.ini relative to cache_dir or current path
            fs::path ini = cache_dir.parent_path() / "data_defs.ini";
            if (!fs::exists(ini)) ini = fs::current_path() / "data_defs.ini";
            if (fs::exists(ini)) {
                allowed_chars = load_datatype_chars_inline(ini, datatype);
                if (allowed_chars.empty())
                    std::cerr << "Warning: No chars defined for '" << datatype
                              << "' in data_defs.ini, using universal\n";
            } else {
                std::cerr << "Warning: data_defs.ini not found, using universal charset\n";
            }
        }

        // --- Convert model alignments to PFF as needed ---
        fs::create_directories(cache_dir);
        {
            std::queue<fs::path> convert_queue;
            for (auto& stem : model_stems) {
                fs::path fasta_path = stem_to_fasta.at(stem);
                fs::path pff_path   = cache_dir / (stem + ".pff");
                fs::path err_path   = cache_dir / (stem + ".err");
                if (fs::exists(err_path)) {
                    std::cerr << "Warning: " << stem << " has a prior conversion error, skipping\n";
                    continue;
                }
                bool needs_convert = true;
                if (fs::exists(pff_path)) {
                    try {
                        auto meta = fasta::read_pff_metadata(pff_path);
                        if (meta.datatype == datatype &&
                            meta.source_path == fs::absolute(fasta_path).string())
                            needs_convert = false;
                    } catch (...) {}
                }
                if (needs_convert) convert_queue.push(fasta_path);
            }

            if (!convert_queue.empty()) {
                int total = static_cast<int>(convert_queue.size());
                std::cout << "Converting " << total << " alignment(s) to PFF...\n";
                std::mutex queue_mutex, print_mutex;
                int converted = 0, failed = 0;

                auto worker = [&]() {
                    while (true) {
                        fs::path fasta_path;
                        {
                            std::lock_guard<std::mutex> lk(queue_mutex);
                            if (convert_queue.empty()) break;
                            fasta_path = convert_queue.front();
                            convert_queue.pop();
                        }
                        fs::path pff_path = cache_dir / (fasta_path.stem().string() + ".pff");
                        fs::path err_path = cache_dir / (fasta_path.stem().string() + ".err");
                        try {
                            fasta::fasta_to_pff(fasta_path, pff_path,
                                pff::Orientation::COLUMN_MAJOR, datatype, allowed_chars);
                            std::lock_guard<std::mutex> lk(print_mutex);
                            ++converted;
                            std::cout << "[" << converted + failed << "/" << total << "] OK: "
                                      << fasta_path.filename() << "\n";
                        } catch (const std::exception& e) {
                            std::ofstream ef(err_path);
                            if (ef) ef << e.what() << "\n";
                            std::lock_guard<std::mutex> lk(print_mutex);
                            ++failed;
                            std::cerr << "[" << converted + failed << "/" << total << "] FAIL: "
                                      << fasta_path.filename() << " -> " << e.what() << "\n";
                        }
                    }
                };

                unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total));
                std::vector<std::thread> threads;
                for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(worker);
                for (auto& t : threads) t.join();
                std::cout << "Converted: " << converted << ", failed: " << failed << "\n";
            }
        }

        // --- Compute predictions ---
        // Build gene order from model_entries (first-appearance order)
        {
            std::set<std::string> seen;
            for (auto& e : model_entries) {
                if (!seen.count(e.stem)) { eval_gene_order.push_back(e.stem); seen.insert(e.stem); }
            }
        }

        std::map<std::string, std::vector<size_t>> stem_entry_indices;
        for (size_t i = 0; i < model_entries.size(); ++i)
            stem_entry_indices[model_entries[i].stem].push_back(i);

        for (auto& [stem, indices] : stem_entry_indices) {
            fs::path pff_path = cache_dir / (stem + ".pff");
            if (!fs::exists(pff_path)) {
                std::cerr << "Warning: no PFF for " << stem << ", skipping its weights\n";
                continue;
            }

            auto meta = fasta::read_pff_metadata(pff_path);
            uint32_t S = meta.num_sequences;
            uint32_t L = meta.alignment_length;

            std::vector<char> raw(meta.get_data_size());
            {
                std::ifstream pf(pff_path, std::ios::binary);
                pf.seekg(static_cast<std::streamoff>(meta.data_offset));
                pf.read(raw.data(), static_cast<std::streamsize>(raw.size()));
            }

            for (auto& id : meta.seq_ids) {
                species_sums.emplace(id, 0.0);
                eval_gene_scores[stem].emplace(id, 0.0);
            }

            bool col_major = (meta.orientation == pff::Orientation::COLUMN_MAJOR);
            for (size_t idx : indices) {
                auto& entry = model_entries[idx];
                if (entry.pos >= L) continue;
                for (uint32_t si = 0; si < S; ++si) {
                    char c = col_major
                        ? raw[static_cast<size_t>(entry.pos) * S + si]
                        : raw[static_cast<size_t>(si) * L + entry.pos];
                    if (c == entry.allele) {
                        species_sums[meta.seq_ids[si]] += entry.weight;
                        eval_gene_scores[stem][meta.seq_ids[si]] += entry.weight;
                    }
                }
            }
        }
    } // end FASTA branch

    result.intercept = intercept_val;

    // -------------------------------------------------------------------------
    // Load hypothesis file (optional)
    // -------------------------------------------------------------------------
    std::map<std::string, double> true_values;
    if (!hyp_path.empty()) {
        std::ifstream hf(hyp_path);
        if (!hf) throw std::runtime_error("Cannot open hypothesis file: " + hyp_path.string());
        std::string line;
        while (std::getline(hf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            auto delim = line.find('\t');
            if (delim == std::string::npos) delim = line.find(' ');
            if (delim == std::string::npos) continue;
            double val = std::stod(line.substr(delim + 1));
            if (val == 0.0) continue;
            true_values[line.substr(0, delim)] = val;
        }
        std::cout << "Hypothesis: " << true_values.size() << " labelled species\n";
    }

    // -------------------------------------------------------------------------
    // Write output_file (eval.txt)
    // -------------------------------------------------------------------------
    {
        fs::path out_dir = output_path.parent_path();
        if (!out_dir.empty()) fs::create_directories(out_dir);

        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("Cannot create output file: " + output_path.string());
        out << std::fixed << std::setprecision(6);
        if (!true_values.empty())
            out << "SequenceID\tPredictedValue\tTrueValue\n";
        else
            out << "SequenceID\tPredictedValue\n";
        for (auto& [species, sum] : species_sums) {
            double pred = intercept_val + sum;
            out << species << '\t' << pred;
            auto it = true_values.find(species);
            if (it != true_values.end())
                out << '\t' << it->second;
            out << '\n';
        }
    }
    std::cout << "Predicted " << species_sums.size() << " species -> " << output_path << "\n";

    // -------------------------------------------------------------------------
    // Write gene_predictions.txt and SPS_SPP.txt
    // -------------------------------------------------------------------------
    fs::path gene_pred_path;
    fs::path svg_path;

    if (!eval_gene_order.empty()) {
        std::string stem_base = output_path.stem().string();
        fs::path out_dir = output_path.parent_path();
        if (out_dir.empty()) out_dir = ".";

        gene_pred_path = out_dir / (stem_base + "_gene_predictions.txt");
        {
            std::ofstream gp(gene_pred_path);
            gp << std::fixed << std::setprecision(15);
            gp << "SeqID\tResponse\tPrediction";
            for (auto& g : eval_gene_order) gp << '\t' << g;
            gp << '\n';
            for (auto& [species, sum] : species_sums) {
                double pred = intercept_val + sum;
                double resp = 0.0;
                auto tv = true_values.find(species);
                if (tv != true_values.end()) resp = tv->second;
                gp << species << '\t' << resp << '\t' << pred;
                for (auto& g : eval_gene_order) {
                    auto gm = eval_gene_scores.find(g);
                    if (gm != eval_gene_scores.end()) {
                        auto sm = gm->second.find(species);
                        if (sm != gm->second.end())
                            gp << '\t' << sm->second;
                        else
                            gp << "\tNaN";
                    } else {
                        gp << "\tNaN";
                    }
                }
                gp << '\n';
            }
        }
        std::cout << "Gene predictions -> " << gene_pred_path.string() << "\n";

        // --- Write SPS_SPP.txt ---
        fs::path sps_path = out_dir / (stem_base + "_SPS_SPP.txt");
        {
            auto expit = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
            double max_expit_pos = 0.5, min_expit_neg = 0.5;
            if (!true_values.empty()) {
                for (auto& [species, truth] : true_values) {
                    auto it = species_sums.find(species);
                    if (it == species_sums.end()) continue;
                    double pred = intercept_val + it->second;
                    double ep = expit(pred);
                    if (truth > 0.0 && ep > max_expit_pos) max_expit_pos = ep;
                    if (truth < 0.0 && ep < min_expit_neg) min_expit_neg = ep;
                }
            }
            double norm_pos = std::max(max_expit_pos - 0.5, 1e-9);
            double norm_neg = std::max(0.5 - min_expit_neg, 1e-9);

            std::ofstream sf(sps_path);
            sf << std::fixed << std::setprecision(6);
            sf << "SeqID\tResponse\tSPS\tSPP\n";
            for (auto& [species, sum] : species_sums) {
                double pred = intercept_val + sum;
                double ep = expit(pred);
                double resp = 0.0;
                auto tv = true_values.find(species);
                if (tv != true_values.end()) resp = tv->second;
                double spp;
                if (resp > 0.0)
                    spp = (ep - 0.5) / norm_pos;
                else if (resp < 0.0)
                    spp = (0.5 - ep) / norm_neg;
                else
                    spp = (ep - 0.5) / norm_pos;
                sf << species << '\t' << resp << '\t' << pred << '\t' << spp << '\n';
            }
        }
        std::cout << "SPS/SPP -> " << sps_path.string() << "\n";

        // --- Auto-invoke SVG visualization unless suppressed ---
        svg_path = out_dir / (stem_base + ".svg");
        if (!no_visualize) {
            try {
                viz::VizOptions vopt;
                vopt.m_grid = opts.m_grid;
                auto gpt = viz::read_gene_predictions(gene_pred_path);
                viz::write_svg(gpt, svg_path, vopt);
                std::cout << "Visualization -> " << svg_path.string() << "\n";
            } catch (const std::exception& ve) {
                std::cerr << "Warning: visualization failed: " << ve.what() << "\n";
                svg_path.clear();
            }
        } else {
            svg_path.clear();
        }
    }

    result.gene_pred_path = gene_pred_path;
    result.svg_path       = svg_path;

    // -------------------------------------------------------------------------
    // Classification metrics (threshold = 0)
    // -------------------------------------------------------------------------
    if (!true_values.empty()) {
        int tp = 0, tn = 0, fp = 0, fn = 0, unmatched = 0;
        for (auto& [species, truth] : true_values) {
            auto it = species_sums.find(species);
            if (it == species_sums.end()) { ++unmatched; continue; }
            double pred = intercept_val + it->second;
            bool pred_pos = pred > 0.0;
            bool true_pos = truth > 0.0;
            if      ( true_pos &&  pred_pos) ++tp;
            else if (!true_pos && !pred_pos) ++tn;
            else if (!true_pos &&  pred_pos) ++fp;
            else                              ++fn;
        }
        if (unmatched > 0)
            std::cerr << "Warning: " << unmatched
                      << " hypothesis species had no prediction\n";

        double tpr = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double tnr = (tn + fp) > 0 ? static_cast<double>(tn) / (tn + fp) : 0.0;
        double fpr = (tn + fp) > 0 ? static_cast<double>(fp) / (tn + fp) : 0.0;
        double fnr = (tp + fn) > 0 ? static_cast<double>(fn) / (tp + fn) : 0.0;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n--- Classification metrics (threshold = 0) ---\n";
        std::cout << "  TP=" << tp << "  TN=" << tn
                  << "  FP=" << fp << "  FN=" << fn << "\n";
        std::cout << "  True positive rate (sensitivity): " << tpr << "\n";
        std::cout << "  True negative rate (specificity): " << tnr << "\n";
        std::cout << "  False positive rate:              " << fpr << "\n";
        std::cout << "  False negative rate:              " << fnr << "\n";

        result.tp  = tp;  result.tn  = tn;
        result.fp  = fp;  result.fn  = fn;
        result.tpr = tpr; result.tnr = tnr;
        result.fpr = fpr; result.fnr = fnr;
    }

    // -------------------------------------------------------------------------
    // AIM feature ranking (if aim_window > 0)
    // -------------------------------------------------------------------------
    if (opts.aim_window > 0) {
        fs::path run_dir    = weights_path.parent_path();
        fs::path bss_file   = run_dir / "bss_median.txt";
        fs::path lam0_wfile = run_dir / "lambda_0" / "weights.txt";

        std::vector<std::pair<double, std::string>> feature_rank; // (signed_weight, label)

        if (fs::exists(bss_file)) {
            std::ifstream bf(bss_file);
            std::string line;
            while (std::getline(bf, line)) {
                if (line.empty()) continue;
                auto tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string lbl = line.substr(0, tab);
                double w = std::stod(line.substr(tab + 1));
                feature_rank.push_back({w, lbl});
            }
        } else if (fs::exists(lam0_wfile)) {
            std::ifstream wf(lam0_wfile);
            std::string line;
            while (std::getline(wf, line)) {
                if (line.empty()) continue;
                auto tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::string lbl = line.substr(0, tab);
                if (lbl == "Intercept") continue;
                double w = std::stod(line.substr(tab + 1));
                feature_rank.push_back({w, lbl});
            }
        }

        if (!feature_rank.empty()) {
            std::sort(feature_rank.begin(), feature_rank.end(),
                [](const auto& a, const auto& b){
                    return std::abs(a.first) > std::abs(b.first); });
            int window = std::min(opts.aim_window, static_cast<int>(feature_rank.size()));
            feature_rank.resize(window);

            result.ranked_features.reserve(window);
            for (auto& [w, lbl] : feature_rank)
                result.ranked_features.push_back({lbl, w});
        }
    }

    std::ostringstream plog_m;
    plog_m << std::fixed << std::setprecision(4);
    plog_m << "hss = " << result.hss << "\n";
    if (!opts.hyp_path.empty()) {
        plog_m << "tp = "  << result.tp  << "\n"
               << "tn = "  << result.tn  << "\n"
               << "fp = "  << result.fp  << "\n"
               << "fn = "  << result.fn  << "\n"
               << "tpr = " << result.tpr << "\n"
               << "tnr = " << result.tnr << "\n";
    }
    plog.finish(plog_m.str());
    return result;

    } catch (const std::exception& e) {
        plog.fail(e.what());
        throw;
    }
}

// ---------------------------------------------------------------------------
// evaluate_drphylo_aggregate()
// ---------------------------------------------------------------------------

DrPhyloAggResult evaluate_drphylo_aggregate(
    const fs::path& run_dir,
    double          grid_rmse_cutoff,
    double          grid_acc_cutoff)
{
    DrPhyloAggResult result;

    // -------------------------------------------------------------------------
    // Scan run_dir for lambda_N dirs in sequence
    // -------------------------------------------------------------------------
    std::vector<fs::path> lambda_dirs;
    for (int li = 0; ; ++li) {
        fs::path ld = run_dir / ("lambda_" + std::to_string(li));
        if (!fs::exists(ld)) break;
        if (fs::exists(ld / "weights.txt"))
            lambda_dirs.push_back(ld);
    }
    if (lambda_dirs.empty()) return result;

    // -------------------------------------------------------------------------
    // Parse a gene_predictions file
    // -------------------------------------------------------------------------
    struct LamGP {
        std::vector<std::string> seq_ids;
        std::vector<double> responses, predictions;
        std::vector<std::string> gene_names;
        std::vector<std::vector<double>> gene_scores; // [gene][seq]
        double rmse = 0.0, acc = 0.0;
    };

    auto parse_gp = [](const fs::path& path) -> LamGP {
        LamGP lam;
        std::ifstream f(path);
        if (!f) return lam;
        std::string line;
        if (!std::getline(f, line)) return lam;
        // Parse header
        {
            std::istringstream ss(line);
            std::string col;
            int ci = 0;
            while (std::getline(ss, col, '\t')) {
                if (ci >= 3) lam.gene_names.push_back(col);
                ++ci;
            }
        }
        lam.gene_scores.resize(lam.gene_names.size());
        // Parse data rows
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> cols;
            {
                std::istringstream ss(line);
                std::string col;
                while (std::getline(ss, col, '\t')) cols.push_back(col);
            }
            if (cols.size() < 3) continue;
            lam.seq_ids.push_back(cols[0]);
            lam.responses.push_back(std::stod(cols[1]));
            lam.predictions.push_back(std::stod(cols[2]));
            for (size_t g = 0; g < lam.gene_names.size(); ++g) {
                if (3 + g < cols.size() && cols[3 + g] != "NaN")
                    lam.gene_scores[g].push_back(std::stod(cols[3 + g]));
                else
                    lam.gene_scores[g].push_back(
                        std::numeric_limits<double>::quiet_NaN());
            }
        }
        // Compute RMSE and accuracy over labelled species
        double sq_sum = 0.0;
        int n_lab = 0, n_cor = 0;
        for (size_t i = 0; i < lam.seq_ids.size(); ++i) {
            double resp = lam.responses[i];
            if (resp == 0.0) continue;
            double diff = lam.predictions[i] - resp;
            sq_sum += diff * diff;
            ++n_lab;
            if ((lam.predictions[i] > 0.0) == (resp > 0.0)) ++n_cor;
        }
        lam.rmse = n_lab > 0 ? std::sqrt(sq_sum / n_lab) : 0.0;
        lam.acc  = n_lab > 0 ? static_cast<double>(n_cor) / n_lab : 0.0;
        return lam;
    };

    // -------------------------------------------------------------------------
    // Filter qualifying lambda models
    // -------------------------------------------------------------------------
    std::vector<LamGP> qualifying;
    for (auto& ld : lambda_dirs) {
        fs::path gp_file = ld / "eval_gene_predictions.txt";
        if (!fs::exists(gp_file)) continue;
        LamGP lam = parse_gp(gp_file);
        if (lam.seq_ids.empty()) continue;
        bool ok = lam.rmse <= grid_rmse_cutoff && lam.acc >= grid_acc_cutoff;
        std::cout << "  " << ld.filename().string()
                  << ": RMSE=" << std::fixed << std::setprecision(4) << lam.rmse
                  << " ACC=" << lam.acc << (ok ? " [OK]" : " [skip]") << "\n";
        if (ok) qualifying.push_back(std::move(lam));
    }

    if (qualifying.empty()) {
        std::cerr << "Warning: no lambda models passed thresholds for " << run_dir << "\n";
        return result;
    }
    std::cout << qualifying.size() << "/" << lambda_dirs.size()
              << " lambda models included in aggregation\n";

    // -------------------------------------------------------------------------
    // Aggregate qualifying gene_predictions
    // -------------------------------------------------------------------------
    // Union of gene names in first-appearance order
    std::vector<std::string> all_genes;
    {
        std::unordered_map<std::string, size_t> seen;
        for (auto& lam : qualifying)
            for (auto& g : lam.gene_names)
                if (seen.find(g) == seen.end()) {
                    seen[g] = all_genes.size();
                    all_genes.push_back(g);
                }
    }

    const LamGP& ref = qualifying[0];
    size_t N = ref.seq_ids.size();
    size_t G = all_genes.size();
    size_t M = qualifying.size();

    // Prediction: arithmetic mean across qualifying models
    std::vector<double> agg_pred(N, 0.0);
    for (auto& lam : qualifying)
        for (size_t i = 0; i < N; ++i)
            agg_pred[i] += lam.predictions[i] / static_cast<double>(M);

    // Gene scores: median of non-zero values per (gene, seq) cell
    auto median_nz = [](std::vector<double> vals) -> double {
        std::vector<double> nz;
        for (double x : vals)
            if (!std::isnan(x) && x != 0.0) nz.push_back(x);
        if (nz.empty()) return 0.0;
        std::sort(nz.begin(), nz.end());
        size_t m = nz.size() / 2;
        return (nz.size() % 2 == 0) ? (nz[m-1] + nz[m]) / 2.0 : nz[m];
    };

    // Build per-model gene→index map for O(1) lookup
    std::vector<std::unordered_map<std::string, size_t>> lam_gene_maps(M);
    for (size_t mi = 0; mi < M; ++mi)
        for (size_t g = 0; g < qualifying[mi].gene_names.size(); ++g)
            lam_gene_maps[mi][qualifying[mi].gene_names[g]] = g;

    std::vector<std::vector<double>> agg_gene(G, std::vector<double>(N, 0.0));
    std::vector<std::vector<bool>>   any_present(G, std::vector<bool>(N, false));
    for (size_t g_out = 0; g_out < G; ++g_out) {
        const std::string& gname = all_genes[g_out];
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> vals;
            for (size_t mi = 0; mi < M; ++mi) {
                auto it = lam_gene_maps[mi].find(gname);
                if (it == lam_gene_maps[mi].end()) continue;
                double v = qualifying[mi].gene_scores[it->second][i];
                if (!std::isnan(v)) {
                    any_present[g_out][i] = true;
                    vals.push_back(v);
                }
            }
            agg_gene[g_out][i] = median_nz(vals);
        }
    }

    // -------------------------------------------------------------------------
    // Write aggregated eval_gene_predictions.txt
    // -------------------------------------------------------------------------
    fs::path gp_out = run_dir / "eval_gene_predictions.txt";
    {
        std::ofstream gp(gp_out);
        gp << std::fixed << std::setprecision(6);
        gp << "SeqID\tResponse\tPrediction_mean";
        for (auto& g : all_genes) gp << '\t' << g;
        gp << '\n';
        for (size_t i = 0; i < N; ++i) {
            gp << ref.seq_ids[i] << '\t' << ref.responses[i] << '\t'
               << agg_pred[i];
            for (size_t g = 0; g < G; ++g) {
                if (!any_present[g][i])
                    gp << "\tNaN";
                else
                    gp << '\t' << agg_gene[g][i];
            }
            gp << '\n';
        }
    }

    // -------------------------------------------------------------------------
    // Derive eval.txt from aggregated predictions
    // -------------------------------------------------------------------------
    {
        std::ofstream ef(run_dir / "eval.txt");
        ef << std::fixed << std::setprecision(6);
        ef << "SequenceID\tPredictedValue\tTrueValue\n";
        for (size_t i = 0; i < N; ++i)
            ef << ref.seq_ids[i] << '\t' << agg_pred[i] << '\t'
               << ref.responses[i] << '\n';
    }

    // -------------------------------------------------------------------------
    // Derive eval_SPS_SPP.txt
    // -------------------------------------------------------------------------
    {
        auto expit = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        double max_ep_pos = 0.5, min_ep_neg = 0.5;
        for (size_t i = 0; i < N; ++i) {
            double ep = expit(agg_pred[i]);
            if (ref.responses[i] > 0.0 && ep > max_ep_pos) max_ep_pos = ep;
            if (ref.responses[i] < 0.0 && ep < min_ep_neg) min_ep_neg = ep;
        }
        double norm_pos = std::max(max_ep_pos - 0.5, 1e-9);
        double norm_neg = std::max(0.5 - min_ep_neg, 1e-9);
        std::ofstream sf(run_dir / "eval_SPS_SPP.txt");
        sf << std::fixed << std::setprecision(6);
        sf << "SeqID\tResponse\tSPS\tSPP\n";
        for (size_t i = 0; i < N; ++i) {
            double ep  = expit(agg_pred[i]);
            double spp = ref.responses[i] > 0.0 ? (ep - 0.5) / norm_pos
                       : ref.responses[i] < 0.0 ? (0.5 - ep) / norm_neg
                       : (ep - 0.5) / norm_pos;
            sf << ref.seq_ids[i] << '\t' << ref.responses[i] << '\t'
               << agg_pred[i] << '\t' << spp << '\n';
        }
    }

    // -------------------------------------------------------------------------
    // Generate SVG from aggregated gene_predictions
    // -------------------------------------------------------------------------
    {
        try {
            fs::path svg_out = run_dir / "eval.svg";
            auto gpt = viz::read_gene_predictions(gp_out);
            viz::VizOptions vopt;
            vopt.m_grid = true;
            viz::write_svg(gpt, svg_out, vopt);
            std::cout << "Visualization -> " << svg_out.string() << "\n";
        } catch (const std::exception& ve) {
            std::cerr << "Warning: aggregated visualization failed: " << ve.what() << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // HSS: sum of all per-lambda GSS values (not just qualifying)
    // -------------------------------------------------------------------------
    double hss = 0.0;
    for (auto& ldir : lambda_dirs) {
        fs::path gss_file = ldir / "gss.txt";
        if (!fs::exists(gss_file)) continue;
        std::ifstream gf(gss_file);
        std::string gline;
        while (std::getline(gf, gline)) {
            auto tab = gline.find('\t');
            if (tab != std::string::npos)
                hss += std::stod(gline.substr(tab + 1));
        }
    }

    std::cout << run_dir.filename().string() << ": HSS=" << hss << "\n";

    result.hss = hss;
    result.ok  = true;
    return result;
}

} // namespace pipeline
