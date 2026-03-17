#include "pipeline_adaptive.hpp"
#include "process_log.hpp"
#include "pipeline_preprocess.hpp"
#include "pipeline_encode.hpp"
#include "pipeline_train.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iostream>

namespace pipeline {

namespace {

// Count non-zero hypothesis samples in hyp_path
static uint32_t count_hyp_samples(const fs::path& hyp_path)
{
    std::ifstream f(hyp_path);
    if (!f) throw std::runtime_error("adaptive_train: cannot open hypothesis file: " + hyp_path.string());
    uint32_t count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto delim = line.find('\t');
        if (delim == std::string::npos) delim = line.find(' ');
        if (delim == std::string::npos) continue;
        double val = std::stod(line.substr(delim + 1));
        if (val != 0.0) ++count;
    }
    return count;
}

// Parse list file → ordered vector of (stem, original_path) pairs.
// Only keeps stems present in the sizes map.
static std::vector<std::pair<std::string, fs::path>>
read_ordered_stems(const fs::path& list_path,
                   const std::map<std::string, uint64_t>& sizes)
{
    std::ifstream f(list_path);
    if (!f) throw std::runtime_error("adaptive_train: cannot open list file: " + list_path.string());

    fs::path list_dir = list_path.parent_path();
    std::vector<std::pair<std::string, fs::path>> result;
    std::unordered_map<std::string, bool> seen;

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        // Take the first token of the comma-split as the stem source
        std::stringstream ss(line);
        std::string token;
        if (!std::getline(ss, token, ',')) continue;

        // trim
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.erase(token.begin());
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\r')) token.pop_back();
        if (token.empty()) continue;

        for (char& c : token) if (c == '\\') c = '/';
        fs::path p = list_dir / token;
        std::string stem = p.stem().string();

        if (seen.count(stem)) continue;
        seen[stem] = true;

        if (sizes.count(stem))
            result.emplace_back(stem, p);
    }
    return result;
}

// Write list file: one original path per line
static void write_list_file(const fs::path& out_path,
                             const std::vector<std::pair<std::string, fs::path>>& stems)
{
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("adaptive_train: cannot write list file: " + out_path.string());
    for (auto& [stem, path] : stems)
        f << path.string() << "\n";
}

// Parse tab-separated "stem\tscore" lines (no header), accumulate into map
static void accumulate_gss(const fs::path& gss_path,
                            std::unordered_map<std::string, double>& summed)
{
    std::ifstream f(gss_path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string stem = line.substr(0, tab);
        double score = 0.0;
        try { score = std::stod(line.substr(tab + 1)); } catch (...) { continue; }
        summed[stem] += score;
    }
}

static std::string fmt_mb(uint64_t bytes)
{
    std::ostringstream oss;
    oss.precision(1);
    oss << std::fixed << bytes / double(1u << 20) << " MB";
    return oss.str();
}

} // anonymous namespace

void adaptive_train(const EncodeOptions& enc_opts, const TrainOptions& train_opts)
{
    process_log::Section plog(enc_opts.output_dir / "process_log.txt", "adaptive_train");
    plog.param("max_mem_bytes",    enc_opts.max_mem)
        .param("method",           train_opts.method.empty() ? std::string("sg_lasso") : train_opts.method)
        .param("adaptive_l1_spec", train_opts.adaptive_l1_spec)
        .param("adaptive_l2_spec", train_opts.adaptive_l2_spec);

    try {

    // ----------------------------------------------------------------
    // 3a. Probe sizes and count N
    // ----------------------------------------------------------------
    auto sizes = pipeline::encode_sizes(enc_opts);

    uint32_t N = count_hyp_samples(enc_opts.hyp_path);
    if (N == 0) throw std::runtime_error("adaptive_train: no non-zero hypothesis samples found");

    // ----------------------------------------------------------------
    // 3b. Read ordered stems from preprocess_config list file
    // ----------------------------------------------------------------
    auto pre_cfg = pipeline::read_preprocess_config(enc_opts.output_dir);
    auto ordered = read_ordered_stems(pre_cfg.list_path, sizes);

    if (ordered.empty())
        throw std::runtime_error("adaptive_sparsification: no genes with cached PFF found");

    // ----------------------------------------------------------------
    // 3c. Chunk the list at max_mem/2
    // ----------------------------------------------------------------
    uint64_t target = enc_opts.max_mem / 2;

    using StemEntry = std::pair<std::string, fs::path>;
    std::vector<std::vector<StemEntry>> chunks;
    std::vector<StemEntry> cur_chunk;
    uint64_t cur_bytes = 0;

    for (auto& entry : ordered) {
        uint64_t cost = sizes.at(entry.first) * static_cast<uint64_t>(N) * sizeof(float);
        if (!cur_chunk.empty() && cur_bytes + cost > target) {
            chunks.push_back(std::move(cur_chunk));
            cur_chunk.clear();
            cur_bytes = 0;
        }
        cur_chunk.push_back(entry);
        cur_bytes += cost;
    }
    if (!cur_chunk.empty()) chunks.push_back(std::move(cur_chunk));

    if (chunks.size() < 2)
        throw std::runtime_error(
            "adaptive_sparsification: dataset cannot be split into 2+ chunks within max_mem/2; "
            "try increasing max_mem");

    size_t n_chunks = chunks.size();
    size_t n_combos = n_chunks * (n_chunks - 1) / 2;
    std::cout << "[adaptive] " << n_chunks << " chunks, " << n_combos << " combo passes\n";

    // ----------------------------------------------------------------
    // 3d. All C(n,2) combo training passes
    // ----------------------------------------------------------------
    fs::path adap_dir = enc_opts.output_dir / "adaptive";
    fs::create_directories(adap_dir);
    std::unordered_map<std::string, double> summed_gss;

    for (size_t i = 0; i < chunks.size(); ++i) {
        for (size_t j = i + 1; j < chunks.size(); ++j) {
            fs::path combo_dir = adap_dir / ("combo_" + std::to_string(i) + "_" + std::to_string(j));
            fs::create_directories(combo_dir);

            // Merge chunks i and j
            std::vector<StemEntry> combo_stems = chunks[i];
            combo_stems.insert(combo_stems.end(), chunks[j].begin(), chunks[j].end());

            // Write list file
            fs::path combo_list = combo_dir / "aln.txt";
            write_list_file(combo_list, combo_stems);

            // Write preprocess_config pointing to combo list
            auto combo_pre = pre_cfg;
            combo_pre.list_path = combo_list;
            pipeline::write_preprocess_config(combo_dir, combo_pre);

            // Encode (no max_mem limit — combo is guaranteed to fit)
            EncodeOptions enc_combo = enc_opts;
            enc_combo.output_dir = combo_dir;
            enc_combo.max_mem    = 0;
            auto enc_result = pipeline::encode(enc_combo);

            // Train with 3x3 exploration grid
            TrainOptions train_combo;
            train_combo.output_dir           = combo_dir;
            train_combo.method               = train_opts.method.empty() ? "sg_lasso" : train_opts.method;
            train_combo.precision            = train_opts.precision;
            train_combo.params               = train_opts.params;
            train_combo.lambda_grid_specs[0] = train_opts.adaptive_l1_spec;
            train_combo.lambda_grid_specs[1] = train_opts.adaptive_l2_spec;
            train_combo.lambda_grid_set      = true;
            pipeline::train(enc_result, train_combo);

            // Accumulate gss_median.txt
            fs::path gss_path = combo_dir / "gss_median.txt";
            if (fs::exists(gss_path)) {
                accumulate_gss(gss_path, summed_gss);
            } else {
                std::cerr << "[adaptive] warning: no gss_median.txt for combo_"
                          << i << "_" << j << "\n";
            }
        }
    }

    // ----------------------------------------------------------------
    // 3e. Build final sublist
    // ----------------------------------------------------------------
    // Sort by summed score descending
    std::vector<std::pair<double, std::string>> ranked;
    for (auto& [stem, score] : summed_gss)
        ranked.emplace_back(score, stem);
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b){ return a.first > b.first; });

    // Greedy pick within max_mem
    std::vector<StemEntry> chosen_stems;
    uint64_t total_bytes = 0;

    // Build a map from stem → original path for fast lookup
    std::unordered_map<std::string, fs::path> stem_to_path;
    for (auto& [stem, path] : ordered)
        stem_to_path[stem] = path;

    for (auto& [score, stem] : ranked) {
        if (!sizes.count(stem)) continue;
        uint64_t cost = sizes.at(stem) * static_cast<uint64_t>(N) * sizeof(float);
        if (total_bytes + cost <= enc_opts.max_mem) {
            chosen_stems.emplace_back(stem, stem_to_path[stem]);
            total_bytes += cost;
        }
    }

    if (chosen_stems.empty())
        throw std::runtime_error(
            "adaptive_sparsification: no genes fit within max_mem budget after aggregation");

    fs::path final_list = adap_dir / "final_sublist.txt";
    write_list_file(final_list, chosen_stems);
    std::cout << "[adaptive] Final sublist: " << chosen_stems.size()
              << " genes, estimated " << fmt_mb(total_bytes) << "\n";

    {
        std::ostringstream plog_m;
        plog_m << "chunks = "      << n_chunks            << "\n"
               << "combos = "      << n_combos             << "\n"
               << "final_genes = " << chosen_stems.size() << "\n";
        plog.finish(plog_m.str());
    }

    // ----------------------------------------------------------------
    // 3f. Final train with original settings
    // ----------------------------------------------------------------
    auto final_pre = pre_cfg;
    final_pre.list_path = final_list;
    pipeline::write_preprocess_config(enc_opts.output_dir, final_pre);

    auto enc_final = pipeline::encode(enc_opts);
    pipeline::train(enc_final, train_opts);

    } catch (const std::exception& e) {
        plog.fail(e.what());
        throw;
    }
}

} // namespace pipeline
