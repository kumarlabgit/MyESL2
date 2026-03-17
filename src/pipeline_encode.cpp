#include "pipeline_encode.hpp"
#include "pipeline_preprocess.hpp"
#include "process_log.hpp"
#include "encoder.hpp"
#include "numeric_parser.hpp"
#include "fasta_parser.hpp"
#include "pff_format.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <cmath>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <atomic>

namespace pipeline {

EncodeResult encode(const EncodeOptions& opts)
{
    // ----------------------------------------------------------------
    // 1. Read preprocess_config
    // ----------------------------------------------------------------
    PreprocessOptions pre = pipeline::read_preprocess_config(opts.output_dir);

    // ----------------------------------------------------------------
    // 2. Effective min_minor
    // ----------------------------------------------------------------
    int min_minor = (opts.min_minor >= 0) ? opts.min_minor : pre.min_minor;

    // ----------------------------------------------------------------
    // 3. num_threads
    // ----------------------------------------------------------------
    unsigned int num_threads = pre.num_threads;
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    process_log::Section plog(opts.output_dir / "process_log.txt", "encode");
    plog.param("hyp_path",      opts.hyp_path)
        .param("max_mem_bytes", opts.max_mem)
        .param("precision",     std::string(opts.precision == regression::Precision::FP64 ? "fp64" : "fp32"));
    if (!opts.class_bal.empty()) plog.param("class_bal",  opts.class_bal);
    if (opts.drop_major)         plog.param("drop_major", true);
    if (opts.auto_bit_ct > 0)    plog.param("auto_bit_ct", opts.auto_bit_ct);

    try {

    // ----------------------------------------------------------------
    // 4. Read hypothesis file
    // ----------------------------------------------------------------
    std::vector<std::string> hyp_seq_names;
    std::vector<float> hyp_values;
    {
        std::ifstream hyp_file(opts.hyp_path);
        if (!hyp_file)
            throw std::runtime_error("Cannot open hypothesis file: " + opts.hyp_path.string());
        std::string line;
        int line_num = 0;
        while (std::getline(hyp_file, line)) {
            ++line_num;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            auto delim = line.find('\t');
            if (delim == std::string::npos) delim = line.find(' ');
            if (delim == std::string::npos) {
                std::cerr << "Warning: hypothesis file line " << line_num
                          << " has no delimiter, skipping: \"" << line << "\"\n";
                continue;
            }
            double val = std::stod(line.substr(delim + 1));
            if (val == 0.0) continue;
            hyp_seq_names.push_back(line.substr(0, delim));
            hyp_values.push_back(static_cast<float>(val));
        }
    }
    std::cout << "Hypothesis sequences (non-zero): " << hyp_seq_names.size() << "\n";

    // --auto-bit-ct: override min_minor as percentage of minority class size
    if (opts.auto_bit_ct > 0.0) {
        int pos_count = 0, neg_count = 0;
        for (float v : hyp_values) {
            if (v > 0.0f) ++pos_count;
            else if (v < 0.0f) ++neg_count;
        }
        int min_class = std::min(pos_count, neg_count);
        min_minor = std::max(1, static_cast<int>(std::ceil(opts.auto_bit_ct / 100.0 * min_class)));
        std::cout << "auto-bit-ct: min-minor set to " << min_minor
                  << " (" << opts.auto_bit_ct << "% of " << min_class << ")\n";
    }

    // ----------------------------------------------------------------
    // 5. Read list file
    // ----------------------------------------------------------------
    fs::path list_path = pre.list_path;
    fs::path cache_dir = pre.cache_dir;

    std::vector<fs::path> all_fasta_paths;
    std::vector<std::vector<fs::path>> groups;
    std::unordered_map<std::string, size_t> stem_to_unique_idx;
    {
        std::ifstream list_file(list_path);
        if (!list_file)
            throw std::runtime_error("Cannot open list file: " + list_path.string());
        fs::path list_dir = list_path.parent_path();
        std::string line;
        while (std::getline(list_file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::vector<fs::path> group;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.erase(token.begin());
                while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\r')) token.pop_back();
                if (token.empty()) continue;
                for (char& c : token) if (c == '\\') c = '/';
                fs::path p = list_dir / token;
                group.push_back(p);
                std::string stem = p.stem().string();
                if (stem_to_unique_idx.find(stem) == stem_to_unique_idx.end()) {
                    stem_to_unique_idx[stem] = all_fasta_paths.size();
                    all_fasta_paths.push_back(p);
                }
            }
            if (!group.empty()) groups.push_back(std::move(group));
        }
    }

    bool is_overlapping = false;
    {
        std::unordered_map<std::string, int> stem_group_count;
        for (auto& g : groups) {
            if (g.size() > 1) is_overlapping = true;
            for (auto& p : g) stem_group_count[p.stem().string()]++;
        }
        for (auto& [s, c] : stem_group_count) if (c > 1) { is_overlapping = true; break; }
    }

    plog.param("genes", (int)all_fasta_paths.size());

    // ----------------------------------------------------------------
    // 6. Phase 2 encoding
    // ----------------------------------------------------------------
    uint32_t N = static_cast<uint32_t>(hyp_seq_names.size());

    uint64_t total_cols = 0;
    int n_aligned = 0, failed_count = 0;
    arma::fmat features;
    arma::mat alg_table;
    std::vector<std::string> all_missing;
    std::vector<std::string> all_stems_ordered;

    auto fmt_bytes = [](uint64_t b) -> std::string {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        if      (b >= uint64_t(1) << 30) oss << b / double(uint64_t(1) << 30) << " GB";
        else if (b >= uint64_t(1) << 20) oss << b / double(uint64_t(1) << 20) << " MB";
        else if (b >= uint64_t(1) << 10) oss << b / double(uint64_t(1) << 10) << " KB";
        else                              oss << b << " B";
        return oss.str();
    };

    if (pre.datatype == "numeric") {
        // ---- Numeric branch ----
        std::vector<fs::path> pnf_paths;
        for (auto& tab_path : all_fasta_paths) {
            fs::path pnf_path = cache_dir / (tab_path.stem().string() + ".pnf");
            if (!fs::exists(pnf_path)) {
                std::cerr << "Warning: no .pnf for " << tab_path.filename() << ", skipping\n";
                continue;
            }
            pnf_paths.push_back(pnf_path);
        }
        int total_files = static_cast<int>(pnf_paths.size());

        std::cout << "\n--- Phase 2: Numeric matrix assembly ---\n";
        std::cout << "  Files to process: " << total_files << "\n";
        std::cout << "  Worker threads:   " << num_threads << "\n\n";

        struct NumericResult {
            std::string stem;
            std::vector<std::string> feature_labels;
            std::vector<std::vector<float>> columns; // columns[j][i]: feat j, hyp-seq i
            std::vector<std::string> missing_sequences;
            bool failed = false;
            std::string error_msg;
        };

        auto encode_start = std::chrono::steady_clock::now();
        std::vector<NumericResult> num_results(total_files);
        uint64_t running_cols = 0;
        uint64_t first_estimate = 0;
        int next_pct = 1;
        std::atomic<bool> mem_exceeded{false};
        std::string mem_err_msg;
        {
            std::mutex queue_mutex, print_mutex;
            std::queue<int> work_queue;
            for (int i = 0; i < total_files; ++i) work_queue.push(i);
            int done_count = 0;

            auto enc_worker = [&]() {
                auto check_milestone = [&]() {
                    while (next_pct <= 10 && running_cols > 0 &&
                           done_count * 10 >= next_pct * total_files) {
                        uint64_t est_cols = running_cols * static_cast<uint64_t>(total_files)
                                            / static_cast<uint64_t>(done_count);
                        uint64_t est_bytes = est_cols * static_cast<uint64_t>(N) * sizeof(float);
                        if (next_pct == 1) first_estimate = est_bytes;
                        std::cout << "  [" << (next_pct * 10) << "%] Estimated matrix size: "
                                  << fmt_bytes(est_bytes) << "\n";
                        ++next_pct;
                    }
                };
                while (true) {
                    int idx;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (work_queue.empty() || mem_exceeded.load()) break;
                        idx = work_queue.front();
                        work_queue.pop();
                    }
                    NumericResult& nr = num_results[idx];
                    nr.stem = pnf_paths[idx].stem().string();
                    try {
                        auto meta = numeric::read_pnf_metadata(pnf_paths[idx]);
                        auto data = numeric::read_pnf_data(pnf_paths[idx], meta);
                        nr.feature_labels = meta.feature_labels;

                        std::unordered_map<std::string, uint32_t> id_to_row;
                        for (uint32_t s = 0; s < meta.num_sequences; ++s)
                            id_to_row[meta.seq_ids[s]] = s;

                        std::vector<int> seq_mapping(N, -1);
                        for (uint32_t i = 0; i < N; ++i) {
                            auto it = id_to_row.find(hyp_seq_names[i]);
                            if (it != id_to_row.end())
                                seq_mapping[i] = static_cast<int>(it->second);
                            else
                                nr.missing_sequences.push_back(hyp_seq_names[i]);
                        }

                        uint32_t F = meta.num_features;
                        nr.columns.resize(F, std::vector<float>(N, 0.0f));
                        for (uint32_t j = 0; j < F; ++j)
                            for (uint32_t i = 0; i < N; ++i)
                                if (seq_mapping[i] >= 0)
                                    nr.columns[j][i] = data[static_cast<uint32_t>(seq_mapping[i])][j];

                        {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            ++done_count;
                            running_cols += F;
                            std::cout << "[" << done_count << "/" << total_files << "] "
                                      << pnf_paths[idx].filename().string()
                                      << " -> " << F << " features\n";
                            check_milestone();
                            if (opts.max_mem > 0 && !mem_exceeded.load()) {
                                uint64_t cur_bytes = running_cols * static_cast<uint64_t>(N) * sizeof(float);
                                if (static_cast<double>(cur_bytes) > static_cast<double>(opts.max_mem) * 0.8) {
                                    uint64_t est_bytes = static_cast<uint64_t>(
                                        static_cast<double>(cur_bytes) * total_files / done_count);
                                    if (static_cast<double>(est_bytes) > static_cast<double>(opts.max_mem) * 1.25) {
                                        mem_err_msg = "max_mem_exceeded: estimated final matrix size "
                                            + fmt_bytes(est_bytes) + " exceeds 125% of max_mem ("
                                            + fmt_bytes(opts.max_mem) + ")";
                                        mem_exceeded = true;
                                        if (opts.disable_mc)
                                            std::cout << "[max_mem] warning: " << mem_err_msg << " (ignored)\n";
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        nr.failed = true;
                        nr.error_msg = e.what();
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++done_count;
                        std::cerr << "[" << done_count << "/" << total_files << "] FAIL: "
                                  << pnf_paths[idx].filename().string()
                                  << " -> " << e.what() << "\n";
                        check_milestone();
                    }
                }
            };

            unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_files));
            if (tc == 0) tc = 1;
            std::vector<std::thread> threads;
            threads.reserve(tc);
            for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(enc_worker);
            for (auto& t : threads) t.join();
        }
        if (mem_exceeded.load() && !opts.disable_mc) throw std::runtime_error(mem_err_msg);
        (void)std::chrono::steady_clock::now(); // encode_start used implicitly above

        for (auto& nr : num_results) {
            for (auto& m : nr.missing_sequences) all_missing.push_back(m);
            if (nr.failed) { ++failed_count; continue; }
            total_cols += nr.columns.size();
            ++n_aligned;
        }
        {
            uint64_t actual_bytes = total_cols * static_cast<uint64_t>(N) * sizeof(float);
            std::cout << "  Matrix size: " << fmt_bytes(actual_bytes) << " actual";
            if (first_estimate > 0)
                std::cout << " (estimated " << fmt_bytes(first_estimate) << " at 10%)";
            std::cout << "\n";
        }

        // Save column counts before freeing intermediate storage
        std::vector<size_t> num_col_counts(num_results.size(), 0);
        for (size_t ri = 0; ri < num_results.size(); ++ri)
            if (!num_results[ri].failed) num_col_counts[ri] = num_results[ri].columns.size();

        features.zeros(N, total_cols);
        {
            uint64_t col_offset = 0;
            for (size_t ri = 0; ri < num_results.size(); ++ri) {
                auto& nr = num_results[ri];
                if (nr.failed) continue;
                size_t ncols = num_col_counts[ri];
                for (size_t j = 0; j < ncols; ++j)
                    for (uint32_t si = 0; si < N; ++si)
                        features(si, col_offset + j) = nr.columns[j][si];
                { decltype(nr.columns) tmp; tmp.swap(nr.columns); } // free after copy
                col_offset += ncols;
            }
        }

        // Write alignment table (per-file, before group_table)
        std::cout << "\nWriting alignment table...\n";
        alg_table.zeros(3, n_aligned);
        {
            uint64_t offset = 0;
            int col = 0;
            for (size_t ri = 0; ri < num_results.size(); ++ri) {
                auto& nr = num_results[ri];
                if (nr.failed) continue;
                uint64_t ncols = num_col_counts[ri];
                alg_table(0, col) = static_cast<double>(offset + 1);
                alg_table(1, col) = static_cast<double>(offset + ncols);
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(6) << std::sqrt(static_cast<double>(ncols));
                    alg_table(2, col) = std::stod(ss.str());
                }
                offset += ncols;
                ++col;
            }
        }

        // Build group_table and write output files
        {
            std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> stem_to_cols;
            int col = 0;
            for (auto& nr : num_results) {
                if (nr.failed) continue;
                stem_to_cols[nr.stem] = {
                    static_cast<uint64_t>(alg_table(0, col)),
                    static_cast<uint64_t>(alg_table(1, col))
                };
                ++col;
            }
            arma::mat group_table(3, groups.size());
            std::vector<uint64_t> field_indices;
            {
                uint64_t field_pos = 1;
                for (size_t gi = 0; gi < groups.size(); ++gi) {
                    uint64_t grp_start = field_pos;
                    for (auto& p : groups[gi]) {
                        auto it = stem_to_cols.find(p.stem().string());
                        if (it == stem_to_cols.end()) continue;
                        for (uint64_t fi = it->second.first; fi <= it->second.second; ++fi) {
                            field_indices.push_back(fi);
                            ++field_pos;
                        }
                    }
                    uint64_t grp_end = field_pos - 1;
                    group_table(0, gi) = static_cast<double>(grp_start);
                    group_table(1, gi) = static_cast<double>(grp_end);
                    {
                        std::ostringstream ss;
                        ss << std::fixed << std::setprecision(6)
                           << std::sqrt(static_cast<double>(grp_end - grp_start + 1));
                        group_table(2, gi) = std::stod(ss.str());
                    }
                }
            }
            {
                std::ofstream table(opts.output_dir / "alignment_table.txt");
                for (int row = 0; row < 3; ++row) {
                    for (size_t c = 0; c < groups.size(); ++c) {
                        if (c > 0) table << '\t';
                        if (row < 2) table << static_cast<uint64_t>(group_table(row, c));
                        else         table << std::fixed << std::setprecision(6) << group_table(row, c);
                    }
                    table << '\n';
                }
            }
            if (is_overlapping) {
                {
                    std::ofstream ff(opts.output_dir / "field.txt");
                    for (size_t i = 0; i < field_indices.size(); ++i) {
                        if (i > 0) ff << ',';
                        ff << field_indices[i];
                    }
                    ff << '\n';
                }
                {
                    std::ofstream gi(opts.output_dir / "group_indices.txt");
                    for (int row = 0; row < 3; ++row) {
                        for (size_t c = 0; c < groups.size(); ++c) {
                            if (c > 0) gi << ',';
                            if (row < 2) gi << static_cast<uint64_t>(group_table(row, c));
                            else         gi << std::fixed << std::setprecision(6) << group_table(row, c);
                        }
                        gi << '\n';
                    }
                }
            }
            alg_table = group_table;
        }

        if (!all_missing.empty()) {
            std::ofstream missing_file(opts.output_dir / "missing_sequences.txt");
            for (auto& m : all_missing) missing_file << m << '\n';
            std::cout << "Missing sequences: " << all_missing.size()
                      << " -> " << (opts.output_dir / "missing_sequences.txt").string() << "\n";
        }

        // Write combined.map
        {
            std::ofstream combined_map(opts.output_dir / "combined.map");
            combined_map << "Position\tLabel\n";
            uint64_t pos = 0;
            for (auto& nr : num_results) {
                if (nr.failed) continue;
                all_stems_ordered.push_back(nr.stem);
                for (auto& feat_name : nr.feature_labels)
                    combined_map << pos++ << '\t' << nr.stem << '_' << feat_name << '\n';
            }
        }

    } else {
        // ---- FASTA branch ----
        std::vector<fs::path> pff_paths;
        for (auto& fasta_path : all_fasta_paths) {
            fs::path pff_path = cache_dir / (fasta_path.stem().string() + ".pff");
            if (!fs::exists(pff_path)) {
                std::cerr << "Warning: no .pff for " << fasta_path.filename() << ", skipping\n";
                continue;
            }
            pff_paths.push_back(pff_path);
        }
        int total_encode = static_cast<int>(pff_paths.size());

        std::cout << "\n--- Phase 2: Encoding ---\n";
        std::cout << "  Alignments to encode: " << total_encode << "\n";
        std::cout << "  Min-minor threshold:  " << min_minor << "\n";
        std::cout << "  Worker threads:       " << num_threads << "\n";
        std::cout << "  Encoder:              " << (pre.use_dlt ? "DLT" : "standard") << "\n\n";

        bool skip_x = (pre.datatype == "protein" || pre.datatype == "nucleotide");

        auto encode_start = std::chrono::steady_clock::now();
        std::vector<encoder::AlignmentResult> results(total_encode);
        uint64_t running_cols = 0;
        uint64_t first_estimate = 0;
        int next_pct = 1;
        std::atomic<bool> mem_exceeded{false};
        std::string mem_err_msg;
        {
            std::mutex queue_mutex, print_mutex;
            std::queue<int> work_queue;
            for (int i = 0; i < total_encode; ++i) work_queue.push(i);
            int done_count = 0;

            auto enc_worker = [&]() {
                auto check_milestone = [&]() {
                    while (next_pct <= 10 && running_cols > 0 &&
                           done_count * 10 >= next_pct * total_encode) {
                        uint64_t est_cols = running_cols * static_cast<uint64_t>(total_encode)
                                            / static_cast<uint64_t>(done_count);
                        uint64_t est_bytes = est_cols * static_cast<uint64_t>(N) * sizeof(float);
                        if (next_pct == 1) first_estimate = est_bytes;
                        std::cout << "  [" << (next_pct * 10) << "%] Estimated matrix size: "
                                  << fmt_bytes(est_bytes) << "\n";
                        ++next_pct;
                    }
                };
                while (true) {
                    int idx;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (work_queue.empty() || mem_exceeded.load()) break;
                        idx = work_queue.front();
                        work_queue.pop();
                    }
                    try {
                        results[idx] = pre.use_dlt
                            ? encoder::encode_pff_dlt(pff_paths[idx], hyp_seq_names, min_minor,
                                                      opts.drop_major, opts.dropout_labels, skip_x)
                            : encoder::encode_pff(pff_paths[idx], hyp_seq_names, min_minor,
                                                  opts.drop_major, opts.dropout_labels, skip_x);
                        {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            ++done_count;
                            running_cols += results[idx].columns.size();
                            std::cout << "[" << done_count << "/" << total_encode << "] "
                                      << pff_paths[idx].filename().string()
                                      << " -> " << results[idx].columns.size() << " columns\n";
                            check_milestone();
                            if (opts.max_mem > 0 && !mem_exceeded.load()) {
                                uint64_t cur_bytes = running_cols * static_cast<uint64_t>(N) * sizeof(float);
                                if (static_cast<double>(cur_bytes) > static_cast<double>(opts.max_mem) * 0.8) {
                                    uint64_t est_bytes = static_cast<uint64_t>(
                                        static_cast<double>(cur_bytes) * total_encode / done_count);
                                    if (static_cast<double>(est_bytes) > static_cast<double>(opts.max_mem) * 1.25) {
                                        mem_err_msg = "max_mem_exceeded: estimated final matrix size "
                                            + fmt_bytes(est_bytes) + " exceeds 125% of max_mem ("
                                            + fmt_bytes(opts.max_mem) + ")";
                                        mem_exceeded = true;
                                        if (opts.disable_mc)
                                            std::cout << "[max_mem] warning: " << mem_err_msg << " (ignored)\n";
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        results[idx].failed    = true;
                        results[idx].error_msg = e.what();
                        results[idx].stem      = pff_paths[idx].stem().string();
                        std::lock_guard<std::mutex> lock(print_mutex);
                        ++done_count;
                        std::cerr << "[" << done_count << "/" << total_encode << "] FAIL: "
                                  << pff_paths[idx].filename().string()
                                  << " -> " << e.what() << "\n";
                        check_milestone();
                    }
                }
            };

            unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_encode));
            if (tc == 0) tc = 1;
            std::vector<std::thread> threads;
            threads.reserve(tc);
            for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(enc_worker);
            for (auto& t : threads) t.join();
        }
        if (mem_exceeded.load() && !opts.disable_mc) throw std::runtime_error(mem_err_msg);
        (void)encode_start; // timing available if needed

        for (int i = 0; i < total_encode; ++i)
            for (auto& m : results[i].missing_sequences)
                all_missing.push_back(m);

        for (auto& r : results) {
            if (r.failed) { ++failed_count; continue; }
            total_cols += r.columns.size();
            ++n_aligned;
        }
        {
            uint64_t actual_bytes = total_cols * static_cast<uint64_t>(N) * sizeof(float);
            std::cout << "  Matrix size: " << fmt_bytes(actual_bytes) << " actual";
            if (first_estimate > 0)
                std::cout << " (estimated " << fmt_bytes(first_estimate) << " at 10%)";
            std::cout << "\n";
        }

        // Save column counts before freeing intermediate storage
        std::vector<size_t> result_col_counts(total_encode, 0);
        for (int ri = 0; ri < total_encode; ++ri)
            if (!results[ri].failed) result_col_counts[ri] = results[ri].columns.size();

        features.zeros(N, total_cols);
        {
            uint64_t col_offset = 0;
            for (int ri = 0; ri < total_encode; ++ri) {
                auto& r = results[ri];
                if (r.failed) continue;
                size_t ncols = result_col_counts[ri];
                for (size_t j = 0; j < ncols; ++j)
                    for (uint32_t si = 0; si < N; ++si)
                        features(si, col_offset + j) = static_cast<float>(r.columns[j][si]);
                { decltype(r.columns) tmp; tmp.swap(r.columns); } // free after copy
                col_offset += ncols;
            }
        }

        std::cout << "\nWriting alignment table...\n";
        alg_table.zeros(3, n_aligned);
        {
            uint64_t offset = 0;
            int col = 0;
            for (int ri = 0; ri < total_encode; ++ri) {
                auto& r = results[ri];
                if (r.failed) continue;
                uint64_t ncols = result_col_counts[ri];
                alg_table(0, col) = static_cast<double>(offset + 1);
                alg_table(1, col) = static_cast<double>(offset + ncols);
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(6) << std::sqrt(static_cast<double>(ncols));
                    alg_table(2, col) = std::stod(ss.str());
                }
                offset += ncols;
                ++col;
            }
        }

        // Build group_table and write output files
        {
            std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> stem_to_cols;
            int col = 0;
            for (auto& r : results) {
                if (r.failed) continue;
                stem_to_cols[r.stem] = {
                    static_cast<uint64_t>(alg_table(0, col)),
                    static_cast<uint64_t>(alg_table(1, col))
                };
                ++col;
            }
            arma::mat group_table(3, groups.size());
            std::vector<uint64_t> field_indices;
            {
                uint64_t field_pos = 1;
                for (size_t gi = 0; gi < groups.size(); ++gi) {
                    uint64_t grp_start = field_pos;
                    for (auto& p : groups[gi]) {
                        auto it = stem_to_cols.find(p.stem().string());
                        if (it == stem_to_cols.end()) continue;
                        for (uint64_t fi = it->second.first; fi <= it->second.second; ++fi) {
                            field_indices.push_back(fi);
                            ++field_pos;
                        }
                    }
                    uint64_t grp_end = field_pos - 1;
                    group_table(0, gi) = static_cast<double>(grp_start);
                    group_table(1, gi) = static_cast<double>(grp_end);
                    {
                        std::ostringstream ss;
                        ss << std::fixed << std::setprecision(6)
                           << std::sqrt(static_cast<double>(grp_end - grp_start + 1));
                        group_table(2, gi) = std::stod(ss.str());
                    }
                }
            }
            {
                std::ofstream table(opts.output_dir / "alignment_table.txt");
                for (int row = 0; row < 3; ++row) {
                    for (size_t c = 0; c < groups.size(); ++c) {
                        if (c > 0) table << '\t';
                        if (row < 2) table << static_cast<uint64_t>(group_table(row, c));
                        else         table << std::fixed << std::setprecision(6) << group_table(row, c);
                    }
                    table << '\n';
                }
            }
            if (is_overlapping) {
                {
                    std::ofstream ff(opts.output_dir / "field.txt");
                    for (size_t i = 0; i < field_indices.size(); ++i) {
                        if (i > 0) ff << ',';
                        ff << field_indices[i];
                    }
                    ff << '\n';
                }
                {
                    std::ofstream gi_file(opts.output_dir / "group_indices.txt");
                    for (int row = 0; row < 3; ++row) {
                        for (size_t c = 0; c < groups.size(); ++c) {
                            if (c > 0) gi_file << ',';
                            if (row < 2) gi_file << static_cast<uint64_t>(group_table(row, c));
                            else         gi_file << std::fixed << std::setprecision(6) << group_table(row, c);
                        }
                        gi_file << '\n';
                    }
                }
            }
            alg_table = group_table;
        }

        if (!all_missing.empty()) {
            std::ofstream missing_file(opts.output_dir / "missing_sequences.txt");
            for (auto& m : all_missing) missing_file << m << '\n';
            std::cout << "Missing sequences: " << all_missing.size()
                      << " -> " << (opts.output_dir / "missing_sequences.txt").string() << "\n";
        }

        {
            std::ofstream combined_map(opts.output_dir / "combined.map");
            combined_map << "Position\tLabel\n";
            for (auto& r : results) {
                if (r.failed) continue;
                all_stems_ordered.push_back(r.stem);
                for (auto& [pos, allele] : r.map)
                    combined_map << pos << '\t' << r.stem << '_' << pos << '_' << allele << '\n';
            }
        }
    }

    // ----------------------------------------------------------------
    // 7. Write feature matrix to file if requested
    // ----------------------------------------------------------------
    auto write_fmat = [&](const std::string& path, bool transposed) {
        std::ofstream out(path);
        if (!out) {
            std::cerr << "Warning: cannot open '" << path << "' for writing\n";
            return;
        }
        if (!transposed) {
            std::cout << "Writing feature matrix (" << features.n_rows
                      << " x " << features.n_cols << ") -> " << path << '\n';
            for (arma::uword r = 0; r < features.n_rows; ++r) {
                for (arma::uword c = 0; c < features.n_cols; ++c) {
                    if (c > 0) out << ',';
                    out << features(r, c);
                }
                out << '\n';
            }
        } else {
            std::cout << "Writing transposed feature matrix (" << features.n_cols
                      << " x " << features.n_rows << ") -> " << path << '\n';
            for (arma::uword c = 0; c < features.n_cols; ++c) {
                for (arma::uword r = 0; r < features.n_rows; ++r) {
                    if (r > 0) out << ',';
                    out << features(r, c);
                }
                out << '\n';
            }
        }
    };
    if (!opts.write_features_path.empty())
        write_fmat(opts.write_features_path, false);
    if (!opts.write_features_transposed_path.empty())
        write_fmat(opts.write_features_transposed_path, true);

    // ----------------------------------------------------------------
    // 8. Class balancing
    // ----------------------------------------------------------------
    std::map<std::string, std::string> extra_params;

    if (!opts.class_bal.empty()) {
        std::vector<uint32_t> pos_idx, neg_idx;
        for (uint32_t i = 0; i < N; ++i) {
            if (hyp_values[i] > 0.0f) pos_idx.push_back(i);
            else if (hyp_values[i] < 0.0f) neg_idx.push_back(i);
        }
        std::mt19937 rng(0);

        if (opts.class_bal == "down") {
            auto& majority = (pos_idx.size() > neg_idx.size()) ? pos_idx : neg_idx;
            auto& minority = (pos_idx.size() <= neg_idx.size()) ? pos_idx : neg_idx;
            std::shuffle(majority.begin(), majority.end(), rng);
            majority.resize(minority.size());
            std::vector<uint32_t> keep;
            keep.insert(keep.end(), pos_idx.begin(), pos_idx.end());
            keep.insert(keep.end(), neg_idx.begin(), neg_idx.end());
            std::sort(keep.begin(), keep.end());
            arma::fmat new_features(keep.size(), features.n_cols);
            arma::frowvec new_responses(keep.size());
            for (size_t ki = 0; ki < keep.size(); ++ki) {
                new_features.row(ki) = features.row(keep[ki]);
                new_responses(ki) = hyp_values[keep[ki]];
            }
            features = std::move(new_features);
            N = static_cast<uint32_t>(keep.size());
            hyp_values.resize(N);
            for (uint32_t i = 0; i < N; ++i) hyp_values[i] = new_responses(i);
            // rebuild seq_names to match kept indices
            std::vector<std::string> new_seq_names(N);
            for (uint32_t i = 0; i < N; ++i) new_seq_names[i] = hyp_seq_names[keep[i]];
            hyp_seq_names = std::move(new_seq_names);
            std::cout << "Class balance (down): " << N << " samples retained\n";
        } else if (opts.class_bal == "up") {
            auto& majority = (pos_idx.size() >= neg_idx.size()) ? pos_idx : neg_idx;
            auto& minority = (pos_idx.size() < neg_idx.size()) ? pos_idx : neg_idx;
            std::uniform_int_distribution<size_t> dist(0, minority.size() - 1);
            std::vector<uint32_t> extra;
            while (minority.size() + extra.size() < majority.size())
                extra.push_back(minority[dist(rng)]);
            minority.insert(minority.end(), extra.begin(), extra.end());
            std::vector<uint32_t> keep;
            keep.insert(keep.end(), pos_idx.begin(), pos_idx.end());
            keep.insert(keep.end(), neg_idx.begin(), neg_idx.end());
            arma::fmat new_features(keep.size(), features.n_cols);
            arma::frowvec new_responses(keep.size());
            for (size_t ki = 0; ki < keep.size(); ++ki) {
                new_features.row(ki) = features.row(keep[ki]);
                new_responses(ki) = hyp_values[keep[ki]];
            }
            features = std::move(new_features);
            N = static_cast<uint32_t>(keep.size());
            hyp_values.resize(N);
            for (uint32_t i = 0; i < N; ++i) hyp_values[i] = new_responses(i);
            std::vector<std::string> new_seq_names(N);
            for (uint32_t i = 0; i < N; ++i) new_seq_names[i] = hyp_seq_names[keep[i]];
            hyp_seq_names = std::move(new_seq_names);
            std::cout << "Class balance (up): " << N << " samples (upsampled minority)\n";
        } else if (opts.class_bal == "weighted") {
            double pos_w = 1.0;
            double neg_w = (neg_idx.size() > 0 && pos_idx.size() > 0)
                           ? static_cast<double>(pos_idx.size()) / neg_idx.size()
                           : 1.0;
            fs::path sw_path = opts.output_dir / "sweights.txt";
            {
                std::ofstream sf(sw_path);
                sf << std::fixed << std::setprecision(6);
                sf << pos_w << "\n" << neg_w << "\n";
            }
            extra_params["sWeight"] = sw_path.string();
            std::cout << "Class balance (weighted): sWeight written -> " << sw_path.string()
                      << " (pos=" << pos_w << " neg=" << neg_w << ")\n";
        }
    }

    // ----------------------------------------------------------------
    // 9. Rebuild responses
    // ----------------------------------------------------------------
    arma::frowvec responses(hyp_values.data(), N);

    // ----------------------------------------------------------------
    // 10. Populate and return result
    // ----------------------------------------------------------------
    EncodeResult result;
    result.features           = std::move(features);
    result.responses          = std::move(responses);
    result.alg_table          = std::move(alg_table);
    result.seq_names          = std::move(hyp_seq_names);
    result.hyp_values         = std::move(hyp_values);
    result.all_stems_ordered  = std::move(all_stems_ordered);
    result.is_overlapping     = is_overlapping;
    result.N                  = N;
    result.total_cols         = total_cols;
    result.precision          = opts.precision;
    result.combined_map_path  = opts.output_dir / "combined.map";
    result.alignment_table_path = opts.output_dir / "alignment_table.txt";
    result.field_path         = is_overlapping ? (opts.output_dir / "field.txt") : fs::path{};
    result.extra_params       = std::move(extra_params);
    result.datatype           = pre.datatype;

    double matrix_mb = static_cast<double>(
        static_cast<uint64_t>(N) * total_cols * sizeof(float)) / (1 << 20);
    std::ostringstream plog_m;
    plog_m << std::fixed << std::setprecision(2);
    plog_m << "samples = "   << N          << "\n"
           << "features = "  << total_cols  << "\n"
           << "matrix_mb = " << matrix_mb   << "\n";
    plog.finish(plog_m.str());
    return result;

    } catch (const std::exception& e) {
        plog.fail(e.what());
        throw;
    }
}

std::map<std::string, uint64_t> encode_sizes(const EncodeOptions& opts)
{
    // ----------------------------------------------------------------
    // 1. Read preprocess_config
    // ----------------------------------------------------------------
    PreprocessOptions pre = pipeline::read_preprocess_config(opts.output_dir);

    // ----------------------------------------------------------------
    // 2. Effective min_minor
    // ----------------------------------------------------------------
    int min_minor = (opts.min_minor >= 0) ? opts.min_minor : pre.min_minor;

    // ----------------------------------------------------------------
    // 3. num_threads
    // ----------------------------------------------------------------
    unsigned int num_threads = pre.num_threads;
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }

    // ----------------------------------------------------------------
    // 4. Read hypothesis file
    // ----------------------------------------------------------------
    std::vector<std::string> hyp_seq_names;
    std::vector<float> hyp_values;
    {
        std::ifstream hyp_file(opts.hyp_path);
        if (!hyp_file)
            throw std::runtime_error("Cannot open hypothesis file: " + opts.hyp_path.string());
        std::string line;
        int line_num = 0;
        while (std::getline(hyp_file, line)) {
            ++line_num;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            auto delim = line.find('\t');
            if (delim == std::string::npos) delim = line.find(' ');
            if (delim == std::string::npos) {
                std::cerr << "Warning: hypothesis file line " << line_num
                          << " has no delimiter, skipping: \"" << line << "\"\n";
                continue;
            }
            double val = std::stod(line.substr(delim + 1));
            if (val == 0.0) continue;
            hyp_seq_names.push_back(line.substr(0, delim));
            hyp_values.push_back(static_cast<float>(val));
        }
    }

    // --auto-bit-ct: override min_minor as percentage of minority class size
    if (opts.auto_bit_ct > 0.0) {
        int pos_count = 0, neg_count = 0;
        for (float v : hyp_values) {
            if (v > 0.0f) ++pos_count;
            else if (v < 0.0f) ++neg_count;
        }
        int min_class = std::min(pos_count, neg_count);
        min_minor = std::max(1, static_cast<int>(std::ceil(opts.auto_bit_ct / 100.0 * min_class)));
    }

    // ----------------------------------------------------------------
    // 5. Read list file
    // ----------------------------------------------------------------
    fs::path list_path = pre.list_path;
    fs::path cache_dir = pre.cache_dir;

    std::vector<fs::path> all_fasta_paths;
    {
        std::unordered_map<std::string, size_t> stem_to_unique_idx;
        std::ifstream list_file(list_path);
        if (!list_file)
            throw std::runtime_error("Cannot open list file: " + list_path.string());
        fs::path list_dir = list_path.parent_path();
        std::string line;
        while (std::getline(list_file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) token.erase(token.begin());
                while (!token.empty() && (token.back() == ' ' || token.back() == '\t' || token.back() == '\r')) token.pop_back();
                if (token.empty()) continue;
                for (char& c : token) if (c == '\\') c = '/';
                fs::path p = list_dir / token;
                std::string stem = p.stem().string();
                if (stem_to_unique_idx.find(stem) == stem_to_unique_idx.end()) {
                    stem_to_unique_idx[stem] = all_fasta_paths.size();
                    all_fasta_paths.push_back(p);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // 6. Encode per file; record column count, discard data
    // ----------------------------------------------------------------
    std::map<std::string, uint64_t> sizes;
    std::mutex queue_mutex, sizes_mutex;

    if (pre.datatype == "numeric") {
        // ---- Numeric branch: metadata only, no read_pnf_data ----
        std::vector<fs::path> pnf_paths;
        for (auto& tab_path : all_fasta_paths) {
            fs::path pnf_path = cache_dir / (tab_path.stem().string() + ".pnf");
            if (!fs::exists(pnf_path)) {
                std::cerr << "Warning: no .pnf for " << tab_path.filename() << ", skipping\n";
                continue;
            }
            pnf_paths.push_back(pnf_path);
        }
        int total_files = static_cast<int>(pnf_paths.size());
        std::queue<int> work_queue;
        for (int i = 0; i < total_files; ++i) work_queue.push(i);
        int done_count = 0;

        auto worker = [&]() {
            while (true) {
                int idx;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (work_queue.empty()) break;
                    idx = work_queue.front();
                    work_queue.pop();
                }
                std::string stem = pnf_paths[idx].stem().string();
                try {
                    auto meta = numeric::read_pnf_metadata(pnf_paths[idx]);
                    uint64_t count = 0;
                    for (auto& feat : meta.feature_labels) {
                        if (!opts.dropout_labels.count(stem + "_" + feat))
                            ++count;
                    }
                    std::lock_guard<std::mutex> lock(sizes_mutex);
                    sizes[stem] = count;
                    ++done_count;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(sizes_mutex);
                    ++done_count;
                    std::cerr << "[" << done_count << "/" << total_files << "] FAIL: "
                              << pnf_paths[idx].filename().string() << " -> " << e.what() << "\n";
                }
            }
        };

        unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_files));
        if (tc == 0) tc = 1;
        std::vector<std::thread> threads;
        threads.reserve(tc);
        for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

    } else {
        // ---- FASTA branch: run encoder, count columns, discard result ----
        std::vector<fs::path> pff_paths;
        for (auto& fasta_path : all_fasta_paths) {
            fs::path pff_path = cache_dir / (fasta_path.stem().string() + ".pff");
            if (!fs::exists(pff_path)) {
                std::cerr << "Warning: no .pff for " << fasta_path.filename() << ", skipping\n";
                continue;
            }
            pff_paths.push_back(pff_path);
        }
        int total_encode = static_cast<int>(pff_paths.size());
        bool skip_x = (pre.datatype == "protein" || pre.datatype == "nucleotide");

        std::queue<int> work_queue;
        for (int i = 0; i < total_encode; ++i) work_queue.push(i);
        int done_count = 0;

        auto worker = [&]() {
            while (true) {
                int idx;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (work_queue.empty()) break;
                    idx = work_queue.front();
                    work_queue.pop();
                }
                std::string stem = pff_paths[idx].stem().string();
                try {
                    auto result = pre.use_dlt
                        ? encoder::encode_pff_dlt(pff_paths[idx], hyp_seq_names, min_minor,
                                                  opts.drop_major, opts.dropout_labels, skip_x)
                        : encoder::encode_pff(pff_paths[idx], hyp_seq_names, min_minor,
                                              opts.drop_major, opts.dropout_labels, skip_x);
                    uint64_t ncols = result.columns.size();
                    // result (including columns) goes out of scope here
                    std::lock_guard<std::mutex> lock(sizes_mutex);
                    sizes[stem] = ncols;
                    ++done_count;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(sizes_mutex);
                    ++done_count;
                    std::cerr << "[" << done_count << "/" << total_encode << "] FAIL: "
                              << pff_paths[idx].filename().string() << " -> " << e.what() << "\n";
                }
            }
        };

        unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_encode));
        if (tc == 0) tc = 1;
        std::vector<std::thread> threads;
        threads.reserve(tc);
        for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    }

    return sizes;
}

} // namespace pipeline
