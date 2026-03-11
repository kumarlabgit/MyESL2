#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <queue>
#include <cmath>
#include <chrono>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <armadillo>
#include "fasta_parser.hpp"
#include "regression.hpp"
#include "pff_format.hpp"
#include "encoder.hpp"
#include "numeric_parser.hpp"
#include "visualizer.hpp"
#include "newick.hpp"

namespace fs = std::filesystem;

static std::unordered_set<char> load_datatype_chars(
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

void print_usage(const char* prog_name) {
    std::cout << "MyESL2 - My Evolutionary Sparse Learning 2\n";
    std::cout << "===========================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " train <list.txt> <hypothesis.txt> <output_dir> [column|row] [--cache-dir DIR] [--min-minor N] [--threads N] [--dlt] [--datatype <type>] [--nfolds N]\n";
    std::cout << "  " << prog_name << " evaluate <weights.txt> <list.txt> <output_file> [--cache-dir DIR] [--datatype <type>] [--threads N] [--hypothesis <file>]\n";
    std::cout << "  " << prog_name << " info <file.pff>\n";
    std::cout << "\nCommands:\n";
    std::cout << "  train      - Convert FASTA files to PFF, encode, then run regression\n";
    std::cout << "               --method <name>:         regression method (omit to skip regression)\n";
    std::cout << "               --lambda <l1> <l2>:      single lambda pair; writes lambda_list.txt (default: 0.1 0.1)\n";
    std::cout << "               --lambda-file <path>:    file of lambda pairs (one 'l1 l2' per line); mutually exclusive with --lambda\n";
    std::cout << "               --param <key>=<value>:   slep option or method-specific param\n";
    std::cout << "                 intercept=false        disable intercept term\n";
    std::cout << "                 field=<path>           group-index CSV (overlapping methods)\n";
    std::cout << "               --nfolds N:              run K-fold cross-validation (N >= 2, requires --method)\n";
    std::cout << "  evaluate   - Apply a trained model to predict response values for new species\n";
    std::cout << "               --hypothesis <file>: compare predictions to true values (TPR/TNR/FPR/FNR)\n";
    std::cout << "  info       - Display PFF file metadata\n";
    std::cout << "\nCommon options:\n";
    std::cout << "  --cache-dir DIR: directory for .pff and .err files (default: pff_cache in CWD)\n";
    std::cout << "  --threads N:     number of worker threads (default: all cores)\n";
    std::cout << "  --min-minor N:   min non-major non-indel count to keep a position (default: 1)\n";
    std::cout << "  --dlt:           use direct lookup table encoder\n";
    std::cout << "  --datatype <type>: universal (default), protein, nucleotide, numeric\n";
    std::cout << "                     numeric: list file points to whitespace-delimited tabular files\n";
    std::cout << "                              (first col = sample name, remaining cols = float features)\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string command = argv[1];

        if (command == "train") {
            if (argc < 5) {
                std::cerr << "Error: train requires <list.txt> <hypothesis.txt> <output_dir>\n";
                print_usage(argv[0]);
                return 1;
            }

            fs::path list_path  = argv[2];
            fs::path hyp_path   = argv[3];
            fs::path output_dir = argv[4];

            pff::Orientation orientation = pff::Orientation::COLUMN_MAJOR;
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
            fs::path cache_dir = fs::current_path() / "pff_cache";
            int min_minor = 1;
            bool use_dlt = false;
            std::string method;
            std::string datatype = "universal";
            std::array<double, 2> lambda = {0.1, 0.1};
            std::map<std::string, std::string> params;
            int nfolds = 0;
            std::string lambda_file_path;
            bool lambda_explicitly_set = false;
            std::array<std::string, 2> lambda_grid_specs;
            bool lambda_grid_set = false;
            double auto_bit_ct = -1.0;
            bool drop_major = false;
            std::string class_bal; // "up", "down", or "weighted"
            std::unordered_set<std::string> dropout_set; // feature labels to exclude

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "row") {
                    orientation = pff::Orientation::ROW_MAJOR;
                } else if (arg == "column") {
                    orientation = pff::Orientation::COLUMN_MAJOR;
                } else if (arg == "--threads" && i + 1 < argc) {
                    num_threads = static_cast<unsigned int>(std::stoi(argv[++i]));
                    if (num_threads == 0) num_threads = 1;
                } else if (arg == "--cache-dir" && i + 1 < argc) {
                    cache_dir = argv[++i];
                } else if (arg == "--min-minor" && i + 1 < argc) {
                    min_minor = std::stoi(argv[++i]);
                } else if (arg == "--dlt") {
                    use_dlt = true;
                } else if (arg == "--method" && i + 1 < argc) {
                    method = argv[++i];
                } else if (arg == "--lambda" && i + 2 < argc) {
                    lambda[0] = std::stod(argv[++i]);
                    lambda[1] = std::stod(argv[++i]);
                    lambda_explicitly_set = true;
                } else if (arg == "--lambda-file" && i + 1 < argc) {
                    lambda_file_path = argv[++i];
                } else if (arg == "--param" && i + 1 < argc) {
                    std::string kv = argv[++i];
                    auto eq = kv.find('=');
                    if (eq != std::string::npos)
                        params[kv.substr(0, eq)] = kv.substr(eq + 1);
                    else
                        std::cerr << "Warning: --param '" << kv << "' has no '=', ignoring\n";
                } else if (arg == "--datatype" && i + 1 < argc) {
                    datatype = argv[++i];
                    if (datatype != "universal" && datatype != "protein" &&
                        datatype != "nucleotide" && datatype != "numeric")
                        throw std::runtime_error("Unknown datatype: " + datatype);
                } else if (arg == "--nfolds" && i + 1 < argc) {
                    nfolds = std::stoi(argv[++i]);
                    if (nfolds < 2)
                        throw std::runtime_error("--nfolds must be >= 2");
                } else if (arg == "--lambda-grid" && i + 2 < argc) {
                    lambda_grid_specs[0] = argv[++i];
                    lambda_grid_specs[1] = argv[++i];
                    lambda_grid_set = true;
                } else if (arg == "--auto-bit-ct" && i + 1 < argc) {
                    auto_bit_ct = std::stod(argv[++i]);
                } else if (arg == "--drop-major-allele") {
                    drop_major = true;
                } else if (arg == "--class-bal" && i + 1 < argc) {
                    class_bal = argv[++i];
                    if (class_bal != "up" && class_bal != "down" && class_bal != "weighted")
                        throw std::runtime_error("--class-bal must be up, down, or weighted");
                } else if (arg == "--dropout" && i + 1 < argc) {
                    std::ifstream df(argv[++i]);
                    if (!df) throw std::runtime_error("Cannot open dropout file: " + std::string(argv[i]));
                    std::string line;
                    while (std::getline(df, line)) {
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        if (!line.empty()) dropout_set.insert(line);
                    }
                    std::cout << "Dropout: " << dropout_set.size() << " features excluded\n";
                } else {
                    std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
                }
            }

            if (!lambda_file_path.empty() && lambda_explicitly_set)
                throw std::runtime_error("--lambda and --lambda-file are mutually exclusive");
            if (lambda_grid_set && lambda_explicitly_set)
                throw std::runtime_error("--lambda-grid and --lambda are mutually exclusive");
            if (lambda_grid_set && !lambda_file_path.empty())
                throw std::runtime_error("--lambda-grid and --lambda-file are mutually exclusive");

            if (nfolds > 0 && method.empty())
                throw std::runtime_error("--nfolds requires --method");

            // Read hypothesis file
            std::vector<std::string> hyp_seq_names;
            std::vector<float> hyp_values;
            {
                std::ifstream hyp_file(hyp_path);
                if (!hyp_file) {
                    std::cerr << "Error: cannot open hypothesis file: " << hyp_path << "\n";
                    return 1;
                }
                std::string line;
                int line_num = 0;
                while (std::getline(hyp_file, line)) {
                    ++line_num;
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    // Accept tab or space as delimiter
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

            // --auto-bit-ct: set min_minor as percentage of minority class size
            if (auto_bit_ct > 0.0) {
                int pos_count = 0, neg_count = 0;
                for (float v : hyp_values) {
                    if (v > 0.0f) ++pos_count;
                    else if (v < 0.0f) ++neg_count;
                }
                int min_class = std::min(pos_count, neg_count);
                min_minor = std::max(1, static_cast<int>(std::ceil(auto_bit_ct / 100.0 * min_class)));
                std::cout << "auto-bit-ct: min-minor set to " << min_minor
                          << " (" << auto_bit_ct << "% of " << min_class << ")\n";
            }

            // Read list file
            std::vector<fs::path> all_fasta_paths;
            {
                std::ifstream list_file(list_path);
                if (!list_file) {
                    std::cerr << "Error: cannot open list file: " << list_path << "\n";
                    return 1;
                }
                fs::path list_dir = list_path.parent_path();
                std::string line;
                while (std::getline(list_file, line)) {
                    if (line.empty()) continue;
                    for (char& c : line) if (c == '\\') c = '/';
                    all_fasta_paths.push_back(list_dir / line);
                }
            }

            fs::create_directories(cache_dir);
            fs::create_directories(output_dir);

            std::unordered_set<char> allowed_chars;
            if (datatype != "universal" && datatype != "numeric") {
                fs::path ini = fs::path(argv[0]).parent_path() / "data_defs.ini";
                if (!fs::exists(ini)) ini = fs::current_path() / "data_defs.ini";
                if (!fs::exists(ini))
                    throw std::runtime_error("data_defs.ini not found");
                allowed_chars = load_datatype_chars(ini, datatype);
                if (allowed_chars.empty())
                    throw std::runtime_error("No chars defined for '" + datatype + "' in data_defs.ini");
            }

            // --- Phase 1: Conversion ---
            int conv_converted = 0, conv_failed = 0;
            double conv_elapsed = 0.0;

            if (datatype == "numeric") {
                // Numeric branch: FASTA paths are actually tabular files; cache as .pnf
                std::queue<fs::path> convert_queue;
                int skipped_done = 0, skipped_error = 0;
                for (auto& tab_path : all_fasta_paths) {
                    fs::path pnf_path = cache_dir / (tab_path.stem().string() + ".pnf");
                    fs::path err_path = cache_dir / (tab_path.stem().string() + ".err");
                    if (fs::exists(err_path)) { ++skipped_error; continue; }
                    if (fs::exists(pnf_path)) {
                        try {
                            auto meta = numeric::read_pnf_metadata(pnf_path);
                            if (meta.source_path == fs::absolute(tab_path).string())
                                { ++skipped_done; continue; }
                        } catch (...) {}
                    }
                    convert_queue.push(tab_path);
                }

                int total_to_convert = static_cast<int>(convert_queue.size());
                std::cout << "\n--- Phase 1: Conversion (numeric) ---\n";
                std::cout << "  Cache directory:       " << cache_dir << "\n";
                std::cout << "  To convert:            " << total_to_convert << "\n";
                std::cout << "  Skipped (done):        " << skipped_done << "\n";
                std::cout << "  Skipped (prior error): " << skipped_error << "\n";
                std::cout << "  Worker threads:        " << num_threads << "\n\n";

                if (total_to_convert > 0) {
                    std::mutex queue_mutex, print_mutex;
                    auto conv_start = std::chrono::steady_clock::now();

                    auto conv_worker = [&]() {
                        while (true) {
                            fs::path tab_path;
                            {
                                std::lock_guard<std::mutex> lock(queue_mutex);
                                if (convert_queue.empty()) break;
                                tab_path = convert_queue.front();
                                convert_queue.pop();
                            }
                            fs::path pnf_path = cache_dir / (tab_path.stem().string() + ".pnf");
                            fs::path err_path = cache_dir / (tab_path.stem().string() + ".err");
                            try {
                                numeric::tabular_to_pnf(tab_path, pnf_path);
                                std::lock_guard<std::mutex> lock(print_mutex);
                                ++conv_converted;
                                std::cout << "[" << conv_converted + conv_failed << "/" << total_to_convert << "] OK: "
                                          << tab_path.filename() << "\n";
                            } catch (const std::exception& e) {
                                std::ofstream err_file(err_path);
                                if (err_file) err_file << e.what() << "\n";
                                std::lock_guard<std::mutex> lock(print_mutex);
                                ++conv_failed;
                                std::cerr << "[" << conv_converted + conv_failed << "/" << total_to_convert << "] FAIL: "
                                          << tab_path.filename() << " -> " << e.what() << "\n";
                            }
                        }
                    };

                    unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_to_convert));
                    std::vector<std::thread> threads;
                    threads.reserve(tc);
                    for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(conv_worker);
                    for (auto& t : threads) t.join();

                    conv_elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - conv_start).count();
                    std::cout << "\nConversion done. Converted: " << conv_converted
                              << ", Failed: " << conv_failed << "\n";
                } else {
                    std::cout << "Nothing to convert.\n";
                }
            } else {
                // FASTA branch
                std::queue<fs::path> convert_queue;
                int skipped_done = 0, skipped_error = 0, skipped_mismatch = 0;
                for (auto& fasta_path : all_fasta_paths) {
                    fs::path pff_path = cache_dir / (fasta_path.stem().string() + ".pff");
                    fs::path err_path = cache_dir / (fasta_path.stem().string() + ".err");
                    if (fs::exists(err_path)) { ++skipped_error; continue; }
                    if (fs::exists(pff_path)) {
                        try {
                            auto meta = fasta::read_pff_metadata(pff_path);
                            if (meta.orientation == orientation && meta.datatype == datatype &&
                                meta.source_path == fs::absolute(fasta_path).string())
                                { ++skipped_done; continue; }
                            ++skipped_mismatch;
                        } catch (...) {}
                    }
                    convert_queue.push(fasta_path);
                }

                int total_to_convert = static_cast<int>(convert_queue.size());
                std::cout << "\n--- Phase 1: Conversion ---\n";
                std::cout << "  Cache directory:       " << cache_dir << "\n";
                std::cout << "  To convert:            " << total_to_convert << "\n";
                std::cout << "  Skipped (done):        " << skipped_done << "\n";
                std::cout << "  Skipped (prior error): " << skipped_error << "\n";
                std::cout << "  Re-converting (orientation mismatch): " << skipped_mismatch << "\n";
                std::cout << "  Worker threads:        " << num_threads << "\n\n";

                if (total_to_convert > 0) {
                    std::mutex queue_mutex, print_mutex;
                    auto conv_start = std::chrono::steady_clock::now();

                    auto conv_worker = [&]() {
                        while (true) {
                            fs::path fasta_path;
                            {
                                std::lock_guard<std::mutex> lock(queue_mutex);
                                if (convert_queue.empty()) break;
                                fasta_path = convert_queue.front();
                                convert_queue.pop();
                            }
                            fs::path pff_path = cache_dir / (fasta_path.stem().string() + ".pff");
                            fs::path err_path = cache_dir / (fasta_path.stem().string() + ".err");
                            try {
                                fasta::fasta_to_pff(fasta_path, pff_path, orientation, datatype, allowed_chars);
                                std::lock_guard<std::mutex> lock(print_mutex);
                                ++conv_converted;
                                std::cout << "[" << conv_converted + conv_failed << "/" << total_to_convert << "] OK: "
                                          << fasta_path.filename() << "\n";
                            } catch (const std::exception& e) {
                                std::ofstream err_file(err_path);
                                if (err_file) err_file << e.what() << "\n";
                                std::lock_guard<std::mutex> lock(print_mutex);
                                ++conv_failed;
                                std::cerr << "[" << conv_converted + conv_failed << "/" << total_to_convert << "] FAIL: "
                                          << fasta_path.filename() << " -> " << e.what() << "\n";
                            }
                        }
                    };

                    unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_to_convert));
                    std::vector<std::thread> threads;
                    threads.reserve(tc);
                    for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(conv_worker);
                    for (auto& t : threads) t.join();

                    conv_elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - conv_start).count();
                    std::cout << "\nConversion done. Converted: " << conv_converted
                              << ", Failed: " << conv_failed << "\n";
                } else {
                    std::cout << "Nothing to convert.\n";
                }
            }

            // --- Phase 2: Encoding / Matrix Building ---
            uint32_t N = static_cast<uint32_t>(hyp_seq_names.size());
            uint64_t total_cols = 0;
            int n_aligned = 0, failed_count = 0, total_processed = 0;
            arma::fmat features;
            arma::mat alg_table;
            std::vector<std::string> all_missing;
            double encode_elapsed = 0.0;
            // Stems in order, for GSS/PSS computation in Phase 3
            std::vector<std::string> all_stems_ordered;

            auto write_start = std::chrono::steady_clock::now();

            auto fmt_bytes = [](uint64_t b) -> std::string {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2);
                if      (b >= uint64_t(1) << 30) oss << b / double(uint64_t(1) << 30) << " GB";
                else if (b >= uint64_t(1) << 20) oss << b / double(uint64_t(1) << 20) << " MB";
                else if (b >= uint64_t(1) << 10) oss << b / double(uint64_t(1) << 10) << " KB";
                else                              oss << b << " B";
                return oss.str();
            };

            if (datatype == "numeric") {
                // Collect .pnf paths
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
                total_processed = total_files;

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
                                if (work_queue.empty()) break;
                                idx = work_queue.front();
                                work_queue.pop();
                            }
                            NumericResult& nr = num_results[idx];
                            nr.stem = pnf_paths[idx].stem().string();
                            try {
                                auto meta = numeric::read_pnf_metadata(pnf_paths[idx]);
                                auto data = numeric::read_pnf_data(pnf_paths[idx], meta);
                                nr.feature_labels = meta.feature_labels;

                                // Build seq_id -> row index map
                                std::unordered_map<std::string, uint32_t> id_to_row;
                                for (uint32_t s = 0; s < meta.num_sequences; ++s)
                                    id_to_row[meta.seq_ids[s]] = s;

                                // Build seq_mapping: hyp index -> pnf row (-1 if missing)
                                std::vector<int> seq_mapping(N, -1);
                                for (uint32_t i = 0; i < N; ++i) {
                                    auto it = id_to_row.find(hyp_seq_names[i]);
                                    if (it != id_to_row.end())
                                        seq_mapping[i] = static_cast<int>(it->second);
                                    else
                                        nr.missing_sequences.push_back(hyp_seq_names[i]);
                                }

                                // Build columns[j][i]
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
                    std::vector<std::thread> threads;
                    threads.reserve(tc);
                    for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(enc_worker);
                    for (auto& t : threads) t.join();
                }
                encode_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - encode_start).count();

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

                features.zeros(N, total_cols);
                {
                    uint64_t col_offset = 0;
                    for (auto& nr : num_results) {
                        if (nr.failed) continue;
                        for (size_t j = 0; j < nr.columns.size(); ++j)
                            for (uint32_t i = 0; i < N; ++i)
                                features(i, col_offset + j) = nr.columns[j][i];
                        col_offset += nr.columns.size();
                    }
                }

                // Write alignment table
                std::cout << "\nWriting alignment table...\n";
                alg_table.zeros(3, n_aligned);
                {
                    uint64_t offset = 0;
                    int col = 0;
                    for (auto& nr : num_results) {
                        if (nr.failed) continue;
                        uint64_t ncols = nr.columns.size();
                        alg_table(0, col) = static_cast<double>(offset + 1);
                        alg_table(1, col) = static_cast<double>(offset + ncols);
                        alg_table(2, col) = std::sqrt(static_cast<double>(ncols));
                        offset += ncols;
                        ++col;
                    }
                    std::ofstream table(output_dir / "alignment_table.txt");
                    for (int row = 0; row < 3; ++row) {
                        for (int c = 0; c < n_aligned; ++c) {
                            if (c > 0) table << '\t';
                            if (row < 2)
                                table << static_cast<uint64_t>(alg_table(row, c));
                            else
                                table << std::fixed << std::setprecision(6) << alg_table(row, c);
                        }
                        table << '\n';
                    }
                }

                if (!all_missing.empty()) {
                    std::ofstream missing_file(output_dir / "missing_sequences.txt");
                    for (auto& m : all_missing) missing_file << m << '\n';
                    std::cout << "Missing sequences: " << all_missing.size()
                              << " -> " << (output_dir / "missing_sequences.txt").string() << "\n";
                }

                // Write combined.map: <stem>_<feature_name>
                {
                    std::ofstream combined_map(output_dir / "combined.map");
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
                // FASTA branch: encoding
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
                total_processed = total_encode;

                std::cout << "\n--- Phase 2: Encoding ---\n";
                std::cout << "  Alignments to encode: " << total_encode << "\n";
                std::cout << "  Min-minor threshold:  " << min_minor << "\n";
                std::cout << "  Worker threads:       " << num_threads << "\n";
                std::cout << "  Encoder:              " << (use_dlt ? "DLT" : "standard") << "\n\n";

                auto encode_start = std::chrono::steady_clock::now();
                std::vector<encoder::AlignmentResult> results(total_encode);
                uint64_t running_cols = 0;
                uint64_t first_estimate = 0;
                int next_pct = 1;
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
                                if (work_queue.empty()) break;
                                idx = work_queue.front();
                                work_queue.pop();
                            }
                            try {
                                results[idx] = use_dlt
                                    ? encoder::encode_pff_dlt(pff_paths[idx], hyp_seq_names, min_minor, drop_major, dropout_set)
                                    : encoder::encode_pff(pff_paths[idx], hyp_seq_names, min_minor, drop_major, dropout_set);
                                {
                                    std::lock_guard<std::mutex> lock(print_mutex);
                                    ++done_count;
                                    running_cols += results[idx].columns.size();
                                    std::cout << "[" << done_count << "/" << total_encode << "] "
                                              << pff_paths[idx].filename().string()
                                              << " -> " << results[idx].columns.size() << " columns\n";
                                    check_milestone();
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
                    std::vector<std::thread> threads;
                    threads.reserve(tc);
                    for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(enc_worker);
                    for (auto& t : threads) t.join();
                }
                encode_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - encode_start).count();

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

                features.zeros(N, total_cols);
                {
                    uint64_t col_offset = 0;
                    for (auto& r : results) {
                        if (r.failed) continue;
                        for (size_t j = 0; j < r.columns.size(); ++j)
                            for (uint32_t i = 0; i < N; ++i)
                                features(i, col_offset + j) = static_cast<float>(r.columns[j][i]);
                        col_offset += r.columns.size();
                    }
                }

                std::cout << "\nWriting alignment table...\n";
                alg_table.zeros(3, n_aligned);
                {
                    uint64_t offset = 0;
                    int col = 0;
                    for (auto& r : results) {
                        if (r.failed) continue;
                        uint64_t ncols = r.columns.size();
                        alg_table(0, col) = static_cast<double>(offset + 1);
                        alg_table(1, col) = static_cast<double>(offset + ncols);
                        alg_table(2, col) = std::sqrt(static_cast<double>(ncols));
                        offset += ncols;
                        ++col;
                    }
                    std::ofstream table(output_dir / "alignment_table.txt");
                    for (int row = 0; row < 3; ++row) {
                        for (int c = 0; c < n_aligned; ++c) {
                            if (c > 0) table << '\t';
                            if (row < 2)
                                table << static_cast<uint64_t>(alg_table(row, c));
                            else
                                table << std::fixed << std::setprecision(6) << alg_table(row, c);
                        }
                        table << '\n';
                    }
                }

                if (!all_missing.empty()) {
                    std::ofstream missing_file(output_dir / "missing_sequences.txt");
                    for (auto& m : all_missing) missing_file << m << '\n';
                    std::cout << "Missing sequences: " << all_missing.size()
                              << " -> " << (output_dir / "missing_sequences.txt").string() << "\n";
                }

                {
                    std::ofstream combined_map(output_dir / "combined.map");
                    combined_map << "Position\tLabel\n";
                    for (auto& r : results) {
                        if (r.failed) continue;
                        all_stems_ordered.push_back(r.stem);
                        for (auto& [pos, allele] : r.map)
                            combined_map << pos << '\t' << r.stem << '_' << pos << '_' << allele << '\n';
                    }
                }
            }

            double write_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - write_start).count();

            // --- Class balancing (applied to features and responses before regression) ---
            if (!class_bal.empty()) {
                std::vector<uint32_t> pos_idx, neg_idx;
                for (uint32_t i = 0; i < N; ++i) {
                    if (hyp_values[i] > 0.0f) pos_idx.push_back(i);
                    else if (hyp_values[i] < 0.0f) neg_idx.push_back(i);
                }
                std::mt19937 rng(0); // fixed seed for reproducibility

                if (class_bal == "down") {
                    // Downsample majority class to minority class size
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
                    std::cout << "Class balance (down): " << N << " samples retained\n";
                } else if (class_bal == "up") {
                    // Upsample minority class to majority class size
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
                    std::cout << "Class balance (up): " << N << " samples (upsampled minority)\n";
                } else if (class_bal == "weighted") {
                    // Write sweights.txt and inject into params
                    double pos_w = 1.0;
                    double neg_w = (neg_idx.size() > 0 && pos_idx.size() > 0)
                                   ? static_cast<double>(pos_idx.size()) / neg_idx.size()
                                   : 1.0;
                    fs::path sw_path = output_dir / "sweights.txt";
                    {
                        std::ofstream sf(sw_path);
                        sf << std::fixed << std::setprecision(6);
                        sf << pos_w << "\n" << neg_w << "\n";
                    }
                    params["sWeight"] = sw_path.string();
                    std::cout << "Class balance (weighted): sWeight written -> " << sw_path.string()
                              << " (pos=" << pos_w << " neg=" << neg_w << ")\n";
                }
            }

            // Rebuild responses from (possibly updated) hyp_values and N
            arma::frowvec responses(hyp_values.data(), N);

            // --- Phase 3: Regression ---
            double regr_elapsed = 0.0;

            if (!method.empty()) {
                // Build lambda list from file or single --lambda pair
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
                if (!lambda_file_path.empty()) {
                    lambdas = load_lambda_list(lambda_file_path);
                } else if (lambda_grid_set) {
                    // Parse "min,max,step" specs and build Cartesian product
                    auto parse_spec = [](const std::string& spec) {
                        std::vector<double> vals;
                        double vmin, vmax, vstep;
                        char c1, c2;
                        std::istringstream ss(spec);
                        if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || c1 != ',' || c2 != ',')
                            throw std::runtime_error("--lambda-grid spec must be 'min,max,step': " + spec);
                        if (vstep <= 0.0) throw std::runtime_error("lambda-grid step must be > 0");
                        for (double v = vmin; v < vmax - vstep * 1e-9; v += vstep)
                            vals.push_back(v);
                        return vals;
                    };
                    auto v1 = parse_spec(lambda_grid_specs[0]);
                    auto v2 = parse_spec(lambda_grid_specs[1]);
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
                        f << lambda[0] << " " << lambda[1] << "\n";
                    }
                    lambdas = load_lambda_list(gen_path);
                }

                std::cout << "\n--- Phase 3: Regression ---\n";
                std::cout << "  Method:  " << method << "\n";
                std::cout << "  Lambdas: " << lambdas.size() << " pair(s)\n";
                if (nfolds > 0)
                    std::cout << "  K-fold CV: " << nfolds << " folds\n";
                if (!params.empty()) {
                    std::cout << "  Params:\n";
                    for (auto& [k, v] : params)
                        std::cout << "    " << k << " = " << v << "\n";
                }

                auto regr_start = std::chrono::steady_clock::now();

                // Build shared read-only structures once
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

                arma::rowvec xval_idxs(N);
                if (nfolds > 0)
                    for (uint32_t i = 0; i < N; ++i)
                        xval_idxs(i) = static_cast<double>(i % nfolds);

                std::mutex queue_mutex;
                std::queue<int> work_queue;
                for (int i = 0; i < (int)lambdas.size(); ++i) work_queue.push(i);

                // Per-lambda output accumulated during parallel phase, printed in order after join
                std::vector<std::string> lambda_outputs(lambdas.size());

                // Build sorted stems list for numeric longest-prefix matching in GSS
                std::vector<std::string> sorted_stems_desc = all_stems_ordered;
                std::sort(sorted_stems_desc.begin(), sorted_stems_desc.end(),
                    [](const std::string& a, const std::string& b){ return a.size() > b.size(); });

                // Helper: compute and write GSS/PSS, return HSS
                auto compute_sig_scores = [&](const fs::path& wpath, const fs::path& lam_dir,
                                              std::ostringstream& sout, int lam_idx) {
                    bool is_numeric_mode = (datatype == "numeric");
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
                            std::string stem = label.substr(0, us1);
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
                        gf << std::fixed << std::setprecision(6);
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
                        pf << std::fixed << std::setprecision(6);
                        for (auto& [key, v] : pss_entries) {
                            auto tp = key.find('\t');
                            pf << key.substr(0, tp) << '_' << key.substr(tp + 1) << '\t' << v << '\n';
                        }
                    }

                    sout << "  [" << lam_idx << "] HSS=" << std::fixed << std::setprecision(4) << hss << "\n";
                };

                auto lambda_worker = [&]() {
                    while (true) {
                        int idx;
                        {
                            std::lock_guard<std::mutex> lk(queue_mutex);
                            if (work_queue.empty()) break;
                            idx = work_queue.front();
                            work_queue.pop();
                        }
                        auto& lam = lambdas[idx];
                        fs::path lam_dir = output_dir / ("lambda_" + std::to_string(idx));
                        fs::create_directories(lam_dir);

                        std::ostringstream out;
                        out << std::fixed << std::setprecision(4);

                        if (nfolds == 0) {
                            // Single model
                            auto regr = regression::createRegressionAnalysis(
                                method, features, responses, alg_table.t(), params, lam);
                            {
                                std::ofstream wo(lam_dir / "weights.txt");
                                std::ifstream mi(output_dir / "combined.map");
                                regr->writeSparseMappedWeightsToStream(wo, mi);
                            }
                            out << "  [" << idx << "] lambda=[" << lam[0] << ","
                                << lam[1] << "] -> " << (lam_dir / "weights.txt").string() << "\n";
                            compute_sig_scores(lam_dir / "weights.txt", lam_dir, out, idx);
                        } else {
                            // K-fold CV
                            std::vector<double> cv_preds(N, 0.0);
                            for (int k = 0; k < nfolds; ++k) {
                                auto regr = regression::createRegressionAnalysisXVal(
                                    method, features, responses, alg_table.t(), params, lam,
                                    xval_idxs, k);
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
                                    cv_out << hyp_seq_names[i] << '\t'
                                           << cv_preds[i] << '\t'
                                           << hyp_values[i] << '\n';
                            }

                            // Classification metrics
                            int tp = 0, tn = 0, fp = 0, fn = 0;
                            for (uint32_t i = 0; i < N; ++i) {
                                bool pred_pos = cv_preds[i] > 0.0;
                                bool true_pos = hyp_values[i] > 0.0f;
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
                        }

                        lambda_outputs[idx] = out.str();
                    }
                };

                unsigned int tc = std::min(num_threads,
                                           static_cast<unsigned int>(lambdas.size()));
                std::vector<std::thread> threads;
                threads.reserve(tc);
                for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(lambda_worker);
                for (auto& t : threads) t.join();

                for (auto& s : lambda_outputs) std::cout << s;

                // Phase 2a: Grid summary — median GSS/PSS/BSS across all lambdas
                if (lambdas.size() > 1 && nfolds == 0) {
                    // Median of non-zero values only; returns 0 if all values are zero
                    auto median_nonzero = [](std::vector<double>& v) -> double {
                        std::vector<double> nz;
                        for (double x : v) if (x != 0.0) nz.push_back(x);
                        if (nz.empty()) return 0.0;
                        std::sort(nz.begin(), nz.end());
                        size_t m = nz.size() / 2;
                        return (nz.size() % 2 == 0) ? (nz[m - 1] + nz[m]) * 0.5 : nz[m];
                    };

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
                            double med = median_nonzero(v);
                            if (std::abs(med) >= 5e-7) med_gss.push_back({med, g});
                        }
                        std::sort(med_gss.rbegin(), med_gss.rend());
                        std::ofstream mf(output_dir / "gss_median.txt");
                        mf << std::fixed << std::setprecision(6);
                        for (auto& [val, g] : med_gss) mf << g << '\t' << val << '\n';
                        std::cout << "  gss_median.txt written (" << med_gss.size() << " genes)\n";
                    }

                    // PSS median (FASTA only)
                    if (datatype != "numeric") {
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
                                double med = median_nonzero(v);
                                if (std::abs(med) >= 5e-7) med_pss.push_back({k, med});
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
                            double med = median_nonzero(bss_all[label]);
                            if (std::abs(med) >= 5e-7) {
                                mf << label << '\t' << med << '\n';
                                ++bss_written;
                            }
                        }
                        std::cout << "  bss_median.txt written (" << bss_written << " weights)\n";
                    }
                }

                regr_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - regr_start).count();
            }

            std::cout << "\n--- Summary ---\n";
            std::cout << "  Files processed:      " << n_aligned << "/" << total_processed << "\n";
            std::cout << "  Total columns:        " << total_cols << "\n";
            if (failed_count > 0)
                std::cout << "  Failed:               " << failed_count << "\n";
            std::cout << "  Features matrix:      " << features.n_rows << " x " << features.n_cols << "  (arma::fmat)\n";
            std::cout << "  Response vector:      1 x " << responses.n_elem << "  (arma::frowvec)\n";
            std::cout << "  Alignment table:      " << alg_table.n_rows << " x " << alg_table.n_cols << "  (arma::mat)\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n  Phase 1 (conversion): " << conv_elapsed << "s\n";
            std::cout << "  Phase 2 (encoding):   " << encode_elapsed << "s\n";
            std::cout << "  Phase 2 (matrix/map): " << write_elapsed << "s\n";
            if (!method.empty())
                std::cout << "  Phase 3 (regression): " << regr_elapsed << "s\n";
            std::cout << "  Total:                "
                      << (conv_elapsed + encode_elapsed + write_elapsed + regr_elapsed) << "s\n";

        } else if (command == "evaluate") {
            if (argc < 5) {
                std::cerr << "Error: evaluate requires <weights.txt> <list.txt> <output_file>\n";
                print_usage(argv[0]);
                return 1;
            }

            fs::path weights_path = argv[2];
            fs::path list_path    = argv[3];
            fs::path output_path  = argv[4];
            fs::path cache_dir    = fs::current_path() / "pff_cache";
            fs::path hyp_path;
            std::string datatype  = "universal";
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;

            bool no_visualize = false;
            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--cache-dir" && i + 1 < argc) {
                    cache_dir = argv[++i];
                } else if (arg == "--hypothesis" && i + 1 < argc) {
                    hyp_path = argv[++i];
                } else if (arg == "--datatype" && i + 1 < argc) {
                    datatype = argv[++i];
                    if (datatype != "universal" && datatype != "protein" &&
                        datatype != "nucleotide" && datatype != "numeric")
                        throw std::runtime_error("Unknown datatype: " + datatype);
                } else if (arg == "--threads" && i + 1 < argc) {
                    num_threads = static_cast<unsigned int>(std::stoi(argv[++i]));
                    if (num_threads == 0) num_threads = 1;
                } else if (arg == "--no-visualize") {
                    no_visualize = true;
                } else {
                    std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
                }
            }

            double intercept_val = 0.0;
            std::map<std::string, double> species_sums;
            // Per-gene per-species scores for gene_predictions table (NaN = species not in file)
            std::vector<std::string> eval_gene_order;
            std::map<std::string, std::map<std::string, double>> eval_gene_scores;

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
                // --- FASTA path ---
                // Parse weight labels as <stem>_<pos>_<allele>
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
                    fs::path ini = fs::path(argv[0]).parent_path() / "data_defs.ini";
                    if (!fs::exists(ini)) ini = fs::current_path() / "data_defs.ini";
                    if (!fs::exists(ini)) throw std::runtime_error("data_defs.ini not found");
                    allowed_chars = load_datatype_chars(ini, datatype);
                    if (allowed_chars.empty())
                        throw std::runtime_error("No chars defined for '" + datatype + "' in data_defs.ini");
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
            }

            // --- Load hypothesis file (optional) ---
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

            // --- Write predictions ---
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

            std::cout << "Predicted " << species_sums.size() << " species -> " << output_path << "\n";

            // --- Write gene_predictions.txt ---
            fs::path gene_pred_path;
            if (!eval_gene_order.empty()) {
                // Derive output stem: strip extension from output_path
                std::string stem_base = output_path.stem().string();
                fs::path out_dir = output_path.parent_path();
                if (out_dir.empty()) out_dir = ".";
                gene_pred_path = out_dir / (stem_base + "_gene_predictions.txt");
                {
                    std::ofstream gp(gene_pred_path);
                    gp << std::fixed << std::setprecision(6);
                    // Header
                    gp << "SeqID\tResponse\tPrediction\tIntercept";
                    for (auto& g : eval_gene_order) gp << '\t' << g;
                    gp << '\n';
                    for (auto& [species, sum] : species_sums) {
                        double pred = intercept_val + sum;
                        double resp = 0.0;
                        auto tv = true_values.find(species);
                        if (tv != true_values.end()) resp = tv->second;
                        gp << species << '\t' << resp << '\t' << pred << '\t' << intercept_val;
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
                    // Compute normalization factor from hypothesis-labelled species
                    double max_expit_pos = 0.5, min_expit_neg = 0.5;
                    auto expit = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
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
                            spp = (ep - 0.5) / norm_neg;
                        else
                            spp = (ep - 0.5) / norm_pos; // default to pos normalization
                        sf << species << '\t' << resp << '\t' << pred << '\t' << spp << '\n';
                    }
                }
                std::cout << "SPS/SPP -> " << sps_path.string() << "\n";

                // Auto-invoke SVG visualization unless suppressed
                if (!no_visualize) {
                    try {
                        fs::path svg_path = out_dir / (stem_base + ".svg");
                        viz::VizOptions vopt;
                        auto gpt = viz::read_gene_predictions(gene_pred_path);
                        viz::write_svg(gpt, svg_path, vopt);
                        std::cout << "Visualization -> " << svg_path.string() << "\n";
                    } catch (const std::exception& ve) {
                        std::cerr << "Warning: visualization failed: " << ve.what() << "\n";
                    }
                }
            }

            // --- Classification metrics (threshold = 0) ---
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
            }

        } else if (command == "info") {
            if (argc < 3) {
                std::cerr << "Error: info requires a PFF file path\n";
                print_usage(argv[0]);
                return 1;
            }

            fs::path pff_path = argv[2];
            auto metadata = fasta::read_pff_metadata(pff_path);

            std::cout << "PFF File Information\n";
            std::cout << "===================\n";
            std::cout << "Data Offset:      " << metadata.data_offset << " bytes\n";
            std::cout << "Num Sequences:    " << metadata.num_sequences << "\n";
            std::cout << "Alignment Length: " << metadata.alignment_length << "\n";
            std::cout << "Orientation:      " << pff::to_string(metadata.orientation) << "\n";
            std::cout << "Data Size:        " << metadata.get_data_size() << " bytes\n";
            std::cout << "\nSequence IDs:\n";
            for (size_t i = 0; i < metadata.seq_ids.size(); ++i) {
                std::cout << "  [" << i << "] " << metadata.seq_ids[i] << "\n";
            }

        } else if (command == "drphylo") {
            // Two calling conventions:
            //   Tree mode: drphylo <list.txt> <tree.nwk> <output_dir> [options]
            //   Hyp  mode: drphylo <list.txt> <output_dir> --hypothesis <file> [options]
            // Pre-scan to detect hyp mode before checking argc.
            std::string direct_hyp_file;
            for (int i = 4; i < argc; ++i) {
                if (std::string(argv[i]) == "--hypothesis" && i + 1 < argc) {
                    direct_hyp_file = argv[i + 1];
                    break;
                }
            }
            bool hyp_mode = !direct_hyp_file.empty();

            if (argc < (hyp_mode ? 4 : 5)) {
                std::cerr << "Error: drphylo requires <list.txt> <tree.nwk> <output_dir> [options]\n"
                          << "       or: drphylo <list.txt> <output_dir> --hypothesis <file> [options]\n";
                return 1;
            }
            fs::path list_path  = argv[2];
            fs::path tree_path;
            fs::path output_dir;
            int      extra_start;
            if (hyp_mode) {
                output_dir   = argv[3];
                extra_start  = 4;
            } else {
                tree_path    = argv[3];
                output_dir   = argv[4];
                extra_start  = 5;
            }

            // Collect pass-through args for train/evaluate
            std::string clade_list_file;
            std::string gen_clade_spec; // "lower,upper"
            std::string class_bal_dp = "phylo";
            std::vector<std::string> train_args_extra;
            std::string datatype_dp = "universal";
            std::string method_dp;

            for (int i = extra_start; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--clade-list" && i + 1 < argc) {
                    clade_list_file = argv[++i];
                } else if (arg == "--gen-clade-list" && i + 1 < argc) {
                    gen_clade_spec = argv[++i];
                } else if (arg == "--class-bal" && i + 1 < argc) {
                    class_bal_dp = argv[++i];
                } else if (arg == "--hypothesis" && i + 1 < argc) {
                    ++i; // already captured in direct_hyp_file
                } else if (arg == "--datatype" && i + 1 < argc) {
                    datatype_dp = argv[++i];
                    train_args_extra.push_back("--datatype");
                    train_args_extra.push_back(argv[i]);
                } else if (arg == "--method" && i + 1 < argc) {
                    method_dp = argv[++i];
                    train_args_extra.push_back("--method");
                    train_args_extra.push_back(argv[i]);
                } else if (arg == "--min-groups" && i + 1 < argc) {
                    train_args_extra.push_back("--param");
                    train_args_extra.push_back(std::string("min_genes=") + argv[++i]);
                } else {
                    // Pass through to train
                    train_args_extra.push_back(arg);
                    if (i + 1 < argc && arg != "--dlt" && arg != "--drop-major-allele") {
                        // Check if next arg is a value (starts without --)
                        std::string next = argv[i + 1];
                        if (next.size() < 2 || next[0] != '-' || !std::isalpha(next[1]))
                            train_args_extra.push_back(argv[++i]);
                    }
                }
            }

            fs::create_directories(output_dir);

            // Build extra train args string
            auto quote_arg = [](const std::string& s) -> std::string {
                if (s.find(' ') != std::string::npos) return "\"" + s + "\"";
                return s;
            };
            std::string train_extra_str;
            if (method_dp.empty())
                train_extra_str = " --method sg_lasso";
            for (auto& a : train_args_extra) train_extra_str += " " + quote_arg(a);

            std::string exe = quote_arg(argv[0]);

            // HSS accumulator
            std::vector<std::pair<std::string, double>> hss_summary;

            // Shared helper: run train + evaluate + visualize for one hypothesis file.
            // Writes results into run_dir; appends HSS entry labelled by label.
            // class_bal_dp == "phylo" means no --class-bal flag is passed to train.
            auto run_one = [&](const fs::path& hyp_file, const fs::path& run_dir,
                               const std::string& label) {
                std::string bal_arg;
                if (class_bal_dp != "phylo")
                    bal_arg = " --class-bal " + class_bal_dp;

                std::string train_cmd = exe + " train "
                    + quote_arg(list_path.string()) + " "
                    + quote_arg(hyp_file.string()) + " "
                    + quote_arg(run_dir.string())
                    + bal_arg + train_extra_str;
                std::cout << "Running: " << train_cmd << "\n" << std::flush;
                int ret = std::system(train_cmd.c_str());
                if (ret != 0) {
                    std::cerr << "Warning: train failed for " << label << "\n";
                    return;
                }

                fs::path lam0_dir     = run_dir / "lambda_0";
                fs::path weights_file = lam0_dir / "weights.txt";
                if (!fs::exists(weights_file)) return;

                fs::path eval_out = run_dir / "eval.txt";
                std::string eval_cmd = exe + " evaluate "
                    + quote_arg(weights_file.string()) + " "
                    + quote_arg(list_path.string()) + " "
                    + quote_arg(eval_out.string())
                    + (datatype_dp != "universal" ? " --datatype " + datatype_dp : "")
                    + " --hypothesis " + quote_arg(hyp_file.string())
                    + " --no-visualize";
                std::cout << "Running: " << eval_cmd << "\n" << std::flush;
                std::system(eval_cmd.c_str());

                fs::path gp_file = run_dir / "eval_gene_predictions.txt";
                if (fs::exists(gp_file)) {
                    fs::path svg_out = run_dir / "eval.svg";
                    std::string viz_cmd = exe + " visualize "
                        + quote_arg(gp_file.string()) + " "
                        + quote_arg(svg_out.string()) + " --m-grid";
                    std::cout << "Running: " << viz_cmd << "\n" << std::flush;
                    std::system(viz_cmd.c_str());
                }

                fs::path gss_file = lam0_dir / "gss.txt";
                double hss = 0.0;
                if (fs::exists(gss_file)) {
                    std::ifstream gf(gss_file);
                    std::string line;
                    while (std::getline(gf, line)) {
                        auto tab = line.find('\t');
                        if (tab != std::string::npos)
                            hss += std::stod(line.substr(tab + 1));
                    }
                }
                hss_summary.push_back({label, hss});
                std::cout << label << ": HSS=" << hss << "\n";
            };

            if (hyp_mode) {
                // ── Hypothesis-file mode: single run with user-supplied hypothesis ──
                std::string label = fs::path(direct_hyp_file).stem().string();
                run_one(direct_hyp_file, output_dir, label);
            } else {
                // ── Tree/clade mode ───────────────────────────────────────────────
                auto tree = newick::read_newick_file(tree_path);

                // Build clade list
                std::vector<std::string> clade_names;
                if (!clade_list_file.empty()) {
                    std::ifstream clf(clade_list_file);
                    if (!clf) throw std::runtime_error("Cannot open clade-list: " + clade_list_file);
                    std::string line;
                    while (std::getline(clf, line)) {
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        if (!line.empty()) clade_names.push_back(line);
                    }
                }
                if (!gen_clade_spec.empty()) {
                    auto comma = gen_clade_spec.find(',');
                    if (comma == std::string::npos)
                        throw std::runtime_error("--gen-clade-list must be 'lower,upper'");
                    int lower = std::stoi(gen_clade_spec.substr(0, comma));
                    int upper = std::stoi(gen_clade_spec.substr(comma + 1));
                    auto gen_names = newick::auto_name_clades(tree, lower, upper);
                    clade_names.insert(clade_names.end(), gen_names.begin(), gen_names.end());
                    std::cout << "Auto-named " << gen_names.size() << " clades in [" << lower << "," << upper << "]\n";
                }
                if (clade_names.empty())
                    throw std::runtime_error("No clades specified. Use --clade-list or --gen-clade-list");

                auto all_tree_leaves = newick::get_leaves(tree);
                std::cout << "Tree has " << all_tree_leaves.size() << " leaves\n";

                auto find_node = [](const newick::NewickNode& root, const std::string& name)
                        -> const newick::NewickNode* {
                    std::vector<const newick::NewickNode*> queue = {&root};
                    for (size_t qi = 0; qi < queue.size(); ++qi) {
                        auto* n = queue[qi];
                        if (n->name == name) return n;
                        for (auto& c : n->children) queue.push_back(&c);
                    }
                    return nullptr;
                };

                for (auto& clade_name : clade_names) {
                    std::cout << "\n=== DrPhylo clade: " << clade_name << " ===\n";
                    fs::path clade_dir = output_dir / clade_name;
                    fs::create_directories(clade_dir);

                    const newick::NewickNode* clade_node = find_node(tree, clade_name);
                    if (!clade_node) {
                        std::cerr << "Warning: clade '" << clade_name << "' not found in tree, skipping\n";
                        continue;
                    }

                    auto pos_leaves = newick::get_leaves(*clade_node);
                    std::set<std::string> pos_set(pos_leaves.begin(), pos_leaves.end());

                    std::vector<std::string> neg_leaves;
                    for (auto& leaf : all_tree_leaves)
                        if (!pos_set.count(leaf)) neg_leaves.push_back(leaf);

                    // Phylo sampling: keep nearest 3 × |pos| negatives by branch distance
                    if (class_bal_dp == "phylo" && neg_leaves.size() > 3 * pos_leaves.size()) {
                        auto leaf_dists = newick::get_leaf_distances(*clade_node);
                        std::unordered_map<std::string, double> dist_map;
                        for (auto& [leaf, d] : leaf_dists)
                            dist_map[leaf] = d;
                        std::sort(neg_leaves.begin(), neg_leaves.end(),
                            [&](const std::string& a, const std::string& b) {
                                double da = dist_map.count(a) ? dist_map[a] : 1e18;
                                double db = dist_map.count(b) ? dist_map[b] : 1e18;
                                return da < db;
                            });
                        neg_leaves.resize(3 * pos_leaves.size());
                    }

                    fs::path hyp_file = clade_dir / "hypothesis.txt";
                    {
                        std::ofstream hf(hyp_file);
                        for (auto& s : pos_leaves) hf << s << "\t1\n";
                        for (auto& s : neg_leaves) hf << s << "\t-1\n";
                    }

                    run_one(hyp_file, clade_dir, clade_name);
                }
            }

            // Write hss_summary.txt (sorted desc)
            std::sort(hss_summary.begin(), hss_summary.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            {
                std::ofstream sf(output_dir / "hss_summary.txt");
                sf << std::fixed << std::setprecision(6);
                for (auto& [name, hss] : hss_summary)
                    sf << name << '\t' << hss << '\n';
            }
            std::cout << "\nhss_summary.txt written for " << hss_summary.size() << " clades\n";

        } else if (command == "aim") {
            if (argc < 5) {
                std::cerr << "Error: aim requires <list.txt> <hypothesis.txt> <output_dir>\n";
                return 1;
            }
            fs::path aim_list_path = argv[2];
            fs::path aim_hyp_path  = argv[3];
            fs::path aim_out_dir   = argv[4];

            double aim_acc_cutoff = 0.9;
            int    aim_max_iter   = 10;
            int    aim_max_ft     = 1000;
            int    aim_window     = 100;
            std::vector<std::string> aim_train_args;

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--aim-acc-cutoff" && i + 1 < argc)
                    aim_acc_cutoff = std::stod(argv[++i]);
                else if (arg == "--aim-max-iter" && i + 1 < argc)
                    aim_max_iter = std::stoi(argv[++i]);
                else if (arg == "--aim-max-ft" && i + 1 < argc)
                    aim_max_ft = std::stoi(argv[++i]);
                else if (arg == "--aim-window" && i + 1 < argc)
                    aim_window = std::stoi(argv[++i]);
                else
                    aim_train_args.push_back(arg);
            }

            // Extract datatype and cache-dir from aim_train_args
            std::string aim_datatype = "universal";
            std::string cache_dir_str = "pff_cache";
            for (size_t k = 0; k < aim_train_args.size(); ++k) {
                if (aim_train_args[k] == "--datatype" && k + 1 < aim_train_args.size())
                    aim_datatype = aim_train_args[k + 1];
                else if (aim_train_args[k] == "--cache-dir" && k + 1 < aim_train_args.size())
                    cache_dir_str = aim_train_args[k + 1];
            }
            bool is_numeric = (aim_datatype == "numeric");
            fs::path aim_cache_dir = cache_dir_str;

            // Check if --lambda / --lambda-grid / --lambda-file already specified
            bool has_lambda = false;
            bool has_method = false;
            for (auto& a : aim_train_args) {
                if (a == "--lambda" || a == "--lambda-grid" || a == "--lambda-file")
                    has_lambda = true;
                if (a == "--method")
                    has_method = true;
            }

            // Build train_common; inject defaults if not specified
            auto quote_aim = [](const std::string& s) -> std::string {
                return "\"" + s + "\"";
            };
            std::string train_common;
            if (!has_lambda)
                train_common = " --lambda-grid \"0.1,0.9,0.1\" \"0.0001,0.0002,0.0001\"";
            if (!has_method)
                train_common += " --method sg_lasso";
            for (auto& a : aim_train_args) train_common += " " + quote_aim(a);

            fs::create_directories(aim_out_dir);
            // Use ""exe" args" quoting pattern so cmd.exe /c passes all arguments correctly
            std::string exe_aim = std::string(argv[0]);

            // Read hypothesis file (skip zero-valued entries, strip \r)
            std::vector<std::string> hyp_species;
            std::vector<double>      hyp_responses;
            {
                std::ifstream hf(aim_hyp_path);
                if (!hf) throw std::runtime_error("Cannot open hypothesis file: " + aim_hyp_path.string());
                std::string line;
                while (std::getline(hf, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    auto delim = line.find('\t');
                    if (delim == std::string::npos) delim = line.find(' ');
                    if (delim == std::string::npos) continue;
                    double val = std::stod(line.substr(delim + 1));
                    if (val == 0.0) continue;
                    hyp_species.push_back(line.substr(0, delim));
                    hyp_responses.push_back(val);
                }
            }
            int N_hyp = static_cast<int>(hyp_species.size());
            int pos_count = 0, neg_count = 0;
            for (double r : hyp_responses) { if (r > 0) ++pos_count; else ++neg_count; }

            // Accumulated selected features (excluded from subsequent iterations)
            std::vector<std::string>      accumulated_selected;
            std::unordered_set<std::string> accumulated_set;
            int total_selected = 0;

            for (int iter = 0; iter < aim_max_iter; ++iter) {
                fs::path iter_dir = aim_out_dir / ("aim_iter_" + std::to_string(iter));

                // Write dropout file (one label per line; may be empty on iter 0)
                fs::path dropout_file = aim_out_dir / ("dropout_" + std::to_string(iter) + ".txt");
                {
                    std::ofstream df(dropout_file);
                    for (auto& lbl : accumulated_selected) df << lbl << '\n';
                }

                // Run train subprocess
                // Use ""exe" args" quoting: outer " so cmd.exe /c passes all args correctly
                std::string tcmd_inner = "\"" + exe_aim + "\" train "
                    + quote_aim(aim_list_path.string()) + " "
                    + quote_aim(aim_hyp_path.string()) + " "
                    + quote_aim(iter_dir.string())
                    + train_common
                    + " --dropout " + quote_aim(dropout_file.string());
                std::string tcmd = "\"" + tcmd_inner + "\"";
                std::cout << "\n[AIM iter " << iter << "] train\n" << tcmd_inner << "\n" << std::flush;
                std::system(tcmd.c_str());

                // Read feature weights: prefer bss_median.txt (multi-lambda), else lambda_0/weights.txt
                std::vector<std::pair<double, std::string>> feature_rank; // (signed_weight, label)
                fs::path bss_file   = iter_dir / "bss_median.txt";
                fs::path lam0_wfile = iter_dir / "lambda_0" / "weights.txt";

                if (fs::exists(bss_file)) {
                    std::ifstream bf(bss_file);
                    std::string line;
                    while (std::getline(bf, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        std::string label = line.substr(0, tab);
                        double w = std::stod(line.substr(tab + 1));
                        feature_rank.push_back({w, label});
                    }
                } else if (fs::exists(lam0_wfile)) {
                    std::ifstream wf(lam0_wfile);
                    std::string line;
                    while (std::getline(wf, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        std::string label = line.substr(0, tab);
                        if (label == "Intercept") continue;
                        double w = std::stod(line.substr(tab + 1));
                        feature_rank.push_back({w, label});
                    }
                } else {
                    std::cerr << "[AIM] No feature weights for iter " << iter << ", terminating\n";
                    break;
                }

                if (feature_rank.empty()) {
                    std::cerr << "[AIM] No features in iter " << iter << ", terminating\n";
                    break;
                }

                // Sort by |weight| descending, take top min(aim_window, total)
                std::sort(feature_rank.begin(), feature_rank.end(),
                    [](const auto& a, const auto& b){
                        return std::abs(a.first) > std::abs(b.first); });
                int window = std::min(aim_window, static_cast<int>(feature_rank.size()));
                feature_rank.resize(window);

                // Read intercept from lambda_0/weights.txt
                double intercept = 0.0;
                if (fs::exists(lam0_wfile)) {
                    std::ifstream wf(lam0_wfile);
                    std::string line;
                    while (std::getline(wf, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        if (line.substr(0, tab) == "Intercept") {
                            intercept = std::stod(line.substr(tab + 1));
                            break;
                        }
                    }
                }

                // ── Precompute x_mat[feat_idx][hyp_idx] ──────────────────────

                std::vector<std::vector<double>> x_mat(
                    window, std::vector<double>(N_hyp, 0.0));

                if (is_numeric) {
                    // Group by stem (label = stem_featname, split at rfind('_'))
                    std::unordered_map<std::string, std::vector<int>> stem_to_feats;
                    for (int fi = 0; fi < window; ++fi) {
                        const auto& label = feature_rank[fi].second;
                        auto us = label.rfind('_');
                        std::string stem = (us != std::string::npos) ? label.substr(0, us) : label;
                        stem_to_feats[stem].push_back(fi);
                    }
                    for (auto& [stem, feat_indices] : stem_to_feats) {
                        fs::path pnf_path = aim_cache_dir / (stem + ".pnf");
                        if (!fs::exists(pnf_path)) continue;
                        try {
                            auto meta = numeric::read_pnf_metadata(pnf_path);
                            auto data = numeric::read_pnf_data(pnf_path, meta);
                            // Build pnf seq_id -> hyp_idx map
                            std::unordered_map<std::string, int> pnf_to_hyp;
                            for (uint32_t pi = 0; pi < meta.num_sequences; ++pi)
                                for (int hi = 0; hi < N_hyp; ++hi)
                                    if (meta.seq_ids[pi] == hyp_species[hi])
                                    { pnf_to_hyp[meta.seq_ids[pi]] = hi; break; }
                            for (int fi : feat_indices) {
                                const auto& label = feature_rank[fi].second;
                                auto us = label.rfind('_');
                                std::string feat_name = (us != std::string::npos)
                                    ? label.substr(us + 1) : label;
                                int feat_col = -1;
                                for (uint32_t j = 0; j < meta.num_features; ++j)
                                    if (meta.feature_labels[j] == feat_name)
                                    { feat_col = static_cast<int>(j); break; }
                                if (feat_col < 0) continue;
                                for (uint32_t pi = 0; pi < meta.num_sequences; ++pi) {
                                    auto it = pnf_to_hyp.find(meta.seq_ids[pi]);
                                    if (it != pnf_to_hyp.end())
                                        x_mat[fi][it->second] = data[pi][feat_col];
                                }
                            }
                        } catch (...) {}
                    }
                } else {
                    // FASTA: label = stem_pos_allele (split at last two '_')
                    std::unordered_map<std::string, std::vector<int>> stem_to_feats;
                    for (int fi = 0; fi < window; ++fi) {
                        const auto& label = feature_rank[fi].second;
                        auto us2 = label.rfind('_');
                        if (us2 == std::string::npos || us2 == 0) continue;
                        auto us1 = label.rfind('_', us2 - 1);
                        if (us1 == std::string::npos) continue;
                        stem_to_feats[label.substr(0, us1)].push_back(fi);
                    }
                    for (auto& [stem, feat_indices] : stem_to_feats) {
                        fs::path pff_path = aim_cache_dir / (stem + ".pff");
                        if (!fs::exists(pff_path)) continue;
                        try {
                            auto meta = fasta::read_pff_metadata(pff_path);
                            // Build pff seq_id -> hyp_idx map
                            std::unordered_map<std::string, int> pff_to_hyp;
                            for (uint32_t pi = 0; pi < meta.num_sequences; ++pi)
                                for (int hi = 0; hi < N_hyp; ++hi)
                                    if (meta.seq_ids[pi] == hyp_species[hi])
                                    { pff_to_hyp[meta.seq_ids[pi]] = hi; break; }
                            // Group by position for bulk reads
                            std::unordered_map<uint32_t, std::vector<int>> pos_to_feats;
                            for (int fi : feat_indices) {
                                const auto& label = feature_rank[fi].second;
                                auto us2 = label.rfind('_');
                                auto us1 = label.rfind('_', us2 - 1);
                                uint32_t pos = static_cast<uint32_t>(
                                    std::stoul(label.substr(us1 + 1, us2 - us1 - 1)));
                                pos_to_feats[pos].push_back(fi);
                            }
                            for (auto& [pos, fis] : pos_to_feats) {
                                std::string pos_str = fasta::read_pff_position(pff_path, pos);
                                for (int fi : fis) {
                                    const auto& label = feature_rank[fi].second;
                                    auto us2 = label.rfind('_');
                                    char allele = label[us2 + 1];
                                    for (uint32_t pi = 0; pi < meta.num_sequences; ++pi) {
                                        auto it = pff_to_hyp.find(meta.seq_ids[pi]);
                                        if (it != pff_to_hyp.end() && pi < pos_str.size())
                                            x_mat[fi][it->second] =
                                                (pos_str[pi] == allele) ? 1.0 : 0.0;
                                    }
                                }
                            }
                        } catch (...) {}
                    }
                }

                // ── Greedy class-balance reorder of window features ───────────

                // Split into pos-weight and neg-weight queues (already sorted desc by |w|)
                std::vector<int> pos_feats, neg_feats;
                for (int fi = 0; fi < window; ++fi) {
                    if (feature_rank[fi].first > 0) pos_feats.push_back(fi);
                    else                              neg_feats.push_back(fi);
                }
                int pf = 0, nf = 0; // front indices into pos_feats / neg_feats

                std::vector<double> running_score(N_hyp, intercept);
                std::vector<int>    resorted;

                for (int k = 0; k < window; ++k) {
                    int tp = 0, tn = 0;
                    for (int s = 0; s < N_hyp; ++s) {
                        if (hyp_responses[s] > 0 && running_score[s] > 0) ++tp;
                        else if (hyp_responses[s] < 0 && running_score[s] < 0) ++tn;
                    }
                    double tp_frac = pos_count > 0 ? static_cast<double>(tp) / pos_count : 1.0;
                    double tn_frac = neg_count > 0 ? static_cast<double>(tn) / neg_count : 1.0;

                    int chosen = -1;
                    if (nf < static_cast<int>(neg_feats.size()) &&
                        (tp_frac >= tn_frac || pf >= static_cast<int>(pos_feats.size())))
                    {
                        chosen = neg_feats[nf++];
                    } else if (pf < static_cast<int>(pos_feats.size())) {
                        chosen = pos_feats[pf++];
                    } else break;

                    for (int s = 0; s < N_hyp; ++s)
                        running_score[s] += feature_rank[chosen].first * x_mat[chosen][s];
                    resorted.push_back(chosen);
                }

                // ── Cumulative TPR/TNR sweep ──────────────────────────────────

                std::vector<double> score(N_hyp, intercept);
                std::vector<viz::AimAccuracyPoint> curve;

                auto make_point = [&]() -> viz::AimAccuracyPoint {
                    int tp = 0, tn = 0, fp = 0, fn = 0;
                    for (int s = 0; s < N_hyp; ++s) {
                        bool pp = score[s] > 0, pt = hyp_responses[s] > 0;
                        if (pt && pp)  ++tp;
                        else if (!pt && !pp) ++tn;
                        else if (pt && !pp)  ++fn;
                        else                 ++fp;
                    }
                    double tpr = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
                    double tnr = (tn + fp) > 0 ? static_cast<double>(tn) / (tn + fp) : 0.0;
                    double acc = N_hyp > 0 ? static_cast<double>(tp + tn) / N_hyp : 0.0;
                    return {tpr, tnr, acc};
                };

                curve.push_back(make_point()); // k=0: intercept only
                for (int ki = 0; ki < static_cast<int>(resorted.size()); ++ki) {
                    int fi = resorted[ki];
                    for (int s = 0; s < N_hyp; ++s)
                        score[s] += feature_rank[fi].first * x_mat[fi][s];
                    curve.push_back(make_point());
                }

                // Find first k where both TPR and TNR >= cutoff
                int cutoff_idx = -1;
                for (int k = 0; k < static_cast<int>(curve.size()); ++k) {
                    if (curve[k].tpr >= aim_acc_cutoff && curve[k].tnr >= aim_acc_cutoff) {
                        cutoff_idx = k;
                        break;
                    }
                }

                if (cutoff_idx == -1) {
                    std::cout << "[AIM] Terminating iter " << iter
                              << ": cutoff not achieved within " << window << " features\n";
                    break;
                }

                // ── Generate AIM SVG ──────────────────────────────────────────

                {
                    viz::AimVizData vd;
                    for (int k : resorted) vd.feature_labels.push_back(feature_rank[k].second);
                    vd.seq_ids   = hyp_species;
                    vd.responses = hyp_responses;
                    vd.curve       = curve;
                    vd.cutoff_idx  = cutoff_idx;
                    vd.contributions.resize(resorted.size(),
                                            std::vector<double>(N_hyp, 0.0));
                    for (int k = 0; k < static_cast<int>(resorted.size()); ++k) {
                        int fi = resorted[k];
                        for (int s = 0; s < N_hyp; ++s)
                            vd.contributions[k][s] = feature_rank[fi].first * x_mat[fi][s];
                    }
                    fs::path svg_out = aim_out_dir / ("aim_iter_" + std::to_string(iter) + ".svg");
                    viz::write_aim_svg(vd, svg_out);
                    std::cout << "[AIM iter " << iter << "] SVG -> " << svg_out.string() << "\n";
                }

                // ── Accumulate selected features (0..cutoff_idx-1) ───────────

                for (int k = 0; k < cutoff_idx; ++k) {
                    const std::string& lbl = feature_rank[resorted[k]].second;
                    if (!accumulated_set.count(lbl)) {
                        accumulated_set.insert(lbl);
                        accumulated_selected.push_back(lbl);
                        ++total_selected;
                    }
                }
                std::cout << "[AIM iter " << iter << "] selected " << cutoff_idx
                          << " features (total=" << total_selected << ")\n";

                if (total_selected >= aim_max_ft) break;
            }

            // Write aim_selected.txt
            {
                std::ofstream sf(aim_out_dir / "aim_selected.txt");
                for (auto& lbl : accumulated_selected) sf << lbl << '\n';
            }
            std::cout << "[AIM] Done. aim_selected.txt written ("
                      << total_selected << " features)\n";

        } else if (command == "visualize") {
            if (argc < 4) {
                std::cerr << "Error: visualize requires <gene_predictions.txt> <output.svg>\n";
                return 1;
            }
            fs::path gp_path  = argv[2];
            fs::path svg_path = argv[3];
            viz::VizOptions opts;
            for (int i = 4; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--gene-limit" && i + 1 < argc) {
                    opts.gene_limit = std::stoi(argv[++i]);
                } else if (arg == "--species-limit" && i + 1 < argc) {
                    opts.species_limit = std::stoi(argv[++i]);
                } else if (arg == "--ssq-threshold" && i + 1 < argc) {
                    opts.ssq_threshold = std::stod(argv[++i]);
                } else if (arg == "--m-grid") {
                    opts.m_grid = true;
                } else {
                    std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
                }
            }
            auto gpt = viz::read_gene_predictions(gp_path);
            viz::write_svg(gpt, svg_path, opts);
            std::cout << "Visualization written -> " << svg_path.string() << "\n";

        } else {
            std::cerr << "Error: unknown command '" << command << "'\n";
            print_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
