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
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <armadillo>
#include "fasta_parser.hpp"
#include "regression.hpp"
#include "pff_format.hpp"
#include "encoder.hpp"

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
    std::cout << "Fungi Genomics Analysis Tool\n";
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
    std::cout << "  --datatype <type>: universal (default), protein, nucleotide\n";
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
                    if (datatype != "universal" && datatype != "protein" && datatype != "nucleotide")
                        throw std::runtime_error("Unknown datatype: " + datatype);
                } else if (arg == "--nfolds" && i + 1 < argc) {
                    nfolds = std::stoi(argv[++i]);
                    if (nfolds < 2)
                        throw std::runtime_error("--nfolds must be >= 2");
                } else {
                    std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
                }
            }

            if (!lambda_file_path.empty() && lambda_explicitly_set)
                throw std::runtime_error("--lambda and --lambda-file are mutually exclusive");

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
                while (std::getline(hyp_file, line)) {
                    if (line.empty()) continue;
                    auto tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    double val = std::stod(line.substr(tab + 1));
                    if (val == 0.0) continue;
                    hyp_seq_names.push_back(line.substr(0, tab));
                    hyp_values.push_back(static_cast<float>(val));
                }
            }
            std::cout << "Hypothesis sequences (non-zero): " << hyp_seq_names.size() << "\n";

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
            if (datatype != "universal") {
                fs::path ini = fs::path(argv[0]).parent_path() / "data_defs.ini";
                if (!fs::exists(ini)) ini = fs::current_path() / "data_defs.ini";
                if (!fs::exists(ini))
                    throw std::runtime_error("data_defs.ini not found");
                allowed_chars = load_datatype_chars(ini, datatype);
                if (allowed_chars.empty())
                    throw std::runtime_error("No chars defined for '" + datatype + "' in data_defs.ini");
            }

            // --- Phase 1: Conversion ---
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

            int conv_converted = 0, conv_failed = 0;
            double conv_elapsed = 0.0;
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

            // --- Phase 2: Encoding ---
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
            std::cout << "  Encoder:              " << (use_dlt ? "DLT" : "standard") << "\n\n";

            uint32_t N = static_cast<uint32_t>(hyp_seq_names.size());
            auto encode_start = std::chrono::steady_clock::now();
            std::vector<encoder::AlignmentResult> results(total_encode);
            {
                std::mutex queue_mutex, print_mutex;
                std::queue<int> work_queue;
                for (int i = 0; i < total_encode; ++i) work_queue.push(i);
                int done_count = 0;

                auto enc_worker = [&]() {
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
                                ? encoder::encode_pff_dlt(pff_paths[idx], hyp_seq_names, min_minor)
                                : encoder::encode_pff(pff_paths[idx], hyp_seq_names, min_minor);
                            {
                                std::lock_guard<std::mutex> lock(print_mutex);
                                ++done_count;
                                std::cout << "[" << done_count << "/" << total_encode << "] "
                                          << pff_paths[idx].filename().string()
                                          << " -> " << results[idx].columns.size() << " columns\n";
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
                        }
                    }
                };

                unsigned int tc = std::min(num_threads, static_cast<unsigned int>(total_encode));
                std::vector<std::thread> threads;
                threads.reserve(tc);
                for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(enc_worker);
                for (auto& t : threads) t.join();
            }
            double encode_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - encode_start).count();

            auto write_start = std::chrono::steady_clock::now();
            std::vector<std::string> all_missing;
            for (int i = 0; i < total_encode; ++i)
                for (auto& m : results[i].missing_sequences)
                    all_missing.push_back(m);

            uint64_t total_cols = 0;
            int n_aligned = 0, failed_count = 0;
            for (auto& r : results) {
                if (r.failed) { ++failed_count; continue; }
                total_cols += r.columns.size();
                ++n_aligned;
            }

            arma::fmat features(N, total_cols, arma::fill::zeros);
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
            arma::frowvec responses(hyp_values.data(), N);

            std::cout << "\nWriting alignment table...\n";
            arma::mat alg_table(3, n_aligned, arma::fill::zeros);
            {
                uint64_t offset = 0;
                int col = 0;
                for (auto& r : results) {
                    if (r.failed) continue;
                    uint64_t ncols = r.columns.size();
                    alg_table(0, col) = static_cast<double>(offset + 1); // 1-based start
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

            // Write combined feature map (column index -> position/allele, in matrix column order)
            {
                std::ofstream combined_map(output_dir / "combined.map");
                combined_map << "Position\tLabel\n"; // header: skipped by writeSparseMappedWeightsToStream
                for (auto& r : results) {
                    if (r.failed) continue;
                    for (auto& [pos, allele] : r.map)
                        combined_map << pos << '\t' << r.stem << '_' << pos << '_' << allele << '\n';
                }
            }

            double write_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - write_start).count();

            // --- Phase 3: Regression ---
            double regr_elapsed = 0.0;
            if (nfolds > 0 && method.empty())
                throw std::runtime_error("--nfolds requires --method");

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

                std::mutex queue_mutex, print_mutex;
                std::queue<int> work_queue;
                for (int i = 0; i < (int)lambdas.size(); ++i) work_queue.push(i);

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

                        if (nfolds == 0) {
                            // Single model
                            auto regr = regression::createRegressionAnalysis(
                                method, features, responses, alg_table.t(), params, lam);
                            std::ofstream wo(lam_dir / "weights.txt");
                            std::ifstream mi(output_dir / "combined.map");
                            regr->writeSparseMappedWeightsToStream(wo, mi);
                            {
                                std::lock_guard<std::mutex> lk(print_mutex);
                                std::cout << "  [" << idx << "] lambda=[" << lam[0] << ","
                                          << lam[1] << "] -> " << (lam_dir / "weights.txt").string() << "\n";
                            }
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

                                {
                                    std::lock_guard<std::mutex> lk(print_mutex);
                                    std::cout << "  [" << idx << "] fold " << k
                                              << ": held-out=" << held_out
                                              << ", non-zero weights=" << fold_weights.size() << "\n";
                                }
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

                            {
                                std::lock_guard<std::mutex> lk(print_mutex);
                                std::cout << std::fixed << std::setprecision(4);
                                std::cout << "  [" << idx << "] lambda=[" << lam[0] << ","
                                          << lam[1] << "] CV -> " << lam_dir.filename().string() << "/\n";
                                std::cout << "    TP=" << tp << " TN=" << tn
                                          << " FP=" << fp << " FN=" << fn << "\n";
                                std::cout << "    TPR=" << tpr << " TNR=" << tnr
                                          << " FPR=" << fpr << " FNR=" << fnr << "\n";
                            }
                        }
                    }
                };

                unsigned int tc = std::min(num_threads,
                                           static_cast<unsigned int>(lambdas.size()));
                std::vector<std::thread> threads;
                threads.reserve(tc);
                for (unsigned int i = 0; i < tc; ++i) threads.emplace_back(lambda_worker);
                for (auto& t : threads) t.join();

                regr_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - regr_start).count();
            }

            std::cout << "\n--- Summary ---\n";
            std::cout << "  Alignments encoded:   " << n_aligned << "/" << total_encode << "\n";
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

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--cache-dir" && i + 1 < argc) {
                    cache_dir = argv[++i];
                } else if (arg == "--hypothesis" && i + 1 < argc) {
                    hyp_path = argv[++i];
                } else if (arg == "--datatype" && i + 1 < argc) {
                    datatype = argv[++i];
                    if (datatype != "universal" && datatype != "protein" && datatype != "nucleotide")
                        throw std::runtime_error("Unknown datatype: " + datatype);
                } else if (arg == "--threads" && i + 1 < argc) {
                    num_threads = static_cast<unsigned int>(std::stoi(argv[++i]));
                    if (num_threads == 0) num_threads = 1;
                } else {
                    std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
                }
            }

            // --- Parse weights file ---
            struct ModelEntry {
                std::string stem;
                uint32_t    pos;
                char        allele;
                double      weight;
            };

            std::vector<ModelEntry> model_entries;
            double intercept_val = 0.0;
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

                    // Parse "<stem>_<pos>_<allele>" from the right:
                    // last '_' separates allele (1 char), second-last separates pos (digits)
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

            // --- Verify all model alignments are present in list ---
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

            // --- Load allowed chars for datatype validation ---
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
            // Group model entries by stem for efficient per-alignment processing
            std::map<std::string, std::vector<size_t>> stem_entry_indices;
            for (size_t i = 0; i < model_entries.size(); ++i)
                stem_entry_indices[model_entries[i].stem].push_back(i);

            // species_sums accumulates weighted contributions; initialized to 0 on first encounter
            std::map<std::string, double> species_sums;

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

                // Ensure every species in this alignment has an entry in the sums map
                for (auto& id : meta.seq_ids)
                    species_sums.emplace(id, 0.0);

                // Accumulate weighted contributions
                bool col_major = (meta.orientation == pff::Orientation::COLUMN_MAJOR);
                for (size_t idx : indices) {
                    auto& entry = model_entries[idx];
                    if (entry.pos >= L) continue;
                    for (uint32_t si = 0; si < S; ++si) {
                        char c = col_major
                            ? raw[static_cast<size_t>(entry.pos) * S + si]
                            : raw[static_cast<size_t>(si) * L + entry.pos];
                        if (c == entry.allele)
                            species_sums[meta.seq_ids[si]] += entry.weight;
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
                    if (line.empty()) continue;
                    auto tab = line.find('\t');
                    if (tab == std::string::npos) continue;
                    double val = std::stod(line.substr(tab + 1));
                    if (val == 0.0) continue;
                    true_values[line.substr(0, tab)] = val;
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
