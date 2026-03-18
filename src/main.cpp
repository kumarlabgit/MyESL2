#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <thread>
#include <set>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <numeric>
#include <limits>
#include <armadillo>
#include "pipeline_preprocess.hpp"
#include "pipeline_encode.hpp"
#include "pipeline_train.hpp"
#include "pipeline_evaluate.hpp"
#include "pipeline_adaptive.hpp"
#include "fasta_parser.hpp"
#include "numeric_parser.hpp"
#include "pff_format.hpp"
#include "visualizer.hpp"
#include "newick.hpp"
#include "regression.hpp"

namespace fs = std::filesystem;

static std::unordered_set<std::string> load_dropout_labels(const char* path) {
    std::ifstream df(path);
    if (!df) throw std::runtime_error("Cannot open dropout file: " + std::string(path));
    std::unordered_set<std::string> labels;
    std::string line;
    while (std::getline(df, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) labels.insert(line);
    }
    return labels;
}

void print_usage(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "MyESL2 - My Evolutionary Sparse Learning 2\n"
        "===========================================\n\n"
        "USAGE\n"
        "  " + p + " <command> [args...]\n\n"
        "COMMANDS\n"
        "  train       Full pipeline: preprocess -> encode -> regression\n"
        "  evaluate    Apply a trained model to new data\n"
        "  drphylo     Per-clade DrPhylo analysis\n"
        "  aim         Iterative AIM feature-selection loop\n"
        "  visualize   SVG heatmap from gene_predictions.txt\n"
        "  info        Display PFF file metadata\n\n"

        "-------------------------------------------\n"
        "TRAIN\n"
        "  " + p + " train <list.txt> <hypothesis.txt> <output_dir> [column|row] [options]\n\n"
        "  Positional:\n"
        "    list.txt        paths to FASTA/numeric input files (one per line)\n"
        "    hypothesis.txt  species labels (tab-delimited: name <TAB> value)\n"
        "    output_dir      directory for all output files\n"
        "    column|row      PFF orientation (default: column)\n\n"
        "  Regression:\n"
        "    --method <name>              regression method (omit to skip regression)\n"
        "    --precision fp32|fp64        arithmetic precision (default: fp32)\n"
        "    --lambda <l1> <l2>           single lambda pair (default: 0.1 0.1)\n"
        "    --lambda-file <path>         file of lambda pairs, one 'l1 l2' per line\n"
        "    --lambda-grid <s1> <s2>      Cartesian product grid; each spec is min,max,step\n"
        "    --nfolds N                   K-fold cross-validation (N >= 2, requires --method)\n"
        "    --min-groups N               skip lambdas selecting fewer than N non-zero groups\n"
        "    --param <key>=<value>        pass option to solver\n"
        "      intercept=false            disable intercept term\n"
        "      field=<path>               group-index CSV (overlapping group methods)\n"
        "  Encoding:\n"
        "    --auto-bit-ct X              set min_minor = ceil(X% x min_class_size)\n"
        "    --drop-major-allele          exclude major-allele column from FASTA encoder\n"
        "    --class-bal up|down|weighted balance classes before regression\n"
        "    --dropout <file>             exclude features listed in file from encoding\n"
        "    --write-features <path>      write encoded feature matrix to file\n"
        "    --write-features-transposed <path>  write transposed feature matrix to file\n"
        "    --max-mem <bytes>            abort if estimated feature matrix exceeds this size (default: 8589934592)\n"
        "      --param disable_mc=1       warn instead of aborting when max_mem is exceeded\n"
        "    --adaptive-sparsification    on max_mem_exceeded, run adaptive sparsification\n"
        "    --adaptive-lambda-grid <l1_spec> <l2_spec>\n"
        "                                 lambda grid for exploration combos (min,max,step each)\n"
        "                                 (default: \"0.1,0.3,0.1\" \"0.1,0.3,0.1\")\n"
        "  Common:\n"
        "    --cache-dir DIR              directory for .pff/.pnf cache (default: ./pff_cache)\n"
        "    --min-minor N                min non-major non-indel count to keep a position (default: 1)\n"
        "    --threads N                  worker threads (default: all cores)\n"
        "    --dlt                        use direct lookup table encoder\n"
        "    --datatype <type>            universal (default), protein, nucleotide, numeric\n"
        "                                 numeric: list file points to whitespace-delimited tabular files\n"
        "                                          (first col = sample name, remaining cols = features)\n\n"

        "-------------------------------------------\n"
        "EVALUATE\n"
        "  " + p + " evaluate <weights.txt> <list.txt> <output_file> [options]\n\n"
        "    --hypothesis <file>     compare predictions to known labels (writes TPR/TNR/FPR/FNR)\n"
        "    --no-visualize          skip automatic SVG generation\n"
        "    --cache-dir DIR\n"
        "    --threads N\n"
        "    --datatype <type>\n\n"

        "-------------------------------------------\n"
        "DRPHYLO\n"
        "  " + p + " drphylo <list.txt> <tree.nwk> <output_dir> [options]   (tree mode)\n"
        "  " + p + " drphylo <list.txt> <output_dir> --hypothesis <file> [options]  (direct mode)\n\n"
        "  Clade selection (required in tree mode, mutually exclusive):\n"
        "    --clade-list <file>          file listing clades to test\n"
        "    --gen-clade-list <spec>      auto-generate clade list from tree\n"
        "  DrPhylo-specific:\n"
        "    --hypothesis <file>          run a single hypothesis instead of tree clades\n"
        "    --grid-rmse-cutoff X         exclude lambda results above RMSE threshold (default: 100)\n"
        "    --grid-acc-cutoff X          exclude lambda results below accuracy threshold (default: 0)\n"
        "  Shared with train (same semantics):\n"
        "    --method, --precision, --lambda, --lambda-file, --lambda-grid\n"
        "    --param, --nfolds, --min-groups, --auto-bit-ct, --drop-major-allele\n"
        "    --class-bal, --cache-dir, --min-minor, --threads, --dlt, --datatype\n\n"

        "-------------------------------------------\n"
        "AIM\n"
        "  " + p + " aim <list.txt> <hypothesis.txt> <output_dir> [options]\n\n"
        "  AIM-specific:\n"
        "    --aim-acc-cutoff X    TPR and TNR threshold to accept a feature set (default: 0.9)\n"
        "    --aim-max-iter N      max AIM iterations (default: 10)\n"
        "    --aim-max-ft N        stop after accumulating this many features total (default: 1000)\n"
        "    --aim-window N        top-N features considered per iteration (default: 100)\n"
        "  Shared with train (same semantics):\n"
        "    --method, --precision, --lambda, --lambda-file, --lambda-grid\n"
        "    --param, --nfolds, --min-groups, --auto-bit-ct, --drop-major-allele\n"
        "    --class-bal, --cache-dir, --min-minor, --threads, --dlt, --datatype\n\n"

        "-------------------------------------------\n"
        "VISUALIZE\n"
        "  " + p + " visualize <gene_predictions.txt> <output.svg> [options]\n\n"
        "    --gene-limit N        max genes displayed\n"
        "    --species-limit N     max species displayed\n"
        "    --ssq-threshold X     hide genes with sum-squared score below X\n"
        "    --m-grid              draw monochrome grid lines\n\n"

        "-------------------------------------------\n"
        "INFO\n"
        "  " + p + " info <file.pff>\n"
        "    Display metadata for a PFF or PNF cache file.\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string command = argv[1];

        // =====================================================================
        if (command == "train") {
        // =====================================================================
            if (argc < 5) {
                std::cerr << "Error: train requires <list.txt> <hypothesis.txt> <output_dir>\n";
                print_usage(argv[0]);
                return 1;
            }

            pipeline::PreprocessOptions pre_opts;
            pre_opts.list_path   = argv[2];
            pre_opts.output_dir  = argv[4];
            pre_opts.binary_dir  = fs::path(argv[0]).parent_path();

            pipeline::EncodeOptions enc_opts;
            enc_opts.output_dir  = argv[4];
            enc_opts.hyp_path    = argv[3];

            pipeline::TrainOptions train_opts;
            train_opts.output_dir = argv[4];

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "row")         pre_opts.orientation = pff::Orientation::ROW_MAJOR;
                else if (arg == "column") pre_opts.orientation = pff::Orientation::COLUMN_MAJOR;
                else if (arg == "--cache-dir"  && i+1<argc) pre_opts.cache_dir = argv[++i];
                else if (arg == "--min-minor"  && i+1<argc) { pre_opts.min_minor = std::stoi(argv[++i]); enc_opts.min_minor = pre_opts.min_minor; }
                else if (arg == "--dlt")        pre_opts.use_dlt = true;
                else if (arg == "--datatype"   && i+1<argc) {
                    pre_opts.datatype = argv[++i];
                    if (pre_opts.datatype != "universal" && pre_opts.datatype != "protein" &&
                        pre_opts.datatype != "nucleotide" && pre_opts.datatype != "numeric")
                        throw std::runtime_error("Unknown datatype: " + pre_opts.datatype);
                }
                else if (arg == "--threads"    && i+1<argc) { pre_opts.num_threads = static_cast<unsigned>(std::stoi(argv[++i])); if (!pre_opts.num_threads) pre_opts.num_threads = 1; }
                else if (arg == "--method"     && i+1<argc) train_opts.method = argv[++i];
                else if (arg == "--precision"  && i+1<argc) {
                    std::string p = argv[++i];
                    if (p == "fp64") enc_opts.precision = train_opts.precision = regression::Precision::FP64;
                    else if (p == "fp32") enc_opts.precision = train_opts.precision = regression::Precision::FP32;
                    else throw std::runtime_error("--precision must be fp32 or fp64");
                }
                else if (arg == "--lambda"     && i+2<argc) { train_opts.lambda[0]=std::stod(argv[++i]); train_opts.lambda[1]=std::stod(argv[++i]); train_opts.lambda_explicitly_set=true; }
                else if (arg == "--lambda-file"&& i+1<argc) train_opts.lambda_file_path = argv[++i];
                else if (arg == "--lambda-grid"&& i+2<argc) { train_opts.lambda_grid_specs[0]=argv[++i]; train_opts.lambda_grid_specs[1]=argv[++i]; train_opts.lambda_grid_set=true; }
                else if (arg == "--param"      && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts.params[kv.substr(0,eq)]=kv.substr(eq+1); else std::cerr<<"Warning: --param '"<<kv<<"' has no '=', ignoring\n"; }
                else if (arg == "--nfolds"     && i+1<argc) { train_opts.nfolds=std::stoi(argv[++i]); if(train_opts.nfolds<2) throw std::runtime_error("--nfolds must be >= 2"); }
                else if (arg == "--min-groups" && i+1<argc) train_opts.min_groups = std::stoi(argv[++i]);
                else if (arg == "--auto-bit-ct"&& i+1<argc) enc_opts.auto_bit_ct = std::stod(argv[++i]);
                else if (arg == "--drop-major-allele") enc_opts.drop_major = true;
                else if (arg == "--class-bal"  && i+1<argc) {
                    enc_opts.class_bal = argv[++i];
                    if (enc_opts.class_bal != "up" && enc_opts.class_bal != "down" && enc_opts.class_bal != "weighted")
                        throw std::runtime_error("--class-bal must be up, down, or weighted");
                }
                else if (arg == "--dropout"    && i+1<argc) {
                    enc_opts.dropout_labels = load_dropout_labels(argv[++i]);
                    std::cout << "Dropout: " << enc_opts.dropout_labels.size() << " features excluded\n";
                }
                else if (arg == "--write-features"           && i+1<argc) enc_opts.write_features_path = argv[++i];
                else if (arg == "--write-features-transposed"&& i+1<argc) enc_opts.write_features_transposed_path = argv[++i];
                else if (arg == "--max-mem"     && i+1<argc) enc_opts.max_mem = std::stoull(argv[++i]);
                else if (arg == "--adaptive-sparsification") train_opts.adaptive_sparsification = true;
                else if (arg == "--adaptive-lambda-grid" && i+2<argc) {
                    train_opts.adaptive_l1_spec = argv[++i];
                    train_opts.adaptive_l2_spec = argv[++i];
                }
                else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
            }
            if (train_opts.params.count("disable_mc") && train_opts.params.at("disable_mc") == "1")
                enc_opts.disable_mc = true;

            if (!train_opts.lambda_file_path.empty() && train_opts.lambda_explicitly_set)
                throw std::runtime_error("--lambda and --lambda-file are mutually exclusive");
            if (train_opts.lambda_grid_set && train_opts.lambda_explicitly_set)
                throw std::runtime_error("--lambda-grid and --lambda are mutually exclusive");
            if (train_opts.lambda_grid_set && !train_opts.lambda_file_path.empty())
                throw std::runtime_error("--lambda-grid and --lambda-file are mutually exclusive");
            if (train_opts.nfolds > 0 && train_opts.method.empty())
                throw std::runtime_error("--nfolds requires --method");

            pipeline::preprocess(pre_opts);
            try {
                auto enc = pipeline::encode(enc_opts);
                pipeline::train(enc, train_opts);

                std::cout << "\n--- Summary ---\n";
                std::cout << "  Features matrix: " << enc.features.n_rows << " x " << enc.features.n_cols << "\n";
                std::cout << "  Response vector: 1 x " << enc.responses.n_elem << "\n";
            } catch (const std::runtime_error& e) {
                if (std::string_view(e.what()).starts_with("max_mem_exceeded")
                    && train_opts.adaptive_sparsification) {
                    std::cout << "[adaptive] max_mem exceeded — starting adaptive sparsification\n";
                    pipeline::adaptive_train(enc_opts, train_opts);
                } else {
                    throw;
                }
            }

        // =====================================================================
        } else if (command == "evaluate") {
        // =====================================================================
            if (argc < 5) {
                std::cerr << "Error: evaluate requires <weights.txt> <list.txt> <output_file>\n";
                print_usage(argv[0]);
                return 1;
            }

            pipeline::EvaluateOptions eval_opts;
            eval_opts.weights_path = argv[2];
            eval_opts.list_path    = argv[3];
            eval_opts.output_file  = argv[4];
            eval_opts.num_threads  = std::thread::hardware_concurrency();
            if (!eval_opts.num_threads) eval_opts.num_threads = 1;
            eval_opts.cache_dir    = fs::current_path() / "pff_cache";

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if      (arg == "--cache-dir"  && i+1<argc) eval_opts.cache_dir = argv[++i];
                else if (arg == "--hypothesis" && i+1<argc) eval_opts.hyp_path  = argv[++i];
                else if (arg == "--datatype"   && i+1<argc) { eval_opts.datatype = argv[++i]; if(eval_opts.datatype!="universal"&&eval_opts.datatype!="protein"&&eval_opts.datatype!="nucleotide"&&eval_opts.datatype!="numeric") throw std::runtime_error("Unknown datatype: "+eval_opts.datatype); }
                else if (arg == "--threads"    && i+1<argc) { eval_opts.num_threads=static_cast<unsigned>(std::stoi(argv[++i])); if(!eval_opts.num_threads) eval_opts.num_threads=1; }
                else if (arg == "--no-visualize") eval_opts.no_visualize = true;
                else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
            }

            pipeline::evaluate(eval_opts);

        // =====================================================================
        } else if (command == "info") {
        // =====================================================================
            if (argc < 3) {
                std::cerr << "Error: info requires a PFF file path\n";
                print_usage(argv[0]);
                return 1;
            }
            fs::path pff_path = argv[2];
            auto metadata = fasta::read_pff_metadata(pff_path);
            std::cout << "PFF File Information\n===================\n";
            std::cout << "Data Offset:      " << metadata.data_offset << " bytes\n";
            std::cout << "Num Sequences:    " << metadata.num_sequences << "\n";
            std::cout << "Alignment Length: " << metadata.alignment_length << "\n";
            std::cout << "Orientation:      " << pff::to_string(metadata.orientation) << "\n";
            std::cout << "Data Size:        " << metadata.get_data_size() << " bytes\n";
            std::cout << "\nSequence IDs:\n";
            for (size_t i = 0; i < metadata.seq_ids.size(); ++i)
                std::cout << "  [" << i << "] " << metadata.seq_ids[i] << "\n";

        // =====================================================================
        } else if (command == "drphylo") {
        // =====================================================================
            std::string direct_hyp_file;
            for (int i = 4; i < argc; ++i)
                if (std::string(argv[i]) == "--hypothesis" && i+1 < argc) { direct_hyp_file = argv[i+1]; break; }
            bool hyp_mode = !direct_hyp_file.empty();

            if (argc < (hyp_mode ? 4 : 5)) {
                std::cerr << "Error: drphylo requires <list.txt> <tree.nwk> <output_dir> [options]\n"
                          << "       or: drphylo <list.txt> <output_dir> --hypothesis <file> [options]\n";
                return 1;
            }

            fs::path list_path   = argv[2];
            fs::path tree_path;
            fs::path output_dir;
            int extra_start;
            if (hyp_mode) { output_dir = argv[3]; extra_start = 4; }
            else          { tree_path = argv[3]; output_dir = argv[4]; extra_start = 5; }

            // Build shared opts with defaults
            pipeline::PreprocessOptions pre_opts;
            pre_opts.list_path  = list_path;
            pre_opts.output_dir = output_dir;
            pre_opts.binary_dir = fs::path(argv[0]).parent_path();
            pre_opts.tree_path  = tree_path;

            pipeline::EncodeOptions enc_opts_base;
            enc_opts_base.class_bal = "weighted";

            pipeline::TrainOptions train_opts_base;
            train_opts_base.method     = "sg_lasso";
            train_opts_base.min_groups = 3;  // DrPhylo default

            double grid_rmse_cutoff = 100.0;
            double grid_acc_cutoff  = 0.0;
            bool min_groups_set = false;

            for (int i = extra_start; i < argc; ++i) {
                std::string arg = argv[i];
                if      (arg == "--clade-list"       && i+1<argc) pre_opts.clade_list_file = argv[++i];
                else if (arg == "--gen-clade-list"   && i+1<argc) pre_opts.gen_clade_spec  = argv[++i];
                else if (arg == "--class-bal"        && i+1<argc) pre_opts.class_bal_phylo = argv[++i];
                else if (arg == "--hypothesis"       && i+1<argc) { ++i; /* already captured */ }
                else if (arg == "--datatype"         && i+1<argc) pre_opts.datatype        = argv[++i];
                else if (arg == "--threads"          && i+1<argc) { pre_opts.num_threads = static_cast<unsigned>(std::stoi(argv[++i])); if(!pre_opts.num_threads) pre_opts.num_threads=1; }
                else if (arg == "--cache-dir"        && i+1<argc) pre_opts.cache_dir       = argv[++i];
                else if (arg == "--dlt")              pre_opts.use_dlt = true;
                else if (arg == "--min-minor"        && i+1<argc) pre_opts.min_minor        = std::stoi(argv[++i]);
                else if (arg == "--method"           && i+1<argc) train_opts_base.method   = argv[++i];
                else if (arg == "--precision"        && i+1<argc) { std::string p=argv[++i]; if(p=="fp64") enc_opts_base.precision=train_opts_base.precision=regression::Precision::FP64; else if(p!="fp32") throw std::runtime_error("--precision must be fp32 or fp64"); }
                else if (arg == "--lambda"           && i+2<argc) { train_opts_base.lambda[0]=std::stod(argv[++i]); train_opts_base.lambda[1]=std::stod(argv[++i]); train_opts_base.lambda_explicitly_set=true; }
                else if (arg == "--lambda-file"      && i+1<argc) train_opts_base.lambda_file_path = argv[++i];
                else if (arg == "--lambda-grid"      && i+2<argc) { train_opts_base.lambda_grid_specs[0]=argv[++i]; train_opts_base.lambda_grid_specs[1]=argv[++i]; train_opts_base.lambda_grid_set=true; }
                else if (arg == "--param"            && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts_base.params[kv.substr(0,eq)]=kv.substr(eq+1); }
                else if (arg == "--nfolds"           && i+1<argc) { train_opts_base.nfolds=std::stoi(argv[++i]); if(train_opts_base.nfolds<2) throw std::runtime_error("--nfolds must be >= 2"); }
                else if (arg == "--min-groups"       && i+1<argc) { train_opts_base.min_groups=std::stoi(argv[++i]); min_groups_set=true; }
                else if (arg == "--grid-rmse-cutoff" && i+1<argc) grid_rmse_cutoff = std::stod(argv[++i]);
                else if (arg == "--grid-acc-cutoff"  && i+1<argc) grid_acc_cutoff  = std::stod(argv[++i]);
                else if (arg == "--auto-bit-ct"      && i+1<argc) enc_opts_base.auto_bit_ct   = std::stod(argv[++i]);
                else if (arg == "--drop-major-allele") enc_opts_base.drop_major = true;
                else if (arg == "--max-mem"           && i+1<argc) enc_opts_base.max_mem = std::stoull(argv[++i]);
                else std::cerr << "Warning: unknown drphylo argument '" << arg << "', ignoring\n";
            }
            if (train_opts_base.params.count("disable_mc") && train_opts_base.params.at("disable_mc") == "1")
                enc_opts_base.disable_mc = true;
            if (!min_groups_set) train_opts_base.min_groups = 3;

            fs::create_directories(output_dir);

            // Determine actual class_bal for encode
            auto bal_for_encode = [&]() -> std::string {
                const std::string& cb = pre_opts.class_bal_phylo;
                if (cb == "phylo" || cb == "phylo_1" || cb == "phylo_2") return "weighted";
                return cb;
            };
            enc_opts_base.class_bal = bal_for_encode();

            // Helper: run encode+train+evaluate for one hypothesis in one run_dir
            std::vector<std::pair<std::string,double>> hss_summary;

            auto run_one = [&](const fs::path& hyp_file, const fs::path& run_dir, const std::string& label) {
                // Write preprocess_config to run_dir so encode() can find it
                auto pre_for_run = pipeline::read_preprocess_config(output_dir);
                pipeline::write_preprocess_config(run_dir, pre_for_run);

                pipeline::EncodeOptions enc_opts = enc_opts_base;
                enc_opts.output_dir = run_dir;
                enc_opts.hyp_path   = hyp_file;

                auto enc = pipeline::encode(enc_opts);

                pipeline::TrainOptions t_opts = train_opts_base;
                t_opts.output_dir = run_dir;
                auto train_result = pipeline::train(enc, t_opts);

                // Per-lambda evaluate (no visualization)
                auto pre_cfg = pipeline::read_preprocess_config(output_dir);
                for (auto& wp : train_result.weights_paths) {
                    fs::path lam_dir = wp.parent_path();
                    pipeline::EvaluateOptions eopts;
                    eopts.weights_path = wp;
                    eopts.list_path    = pre_cfg.list_path;
                    eopts.output_file  = lam_dir / "eval.txt";
                    eopts.hyp_path     = hyp_file;
                    eopts.no_visualize = true;
                    eopts.datatype     = pre_cfg.datatype;
                    eopts.num_threads  = pre_cfg.num_threads ? pre_cfg.num_threads : std::thread::hardware_concurrency();
                    eopts.cache_dir    = pre_cfg.cache_dir;
                    pipeline::evaluate(eopts);
                }

                auto agg = pipeline::evaluate_drphylo_aggregate(run_dir, grid_rmse_cutoff, grid_acc_cutoff);
                hss_summary.push_back({label, agg.hss});
                std::cout << label << ": HSS=" << agg.hss << "\n";
            };

            if (hyp_mode) {
                // Phase 1 conversion only (no tree)
                pipeline::preprocess(pre_opts);
                std::string label = fs::path(direct_hyp_file).stem().string();
                run_one(direct_hyp_file, output_dir, label);
            } else {
                if (pre_opts.clade_list_file.empty() && pre_opts.gen_clade_spec.empty())
                    throw std::runtime_error("No clades specified. Use --clade-list or --gen-clade-list");

                // Phase 1 conversion + hypothesis generation
                auto hyp_files = pipeline::preprocess(pre_opts);

                for (auto& hyp_file : hyp_files) {
                    fs::path clade_dir  = hyp_file.parent_path();
                    std::string clade_name = clade_dir.filename().string();
                    std::cout << "\n=== DrPhylo clade: " << clade_name << " ===\n";
                    run_one(hyp_file, clade_dir, clade_name);
                }
            }

            // Write hss_summary.txt (sorted desc)
            std::sort(hss_summary.begin(), hss_summary.end(),
                [](const auto& a, const auto& b){ return a.second > b.second; });
            {
                std::ofstream sf(output_dir / "hss_summary.txt");
                sf << std::fixed << std::setprecision(6);
                for (auto& [name, hss] : hss_summary) sf << name << '\t' << hss << '\n';
            }
            std::cout << "\nhss_summary.txt written for " << hss_summary.size() << " clades\n";

        // =====================================================================
        } else if (command == "aim") {
        // =====================================================================
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

            pipeline::PreprocessOptions pre_opts;
            pre_opts.list_path  = aim_list_path;
            pre_opts.output_dir = aim_out_dir;
            pre_opts.binary_dir = fs::path(argv[0]).parent_path();

            pipeline::EncodeOptions enc_opts_base;

            pipeline::TrainOptions train_opts_base;
            train_opts_base.method = "sg_lasso";
            bool has_lambda = false, has_method = false;

            for (int i = 5; i < argc; ++i) {
                std::string arg = argv[i];
                if      (arg == "--aim-acc-cutoff" && i+1<argc) aim_acc_cutoff = std::stod(argv[++i]);
                else if (arg == "--aim-max-iter"   && i+1<argc) aim_max_iter   = std::stoi(argv[++i]);
                else if (arg == "--aim-max-ft"     && i+1<argc) aim_max_ft     = std::stoi(argv[++i]);
                else if (arg == "--aim-window"     && i+1<argc) aim_window     = std::stoi(argv[++i]);
                else if (arg == "--cache-dir"      && i+1<argc) pre_opts.cache_dir  = argv[++i];
                else if (arg == "--datatype"       && i+1<argc) pre_opts.datatype   = argv[++i];
                else if (arg == "--threads"        && i+1<argc) { pre_opts.num_threads=static_cast<unsigned>(std::stoi(argv[++i])); if(!pre_opts.num_threads) pre_opts.num_threads=1; }
                else if (arg == "--dlt")            pre_opts.use_dlt = true;
                else if (arg == "--min-minor"      && i+1<argc) pre_opts.min_minor = std::stoi(argv[++i]);
                else if (arg == "--method"         && i+1<argc) { train_opts_base.method = argv[++i]; has_method = true; }
                else if (arg == "--precision"      && i+1<argc) { std::string p=argv[++i]; if(p=="fp64") enc_opts_base.precision=train_opts_base.precision=regression::Precision::FP64; else if(p!="fp32") throw std::runtime_error("--precision must be fp32 or fp64"); }
                else if (arg == "--lambda"         && i+2<argc) { train_opts_base.lambda[0]=std::stod(argv[++i]); train_opts_base.lambda[1]=std::stod(argv[++i]); train_opts_base.lambda_explicitly_set=true; has_lambda=true; }
                else if (arg == "--lambda-file"    && i+1<argc) { train_opts_base.lambda_file_path=argv[++i]; has_lambda=true; }
                else if (arg == "--lambda-grid"    && i+2<argc) { train_opts_base.lambda_grid_specs[0]=argv[++i]; train_opts_base.lambda_grid_specs[1]=argv[++i]; train_opts_base.lambda_grid_set=true; has_lambda=true; }
                else if (arg == "--param"          && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts_base.params[kv.substr(0,eq)]=kv.substr(eq+1); }
                else if (arg == "--nfolds"         && i+1<argc) train_opts_base.nfolds = std::stoi(argv[++i]);
                else if (arg == "--min-groups"     && i+1<argc) train_opts_base.min_groups = std::stoi(argv[++i]);
                else if (arg == "--class-bal"      && i+1<argc) enc_opts_base.class_bal = argv[++i];
                else if (arg == "--drop-major-allele") enc_opts_base.drop_major = true;
                else if (arg == "--auto-bit-ct"    && i+1<argc) enc_opts_base.auto_bit_ct = std::stod(argv[++i]);
                else if (arg == "--max-mem"         && i+1<argc) enc_opts_base.max_mem = std::stoull(argv[++i]);
                else std::cerr << "Warning: unknown aim argument '" << arg << "', ignoring\n";
            }
            if (train_opts_base.params.count("disable_mc") && train_opts_base.params.at("disable_mc") == "1")
                enc_opts_base.disable_mc = true;
            if (!has_lambda) { train_opts_base.lambda_grid_specs[0]="0.1,0.9,0.1"; train_opts_base.lambda_grid_specs[1]="0.0001,0.0002,0.0001"; train_opts_base.lambda_grid_set=true; }
            if (!has_method) train_opts_base.method = "sg_lasso";

            // Read hypothesis
            std::vector<std::string> hyp_species;
            std::vector<double>      hyp_responses;
            {
                std::ifstream hf(aim_hyp_path);
                if (!hf) throw std::runtime_error("Cannot open hypothesis: " + aim_hyp_path.string());
                std::string line;
                while (std::getline(hf, line)) {
                    if (!line.empty() && line.back()=='\r') line.pop_back();
                    if (line.empty()) continue;
                    auto delim = line.find('\t');
                    if (delim == std::string::npos) delim = line.find(' ');
                    if (delim == std::string::npos) continue;
                    double val = std::stod(line.substr(delim+1));
                    if (val == 0.0) continue;
                    hyp_species.push_back(line.substr(0, delim));
                    hyp_responses.push_back(val);
                }
            }
            int N_hyp = static_cast<int>(hyp_species.size());
            int pos_count = 0, neg_count = 0;
            for (double r : hyp_responses) { if (r > 0) ++pos_count; else ++neg_count; }

            fs::create_directories(aim_out_dir);

            // Phase 1 conversion (once)
            pipeline::preprocess(pre_opts);
            auto pre_cfg = pipeline::read_preprocess_config(aim_out_dir);
            fs::path aim_cache_dir = pre_cfg.cache_dir;
            bool is_numeric = (pre_cfg.datatype == "numeric");

            std::vector<std::string>         accumulated_selected;
            std::unordered_set<std::string>  accumulated_set;
            int total_selected = 0;

            for (int iter = 0; iter < aim_max_iter; ++iter) {
                fs::path iter_dir    = aim_out_dir / ("aim_iter_" + std::to_string(iter));
                fs::path dropout_file = aim_out_dir / ("dropout_" + std::to_string(iter) + ".txt");
                {
                    std::ofstream df(dropout_file);
                    for (auto& lbl : accumulated_selected) df << lbl << '\n';
                }

                // Write preprocess_config to iter_dir for encode()
                fs::create_directories(iter_dir);
                pipeline::write_preprocess_config(iter_dir, pre_cfg);

                pipeline::EncodeOptions enc_opts = enc_opts_base;
                enc_opts.output_dir    = iter_dir;
                enc_opts.hyp_path      = aim_hyp_path;
                enc_opts.dropout_labels = accumulated_set;
                auto enc = pipeline::encode(enc_opts);

                pipeline::TrainOptions t_opts = train_opts_base;
                t_opts.output_dir = iter_dir;
                pipeline::train(enc, t_opts);

                // Read feature weights
                std::vector<std::pair<double,std::string>> feature_rank;
                fs::path bss_file   = iter_dir / "bss_median.txt";
                fs::path lam0_wfile = iter_dir / "lambda_0" / "weights.txt";
                if (fs::exists(bss_file)) {
                    std::ifstream bf(bss_file);
                    std::string line;
                    while (std::getline(bf, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        feature_rank.push_back({std::stod(line.substr(tab+1)), line.substr(0, tab)});
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
                        feature_rank.push_back({std::stod(line.substr(tab+1)), label});
                    }
                } else {
                    std::cerr << "[AIM] No feature weights for iter " << iter << ", terminating\n";
                    break;
                }
                if (feature_rank.empty()) { std::cerr << "[AIM] No features in iter " << iter << ", terminating\n"; break; }

                std::sort(feature_rank.begin(), feature_rank.end(),
                    [](const auto& a, const auto& b){ return std::abs(a.first) > std::abs(b.first); });
                int window = std::min(aim_window, static_cast<int>(feature_rank.size()));
                feature_rank.resize(window);

                // Read intercept
                double intercept = 0.0;
                if (fs::exists(lam0_wfile)) {
                    std::ifstream wf(lam0_wfile);
                    std::string line;
                    while (std::getline(wf, line)) {
                        if (line.empty()) continue;
                        auto tab = line.find('\t');
                        if (tab == std::string::npos) continue;
                        if (line.substr(0, tab) == "Intercept") { intercept = std::stod(line.substr(tab+1)); break; }
                    }
                }

                // Build x_mat[feat_idx][hyp_idx]
                std::vector<std::vector<double>> x_mat(window, std::vector<double>(N_hyp, 0.0));
                if (is_numeric) {
                    std::unordered_map<std::string, std::vector<int>> stem_to_feats;
                    for (int fi = 0; fi < window; ++fi) {
                        const auto& label = feature_rank[fi].second;
                        auto us = label.rfind('_');
                        stem_to_feats[(us!=std::string::npos)?label.substr(0,us):label].push_back(fi);
                    }
                    for (auto& [stem, feat_indices] : stem_to_feats) {
                        fs::path pnf_path = aim_cache_dir / (stem + ".pnf");
                        if (!fs::exists(pnf_path)) continue;
                        try {
                            auto meta = numeric::read_pnf_metadata(pnf_path);
                            auto data = numeric::read_pnf_data(pnf_path, meta);
                            std::unordered_map<std::string,int> pnf_to_hyp;
                            for (uint32_t pi = 0; pi < meta.num_sequences; ++pi)
                                for (int hi = 0; hi < N_hyp; ++hi)
                                    if (meta.seq_ids[pi] == hyp_species[hi]) { pnf_to_hyp[meta.seq_ids[pi]] = hi; break; }
                            for (int fi : feat_indices) {
                                const auto& label = feature_rank[fi].second;
                                auto us = label.rfind('_');
                                std::string feat_name = (us!=std::string::npos)?label.substr(us+1):label;
                                int feat_col = -1;
                                for (uint32_t j = 0; j < meta.num_features; ++j)
                                    if (meta.feature_labels[j] == feat_name) { feat_col = static_cast<int>(j); break; }
                                if (feat_col < 0) continue;
                                for (uint32_t pi = 0; pi < meta.num_sequences; ++pi) {
                                    auto it = pnf_to_hyp.find(meta.seq_ids[pi]);
                                    if (it != pnf_to_hyp.end()) x_mat[fi][it->second] = data[pi][feat_col];
                                }
                            }
                        } catch (...) {}
                    }
                } else {
                    std::unordered_map<std::string, std::vector<int>> stem_to_feats;
                    for (int fi = 0; fi < window; ++fi) {
                        const auto& label = feature_rank[fi].second;
                        auto us2 = label.rfind('_');
                        if (us2==std::string::npos || us2==0) continue;
                        auto us1 = label.rfind('_', us2-1);
                        if (us1==std::string::npos) continue;
                        stem_to_feats[label.substr(0, us1)].push_back(fi);
                    }
                    for (auto& [stem, feat_indices] : stem_to_feats) {
                        fs::path pff_path = aim_cache_dir / (stem + ".pff");
                        if (!fs::exists(pff_path)) continue;
                        try {
                            auto meta = fasta::read_pff_metadata(pff_path);
                            std::unordered_map<std::string,int> pff_to_hyp;
                            for (uint32_t pi = 0; pi < meta.num_sequences; ++pi)
                                for (int hi = 0; hi < N_hyp; ++hi)
                                    if (meta.seq_ids[pi] == hyp_species[hi]) { pff_to_hyp[meta.seq_ids[pi]] = hi; break; }
                            std::unordered_map<uint32_t, std::vector<int>> pos_to_feats;
                            for (int fi : feat_indices) {
                                const auto& label = feature_rank[fi].second;
                                auto us2 = label.rfind('_'), us1 = label.rfind('_', us2-1);
                                pos_to_feats[static_cast<uint32_t>(std::stoul(label.substr(us1+1, us2-us1-1)))].push_back(fi);
                            }
                            for (auto& [pos, fis] : pos_to_feats) {
                                std::string pos_str = fasta::read_pff_position(pff_path, pos);
                                for (int fi : fis) {
                                    const auto& label = feature_rank[fi].second;
                                    char allele = label[label.rfind('_')+1];
                                    for (uint32_t pi = 0; pi < meta.num_sequences; ++pi) {
                                        auto it = pff_to_hyp.find(meta.seq_ids[pi]);
                                        if (it != pff_to_hyp.end() && pi < pos_str.size())
                                            x_mat[fi][it->second] = (pos_str[pi] == allele) ? 1.0 : 0.0;
                                    }
                                }
                            }
                        } catch (...) {}
                    }
                }

                // Greedy class-balance reorder
                std::vector<int> pos_feats, neg_feats;
                for (int fi = 0; fi < window; ++fi)
                    (feature_rank[fi].first > 0 ? pos_feats : neg_feats).push_back(fi);
                int pf = 0, nf = 0;
                std::vector<double> running_score(N_hyp, intercept);
                std::vector<int> resorted;
                for (int k = 0; k < window; ++k) {
                    int tp = 0, tn = 0;
                    for (int s = 0; s < N_hyp; ++s) {
                        if (hyp_responses[s] > 0 && running_score[s] > 0) ++tp;
                        else if (hyp_responses[s] < 0 && running_score[s] < 0) ++tn;
                    }
                    double tp_frac = pos_count > 0 ? static_cast<double>(tp)/pos_count : 1.0;
                    double tn_frac = neg_count > 0 ? static_cast<double>(tn)/neg_count : 1.0;
                    int chosen = -1;
                    if (nf < static_cast<int>(neg_feats.size()) && (tp_frac >= tn_frac || pf >= static_cast<int>(pos_feats.size())))
                        chosen = neg_feats[nf++];
                    else if (pf < static_cast<int>(pos_feats.size()))
                        chosen = pos_feats[pf++];
                    else break;
                    for (int s = 0; s < N_hyp; ++s) running_score[s] += feature_rank[chosen].first * x_mat[chosen][s];
                    resorted.push_back(chosen);
                }

                // Cumulative TPR/TNR sweep
                std::vector<double> score(N_hyp, intercept);
                std::vector<viz::AimAccuracyPoint> curve;
                auto make_point = [&]() -> viz::AimAccuracyPoint {
                    int tp=0,tn=0,fp=0,fn=0;
                    for (int s = 0; s < N_hyp; ++s) {
                        bool pp = score[s]>0, pt = hyp_responses[s]>0;
                        if(pt&&pp)++tp; else if(!pt&&!pp)++tn; else if(pt&&!pp)++fn; else ++fp;
                    }
                    double tpr=(tp+fn)>0?static_cast<double>(tp)/(tp+fn):0.0;
                    double tnr=(tn+fp)>0?static_cast<double>(tn)/(tn+fp):0.0;
                    double acc=N_hyp>0?static_cast<double>(tp+tn)/N_hyp:0.0;
                    return {tpr,tnr,acc};
                };
                curve.push_back(make_point());
                for (int ki = 0; ki < static_cast<int>(resorted.size()); ++ki) {
                    int fi = resorted[ki];
                    for (int s = 0; s < N_hyp; ++s) score[s] += feature_rank[fi].first * x_mat[fi][s];
                    curve.push_back(make_point());
                }

                // Find cutoff
                int cutoff_idx = -1;
                for (int k = 0; k < static_cast<int>(curve.size()); ++k)
                    if (curve[k].tpr >= aim_acc_cutoff && curve[k].tnr >= aim_acc_cutoff) { cutoff_idx = k; break; }
                if (cutoff_idx == -1) {
                    std::cout << "[AIM] Terminating iter " << iter << ": cutoff not achieved within " << window << " features\n";
                    break;
                }

                // Generate SVG
                {
                    viz::AimVizData vd;
                    for (int k : resorted) vd.feature_labels.push_back(feature_rank[k].second);
                    vd.seq_ids = hyp_species; vd.responses = hyp_responses;
                    vd.curve = curve; vd.cutoff_idx = cutoff_idx;
                    vd.contributions.resize(resorted.size(), std::vector<double>(N_hyp, 0.0));
                    for (int k = 0; k < static_cast<int>(resorted.size()); ++k) {
                        int fi = resorted[k];
                        for (int s = 0; s < N_hyp; ++s) vd.contributions[k][s] = feature_rank[fi].first * x_mat[fi][s];
                    }
                    fs::path svg_out = aim_out_dir / ("aim_iter_" + std::to_string(iter) + ".svg");
                    viz::write_aim_svg(vd, svg_out);
                    std::cout << "[AIM iter " << iter << "] SVG -> " << svg_out.string() << "\n";
                }

                // Accumulate selected features
                for (int k = 0; k < cutoff_idx; ++k) {
                    const std::string& lbl = feature_rank[resorted[k]].second;
                    if (!accumulated_set.count(lbl)) { accumulated_set.insert(lbl); accumulated_selected.push_back(lbl); ++total_selected; }
                }
                std::cout << "[AIM iter " << iter << "] selected " << cutoff_idx << " features (total=" << total_selected << ")\n";
                if (total_selected >= aim_max_ft) break;
            }

            {
                std::ofstream sf(aim_out_dir / "aim_selected.txt");
                for (auto& lbl : accumulated_selected) sf << lbl << '\n';
            }
            std::cout << "[AIM] Done. aim_selected.txt written (" << total_selected << " features)\n";

        // =====================================================================
        } else if (command == "encode-sizes") {
        // =====================================================================
            if (argc < 4) {
                std::cerr << "Error: encode-sizes requires <output_dir> <hypothesis.txt>\n";
                return 1;
            }
            pipeline::EncodeOptions enc_opts;
            enc_opts.output_dir = argv[2];
            enc_opts.hyp_path   = argv[3];
            for (int i = 4; i < argc; ++i) {
                std::string arg = argv[i];
                if      (arg == "--min-minor"       && i+1<argc) enc_opts.min_minor    = std::stoi(argv[++i]);
                else if (arg == "--auto-bit-ct"     && i+1<argc) enc_opts.auto_bit_ct  = std::stod(argv[++i]);
                else if (arg == "--drop-major-allele")            enc_opts.drop_major   = true;
                else if (arg == "--dropout"         && i+1<argc) {
                    enc_opts.dropout_labels = load_dropout_labels(argv[++i]);
                }
                else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
            }
            auto sizes = pipeline::encode_sizes(enc_opts);
            uint64_t total = 0;
            for (auto& [stem, ncols] : sizes) {
                std::cout << stem << '\t' << ncols << '\n';
                total += ncols;
            }
            std::cout << "\nFiles: " << sizes.size() << "  Total columns: " << total << "\n";

        // =====================================================================
        } else if (command == "visualize") {
        // =====================================================================
            if (argc < 4) { std::cerr << "Error: visualize requires <gene_predictions.txt> <output.svg>\n"; return 1; }
            fs::path gp_path  = argv[2];
            fs::path svg_path = argv[3];
            viz::VizOptions vopts;
            for (int i = 4; i < argc; ++i) {
                std::string arg = argv[i];
                if      (arg == "--gene-limit"    && i+1<argc) vopts.gene_limit    = std::stoi(argv[++i]);
                else if (arg == "--species-limit" && i+1<argc) vopts.species_limit = std::stoi(argv[++i]);
                else if (arg == "--ssq-threshold" && i+1<argc) vopts.ssq_threshold = std::stod(argv[++i]);
                else if (arg == "--m-grid")         vopts.m_grid = true;
                else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
            }
            auto gpt = viz::read_gene_predictions(gp_path);
            viz::write_svg(gpt, svg_path, vopts);
            std::cout << "Visualization written -> " << svg_path.string() << "\n";

        // =====================================================================
        } else {
        // =====================================================================
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
