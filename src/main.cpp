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
#include "model_log.hpp"
#include "pipeline_evaluate.hpp"
#include "pipeline_adaptive.hpp"
#include "pipeline_utils.hpp"
#include "fasta_parser.hpp"
#include "numeric_parser.hpp"
#include "pff_format.hpp"
#include "visualizer.hpp"
#include "newick.hpp"
#include "regression.hpp"
#include "pipeline_psc.hpp"
#include "vcf_parser.hpp"

namespace fs = std::filesystem;

// Shared by every command that accepts --datatype, so the accepted set cannot
// drift between them (drphylo and aim previously validated nothing and failed
// later inside the data_defs.ini lookup with a confusing message).
static void validate_datatype(const std::string& dt) {
    if (dt != "universal" && dt != "protein" && dt != "nucleotide" &&
        dt != "numeric" && dt != "vcf")
        throw std::runtime_error(
            "Unknown datatype: " + dt +
            " (expected: universal, protein, nucleotide, numeric, vcf)");
}

// --het-mode only has meaning for VCF input; validated here so a typo or a
// misplaced flag surfaces before any conversion work happens.
static void validate_het_mode(const std::string& mode, const std::string& datatype) {
    vcf::HetMode parsed;
    if (!vcf::parse_het_mode(mode, parsed))
        throw std::runtime_error(
            "--het-mode must be one of: dosage, presence, alt-dominant (got '" + mode + "')");
    if (datatype != "vcf")
        throw std::runtime_error("--het-mode requires --datatype vcf");
}

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

void print_overview(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "MyESL2 - My Evolutionary Sparse Learning 2\n"
        "===========================================\n\n"
        "USAGE\n"
        "  " + p + " <command> [args...]\n"
        "  " + p + " --help <command>     show detailed help for <command>\n\n"
        "COMMANDS\n"
        "  train       Full pipeline: preprocess -> encode -> regression\n"
        "  evaluate    Apply a trained model to new data\n"
        "  drphylo     Per-clade DrPhylo analysis\n"
        "  aim         Iterative AIM feature-selection loop\n"
        "  psc         Paired Species Contrast analysis\n"
        "  visualize   SVG heatmap from gene_predictions.txt\n"
        "  info        Display PFF file metadata\n"
        "  taskfile    Run any task from a YAML control file\n";
}

void print_help_train(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "TRAIN\n"
        "  " + p + " train <list.txt> <hypothesis.txt> <output_dir> [column|row] [options]\n\n"
        "  Positional:\n"
        "    list.txt        paths to FASTA/numeric input files (one per line)\n"
        "    hypothesis.txt  species labels (tab-delimited: name <TAB> value)\n"
        "    output_dir      directory for all output files\n"
        "    column|row      PFF orientation (default: column)\n\n"
        "  Regression:\n"
        "    --method <name>              regression method (default: sg_lasso_logisticr;\n"
        "                                 use --method none to skip regression)\n"
        "    --precision fp32|fp64        arithmetic precision (default: fp32)\n"
        "    --lambda <l1> <l2>           single lambda pair (default: 0.1 0.1)\n"
        "    --lambda-file <path>         file of lambda pairs, one 'l1 l2' per line\n"
        "    --lambda-grid <s1> <s2>      Cartesian product grid; each spec is min,max,step\n"
        "    --use-logspace               replace linear sweep with log-spaced grid anchored at\n"
        "                                 [vmin, largest sweep value < vmax and < 1]\n"
        "      Note: every lambda value (single, file, or grid) must lie in the open\n"
        "      interval (0,1); out-of-range values raise an error.\n"
        "    --no-evaluate                skip the automatic post-training evaluation.\n"
        "                                 By default each fitted model is scored against the\n"
        "                                 training data, writing eval.txt, eval_SPS_SPP.txt\n"
        "                                 (SPS/SPP/SCP per species) and eval_gene_predictions.txt\n"
        "                                 into every lambda_<N>/ dir. Implicitly skipped when\n"
        "                                 --method none leaves no model to score.\n"
        "    --nfolds N                   K-fold cross-validation (N >= 2, requires --method)\n"
        "    --cv-seed N                  shuffle samples with mt19937(N) before fold round-robin\n"
        "                                 (default: -1 = legacy unshuffled i % nfolds assignment)\n"
        "    --cv-assignments <file>      load fold assignments from TSV (cols: SequenceID, Fold;\n"
        "                                 header auto-detected; cv_predictions.txt from a prior\n"
        "                                 run can be passed directly). Overrides --cv-seed.\n"
        "    --min-groups N               skip lambdas selecting fewer than N non-zero groups\n"
        "    --prune-skipped-lambda       with --threads > 1 and --min-groups > 0, after the\n"
        "                                 parallel grid finishes, delete contents of lambda_N/\n"
        "                                 dirs that the single-threaded run would have skipped\n"
        "                                 (preserves dirs as drphylo sentinels)\n"
        "    --param <key>=<value>        pass option to solver\n"
        "      intercept=false            disable intercept term\n"
        "      field=<path>               group-index CSV (overlapping group methods)\n"
        "  Group penalty:\n"
        "    --group-penalty-type <type>  std|sqrt|linear|median (default: std)\n"
        "    --initial-gp-value X         initial penalty term for linear mode (default: 1)\n"
        "    --final-gp-value X           final penalty term for linear mode (default: 1)\n"
        "    --gp-step X                  penalty term step for linear mode (default: 1)\n"
        "  Encoding:\n"
        "    --auto-bit-ct X              set min_minor = ceil(X% x min_class_size)\n"
        "    --drop-major-allele          exclude major-allele column from FASTA encoder\n"
        "    --minor-column               add per-gene binary minor allele summary column\n"
        "    --tiered-minor-col           add per-gene tiered minor allele columns (0%, 0.1%, 1%, 5%)\n"
        "    --class-bal up|down|weighted balance classes before regression\n"
        "    --dropout <file>             exclude features listed in file from encoding\n"
        "    --het-mode <mode>            how heterozygous genotypes are weighted during\n"
        "                                 one-hot encoding (--datatype vcf only). Modes:\n"
        "                                   dosage       copies/called copies, so a diploid\n"
        "                                                het gives 0.5 to each observed\n"
        "                                                allele (default)\n"
        "                                   presence     1.0 to each observed allele\n"
        "                                   alt-dominant 0.0 to the reference allele and 1.0\n"
        "                                                to each observed alternate\n"
        "                                 Homozygous calls encode identically in all modes;\n"
        "                                 missing calls (./.) are all-zero. Recorded in\n"
        "                                 vcf_encoding.txt so evaluate re-encodes to match.\n"
        "    --feature-normalize <mode>   column-wise transform of the numeric feature matrix\n"
        "                                 (--datatype numeric or vcf only). Modes:\n"
        "                                   none   no transform\n"
        "                                   center subtract column mean (mean-shifting)\n"
        "                                   zscore (x - mean) / stddev (auto-scaling)\n"
        "                                   slep   (x - mean) / sqrt(sum(x^2)/N), matches\n"
        "                                          SLEP's opts.nFlag=1 column normalization\n"
        "                                 Default: slep for --datatype numeric, otherwise none.\n"
        "                                 Writes feature_normalization.txt to output_dir;\n"
        "                                 evaluate reads it to apply the same transform.\n"
        "    --write-features <path>      write encoded feature matrix to file\n"
        "    --write-features-transposed <path>  write transposed feature matrix to file\n"
        "    --max-mem <bytes>            abort if estimated feature matrix exceeds this size (default: 8589934592)\n"
        "      --param disable_mc=1       warn instead of aborting when max_mem is exceeded\n"
        "    --adaptive-sparsification    on max_mem_exceeded, run adaptive sparsification\n"
        "    --adaptive-lambda-grid <l1_spec> <l2_spec>\n"
        "                                 lambda grid for exploration combos (min,max,step each)\n"
        "                                 (default: \"0.1,0.3,0.1\" \"0.1,0.3,0.1\")\n"
        "  Common:\n"
        "    --cache-dir DIR              directory for .pff/.pnf/.vnf cache (default: ./pff_cache)\n"
        "    --min-minor N                min non-major non-indel count to keep a position (default: 1)\n"
        "    --threads N                  worker threads for preprocessing and the\n"
        "                                 lambda grid loop (default: all cores). When >1,\n"
        "                                 consider OPENBLAS_NUM_THREADS=1 to avoid\n"
        "                                 nested-parallelism oversubscription in the solver.\n"
        "                                 Grid-loop parallelism is disabled when --nfolds > 0.\n"
        "    --dlt                        use direct lookup table encoder\n"
        "    --datatype <type>            universal (default), protein, nucleotide, numeric, vcf\n"
        "                                 numeric: list file points to whitespace-delimited tabular files\n"
        "                                          (first col = sample name, remaining cols = features)\n"
        "                                 vcf:     list file points to .vcf/.vcf.gz files, one group each.\n"
        "                                          Sample IDs come from the #CHROM header; every\n"
        "                                          (site, allele) pair becomes a feature column named\n"
        "                                          {stem}_{chrom}:{pos}_{allele}. Multiallelic sites and\n"
        "                                          indels are supported. See --het-mode.\n";
}

void print_help_evaluate(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "EVALUATE\n"
        "  " + p + " evaluate <weights.txt> <list.txt> <output_file> [options]\n"
        "  " + p + " evaluate --from-run <run_dir> [options]\n\n"
        "  Apply a trained model to new data and write per-species predictions.\n\n"
        "  Batch mode:\n"
        "    --from-run <run_dir>    Score every model that <run_dir>/models.tsv records\n"
        "                            as complete, writing eval.txt / eval_SPS_SPP.txt /\n"
        "                            eval_gene_predictions.txt into each lambda dir.\n"
        "                            Useful after a training run was cancelled partway:\n"
        "                            the finished grid points can be scored without\n"
        "                            re-running the solver. The list file, datatype,\n"
        "                            cache dir and het-mode are read from the run's\n"
        "                            preprocess_config, and the hypothesis file from its\n"
        "                            process_log.txt, so no positionals are needed.\n"
        "                            Models already carrying eval output are left alone\n"
        "                            unless --re-evaluate is given.\n"
        "    --re-evaluate           (--from-run only) re-score models that already have\n"
        "                            evaluation output instead of skipping them.\n"
        "    --visualize             (--from-run only) emit an SVG per model; off by\n"
        "                            default since a 9x9 sweep would produce 81 of them.\n\n"
        "  Positional (single-model mode):\n"
        "    weights.txt     INPUT: trained model weights (e.g. <train_out>/lambda_N/weights.txt)\n"
        "    list.txt        INPUT: paths to FASTA/numeric files for the species to predict\n"
        "                    (one per line; same format as train's list.txt)\n"
        "    output_file     OUTPUT: per-species predictions written here (TSV).\n"
        "                    Auxiliary outputs (gene_predictions.txt, SPS_SPP.txt, .svg)\n"
        "                    are written alongside it in the same directory.\n\n"
        "  Optional:\n"
        "    --hypothesis <file>     INPUT: known labels for the species in list.txt.\n"
        "                            When supplied, evaluate also prints classification\n"
        "                            metrics (TP/TN/FP/FN, TPR/TNR/FPR/FNR, accuracy, AUC)\n"
        "                            and writes them to process_log.txt. Omit to run\n"
        "                            prediction-only with no accuracy assessment.\n"
        "    --no-visualize          skip automatic SVG generation\n"
        "    --minor-alleles <file>  minor_alleles.txt from training (auto-detected if omitted)\n"
        "    --tiered-minor-alleles <file>  tiered_minor_alleles.txt from training (auto-detected)\n"
        "    --gene-limit N          max genes displayed in auto-generated SVG (default: 20)\n"
        "    --species-limit N       max species displayed in auto-generated SVG (default: 20)\n"
        "    --het-mode <mode>       (--datatype vcf only) genotype encoding to re-apply to the\n"
        "                            input VCFs. Omit to read it from vcf_encoding.txt beside\n"
        "                            weights.txt; it MUST match the mode used for training.\n"
        "    --cache-dir DIR\n"
        "    --threads N\n"
        "    --datatype <type>       universal (default), protein, nucleotide, numeric, vcf.\n"
        "                            Must match the datatype the model was trained on.\n";
}

void print_help_drphylo(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "DRPHYLO\n"
        "  " + p + " drphylo <list.txt> <hypothesis.txt> <output_dir> [options]  (direct mode)\n"
        "  " + p + " drphylo <list.txt> <output_dir> --tree <tree.nwk> [options] (tree mode)\n\n"
        "  Tree mode (requires --tree):\n"
        "    --tree <tree.nwk>            Newick tree file (required for tree mode)\n"
        "    --clade-list <file>          file listing clades to test\n"
        "    --gen-clade-list <spec>      auto-generate clade list from tree\n"
        "  DrPhylo-specific:\n"
        "    --grid-rmse-cutoff X         exclude lambda results above RMSE threshold (default: 100)\n"
        "    --grid-acc-cutoff X          exclude lambda results below accuracy threshold (default: 0)\n"
        "    --gene-limit N               max genes displayed in aggregated eval.svg (default: 20)\n"
        "    --species-limit N            max species displayed in aggregated eval.svg (default: 20)\n"
        "  Default lambda grid (when none of --lambda/--lambda-grid/--lambda-file given):\n"
        "    --lambda-grid 0.1,0.9,0.1 0.1,0.9,0.1   (81-cell sweep)\n"
        "  Shared with train (same semantics):\n"
        "    --method, --precision, --lambda, --lambda-file, --lambda-grid, --use-logspace\n"
        "    --param, --nfolds, --cv-seed, --cv-assignments, --min-groups, --prune-skipped-lambda\n"
        "    --group-penalty-type, --initial-gp-value, --final-gp-value, --gp-step\n"
        "    --auto-bit-ct, --drop-major-allele, --minor-column\n"
        "    --class-bal, --cache-dir, --min-minor, --threads, --dlt, --datatype\n"
        "    --feature-normalize  (numeric/vcf input only; see train help)\n"
        "    --het-mode           (--datatype vcf only; see train help)\n";
}

void print_help_aim(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "AIM\n"
        "  " + p + " aim <list.txt> <hypothesis.txt> <output_dir> [options]\n\n"
        "  AIM-specific:\n"
        "    --aim-acc-cutoff X    TPR and TNR threshold to accept a feature set (default: 0.9)\n"
        "    --aim-max-iter N      max AIM iterations (default: 10)\n"
        "    --aim-max-ft N        stop after accumulating this many features total (default: 1000)\n"
        "    --aim-window N        top-N features considered per iteration (default: 100)\n"
        "  Shared with train (same semantics):\n"
        "    --method, --precision, --lambda, --lambda-file, --lambda-grid, --use-logspace\n"
        "    --param, --nfolds, --cv-seed, --cv-assignments, --min-groups, --prune-skipped-lambda\n"
        "    --group-penalty-type, --initial-gp-value, --final-gp-value, --gp-step\n"
        "    --auto-bit-ct, --drop-major-allele, --minor-column\n"
        "    --class-bal, --cache-dir, --min-minor, --threads, --dlt, --datatype\n"
        "    --feature-normalize  (numeric/vcf input only; see train help)\n"
        "    --het-mode           (--datatype vcf only; see train help)\n";
}

void print_help_psc(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "PSC\n"
        "  " + p + " psc <alignments_dir> <output_dir> [options]\n\n"
        "  Input:\n"
        "    --alignments-list <file>      list file specifying overlapping groups of alignments\n"
        "                                  (one group per line, comma-separated paths relative to\n"
        "                                  alignments_dir; requires --method olsg_lasso_logisticr,\n"
        "                                  olsg_lasso_leastr, ol_sg_lasso_logisticr, or ol_sg_lasso_leastr when any group has >1 entry)\n\n"
        "  Species contrast source (exactly one required):\n"
        "    --species-groups <file>       species contrast pairs file\n"
        "    --response-file <file>        single response matrix\n"
        "    --response-dir <dir>          directory of response matrices\n\n"
        // AUTO-PAIRS DISABLED — see CLAUDE.md "Disabled features"
        // "    --auto-pairs-tree <file>      auto-generate from tree (requires --species-pheno-path)\n\n"
        "  Lambda grid:\n"
        "    --initial-lambda1 X           (default: 0.01)\n"
        "    --final-lambda1 X             (default: 0.99)\n"
        "    --initial-lambda2 X           (default: 0.01)\n"
        "    --final-lambda2 X             (default: 0.99)\n"
        "    --lambda-step X               (default: 0.05)\n"
        "    --use-logspace                use log10 scale instead of linear\n"
        "    --num-log-points N            (default: 20)\n\n"
        "  Group penalty:\n"
        "    --group-penalty-type <type>   median|linear|sqrt|std (default: median)\n"
        "    --initial-gp-value X          (default: 1)\n"
        "    --final-gp-value X            (default: 1)\n"
        "    --gp-step X                   (default: 1)\n"
        "    --use-default-gp              use sqrt(n_features) regardless\n\n"
        "  Gap cancellation:\n"
        "    --use-uncanceled-alignments   skip gap cancellation\n"
        "    --cancel-only-partner         only cancel affected pairs\n"
        "    --cancel-tri-allelic          cancel 3+ unique residue positions\n"
        "    --nix-full-deletions          remove fully deleted positions\n"
        "    --outgroup-species <name>     outgroup for ancestral filtering\n"
        "    --min-pairs N                 (default: 2)\n\n"
        "  Solver:\n"
        "    --method <name>               (default: sg_lasso_logisticr)\n"
        "    --precision fp32|fp64         (default: fp32)\n"
        "    --maxiter N                   (default: 100)\n"
        "    --threads N\n"
        "    --param <key>=<value>         pass option to solver\n\n"
        "  Output/prediction:\n"
        "    --output-base-name <name>     (required)\n"
        "    --prediction-alignments-dir <dir>\n"
        "    --species-pheno-path <file>\n"
        "    --no-pred-output              skip species predictions\n"
        "    --no-genes-output             skip gene ranks\n"
        "    --show-selected-sites         output selected sites CSV\n"
        "    --dump-weights                dump per-run feature weights as TSV under\n"
        "                                  <output_dir>/[combo_N/][penalty_P/]lambda_L/weights.tsv\n"
        "    --top-rank-frac X             (default: 0.01)\n"
        "    --limited-genes-list <file>\n\n"
        "  Null models:\n"
        "    --make-null-models            response-flipped null\n"
        "    --make-pair-randomized-null-models\n"
        "    --num-randomized-alignments N (default: 10)\n\n"
        // AUTO-PAIRS DISABLED — see CLAUDE.md "Disabled features"
        // "  Auto-pairs:\n"
        // "    --auto-pairs-method <method>  (default: simple_deterministic)\n"
        // "    --auto-pairs-num-alternates N (default: 0)\n"
        // "    --auto-pairs-max-combinations N (default: 1)\n"
        ;
}

void print_help_visualize(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "VISUALIZE\n"
        "  " + p + " visualize <gene_predictions.txt> <output.svg> [options]\n\n"
        "    --gene-limit N        max genes displayed\n"
        "    --species-limit N     max species displayed\n"
        "    --ssq-threshold X     hide genes with sum-squared score below X\n"
        "    --m-grid              DrPhylo mode: show only positive-class samples\n";
}

void print_help_info(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "INFO\n"
        "  " + p + " info <file.pff>\n"
        "    Display metadata for a PFF or PNF cache file.\n";
}

void print_help_taskfile(const char* prog_name) {
    const std::string p = prog_name;
    std::cout <<
        "TASKFILE\n"
        "  " + p + " taskfile <control.yaml> [overrides...]\n\n"
        "    Run any MyESL2 task from a YAML control file. The YAML must contain\n"
        "    a 'task_type:' key (train, evaluate, drphylo, aim, psc, visualize,\n"
        "    encode-sizes, or info). All other keys map to CLI flags (without\n"
        "    the leading '--'), with hyphens preserved. Positional arguments\n"
        "    have named keys (e.g. list_path, hypothesis_path, output_dir).\n"
        "    Boolean flags are YAML booleans; multi-value flags are YAML\n"
        "    sequences; repeatable --param k=v is a YAML mapping under 'param:'.\n"
        "    CLI flags after the YAML path override matching YAML keys (warning\n"
        "    is printed on conflict). YAML keys that do not apply to the\n"
        "    selected task_type are rejected.\n";
}

void print_help(const char* prog_name, const std::string& command) {
    if      (command == "train")        print_help_train(prog_name);
    else if (command == "evaluate")     print_help_evaluate(prog_name);
    else if (command == "drphylo")      print_help_drphylo(prog_name);
    else if (command == "aim")          print_help_aim(prog_name);
    else if (command == "psc")          print_help_psc(prog_name);
    else if (command == "visualize")    print_help_visualize(prog_name);
    else if (command == "info")         print_help_info(prog_name);
    else if (command == "taskfile" ||
             command == "encode-sizes") print_help_taskfile(prog_name);
    else                                print_overview(prog_name);
}

int run_taskfile(int argc, char* argv[]);

int run_train(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Error: train requires <list.txt> <hypothesis.txt> <output_dir>\n";
        print_help(argv[0], "train");
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
    train_opts.method     = "sg_lasso_logisticr";  // default; --method none skips regression

    bool het_mode_set = false;
    bool no_evaluate  = false;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "row")         pre_opts.orientation = pff::Orientation::ROW_MAJOR;
        else if (arg == "column") pre_opts.orientation = pff::Orientation::COLUMN_MAJOR;
        else if (arg == "--cache-dir"  && i+1<argc) pre_opts.cache_dir = argv[++i];
        else if (arg == "--min-minor"  && i+1<argc) { pre_opts.min_minor = std::stoi(argv[++i]); enc_opts.min_minor = pre_opts.min_minor; }
        else if (arg == "--dlt")        pre_opts.use_dlt = true;
        else if (arg == "--datatype"   && i+1<argc) {
            pre_opts.datatype = argv[++i];
            validate_datatype(pre_opts.datatype);
        }
        else if (arg == "--het-mode"   && i+1<argc) { pre_opts.het_mode = argv[++i]; het_mode_set = true; }
        else if (arg == "--threads"    && i+1<argc) { pre_opts.num_threads = static_cast<unsigned>(std::stoi(argv[++i])); if (!pre_opts.num_threads) pre_opts.num_threads = 1; train_opts.threads = pre_opts.num_threads; }
        else if (arg == "--prune-skipped-lambda") train_opts.prune_skipped_lambda = true;
        else if (arg == "--no-evaluate") no_evaluate = true;
        else if (arg == "--method"     && i+1<argc) { train_opts.method = argv[++i]; if (train_opts.method == "none") train_opts.method.clear(); }
        else if (arg == "--precision"  && i+1<argc) {
            std::string p = argv[++i];
            if (p == "fp64") enc_opts.precision = train_opts.precision = regression::Precision::FP64;
            else if (p == "fp32") enc_opts.precision = train_opts.precision = regression::Precision::FP32;
            else throw std::runtime_error("--precision must be fp32 or fp64");
        }
        else if (arg == "--lambda"     && i+2<argc) { train_opts.lambda[0]=std::stod(argv[++i]); train_opts.lambda[1]=std::stod(argv[++i]); train_opts.lambda_explicitly_set=true; }
        else if (arg == "--lambda-file"&& i+1<argc) train_opts.lambda_file_path = argv[++i];
        else if (arg == "--lambda-grid"&& i+2<argc) { train_opts.lambda_grid_specs[0]=argv[++i]; train_opts.lambda_grid_specs[1]=argv[++i]; train_opts.lambda_grid_set=true; }
        else if (arg == "--use-logspace") train_opts.use_logspace = true;
        else if (arg == "--param"      && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts.params[kv.substr(0,eq)]=kv.substr(eq+1); else std::cerr<<"Warning: --param '"<<kv<<"' has no '=', ignoring\n"; }
        else if (arg == "--nfolds"     && i+1<argc) { train_opts.nfolds=std::stoi(argv[++i]); if(train_opts.nfolds<2) throw std::runtime_error("--nfolds must be >= 2"); }
        else if (arg == "--cv-seed"    && i+1<argc) train_opts.cv_seed = std::stoi(argv[++i]);
        else if (arg == "--cv-assignments" && i+1<argc) train_opts.cv_assignments_path = argv[++i];
        else if (arg == "--min-groups" && i+1<argc) train_opts.min_groups = std::stoi(argv[++i]);
        else if (arg == "--auto-bit-ct"&& i+1<argc) enc_opts.auto_bit_ct = std::stod(argv[++i]);
        else if (arg == "--drop-major-allele") enc_opts.drop_major = true;
        else if (arg == "--minor-column") enc_opts.minor_column = true;
        else if (arg == "--tiered-minor-col") enc_opts.tiered_minor_col = true;
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
        else if (arg == "--group-penalty-type" && i+1<argc) train_opts.group_penalty_type = argv[++i];
        else if (arg == "--initial-gp-value"   && i+1<argc) train_opts.initial_gp_value = std::stod(argv[++i]);
        else if (arg == "--final-gp-value"     && i+1<argc) train_opts.final_gp_value   = std::stod(argv[++i]);
        else if (arg == "--gp-step"            && i+1<argc) train_opts.gp_step          = std::stod(argv[++i]);
        else if (arg == "--feature-normalize"  && i+1<argc) enc_opts.feature_normalize = argv[++i];
        else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
    }
    if (train_opts.params.count("disable_mc") && train_opts.params.at("disable_mc") == "1")
        enc_opts.disable_mc = true;

    if (enc_opts.feature_normalize == "auto")
        enc_opts.feature_normalize = (pre_opts.datatype == "numeric") ? "slep" : "none";
    if (enc_opts.feature_normalize != "none") {
        if (enc_opts.feature_normalize != "center" && enc_opts.feature_normalize != "zscore" &&
            enc_opts.feature_normalize != "slep")
            throw std::runtime_error("--feature-normalize must be one of: none, center, zscore, slep");
        if (pre_opts.datatype != "numeric" && pre_opts.datatype != "vcf")
            throw std::runtime_error(
                "--feature-normalize is only supported with --datatype numeric or vcf");
    }

    {
        std::string gpt = train_opts.group_penalty_type;
        for (auto& c : gpt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (gpt != "std" && gpt != "sqrt" && gpt != "linear" && gpt != "median")
            throw std::runtime_error("--group-penalty-type must be one of: std, sqrt, linear, median");
    }

    // Resolve deprecated method aliases
    if (!train_opts.method.empty()) {
        bool was_alias = false;
        auto resolved = regression::resolve_method_alias(train_opts.method, &was_alias);
        if (was_alias) {
            std::cerr << "Note: --method " << train_opts.method
                      << " is deprecated, use " << resolved
                      << " (may stop working in a future release)\n";
            train_opts.method = resolved;
        }
    }

    if (!train_opts.lambda_file_path.empty() && train_opts.lambda_explicitly_set)
        throw std::runtime_error("--lambda and --lambda-file are mutually exclusive");
    if (train_opts.lambda_grid_set && train_opts.lambda_explicitly_set)
        throw std::runtime_error("--lambda-grid and --lambda are mutually exclusive");
    if (train_opts.lambda_grid_set && !train_opts.lambda_file_path.empty())
        throw std::runtime_error("--lambda-grid and --lambda-file are mutually exclusive");
    if (train_opts.nfolds > 0 && train_opts.method.empty())
        throw std::runtime_error("--nfolds requires --method");

    // Compute lambda count for peak memory estimation
    if (train_opts.lambda_grid_set) {
        auto count_steps = [](const std::string& spec) -> uint32_t {
            double vmin, vmax, vstep; char c1, c2;
            std::istringstream ss(spec);
            if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || vstep <= 0) return 1;
            uint32_t n = 0;
            for (double v = vmin; v <= vmax + vstep * 1e-9; v += vstep) ++n;
            return n ? n : 1;
        };
        enc_opts.lambda_count = count_steps(train_opts.lambda_grid_specs[0])
                              * count_steps(train_opts.lambda_grid_specs[1]);
    } else if (!train_opts.lambda_file_path.empty()) {
        std::ifstream lf(train_opts.lambda_file_path);
        uint32_t n = 0; std::string line;
        while (std::getline(lf, line))
            if (!line.empty() && line[0] != '#') ++n;
        enc_opts.lambda_count = n ? n : 1;
    }

    if (het_mode_set) validate_het_mode(pre_opts.het_mode, pre_opts.datatype);
    // The per-gene minor-allele summary columns are produced by the FASTA
    // encoder only; VCF input goes through the float-matrix path where they
    // have no implementation. Reject rather than silently ignore.
    if (pre_opts.datatype == "vcf" && (enc_opts.minor_column || enc_opts.tiered_minor_col))
        throw std::runtime_error(
            "--minor-column / --tiered-minor-col are not supported with --datatype vcf");
    if (enc_opts.minor_column && enc_opts.tiered_minor_col)
        throw std::runtime_error("--minor-column and --tiered-minor-col are mutually exclusive");

    pipeline::preprocess(pre_opts);
    try {
        auto enc = pipeline::encode(enc_opts);
        auto train_result = pipeline::train(enc, train_opts);

        // Score every fitted model against the training data, so each lambda dir
        // gets eval.txt / eval_SPS_SPP.txt / eval_gene_predictions.txt without a
        // separate evaluate invocation. Mirrors drphylo's per-lambda evaluate.
        // Skipped when regression was skipped (--method none): there are no
        // weights to score.
        if (!no_evaluate && !train_result.weights_paths.empty()) {
            std::cout << "\n--- Phase 4: Evaluation ---\n";
            std::cout << "  Models to score: " << train_result.weights_paths.size() << "\n";
            // Read back the resolved config rather than reusing pre_opts, whose
            // cache_dir/num_threads may still hold pre-resolution defaults.
            auto pre_cfg = pipeline::read_preprocess_config(enc_opts.output_dir);
            for (auto& wp : train_result.weights_paths) {
                pipeline::EvaluateOptions eopts;
                eopts.weights_path = wp;
                eopts.list_path    = pre_cfg.list_path;
                eopts.output_file  = wp.parent_path() / "eval.txt";
                eopts.hyp_path     = enc_opts.hyp_path;   // enables accuracy metrics
                eopts.no_visualize = true;                // one SVG per grid point is rarely wanted
                eopts.datatype     = pre_cfg.datatype;
                eopts.het_mode     = pre_cfg.het_mode;
                eopts.num_threads  = pre_cfg.num_threads;
                eopts.cache_dir    = pre_cfg.cache_dir;
                eopts.minor_alleles_path = enc_opts.output_dir / "minor_alleles.txt";
                eopts.tiered_minor_alleles_path =
                    enc_opts.output_dir / "tiered_minor_alleles.txt";
                pipeline::evaluate(eopts);
            }
        }

        std::cout << "\n--- Summary ---\n";
        std::cout << "  Features matrix: " << enc.features.n_rows << " x " << enc.features.n_cols << "\n";
        std::cout << "  Response vector: 1 x " << enc.responses.n_elem << "\n";
    } catch (const std::runtime_error& e) {
        if (std::string_view(e.what()).starts_with("max_mem_exceeded")
            && train_opts.adaptive_sparsification) {
            // The failed encode's metadata pages are still retained by
            // glibc; release them before the adaptive path starts.
            pipeline_utils::release_freed_heap();
            std::cout << "[adaptive] max_mem exceeded — starting adaptive sparsification\n";
            pipeline::adaptive_train(enc_opts, train_opts);
        } else {
            throw;
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// evaluate --from-run <dir>
//
// Scores every model that <dir>/models.tsv records as `complete`. Intended for
// a training run that was cancelled partway through: the finished grid points
// can be evaluated without re-running the solver, and without restating the
// list file, datatype, cache dir or het-mode by hand.
//
// Settings come from the run directory itself:
//   models.tsv          which models exist and which of them finished
//   preprocess_config   list_path, datatype, cache_dir, het_mode, num_threads
//   process_log.txt     hyp_path (the only one of these not in preprocess_config;
//                       needed for the accuracy metrics)
// ---------------------------------------------------------------------------

// Recover hyp_path from the most recent `encode` section of process_log.txt.
// Returns an empty path when absent, in which case evaluation still runs but
// reports predictions only.
static fs::path find_encode_hyp_path(const fs::path& run_dir) {
    std::ifstream f(run_dir / "process_log.txt");
    if (!f) return {};
    std::string line, found;
    bool in_encode = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("=== ", 0) == 0) {
            in_encode = line.rfind("=== encode ", 0) == 0;
            continue;
        }
        if (!in_encode) continue;
        const std::string key = "hyp_path = ";
        if (line.rfind(key, 0) == 0) found = line.substr(key.size());  // keep the last
    }
    if (found.empty()) return {};

    // The path was logged exactly as it was typed, so it may be relative to the
    // working directory the training run used. Try that first, then relative to
    // the run directory and its parent.
    for (const fs::path& cand : {fs::path(found), run_dir / found, run_dir.parent_path() / found})
        if (fs::exists(cand)) return cand;
    return fs::path(found);   // let evaluate report the failure with the original text
}

int run_evaluate_from_run(const fs::path& run_dir, int argc, char* argv[]) {
    if (!fs::exists(run_dir))
        throw std::runtime_error("--from-run directory does not exist: " + run_dir.string());

    fs::path log_path = run_dir / "models.tsv";
    if (!fs::exists(log_path))
        throw std::runtime_error(
            "No models.tsv in " + run_dir.string() +
            " — --from-run needs a run directory produced by train/drphylo/aim.");

    auto rows = model_log::read(log_path);
    auto pre  = pipeline::read_preprocess_config(run_dir);

    // CLI overrides; anything left unset falls back to the recorded settings.
    fs::path     hyp_override, cache_override;
    std::string  datatype_override, het_override;
    unsigned int threads_override = 0;
    bool         re_evaluate = false, no_visualize = true;
    int          gene_limit = -1, species_limit = -1;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--from-run"    && i+1<argc) ++i;   // already consumed
        else if (arg == "--re-evaluate")             re_evaluate = true;
        else if (arg == "--visualize")               no_visualize = false;
        else if (arg == "--no-visualize")            no_visualize = true;
        else if (arg == "--hypothesis"  && i+1<argc) hyp_override      = argv[++i];
        else if (arg == "--cache-dir"   && i+1<argc) cache_override    = argv[++i];
        else if (arg == "--datatype"    && i+1<argc) { datatype_override = argv[++i]; validate_datatype(datatype_override); }
        else if (arg == "--het-mode"    && i+1<argc) het_override      = argv[++i];
        else if (arg == "--threads"     && i+1<argc) { threads_override = static_cast<unsigned>(std::stoi(argv[++i])); if(!threads_override) threads_override = 1; }
        else if (arg == "--gene-limit"    && i+1<argc) gene_limit    = std::stoi(argv[++i]);
        else if (arg == "--species-limit" && i+1<argc) species_limit = std::stoi(argv[++i]);
        else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
    }

    fs::path hyp_path = !hyp_override.empty() ? hyp_override : find_encode_hyp_path(run_dir);

    size_t n_complete = 0, n_pending = 0, n_failed = 0, n_skipped = 0;
    for (const auto& r : rows) {
        switch (r.status) {
            case model_log::Status::Complete: ++n_complete; break;
            case model_log::Status::Pending:  ++n_pending;  break;
            case model_log::Status::Failed:   ++n_failed;   break;
            case model_log::Status::Skipped:  ++n_skipped;  break;
        }
    }

    std::cout << "--- Evaluate from run: " << run_dir.string() << " ---\n";
    std::cout << "  Models in log:  " << rows.size()
              << "  (complete=" << n_complete << ", pending=" << n_pending
              << ", failed=" << n_failed << ", skipped=" << n_skipped << ")\n";
    std::cout << "  Datatype:       " << (datatype_override.empty() ? pre.datatype : datatype_override) << "\n";
    std::cout << "  List:           " << pre.list_path.string() << "\n";
    if (hyp_path.empty())
        std::cout << "  Hypothesis:     (none found — predictions only, no accuracy metrics)\n";
    else
        std::cout << "  Hypothesis:     " << hyp_path.string() << "\n";

    size_t scored = 0, reused = 0, missing = 0, errored = 0, unscoreable = 0;
    for (const auto& r : rows) {
        if (r.status != model_log::Status::Complete) continue;

        if (r.weights_path.empty()) {
            // k-fold CV rows: no single model to score.
            ++unscoreable;
            continue;
        }
        fs::path wp = run_dir / r.weights_path;
        if (!fs::exists(wp)) {
            std::cerr << "Warning: [" << r.index << "] weights missing, skipping: "
                      << wp.string() << "\n";
            ++missing;
            continue;
        }
        fs::path lam_dir = wp.parent_path();
        if (!re_evaluate && fs::exists(lam_dir / "eval_SPS_SPP.txt")) {
            ++reused;
            continue;
        }

        pipeline::EvaluateOptions eopts;
        eopts.weights_path = wp;
        eopts.list_path    = pre.list_path;
        eopts.output_file  = lam_dir / "eval.txt";
        eopts.hyp_path     = hyp_path;
        eopts.no_visualize = no_visualize;
        eopts.datatype     = datatype_override.empty() ? pre.datatype : datatype_override;
        eopts.het_mode     = het_override.empty()      ? pre.het_mode : het_override;
        eopts.num_threads  = threads_override ? threads_override : pre.num_threads;
        eopts.cache_dir    = cache_override.empty()    ? pre.cache_dir : cache_override;
        eopts.minor_alleles_path        = run_dir / "minor_alleles.txt";
        eopts.tiered_minor_alleles_path = run_dir / "tiered_minor_alleles.txt";
        if (gene_limit    >= 0) eopts.gene_limit    = gene_limit;
        if (species_limit >= 0) eopts.species_limit = species_limit;

        std::cout << "\n[" << r.index << "] lambda=[" << r.lambda1 << "," << r.lambda2
                  << "] -> " << r.weights_path << "\n";
        try {
            pipeline::evaluate(eopts);
            ++scored;
        } catch (const std::exception& e) {
            // Keep going: the point of this mode is salvaging a partial run, so
            // one bad model must not hide the results of every later one.
            std::cerr << "Error: [" << r.index << "] evaluate failed: " << e.what() << "\n";
            ++errored;
        }
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "  Scored:            " << scored << "\n";
    if (reused)      std::cout << "  Already evaluated: " << reused << " (use --re-evaluate to redo)\n";
    if (unscoreable) std::cout << "  No single model:   " << unscoreable << " (cross-validation rows)\n";
    if (missing)     std::cout << "  Weights missing:   " << missing << "\n";
    if (errored)     std::cout << "  Failed:            " << errored << "\n";
    if (n_pending)   std::cout << "  Still pending:     " << n_pending << " (never solved)\n";

    return errored ? 1 : 0;
}

int run_evaluate(int argc, char* argv[]) {
    // Detect from-run mode before touching positionals, mirroring how
    // run_drphylo detects tree mode by pre-scanning for --tree.
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--from-run")
            return run_evaluate_from_run(argv[i + 1], argc, argv);

    if (argc < 5) {
        std::cerr << "Error: evaluate requires <weights.txt> <list.txt> <output_file>\n";
        print_help(argv[0], "evaluate");
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
        else if (arg == "--datatype"   && i+1<argc) { eval_opts.datatype = argv[++i]; validate_datatype(eval_opts.datatype); }
        else if (arg == "--het-mode"   && i+1<argc) eval_opts.het_mode = argv[++i];
        else if (arg == "--threads"    && i+1<argc) { eval_opts.num_threads=static_cast<unsigned>(std::stoi(argv[++i])); if(!eval_opts.num_threads) eval_opts.num_threads=1; }
        else if (arg == "--no-visualize") eval_opts.no_visualize = true;
        else if (arg == "--minor-alleles" && i+1<argc) eval_opts.minor_alleles_path = argv[++i];
        else if (arg == "--tiered-minor-alleles" && i+1<argc) eval_opts.tiered_minor_alleles_path = argv[++i];
        else if (arg == "--gene-limit"    && i+1<argc) eval_opts.gene_limit    = std::stoi(argv[++i]);
        else if (arg == "--species-limit" && i+1<argc) eval_opts.species_limit = std::stoi(argv[++i]);
        else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
    }

    pipeline::evaluate(eval_opts);

    return 0;
}

int run_info(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: info requires a PFF file path\n";
        print_help(argv[0], "info");
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

    return 0;
}

int run_drphylo(int argc, char* argv[]) {
    // Detect tree mode: --tree flag present anywhere in args
    std::string tree_file;
    bool has_clade_flag = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tree" && i+1 < argc) tree_file = argv[i+1];
        if (a == "--clade-list" || a == "--gen-clade-list") has_clade_flag = true;
    }
    bool tree_mode = !tree_file.empty();

    // --clade-list / --gen-clade-list require --tree
    if (!tree_mode && has_clade_flag) {
        std::cerr << "Error: --clade-list and --gen-clade-list require tree mode.\n"
                  << "       Add --tree <tree.nwk> to use tree mode.\n";
        return 1;
    }

    if (argc < (tree_mode ? 4 : 5)) {
        std::cerr << "Error: drphylo usage:\n"
                  << "  drphylo <list.txt> <hypothesis.txt> <output_dir> [options]  (direct mode)\n"
                  << "  drphylo <list.txt> <output_dir> --tree <tree.nwk> [options] (tree mode)\n";
        print_help(argv[0], "drphylo");
        return 1;
    }

    fs::path list_path   = argv[2];
    fs::path tree_path;
    fs::path output_dir;
    fs::path direct_hyp_file;
    int extra_start;
    if (tree_mode) { output_dir = argv[3]; tree_path = tree_file; extra_start = 4; }
    else           { direct_hyp_file = argv[3]; output_dir = argv[4]; extra_start = 5; }

    // Build shared opts with defaults
    pipeline::PreprocessOptions pre_opts;
    pre_opts.list_path  = list_path;
    pre_opts.output_dir = output_dir;
    pre_opts.binary_dir = fs::path(argv[0]).parent_path();
    pre_opts.tree_path  = tree_path;

    pipeline::EncodeOptions enc_opts_base;
    enc_opts_base.class_bal = "weighted";

    pipeline::TrainOptions train_opts_base;
    train_opts_base.method     = "sg_lasso_logisticr";
    train_opts_base.min_groups = 3;  // DrPhylo default

    double grid_rmse_cutoff = 100.0;
    double grid_acc_cutoff  = 0.0;
    int    viz_gene_limit    = 20;
    int    viz_species_limit = 20;
    bool min_groups_set = false;
    bool het_mode_set = false;

    for (int i = extra_start; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--clade-list"       && i+1<argc) pre_opts.clade_list_file = argv[++i];
        else if (arg == "--gen-clade-list"   && i+1<argc) pre_opts.gen_clade_spec  = argv[++i];
        else if (arg == "--class-bal"        && i+1<argc) pre_opts.class_bal_phylo = argv[++i];
        else if (arg == "--tree"             && i+1<argc) { ++i; /* already captured */ }
        else if (arg == "--datatype"         && i+1<argc) { pre_opts.datatype = argv[++i]; validate_datatype(pre_opts.datatype); }
        else if (arg == "--het-mode"         && i+1<argc) { pre_opts.het_mode = argv[++i]; het_mode_set = true; }
        else if (arg == "--threads"          && i+1<argc) { pre_opts.num_threads = static_cast<unsigned>(std::stoi(argv[++i])); if(!pre_opts.num_threads) pre_opts.num_threads=1; train_opts_base.threads = pre_opts.num_threads; }
        else if (arg == "--prune-skipped-lambda") train_opts_base.prune_skipped_lambda = true;
        else if (arg == "--cache-dir"        && i+1<argc) pre_opts.cache_dir       = argv[++i];
        else if (arg == "--dlt")              pre_opts.use_dlt = true;
        else if (arg == "--min-minor"        && i+1<argc) pre_opts.min_minor        = std::stoi(argv[++i]);
        else if (arg == "--method"           && i+1<argc) train_opts_base.method   = argv[++i];
        else if (arg == "--precision"        && i+1<argc) { std::string p=argv[++i]; if(p=="fp64") enc_opts_base.precision=train_opts_base.precision=regression::Precision::FP64; else if(p!="fp32") throw std::runtime_error("--precision must be fp32 or fp64"); }
        else if (arg == "--lambda"           && i+2<argc) { train_opts_base.lambda[0]=std::stod(argv[++i]); train_opts_base.lambda[1]=std::stod(argv[++i]); train_opts_base.lambda_explicitly_set=true; }
        else if (arg == "--lambda-file"      && i+1<argc) train_opts_base.lambda_file_path = argv[++i];
        else if (arg == "--lambda-grid"      && i+2<argc) { train_opts_base.lambda_grid_specs[0]=argv[++i]; train_opts_base.lambda_grid_specs[1]=argv[++i]; train_opts_base.lambda_grid_set=true; }
        else if (arg == "--use-logspace") train_opts_base.use_logspace = true;
        else if (arg == "--param"            && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts_base.params[kv.substr(0,eq)]=kv.substr(eq+1); }
        else if (arg == "--nfolds"           && i+1<argc) { train_opts_base.nfolds=std::stoi(argv[++i]); if(train_opts_base.nfolds<2) throw std::runtime_error("--nfolds must be >= 2"); }
        else if (arg == "--cv-seed"          && i+1<argc) train_opts_base.cv_seed = std::stoi(argv[++i]);
        else if (arg == "--cv-assignments"   && i+1<argc) train_opts_base.cv_assignments_path = argv[++i];
        else if (arg == "--min-groups"       && i+1<argc) { train_opts_base.min_groups=std::stoi(argv[++i]); min_groups_set=true; }
        else if (arg == "--grid-rmse-cutoff" && i+1<argc) grid_rmse_cutoff = std::stod(argv[++i]);
        else if (arg == "--grid-acc-cutoff"  && i+1<argc) grid_acc_cutoff  = std::stod(argv[++i]);
        else if (arg == "--gene-limit"       && i+1<argc) viz_gene_limit    = std::stoi(argv[++i]);
        else if (arg == "--species-limit"    && i+1<argc) viz_species_limit = std::stoi(argv[++i]);
        else if (arg == "--auto-bit-ct"      && i+1<argc) enc_opts_base.auto_bit_ct   = std::stod(argv[++i]);
        else if (arg == "--drop-major-allele") enc_opts_base.drop_major = true;
        else if (arg == "--minor-column") enc_opts_base.minor_column = true;
        else if (arg == "--tiered-minor-col") enc_opts_base.tiered_minor_col = true;
        else if (arg == "--max-mem"           && i+1<argc) enc_opts_base.max_mem = std::stoull(argv[++i]);
        else if (arg == "--group-penalty-type" && i+1<argc) train_opts_base.group_penalty_type = argv[++i];
        else if (arg == "--initial-gp-value"   && i+1<argc) train_opts_base.initial_gp_value = std::stod(argv[++i]);
        else if (arg == "--final-gp-value"     && i+1<argc) train_opts_base.final_gp_value   = std::stod(argv[++i]);
        else if (arg == "--gp-step"            && i+1<argc) train_opts_base.gp_step          = std::stod(argv[++i]);
        else if (arg == "--feature-normalize"  && i+1<argc) enc_opts_base.feature_normalize = argv[++i];
        else std::cerr << "Warning: unknown drphylo argument '" << arg << "', ignoring\n";
    }
    if (enc_opts_base.feature_normalize == "auto")
        enc_opts_base.feature_normalize = (pre_opts.datatype == "numeric") ? "slep" : "none";
    if (enc_opts_base.feature_normalize != "none") {
        if (enc_opts_base.feature_normalize != "center" && enc_opts_base.feature_normalize != "zscore" &&
            enc_opts_base.feature_normalize != "slep")
            throw std::runtime_error("--feature-normalize must be one of: none, center, zscore, slep");
        if (pre_opts.datatype != "numeric" && pre_opts.datatype != "vcf")
            throw std::runtime_error(
                "--feature-normalize is only supported with --datatype numeric or vcf");
    }
    // Resolve deprecated method aliases
    {
        bool was_alias = false;
        auto resolved = regression::resolve_method_alias(train_opts_base.method, &was_alias);
        if (was_alias) {
            std::cerr << "Note: --method " << train_opts_base.method
                      << " is deprecated, use " << resolved
                      << " (may stop working in a future release)\n";
            train_opts_base.method = resolved;
        }
    }
    {
        std::string gpt = train_opts_base.group_penalty_type;
        for (auto& c : gpt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (gpt != "std" && gpt != "sqrt" && gpt != "linear" && gpt != "median")
            throw std::runtime_error("--group-penalty-type must be one of: std, sqrt, linear, median");
    }
    if (het_mode_set) validate_het_mode(pre_opts.het_mode, pre_opts.datatype);
    // The per-gene minor-allele summary columns are produced by the FASTA
    // encoder only; VCF input goes through the float-matrix path where they
    // have no implementation. Reject rather than silently ignore.
    if (pre_opts.datatype == "vcf" && (enc_opts_base.minor_column || enc_opts_base.tiered_minor_col))
        throw std::runtime_error(
            "--minor-column / --tiered-minor-col are not supported with --datatype vcf");
    if (enc_opts_base.minor_column && enc_opts_base.tiered_minor_col)
        throw std::runtime_error("--minor-column and --tiered-minor-col are mutually exclusive");
    if (train_opts_base.params.count("disable_mc") && train_opts_base.params.at("disable_mc") == "1")
        enc_opts_base.disable_mc = true;
    if (!min_groups_set) train_opts_base.min_groups = 3;

    // DrPhylo default lambda grid: 9x9 sweep matching original MyESL's
    // DrPhylo default ("0.1,1.0,0.1" with strict <1.0 filter -> 0.1..0.9).
    if (!train_opts_base.lambda_grid_set
        && !train_opts_base.lambda_explicitly_set
        && train_opts_base.lambda_file_path.empty()) {
        train_opts_base.lambda_grid_specs[0] = "0.1,0.9,0.1";
        train_opts_base.lambda_grid_specs[1] = "0.1,0.9,0.1";
        train_opts_base.lambda_grid_set = true;
    }

    // Lambda count for peak memory estimation
    if (train_opts_base.lambda_grid_set) {
        auto count_steps = [](const std::string& spec) -> uint32_t {
            double vmin, vmax, vstep; char c1, c2;
            std::istringstream ss(spec);
            if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || vstep <= 0) return 1;
            uint32_t n = 0;
            for (double v = vmin; v <= vmax + vstep * 1e-9; v += vstep) ++n;
            return n ? n : 1;
        };
        enc_opts_base.lambda_count = count_steps(train_opts_base.lambda_grid_specs[0])
                                   * count_steps(train_opts_base.lambda_grid_specs[1]);
    } else if (!train_opts_base.lambda_file_path.empty()) {
        std::ifstream lf(train_opts_base.lambda_file_path);
        uint32_t n = 0; std::string line;
        while (std::getline(lf, line))
            if (!line.empty() && line[0] != '#') ++n;
        enc_opts_base.lambda_count = n ? n : 1;
    }

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
            // Without this the per-clade evaluate would silently fall back to the
            // default genotype encoding and rescale every dosage.
            eopts.het_mode     = pre_cfg.het_mode;
            eopts.num_threads  = pre_cfg.num_threads ? pre_cfg.num_threads : std::thread::hardware_concurrency();
            eopts.cache_dir    = pre_cfg.cache_dir;
            eopts.minor_alleles_path = run_dir / "minor_alleles.txt";
            eopts.tiered_minor_alleles_path = run_dir / "tiered_minor_alleles.txt";
            pipeline::evaluate(eopts);
        }

        auto agg = pipeline::evaluate_drphylo_aggregate(run_dir, grid_rmse_cutoff, grid_acc_cutoff,
                                                        viz_gene_limit, viz_species_limit);
        hss_summary.push_back({label, agg.hss});
        std::cout << label << ": HSS=" << agg.hss << "\n";
    };

    if (!tree_mode) {
        // Direct mode: 3 positional args (list, hypothesis, output_dir)
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

    return 0;
}

int run_aim(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Error: aim requires <list.txt> <hypothesis.txt> <output_dir>\n";
        print_help(argv[0], "aim");
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
    train_opts_base.method = "sg_lasso_logisticr";
    bool has_lambda = false, has_method = false;
    bool het_mode_set = false;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--aim-acc-cutoff" && i+1<argc) aim_acc_cutoff = std::stod(argv[++i]);
        else if (arg == "--aim-max-iter"   && i+1<argc) aim_max_iter   = std::stoi(argv[++i]);
        else if (arg == "--aim-max-ft"     && i+1<argc) aim_max_ft     = std::stoi(argv[++i]);
        else if (arg == "--aim-window"     && i+1<argc) aim_window     = std::stoi(argv[++i]);
        else if (arg == "--cache-dir"      && i+1<argc) pre_opts.cache_dir  = argv[++i];
        else if (arg == "--datatype"       && i+1<argc) { pre_opts.datatype = argv[++i]; validate_datatype(pre_opts.datatype); }
        else if (arg == "--het-mode"       && i+1<argc) { pre_opts.het_mode = argv[++i]; het_mode_set = true; }
        else if (arg == "--threads"        && i+1<argc) { pre_opts.num_threads=static_cast<unsigned>(std::stoi(argv[++i])); if(!pre_opts.num_threads) pre_opts.num_threads=1; train_opts_base.threads = pre_opts.num_threads; }
        else if (arg == "--prune-skipped-lambda") train_opts_base.prune_skipped_lambda = true;
        else if (arg == "--dlt")            pre_opts.use_dlt = true;
        else if (arg == "--min-minor"      && i+1<argc) pre_opts.min_minor = std::stoi(argv[++i]);
        else if (arg == "--method"         && i+1<argc) { train_opts_base.method = argv[++i]; has_method = true; }
        else if (arg == "--precision"      && i+1<argc) { std::string p=argv[++i]; if(p=="fp64") enc_opts_base.precision=train_opts_base.precision=regression::Precision::FP64; else if(p!="fp32") throw std::runtime_error("--precision must be fp32 or fp64"); }
        else if (arg == "--lambda"         && i+2<argc) { train_opts_base.lambda[0]=std::stod(argv[++i]); train_opts_base.lambda[1]=std::stod(argv[++i]); train_opts_base.lambda_explicitly_set=true; has_lambda=true; }
        else if (arg == "--lambda-file"    && i+1<argc) { train_opts_base.lambda_file_path=argv[++i]; has_lambda=true; }
        else if (arg == "--lambda-grid"    && i+2<argc) { train_opts_base.lambda_grid_specs[0]=argv[++i]; train_opts_base.lambda_grid_specs[1]=argv[++i]; train_opts_base.lambda_grid_set=true; has_lambda=true; }
        else if (arg == "--use-logspace") train_opts_base.use_logspace = true;
        else if (arg == "--param"          && i+1<argc) { std::string kv=argv[++i]; auto eq=kv.find('='); if(eq!=std::string::npos) train_opts_base.params[kv.substr(0,eq)]=kv.substr(eq+1); }
        else if (arg == "--nfolds"         && i+1<argc) train_opts_base.nfolds = std::stoi(argv[++i]);
        else if (arg == "--cv-seed"        && i+1<argc) train_opts_base.cv_seed = std::stoi(argv[++i]);
        else if (arg == "--cv-assignments" && i+1<argc) train_opts_base.cv_assignments_path = argv[++i];
        else if (arg == "--min-groups"     && i+1<argc) train_opts_base.min_groups = std::stoi(argv[++i]);
        else if (arg == "--class-bal"      && i+1<argc) enc_opts_base.class_bal = argv[++i];
        else if (arg == "--drop-major-allele") enc_opts_base.drop_major = true;
        else if (arg == "--minor-column") enc_opts_base.minor_column = true;
        else if (arg == "--tiered-minor-col") enc_opts_base.tiered_minor_col = true;
        else if (arg == "--auto-bit-ct"    && i+1<argc) enc_opts_base.auto_bit_ct = std::stod(argv[++i]);
        else if (arg == "--max-mem"         && i+1<argc) enc_opts_base.max_mem = std::stoull(argv[++i]);
        else if (arg == "--group-penalty-type" && i+1<argc) train_opts_base.group_penalty_type = argv[++i];
        else if (arg == "--initial-gp-value"   && i+1<argc) train_opts_base.initial_gp_value = std::stod(argv[++i]);
        else if (arg == "--final-gp-value"     && i+1<argc) train_opts_base.final_gp_value   = std::stod(argv[++i]);
        else if (arg == "--gp-step"            && i+1<argc) train_opts_base.gp_step          = std::stod(argv[++i]);
        else if (arg == "--feature-normalize"  && i+1<argc) enc_opts_base.feature_normalize = argv[++i];
        else std::cerr << "Warning: unknown aim argument '" << arg << "', ignoring\n";
    }
    if (enc_opts_base.feature_normalize == "auto")
        enc_opts_base.feature_normalize = (pre_opts.datatype == "numeric") ? "slep" : "none";
    if (enc_opts_base.feature_normalize != "none") {
        if (enc_opts_base.feature_normalize != "center" && enc_opts_base.feature_normalize != "zscore" &&
            enc_opts_base.feature_normalize != "slep")
            throw std::runtime_error("--feature-normalize must be one of: none, center, zscore, slep");
        if (pre_opts.datatype != "numeric" && pre_opts.datatype != "vcf")
            throw std::runtime_error(
                "--feature-normalize is only supported with --datatype numeric or vcf");
    }
    // Resolve deprecated method aliases
    {
        bool was_alias = false;
        auto resolved = regression::resolve_method_alias(train_opts_base.method, &was_alias);
        if (was_alias) {
            std::cerr << "Note: --method " << train_opts_base.method
                      << " is deprecated, use " << resolved
                      << " (may stop working in a future release)\n";
            train_opts_base.method = resolved;
        }
    }
    {
        std::string gpt = train_opts_base.group_penalty_type;
        for (auto& c : gpt) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (gpt != "std" && gpt != "sqrt" && gpt != "linear" && gpt != "median")
            throw std::runtime_error("--group-penalty-type must be one of: std, sqrt, linear, median");
    }
    if (het_mode_set) validate_het_mode(pre_opts.het_mode, pre_opts.datatype);
    // The per-gene minor-allele summary columns are produced by the FASTA
    // encoder only; VCF input goes through the float-matrix path where they
    // have no implementation. Reject rather than silently ignore.
    if (pre_opts.datatype == "vcf" && (enc_opts_base.minor_column || enc_opts_base.tiered_minor_col))
        throw std::runtime_error(
            "--minor-column / --tiered-minor-col are not supported with --datatype vcf");
    if (enc_opts_base.minor_column && enc_opts_base.tiered_minor_col)
        throw std::runtime_error("--minor-column and --tiered-minor-col are mutually exclusive");
    if (train_opts_base.params.count("disable_mc") && train_opts_base.params.at("disable_mc") == "1")
        enc_opts_base.disable_mc = true;
    if (!has_lambda) { train_opts_base.lambda_grid_specs[0]="0.1,0.9,0.1"; train_opts_base.lambda_grid_specs[1]="0.0001,0.0002,0.0001"; train_opts_base.lambda_grid_set=true; }
    if (!has_method) train_opts_base.method = "sg_lasso_logisticr";

    // Lambda count for peak memory estimation
    if (train_opts_base.lambda_grid_set) {
        auto count_steps = [](const std::string& spec) -> uint32_t {
            double vmin, vmax, vstep; char c1, c2;
            std::istringstream ss(spec);
            if (!(ss >> vmin >> c1 >> vmax >> c2 >> vstep) || vstep <= 0) return 1;
            uint32_t n = 0;
            for (double v = vmin; v <= vmax + vstep * 1e-9; v += vstep) ++n;
            return n ? n : 1;
        };
        enc_opts_base.lambda_count = count_steps(train_opts_base.lambda_grid_specs[0])
                                   * count_steps(train_opts_base.lambda_grid_specs[1]);
    } else if (!train_opts_base.lambda_file_path.empty()) {
        std::ifstream lf(train_opts_base.lambda_file_path);
        uint32_t n = 0; std::string line;
        while (std::getline(lf, line))
            if (!line.empty() && line[0] != '#') ++n;
        enc_opts_base.lambda_count = n ? n : 1;
    }

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

        // Read feature weights — check penalty subdirs when multi-penalty output exists
        std::vector<std::pair<double,std::string>> feature_rank;
        fs::path aim_base = fs::exists(iter_dir / "penalty_0") ? iter_dir / "penalty_0" : iter_dir;
        fs::path bss_file   = aim_base / "bss_median.txt";
        fs::path lam0_wfile = aim_base / "lambda_0" / "weights.txt";
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

    return 0;
}

int run_psc(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Error: psc requires <alignments_dir> <output_dir>\n";
        print_help(argv[0], "psc");
        return 1;
    }

    pipeline::psc::PscOptions psc_opts;
    psc_opts.alignments_dir = argv[2];
    psc_opts.output_dir     = argv[3];

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        // Input
        if      (arg == "--alignments-list"   && i+1<argc) psc_opts.alignments_list_file = argv[++i];
        // Species contrast source
        else if (arg == "--species-groups"    && i+1<argc) psc_opts.species_groups_file = argv[++i];
        else if (arg == "--response-file"     && i+1<argc) psc_opts.response_file       = argv[++i];
        else if (arg == "--response-dir"      && i+1<argc) psc_opts.response_dir        = argv[++i];
        // AUTO-PAIRS DISABLED — see CLAUDE.md "Disabled features". Re-enable by uncommenting:
        // else if (arg == "--auto-pairs-tree"   && i+1<argc) psc_opts.auto_pairs_tree     = argv[++i];
        // Lambda grid
        else if (arg == "--initial-lambda1"   && i+1<argc) psc_opts.initial_lambda1 = std::stod(argv[++i]);
        else if (arg == "--final-lambda1"     && i+1<argc) psc_opts.final_lambda1   = std::stod(argv[++i]);
        else if (arg == "--initial-lambda2"   && i+1<argc) psc_opts.initial_lambda2 = std::stod(argv[++i]);
        else if (arg == "--final-lambda2"     && i+1<argc) psc_opts.final_lambda2   = std::stod(argv[++i]);
        else if (arg == "--lambda-step"       && i+1<argc) psc_opts.lambda_step     = std::stod(argv[++i]);
        else if (arg == "--use-logspace")                   psc_opts.use_logspace    = true;
        else if (arg == "--num-log-points"    && i+1<argc) psc_opts.num_log_points  = static_cast<size_t>(std::stoi(argv[++i]));
        // Group penalty
        else if (arg == "--group-penalty-type" && i+1<argc) psc_opts.group_penalty_type = argv[++i];
        else if (arg == "--initial-gp-value"  && i+1<argc) psc_opts.initial_gp_value = std::stod(argv[++i]);
        else if (arg == "--final-gp-value"    && i+1<argc) psc_opts.final_gp_value   = std::stod(argv[++i]);
        else if (arg == "--gp-step"           && i+1<argc) psc_opts.gp_step          = std::stod(argv[++i]);
        else if (arg == "--use-default-gp")                 psc_opts.use_default_gp  = true;
        // Gap cancellation
        else if (arg == "--use-uncanceled-alignments")      psc_opts.use_uncanceled_alignments = true;
        else if (arg == "--cancel-only-partner")            psc_opts.cancel_only_partner       = true;
        else if (arg == "--cancel-tri-allelic")             psc_opts.cancel_tri_allelic        = true;
        else if (arg == "--nix-full-deletions")             psc_opts.nix_full_deletions        = true;
        else if (arg == "--outgroup-species"   && i+1<argc) psc_opts.outgroup_species          = argv[++i];
        else if (arg == "--min-pairs"          && i+1<argc) psc_opts.min_pairs = static_cast<size_t>(std::stoi(argv[++i]));
        // Solver
        else if (arg == "--method"             && i+1<argc) psc_opts.method         = argv[++i];
        else if (arg == "--precision"          && i+1<argc) {
            std::string p = argv[++i];
            if (p != "fp32" && p != "fp64") throw std::runtime_error("--precision must be fp32 or fp64");
            psc_opts.precision_str = p;
        }
        else if (arg == "--maxiter"            && i+1<argc) psc_opts.maxiter  = std::stoi(argv[++i]);
        else if (arg == "--threads"            && i+1<argc) psc_opts.threads  = static_cast<unsigned>(std::stoi(argv[++i]));
        else if (arg == "--param"              && i+1<argc) {
            std::string kv = argv[++i];
            auto eq = kv.find('=');
            if (eq != std::string::npos)
                psc_opts.params[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        // Output/prediction
        else if (arg == "--output-base-name"          && i+1<argc) psc_opts.output_base_name         = argv[++i];
        else if (arg == "--prediction-alignments-dir" && i+1<argc) psc_opts.prediction_alignments_dir = argv[++i];
        else if (arg == "--species-pheno-path"        && i+1<argc) psc_opts.species_pheno_path        = argv[++i];
        else if (arg == "--no-pred-output")                         psc_opts.no_pred_output            = true;
        else if (arg == "--no-genes-output")                        psc_opts.no_genes_output           = true;
        else if (arg == "--show-selected-sites")                    psc_opts.show_selected_sites       = true;
        else if (arg == "--dump-weights")                           psc_opts.dump_weights              = true;
        else if (arg == "--top-rank-frac"             && i+1<argc) psc_opts.top_rank_frac             = std::stod(argv[++i]);
        else if (arg == "--limited-genes-list"        && i+1<argc) psc_opts.limited_genes_list        = argv[++i];
        // Null models
        else if (arg == "--make-null-models")                        psc_opts.make_null_models                    = true;
        else if (arg == "--make-pair-randomized-null-models")        psc_opts.make_pair_randomized_null_models    = true;
        else if (arg == "--num-randomized-alignments" && i+1<argc) psc_opts.num_randomized_alignments = static_cast<size_t>(std::stoi(argv[++i]));
        // AUTO-PAIRS DISABLED — see CLAUDE.md "Disabled features". Re-enable by uncommenting the four lines below:
        // // Auto-pairs
        // else if (arg == "--auto-pairs-method"         && i+1<argc) psc_opts.auto_pairs_method          = argv[++i];
        // else if (arg == "--auto-pairs-num-alternates" && i+1<argc) psc_opts.auto_pairs_num_alternates  = std::stoi(argv[++i]);
        // else if (arg == "--auto-pairs-max-combinations" && i+1<argc) psc_opts.auto_pairs_max_combinations = std::stoi(argv[++i]);
        else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
    }

    // Validate: exactly one contrast source
    int sources = 0;
    if (!psc_opts.species_groups_file.empty()) ++sources;
    if (!psc_opts.response_file.empty())       ++sources;
    if (!psc_opts.response_dir.empty())        ++sources;
    // AUTO-PAIRS DISABLED — see CLAUDE.md "Disabled features". Re-enable by uncommenting:
    // if (!psc_opts.auto_pairs_tree.empty())     ++sources;
    if (sources == 0)
        throw std::runtime_error("PSC requires one of: --species-groups, --response-file, --response-dir");
    if (sources > 1)
        throw std::runtime_error("PSC: specify only one of --species-groups, --response-file, --response-dir");

    if (psc_opts.output_base_name.empty())
        throw std::runtime_error("PSC requires --output-base-name");

    // Resolve deprecated method aliases
    {
        bool was_alias = false;
        auto resolved = regression::resolve_method_alias(psc_opts.method, &was_alias);
        if (was_alias) {
            std::cerr << "Note: --method " << psc_opts.method
                      << " is deprecated, use " << resolved
                      << " (may stop working in a future release)\n";
            psc_opts.method = resolved;
        }
    }

    pipeline::psc::run_psc(psc_opts);

    return 0;
}

int run_encode_sizes(int argc, char* argv[]) {
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
        else if (arg == "--minor-column")                  enc_opts.minor_column = true;
        else if (arg == "--tiered-minor-col")              enc_opts.tiered_minor_col = true;
        else if (arg == "--dropout"         && i+1<argc) {
            enc_opts.dropout_labels = load_dropout_labels(argv[++i]);
        }
        else std::cerr << "Warning: unknown argument '" << arg << "', ignoring\n";
    }
    if (enc_opts.minor_column && enc_opts.tiered_minor_col)
        throw std::runtime_error("--minor-column and --tiered-minor-col are mutually exclusive");
    auto sizes = pipeline::encode_sizes(enc_opts);
    uint64_t total = 0;
    for (auto& [stem, ncols] : sizes) {
        std::cout << stem << '\t' << ncols << '\n';
        total += ncols;
    }
    std::cout << "\nFiles: " << sizes.size() << "  Total columns: " << total << "\n";

    return 0;
}

int run_visualize(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Error: visualize requires <gene_predictions.txt> <output.svg>\n";
        print_help(argv[0], "visualize");
        return 1;
    }
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
    if (svg_path.has_parent_path())
        fs::create_directories(svg_path.parent_path());
    viz::write_svg(gpt, svg_path, vopts);
    std::cout << "Visualization written -> " << svg_path.string() << "\n";

    return 0;
}

static bool is_help_flag(const std::string& s) {
    return s == "--help" || s == "-h" || s == "-?" || s == "/?";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            print_overview(argv[0]);
            return 0;
        }
        std::string command = argv[1];

        // `myesl2 --help` or `myesl2 --help <command>`
        if (is_help_flag(command)) {
            if (argc >= 3) print_help(argv[0], argv[2]);
            else           print_overview(argv[0]);
            return 0;
        }

        // `myesl2 <command> --help` (or -h, anywhere in argv)
        for (int i = 2; i < argc; ++i) {
            if (is_help_flag(argv[i])) {
                print_help(argv[0], command);
                return 0;
            }
        }

        if      (command == "train")        return run_train(argc, argv);
        else if (command == "evaluate")     return run_evaluate(argc, argv);
        else if (command == "info")         return run_info(argc, argv);
        else if (command == "drphylo")      return run_drphylo(argc, argv);
        else if (command == "aim")          return run_aim(argc, argv);
        else if (command == "psc")          return run_psc(argc, argv);
        else if (command == "encode-sizes") return run_encode_sizes(argc, argv);
        else if (command == "visualize")    return run_visualize(argc, argv);
        else if (command == "taskfile")     return run_taskfile(argc, argv);
        else {
            std::cerr << "Error: unknown command '" << command << "'\n";
            print_overview(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
